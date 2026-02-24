"""A module for executing a planner on a scenario."""
import os.path
import pickle
from typing import List

import cv2
import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.util import Interval
from commonroad.geometry.shape import ShapeGroup, Polygon
from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblemSet, PlanningProblem
from commonroad.scenario.trajectory import State

from planner.utils.timers import ExecTimer
from planner.utils.vehicleparams import VehicleParameters
from planner.utils.timeout import Timeout
from planner.Frenet.utils.visualization import draw_global_map
from agent_sim.agent import clean_scenario, update_scenario

from planner.planning import (
    PlanningAgent,
    add_ego_vehicles_to_scenario,
)

import copy
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import (
    create_collision_checker,
    create_collision_object,
)
from commonroad_helper_functions.exceptions import NoLocalTrajectoryFoundError
from risk_assessment.harm_estimation import harm_model
from risk_assessment.visualization.collision_visualization import (
    collision_vis,
)
from risk_assessment.helpers.collision_helper_function import (
    angle_range,
)
from risk_assessment.utils.logistic_regression_symmetrical import get_protected_inj_prob_log_reg_ignore_angle
from planner.Frenet.utils.helper_functions import create_tvobstacle
from planner.Frenet.configs.load_json import load_harm_parameter_json
from commonroad.scenario.obstacle import (
    ObstacleRole,
    ObstacleType,
)

from planner.Frenet.utils.traj_evaluate import TrajLogger


class ScenarioHandler:
    """Generic class for looping a scenario with a planner."""

    def __init__(
        self,
        planner_creator,
        vehicle_type,
        path_to_scenarios,
        log_path,
        collision_report_path,
        timing_enabled=False,
        active_learning=False,
        prepare_num_data=False,
        label_path=None,
        fig_path=None,
        traverse_future_points=30,
    ):
        """Create scenario handler.

        Args:
            evaluation_settings (dict): Documentation in work ;)
            collision_report_path (Str): Path to save collision reports.
        """
        self.planner_creator = planner_creator
        self.vehicle_type = vehicle_type
        self.exec_timer = ExecTimer(timing_enabled=timing_enabled)
        self.timing_enabled = timing_enabled
        self.active_learning = active_learning
        self.prepare_num_data = prepare_num_data
        self.path_to_scenarios = path_to_scenarios

        self.log_path = log_path
        self.collision_report_path = collision_report_path
        self.label_path = label_path
        self.fig_path = fig_path
        self.scenario_path = None
        self.scenario = None
        self.planning_problem_set = None
        self.vehicle_params = None
        self.agent_list = None
        self.vel_list = []
        self.bev = None
        self.harm = {
            "Ego": 0.0,
            "Unknown": 0.0,
            "Car": 0.0,
            "Truck": 0.0,
            "Bus": 0.0,
            "Bicycle": 0.0,
            "Pedestrian": 0.0,
            "Priority_vehicle": 0.0,
            "Parked_vehicle": 0.0,
            "Construction_zone": 0.0,
            "Train": 0.0,
            "Road_boundary": 0.0,
            "Motorcycle": 0.0,
            "Taxi": 0.0,
            "Building": 0.0,
            "Pillar": 0.0,
            "Median_strip": 0.0,
            "Total": 0.0,
        }

    def _initialize(self):
        """WIP."""
        with self.exec_timer.time_with_cm("read scenario"):
            self.scenario, self.planning_problem_set = CommonRoadFileReader(
                self.scenario_path
            ).open()
        if self.active_learning:
            # set planning problem from states in trajectory
            self.planning_problem_set = PlanningProblemSet()
            for obs in self.scenario.dynamic_obstacles:
                initial_state = State(position=obs.initial_state.position,
                                      time_step=obs.initial_state.time_step,
                                      velocity=obs.initial_state.velocity,
                                      orientation=obs.initial_state.orientation,
                                      acceleration=0.0,
                                      yaw_rate=0.0,
                                      slip_angle=0.0)
                if isinstance(obs.prediction.occupancy_set[-1].shape, ShapeGroup):
                    goal_state_shape = obs.prediction.occupancy_set[-1].shape.shapes[0].vertices
                else:
                    goal_state_shape = obs.prediction.occupancy_set[-1].shape.vertices
                # goal_state = obs.prediction.trajectory.final_state
                goal_state_time = obs.prediction.trajectory.final_state.time_step
                goal_state_velocity_Interval = Interval(-2.3652294, 10.634771)
                goal_state = State(position=Polygon(goal_state_shape),
                                   time_step=Interval(goal_state_time-1, goal_state_time),
                                   velocity=goal_state_velocity_Interval,
                                   )
                goal_region = GoalRegion(state_list=[goal_state])
                new_planning_problem = PlanningProblem(
                    planning_problem_id=obs.obstacle_id + 70000, initial_state=initial_state, goal_region=goal_region
                )
                self.planning_problem_set.add_planning_problem(new_planning_problem)

        with self.exec_timer.time_with_cm("read vehicle parameters"):
            # get the parameters of the vehicle
            self.vehicle_params = VehicleParameters(self.vehicle_type)


        with self.exec_timer.time_with_cm("add vehicle to scenario"):
            (
                self.scenario,
                self.agent_planning_problem_id_assignment
            ) = add_ego_vehicles_to_scenario(
                scenario=self.scenario,
                planning_problem_set=self.planning_problem_set,
                vehicle_params=self.vehicle_params,
                active_learning=self.active_learning
            )

        self.agent_list = []

        for dynamic_obstacle in self.scenario.dynamic_obstacles:

            if (
                dynamic_obstacle.obstacle_id
                in self.agent_planning_problem_id_assignment
            ):
                self._create_planner_agent_for_ego_vehicle(dynamic_obstacle.obstacle_id)


        if self.active_learning:
            traj_length = 10
            if len(self.agent_list) != 0:
                traj_length = self.agent_list[0].planner.traj_length
                self.traverse_future_steps = traj_length * 3
            self.search_traj_logger = TrajLogger(log_prefix="scenario search tree traj logger", points_num=traj_length * 3)
            self.ground_truth_traj_logger = TrajLogger(log_prefix="scenario ground truth traj logger", points_num=traj_length * 3)
            for dynamic_obstacle in self.scenario.dynamic_obstacles:
                for s in dynamic_obstacle.prediction.trajectory.state_list:
                    if (
                            len(
                                self.scenario.lanelet_network.find_lanelet_by_position(
                                    [s.position]
                                )[0]
                            )
                            > 0
                    ):
                        lanelet_id = (
                            self.scenario.lanelet_network.find_lanelet_by_position(
                                [s.position]
                            )[0][0]
                        )
                    if lanelet_id is not None:
                        if (
                            self.scenario.lanelet_network.find_lanelet_by_id(
                                lanelet_id
                            ).dynamic_obstacles_on_lanelet.get(s.time_step)
                            is None
                        ):
                            self.scenario.lanelet_network.find_lanelet_by_id(
                                lanelet_id
                            ).dynamic_obstacles_on_lanelet[s.time_step] = set()
                        self.scenario.lanelet_network.find_lanelet_by_id(
                            lanelet_id
                        ).dynamic_obstacles_on_lanelet[s.time_step].add(dynamic_obstacle.obstacle_id)


        if self.prepare_num_data:
            obs_data = {}
            for agent in self.agent_list:
                obs_data.setdefault(agent.agent_id, agent.planner.numerical_dict)
            with open("../../data/{}/numerical_data.pkl".format(self.scenario.scenario_id), "wb") as f:
                pickle.dump(obs_data, f)

        if self.label_path is not None:
            label_scenario_path = os.path.join(self.label_path, self.scenario.benchmark_id)
            if not os.path.exists(label_scenario_path):
                os.makedirs(label_scenario_path)
            self.label_path = label_scenario_path
        if self.fig_path is not None:
            fig_scenario_path = os.path.join(self.fig_path, self.scenario.benchmark_id)
            if not os.path.exists(fig_scenario_path):
                os.makedirs(fig_scenario_path)
            global_path = os.path.join(fig_scenario_path, "global_path")
            hist_path = os.path.join(fig_scenario_path, "hist_traj")
            if not os.path.exists(global_path):
                os.makedirs(global_path)
            if not os.path.exists(hist_path):
                os.makedirs(hist_path)
            self.fig_path = fig_scenario_path
        # create a collision checker
        # remove the ego vehicle from the scenario
        with self.exec_timer.time_with_cm(
            "initialization/initialize collision checker"
        ):
            cc_scenario = copy.deepcopy(self.scenario)
            for agent in self.agent_list:
                cc_scenario.remove_obstacle(
                    obstacle=[cc_scenario.obstacle_by_id(agent.agent_id)]
                )
            try:
                self.collision_checker = create_collision_checker(cc_scenario)
            except Exception:
                raise BrokenPipeError("Collision Checker fails.") from None

    def _simulate(self):
        """WIP.

        Raises:
            Exception: [description]
        """
        if len(self.agent_list) == 0:
            return

        with Timeout(30, "Simulation preparation"):
            if not self.active_learning:
                self.scenario = clean_scenario(
                    scenario=self.scenario, agent_list=self.agent_list
                )

            # get the max time for the simulation
            max_time_steps = max([
                agent.planner.planning_problem.goal.state_list[0]
                .time_step.end for agent in self.agent_list]
            )
            # run the simulation not longer to avoid simulating forever
            max_simulation_time_steps = int(max_time_steps) if self.active_learning else int(max_time_steps * 2.0)

        if self.label_path is not None:
            label_count = 0
        if self.fig_path is not None:
            fig_count = 0

        # draw_global_map(self.scenario)
        for time_step in range(max_simulation_time_steps):
            for agent in self.agent_list:
                # Also pass in timestep because it is needed in the recorder
                # That gets derived from the evaluator
                if not self.active_learning:
                    self._check_collision(agent=agent, time_step=time_step)
                self._do_simulation_step(agent=agent, time_step=time_step)

            if self.label_path is not None:
                if time_step == (max_simulation_time_steps - self.traverse_future_steps):
                    for agent in self.agent_list:
                        for maps in agent.planner.saved_maps:
                            for i, map in enumerate(maps):
                                map_path = os.path.join(self.label_path, self.scenario.benchmark_id + "_" +
                                                        str(label_count).zfill(8)+"_"+str(i)+".png")
                                map.save_map_vertical(map_path)
                            label_count += 1
            if self.fig_path is not None:
                if time_step == (max_simulation_time_steps - self.traverse_future_steps):
                    for agent in self.agent_list:
                        for i in range(len(agent.planner.saved_global_path)):
                            global_p_img = agent.planner.saved_global_path[i][0]
                            img_path = os.path.join(self.fig_path, "global_path", self.scenario.benchmark_id + "_" +
                                                    str(fig_count).zfill(8)+".png")
                            img = global_p_img.astype(np.int16)
                            cv2.imwrite(filename=img_path, img=img)
                            hist_t_img = agent.planner.saved_hist_traj[i][0]
                            img_path = os.path.join(self.fig_path, "hist_traj", self.scenario.benchmark_id + "_" +
                                                    str(fig_count).zfill(8) + ".png")
                            img = hist_t_img.astype(np.int16)
                            cv2.imwrite(filename=img_path, img=img)
                            fig_count += 1

            # stop if the max_simulation_time is reached and no reason was found
            if time_step == (max_simulation_time_steps - 1):
                if self.active_learning:
                    for agent in self.agent_list:
                        self.search_traj_logger.extend_logger(agent.planner.search_traj_logger)
                        self.ground_truth_traj_logger.extend_logger(agent.planner.gt_traj_logger)
                    self.search_traj_logger.log_average_traj_data()
                    self.ground_truth_traj_logger.log_average_traj_data()
                if not self.active_learning:
                    raise Exception("Goal was not reached in time")

            # update the scenario with new states of the agents
            self.scenario = update_scenario(
                scenario=self.scenario, agent_list=self.agent_list
            )

    def _do_simulation_step(self, **kwargs):
        # This does the planning of the trajectory nothing else
        agent = kwargs["agent"]
        agent.step(scenario=self.scenario)

    def _create_planner_agent_for_ego_vehicle(self, ego_vehicle_id):
        # TimeOut 10 seconds to create a planner
        with Timeout(1000, "Creation of planner object"):

            planner = self.planner_creator.get_planner(
                scenario_handler=self,
                ego_vehicle_id=ego_vehicle_id,
            )

        with Timeout(10000, "Creation of agent object"):
            self.agent_list.append(
                PlanningAgent(
                    scenario=self.scenario,
                    agent_id=ego_vehicle_id,
                    planner=planner,
                    control_dynamics=None,
                    enable_logging=False,
                    log_path=self.log_path,
                    debug_step=False,
                )
            )

    def _check_collision(self, agent, time_step):
        # check if the current state is collision-free
        vis = self.planner_creator.settings["risk_dict"]
        coeffs = load_harm_parameter_json()

        # get ego position and orientation
        if self.active_learning:
            try:
                ego_pos = self.scenario.obstacle_by_id(obstacle_id=agent.agent_id).prediction.trajectory.state_list[time_step].position
            # ego_pos = (
            #     self.scenario.obstacle_by_id(obstacle_id=agent.agent_id)
            #     .occupancy_at_time(time_step=time_step)
            #     .shape.center
            # )
            # print(ego_pos)
            except Exception as e:
                print(e)
        else:
            try:
                ego_pos = self.scenario.obstacle_by_id(obstacle_id=agent.agent_id).prediction.trajectory.state_list[-1].position
            # ego_pos = (
            #     self.scenario.obstacle_by_id(obstacle_id=agent.agent_id)
            #     .occupancy_at_time(time_step=time_step)
            #     .shape.center
            # )
            # print(ego_pos)
            except Exception as e:
                print(e)

        road_boundary = agent.planner.road_boundary

        if time_step == 0:
            ego_vel = self.scenario.obstacle_by_id(
                obstacle_id=agent.agent_id
            ).initial_state.velocity
            ego_yaw = self.scenario.obstacle_by_id(
                obstacle_id=agent.agent_id
            ).initial_state.orientation

            self.vel_list.append(ego_vel)
        else:
            try:
                ego_pos_last = (
                    self.scenario.obstacle_by_id(obstacle_id=agent.agent_id)
                    .occupancy_at_time(time_step=time_step - 1)
                    .shape.center
                )
            except:
                print("hello")
            delta_ego_pos = ego_pos - ego_pos_last

            ego_vel = np.linalg.norm(delta_ego_pos) / agent.dt

            self.vel_list.append(ego_vel)

            ego_yaw = np.arctan2(delta_ego_pos[1], delta_ego_pos[0])

        current_state_collision_object = create_tvobstacle(
            traj_list=[
                [
                    ego_pos[0],
                    ego_pos[1],
                    ego_yaw,
                ]
            ],
            box_length=self.vehicle_params.l / 2,
            box_width=self.vehicle_params.w / 2,
            start_time_step=time_step,
        )

        # Add road boundary to collision checker
        self.collision_checker.add_collision_object(road_boundary)

        if not self.collision_checker.collide(current_state_collision_object):
            return

        # get the colliding obstacle
        obs_id = None
        for obs in self.scenario.obstacles:
            co = create_collision_object(obs)
            if current_state_collision_object.collide(co):
                if obs.obstacle_id != agent.agent_id:
                    if obs_id is None:
                        obs_id = obs.obstacle_id
                    else:
                        raise Exception("More than one collision detected")

        # Collisoin with boundary
        if obs_id is None:
            self.harm["Ego"] = get_protected_inj_prob_log_reg_ignore_angle(
                velocity=ego_vel, coeff=coeffs
            )
            self.harm["Total"] = self.harm["Ego"]

            raise NoLocalTrajectoryFoundError("Collision with road boundary. (Harm: {:.2f})".format(self.harm["Ego"]))

        # get information of colliding obstace
        obs_pos = (
            self.scenario.obstacle_by_id(obstacle_id=obs_id)
            .occupancy_at_time(time_step=time_step)
            .shape.center
        )
        obs_pos_last = (
            self.scenario.obstacle_by_id(obstacle_id=obs_id)
            .occupancy_at_time(time_step=time_step - 1)
            .shape.center
        )
        obs_size = (
            self.scenario.obstacle_by_id(obstacle_id=obs_id).obstacle_shape.length
            * self.scenario.obstacle_by_id(obstacle_id=obs_id).obstacle_shape.width
        )

        # filter initial collisions
        if time_step < 1:
            raise ValueError("Collision at initial state")

        if (
            self.scenario.obstacle_by_id(obstacle_id=obs_id).obstacle_role
            == ObstacleRole.ENVIRONMENT
        ):
            obs_vel = 0
            obs_yaw = 0
        else:
            pos_delta = obs_pos - obs_pos_last

            obs_vel = np.linalg.norm(pos_delta) / agent.dt
            if (
                self.scenario.obstacle_by_id(obstacle_id=obs_id).obstacle_role
                == ObstacleRole.DYNAMIC
            ):
                obs_yaw = np.arctan2(pos_delta[1], pos_delta[0])
            else:
                obs_yaw = self.scenario.obstacle_by_id(
                    obstacle_id=obs_id
                ).initial_state.orientation

        # calculate crash angle
        pdof = angle_range(obs_yaw - ego_yaw + np.pi)
        rel_angle = np.arctan2(
            obs_pos_last[1] - ego_pos_last[1], obs_pos_last[0] - ego_pos_last[0]
        )
        ego_angle = angle_range(rel_angle - ego_yaw)
        obs_angle = angle_range(np.pi + rel_angle - obs_yaw)

        # calculate harm
        self.ego_harm, self.obs_harm, ego_obj, obs_obj = harm_model(
            scenario=self.scenario,
            ego_vehicle_id=agent.agent_id,
            vehicle_params=self.vehicle_params,
            ego_velocity=ego_vel,
            ego_yaw=ego_yaw,
            obstacle_id=obs_id,
            obstacle_size=obs_size,
            obstacle_velocity=obs_vel,
            obstacle_yaw=obs_yaw,
            pdof=pdof,
            ego_angle=ego_angle,
            obs_angle=obs_angle,
            modes=vis,
            coeffs=coeffs,
        )

        # if collision report should be shown
        if vis["collision_report"]:
            collision_vis(
                scenario=self.scenario,
                destination=self.collision_report_path,
                ego_harm=self.ego_harm,
                ego_type=ego_obj.type,
                ego_v=ego_vel,
                ego_mass=ego_obj.mass,
                obs_harm=self.obs_harm,
                obs_type=obs_obj.type,
                obs_v=obs_vel,
                obs_mass=obs_obj.mass,
                pdof=pdof,
                ego_angle=ego_angle,
                obs_angle=obs_angle,
                time_step=time_step,
                modes=vis,
                marked_vehicle=agent.agent_id,
                planning_problem=agent.planner.planning_problem,
                global_path=None,
                driven_traj=None,
            )

        # add ego harm to dict
        self.harm["Ego"] = self.ego_harm
        self.harm["Total"] = self.ego_harm + self.obs_harm

        # add obstacle harm of respective type to dict
        if obs_obj.type is ObstacleType.UNKNOWN:  # worst case assumption
            self.harm["Pedestrian"] = self.obs_harm
        elif obs_obj.type is ObstacleType.CAR:
            self.harm["Car"] = self.obs_harm
        elif obs_obj.type is ObstacleType.BUS:
            self.harm["Bus"] = self.obs_harm
        elif obs_obj.type is ObstacleType.TRUCK:
            self.harm["Truck"] = self.obs_harm
        elif obs_obj.type is ObstacleType.BICYCLE:
            self.harm["Bicycle"] = self.obs_harm
        elif obs_obj.type is ObstacleType.PEDESTRIAN:
            self.harm["Pedestrian"] = self.obs_harm
        elif obs_obj.type is ObstacleType.PRIORITY_VEHICLE:
            self.harm["Priority_vehicle"] = self.obs_harm
        elif obs_obj.type is ObstacleType.PARKED_VEHICLE:
            self.harm["Parked_vehicle"] = self.obs_harm
        elif obs_obj.type is ObstacleType.CONSTRUCTION_ZONE:
            self.harm["Construction_zone"] = self.obs_harm
        elif obs_obj.type is ObstacleType.TRAIN:
            self.harm["Train"] = self.obs_harm
        elif obs_obj.type is ObstacleType.ROAD_BOUNDARY:
            self.harm["Road_boundary"] = self.obs_harm
        elif obs_obj.type is ObstacleType.MOTORCYCLE:
            self.harm["Motorcycle"] = self.obs_harm
        elif obs_obj.type is ObstacleType.TAXI:
            self.harm["Taxi"] = self.obs_harm
        elif obs_obj.type is ObstacleType.BUILDING:
            self.harm["Building"] = self.obs_harm
        elif obs_obj.type is ObstacleType.PILLAR:
            self.harm["Pillar"] = self.obs_harm
        elif obs_obj.type is ObstacleType.MEDIAN_STRIP:
            self.harm["Median_strip"] = self.obs_harm
        else:
            raise AttributeError("Error in obstacle type")

        raise NoLocalTrajectoryFoundError("Collision in the driven path with {0}. Total harm: {1:.2f}".format(obs_obj.type, self.ego_harm + self.obs_harm))




class PlannerCreator:
    """Class for constructing a planner object from a Handler object."""

    def __init__(self):
        """__init__ function to constuct the object.

        This function is called from the user.
        """

    def get_planner(self, scenario_handler, ego_vehicle_id):
        """Create the planner from the scenario handler object.

        Args:
            scenario_handler (obj): scenario handler object

        Raises:
            NotImplementedError: Abstract Method

        Returns:
            obj: a planner object.
        """
        raise NotImplementedError("Overwrite this method.")

    @staticmethod
    def get_blacklist():
        """Create the blacklist for the planner.

        Raises:
            NotImplementedError: Abstract Method

        Returns:
            list(str): scenario blacklist
        """
        raise NotImplementedError("Overwrite this method.")
