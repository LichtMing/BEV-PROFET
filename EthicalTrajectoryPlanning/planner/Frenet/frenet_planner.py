#!/user/bin/env python

"""Sampling-based trajectory planning in a frenet frame considering ethical implications."""

import copy
import json
# Standard imports
import os
import pathlib
import sys
import time
import warnings
from datetime import datetime
from inspect import currentframe, getframeinfo
from pyinstrument import Profiler
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Third party imports
import numpy as np
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import State
from commonroad_dc.boundary import boundary
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import (
    create_collision_checker,
)
from commonroad_helper_functions.exceptions import (
    ExecutionTimeoutError,
)
from commonroad_helper_functions.sensor_model import get_visible_objects
from prediction import WaleNet

from utils.traj_evaluate import find_best_traj, TrajLogger, calc_eval_values

# Custom imports
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

mopl_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(mopl_path)

from planner.planning import Planner
from planner.utils.timeout import Timeout
from planner.Frenet.utils.visualization import draw_frenet_trajectories, draw_bev_map, get_current_graphs, \
    draw_future_situations
from planner.Frenet.utils.validity_checks import (
    VALIDITY_LEVELS, check_validity, create_collision_object, boundary_valid,
)
from planner.Frenet.utils.helper_functions import (
    get_goal_area_shape_group,
)
from planner.Frenet.utils.prediction_helpers import (
    add_static_obstacle_to_prediction,
    get_dyn_and_stat_obstacles,
    get_ground_truth_prediction,
    get_obstacles_in_radius,
    get_orientation_velocity_and_shape_of_prediction,
)
from planner.Frenet.configs.load_json import (
    load_harm_parameter_json,
    load_planning_json,
    load_risk_json,
    load_weight_json,
)
from planner.Frenet.utils.frenet_functions import (
    calc_frenet_trajectories,
    get_v_list,
    sort_frenet_trajectories, calc_frenet_trajectories_combination_based,
)
from planner.Frenet.utils.frenet_logging import FrenetLogging
from planner.utils.responsibility import assign_responsibility_by_action_space
from planner.utils import (reachable_set)
from planner.Frenet.utils.InteractionMap import InteractionMap
from planner.Frenet.utils.BehaviorTree import BehaviorTree
from planner.Frenet.utils.bev_prob_loader import BEVProbLoader

from risk_assessment.visualization.risk_visualization import (
    create_risk_files,
)
from risk_assessment.visualization.risk_dashboard import risk_dashboard


class FrenetPlanner(Planner):
    """Jerk optimal planning in frenet coordinates with quintic polynomials in lateral direction and quartic polynomials in longitudinal direction."""

    def __init__(
            self,
            scenario: Scenario,
            planning_problem: PlanningProblem,
            ego_id: int,
            vehicle_params,
            mode,
            exec_timer=None,
            frenet_parameters: dict = None,
            sensor_radius: float = 50.0,
            plot_frenet_trajectories: bool = False,
            active_learning=False,
            weights=None,
            settings=None,
    ):
        """
        Initialize a frenét planner.

        Args:
            scenario (Scenario): Scenario.
            planning_problem (PlanningProblem): Given planning problem.
            ego_id (int): ID of the ego vehicle.
            vehicle_params (VehicleParameters): Parameters of the ego vehicle.
            mode (Str): Mode of the frenét planner.
            timing (bool): True if the execution times should be saved. Defaults to False.
            frenet_parameters (dict): Parameters for the frenét planner. Defaults to None.
            sensor_radius (float): Radius of the sensor model. Defaults to 30.0.
            plot_frenet_trajectories (bool): True if the frenét paths should be visualized. Defaults to False.
            weights(dict): the weights of the costfunction. Defaults to None.
        """
        super().__init__(scenario, planning_problem, ego_id, vehicle_params, exec_timer, active_learning)

        # Set up logger
        self.logger = FrenetLogging(
            log_path=f"./planner/Frenet/results/logs/{scenario.benchmark_id}.csv"
        )

        try:
            with Timeout(1000, "Frenet Planner initialization"):

                self.exec_timer.start_timer("initialization/total")
                if frenet_parameters is None:
                    print(
                        "No frenet parameters found. Swichting to default parameters."
                    )
                    frenet_parameters = {
                        "t_list": [1.0],
                        "v_list_generation_mode": "linspace",
                        "n_v_samples": 5,
                        "d_list": np.linspace(-3.5, 3.5, 15),
                        "dt": 0.1,
                        "v_thr": 3.0,
                    }

                # parameters for frenet planner
                self.frenet_parameters = frenet_parameters
                # vehicle parameters
                self.p = vehicle_params

                # load parameters
                self.params_harm = load_harm_parameter_json()
                if weights is None:
                    self.params_weights = load_weight_json()
                else:
                    self.params_weights = weights

                self.active_learning = active_learning
                if self.active_learning:
                    # Store active_learning config for InteractionMap parameters
                    self.planning_config = settings.get("active_learning", {}) if settings else {}
                    # Store original scenario with full trajectories for prediction
                    # update_scenario() truncates all agent trajectories during simulation,
                    # which makes WaleNet/ground_truth prediction unable to see future positions.
                    self.original_scenario = copy.deepcopy(scenario)
                    self.max_exploration_time = settings["active_learning"]["max_exploration_time"]
                    self.search_step = settings["active_learning"]["search_step"]
                    self.decay_rate = settings["active_learning"]["decay_rate"]
                    self.save_irl_feature = settings["active_learning"]["save_irl_feature"]
                    self.prepare_fig_data = settings["active_learning"]["prepare_figure_data"]
                    self.prepare_num_data = settings["active_learning"]["prepare_numerical_data"]
                    self.prepare_label = settings["active_learning"]["prepare_label"]
                    self.store_interval = settings["active_learning"]["store_interval"]
                    self.trajectory_collect = settings["active_learning"]["trajectory_collect"]
                    self.tc_time_step = settings["active_learning"]["tc_time_step"]
                    self.tc_ego_id = settings["active_learning"]["tc_ego_id"]
                    self.tc_target = settings["active_learning"]["tc_target"]
                    # Initialize BEV probability loader for risk overlay
                    bev_prob_dir = settings["active_learning"].get(
                        "bev_prob_dir",
                        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                                     "bev_v2x_transformer", "BEVPredProb")
                    )
                    bev_weight = settings["active_learning"].get("bev_weight", 1.0)
                    self.bev_prob_loader = BEVProbLoader(
                        bev_prob_dir=bev_prob_dir,
                        benchmark_id=scenario.benchmark_id,
                        bev_weight=bev_weight,
                    )
                    if self.prepare_fig_data:
                        self.saved_global_path = []
                        self.saved_hist_traj = []
                        self.obs_id_color_mapping = {}
                        i = 1
                        for obs in self.scenario.dynamic_obstacles:
                            self.obs_id_color_mapping[obs.obstacle_id] = i
                            i += 1
                        norm_for_obs = Normalize(vmin=0, vmax=len(self.obs_id_color_mapping)+1)
                        cmap_for_obs = plt.get_cmap("binary")
                        for obs_id, mapping_id in self.obs_id_color_mapping.items():
                            self.obs_id_color_mapping[obs_id] = cmap_for_obs(norm_for_obs(mapping_id))
                    if self.prepare_num_data:
                        state_list_tmp = scenario.obstacle_by_id(
                            self.ego_id
                        ).prediction.trajectory.state_list
                        length = len(state_list_tmp)
                        self.numerical_dict = {"global_path": self.global_path_to_goal, "time": np.empty(shape=length),
                                               "x": np.empty(shape=length),
                                               "y": np.empty(shape=length), "ori": np.empty(shape=length),
                                               "vel": np.empty(shape=length)}
                        for i, state in enumerate(state_list_tmp):
                            self.numerical_dict["time"][i] = state.time_step
                            self.numerical_dict["x"][i] = state.position[0]
                            self.numerical_dict["y"][i] = state.position[1]
                            self.numerical_dict["ori"][i] = state.orientation
                            self.numerical_dict["vel"][i] = state.velocity
                    if self.prepare_label:
                        self.saved_maps = []

                if settings is not None:
                    if "risk_dict" in settings:
                        self.params_mode = settings["risk_dict"]
                    else:
                        self.params_mode = load_risk_json()

                self.params_dict = {
                    'weights': self.params_weights,
                    'modes': self.params_mode,
                    'harm': self.params_harm,
                }

                # check if the planning problem has a goal velocity and safe it
                if hasattr(planning_problem.goal.state_list[0], "velocity"):
                    self.v_goal_min = planning_problem.goal.state_list[0].velocity.start
                    self.v_goal_max = planning_problem.goal.state_list[0].velocity.end
                else:
                    self.v_goal_min = None
                    self.v_goal_max = None

                self.cost_dict = {}

                # check if the planning problem has an initial acceleration, else set it to zero
                if not hasattr(self.ego_state, "acceleration"):
                    self.ego_state.acceleration = 0.0

                # initialize the driven trajectory with the initial position
                self.driven_traj = [
                    State(
                        position=self.ego_state.position,
                        orientation=self.ego_state.orientation,
                        time_step=self.ego_state.time_step,
                        velocity=self.ego_state.velocity,
                        acceleration=self.ego_state.acceleration,
                    )
                ]

                # get sensor radius param, and planner mode
                self.sensor_radius = sensor_radius
                self.mode = mode

                # get visualization marker
                self.plot_frenet_trajectories = plot_frenet_trajectories

                # initialize the prediction network if necessary
                if self.mode == "WaleNet" or self.mode == "risk":

                    prediction_config_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "configs",
                        "prediction.json",
                    )
                    with open(prediction_config_path, "r") as f:
                        online_args = json.load(f)

                    self.predictor = WaleNet(scenario=scenario, online_args=online_args, verbose=False)
                elif self.mode == "ground_truth":
                    self.predictor = None
                else:
                    raise ValueError("mode must be ground_truth, WaleNet, or risk")

                # check whether reachable sets have to be calculated for responsibility
                if (
                        'responsibility' in self.params_weights
                        and self.params_weights['responsibility'] > 0
                ):
                    self.responsibility = True
                    self.reach_set = reachable_set.ReachSet(
                        scenario=self.scenario,
                        ego_id=self.ego_id,
                        ego_length=self.p.l,
                        ego_width=self.p.w,
                    )
                else:
                    self.responsibility = False
                    self.reach_set = None

                # create a collision object of the non-lanelet area of the scenario to check if a trajectory leaves the road
                # for some scenarios, this does not work/takes forever
                # to avoid that, abort after 5 seconds and raise an error
                with self.exec_timer.time_with_cm(
                        "initialization/initialize road boundary"
                ):
                    try:
                        with Timeout(5, "Initializing roud boundary"):
                            (
                                _,
                                self.road_boundary,
                            ) = boundary.create_road_boundary_obstacle(
                                scenario=scenario,
                                method="obb_rectangles",
                            )
                    except ExecutionTimeoutError:
                        raise RuntimeError("Road Boundary can not be created")

                # create a collision checker
                # remove the ego vehicle from the scenario
                with self.exec_timer.time_with_cm(
                        "initialization/initialize collision checker"
                ):
                    cc_scenario = copy.deepcopy(self.scenario)
                    cc_scenario.remove_obstacle(
                        obstacle=[cc_scenario.obstacle_by_id(ego_id)]
                    )
                    try:
                        self.collision_checker = create_collision_checker(cc_scenario)
                    except Exception:
                        raise BrokenPipeError("Collision Checker fails.") from None

                with self.exec_timer.time_with_cm(
                        "initialization/initialize goal area"
                ):
                    # get the shape group of the goal area
                    self.goal_area = get_goal_area_shape_group(
                        planning_problem=self.planning_problem, scenario=self.scenario
                    )

                # Initialize BEV probability loader for risk overlay (both modes)
                if not self.active_learning:
                    bev_prob_dir = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                        "bev_v2x_transformer", "BEVPredProb"
                    )
                    if settings is not None and "active_learning" in settings:
                        bev_prob_dir = settings["active_learning"].get("bev_prob_dir", bev_prob_dir)
                        bev_weight = settings["active_learning"].get("bev_weight", 1.0)
                    else:
                        bev_weight = 1.0
                    self.bev_prob_loader = BEVProbLoader(
                        bev_prob_dir=bev_prob_dir,
                        benchmark_id=scenario.benchmark_id,
                        bev_weight=bev_weight,
                    )

                self.initial_step = scenario.obstacle_by_id(self.ego_id).initial_state.time_step
                self.final_step = scenario.obstacle_by_id(self.ego_id).prediction.final_time_step
                self.traj_length = int(self.frenet_parameters["t_list"][0] // self.frenet_parameters["dt"])
                if self.active_learning:
                    self.behavior_tree = BehaviorTree()
                    self.reference_traj = copy.deepcopy(scenario.obstacle_by_id(
                        self.ego_id
                    ).prediction.trajectory)
                    eval_save_dir = os.path.join("../../saved_fig", scenario.benchmark_id)
                    self.gt_traj_logger = TrajLogger(log_prefix="car " + str(self.ego_id) + " dataset", points_num=self.traj_length * 3, save_dir=eval_save_dir)
                    self.search_traj_logger = TrajLogger(log_prefix="car " + str(self.ego_id) + " search tree", points_num=self.traj_length * 3, save_dir=eval_save_dir)
                    self.exec_timer.stop_timer("initialization/total")
        except ExecutionTimeoutError:
            raise TimeoutError

    def _step_planner(self):
        """Frenet Planner step function.

        This methods overloads the basic step method. It generates a new trajectory with the jerk optimal polynomials.
        """
        self.exec_timer.start_timer("simulation/total")

        with self.exec_timer.time_with_cm("simulation/update driven trajectory"):
            # update the driven trajectory
            # add the current state to the driven path
            if self.ego_state.time_step > self.initial_step:
                current_state = State(
                    position=self.ego_state.position,
                    orientation=self.ego_state.orientation,
                    time_step=self.ego_state.time_step,
                    velocity=self.ego_state.velocity,
                    acceleration=self.ego_state.acceleration,
                )

                self.driven_traj.append(current_state)

                # # if current position derives more than 1m from global path, replan global path from there
                # if self.trajectory["d_loc_m"][0] > 1.0:
                #     print("Replanning global path")
                #     super().plan_global_path(self.scenario, self.planning_problem, self.p, initial_state=current_state)

        # find position along the reference spline (s, s_d, s_dd, d, d_d, d_dd)
        c_s = self.trajectory["s_loc_m"][1]
        c_s_d = self.ego_state.velocity
        c_s_dd = self.ego_state.acceleration
        c_d = self.trajectory["d_loc_m"][1]
        c_d_d = self.trajectory["d_d_loc_mps"][1]
        c_d_dd = self.trajectory["d_dd_loc_mps2"][1]

        # get the end velocities for the frenét paths
        current_v = self.ego_state.velocity
        max_acceleration = self.p.longitudinal.a_max
        t_min = min(self.frenet_parameters["t_list"])
        t_max = max(self.frenet_parameters["t_list"])
        max_v = min(
            current_v + (max_acceleration / 2.0) * t_max, self.p.longitudinal.v_max
        )
        min_v = max(0.01, current_v - max_acceleration * t_min)

        if self.active_learning:
            c_s, c_s_d, c_d, c_d_d = self.reference_spline.cartesian_to_frenet(
                self.ego_state.position, self.ego_state.velocity, self.ego_state.orientation
            )
            c_s_dd = 0
            c_d_dd = 0

            next_state = self.reference_traj.state_at_time_step(self.time_step + 1)
            if next_state is None:
                # Trajectory ended, use current state as fallback
                next_state = self.ego_state
            # store the best trajectory
            self._trajectory = {
                "s_loc_m": [0, 0],
                "d_loc_m": [0, 0],
                "d_d_loc_mps": [0, 0],
                "d_dd_loc_mps2": [0, 0],
                "x_m": [self.ego_state.position[0], next_state.position[0]],
                "y_m": [self.ego_state.position[1], next_state.position[1]],
                "psi_rad": [self.ego_state.orientation, next_state.orientation],
                "kappa_radpm": [0, 0],
                "v_mps": [self.ego_state.velocity, next_state.velocity],
                "ax_mps2": [0, 0],
                "time_s": [0, 0.1],
            }

            if self.time_step == self.initial_step:
                return
            #
            #

            if self.time_step < self.final_step - self.max_exploration_time / 0.1:
                matplotlib.use("Agg")
                print(
                    "Time step: {} | Velocity: {:.2f} km/h | Acceleration: {:.2f} m/s2".format(
                        self.time_step, current_v * 3.6, c_s_dd
                    )
                )
                if self.prepare_label and self.time_step % int(self.store_interval / 0.1) == 0:
                    search_length = int(self.search_step / self.frenet_parameters["dt"])
                    try:
                        bev_map, fig, ax = draw_bev_map(
                            scenario=self.scenario,
                            ego_id=self.ego_id,
                            time_step=self.ego_state.time_step + search_length * 2,
                            ego_time_step=self.ego_state.time_step,
                            marked_vehicle=self.ego_id,
                            planning_problem=self.planning_problem,
                            traj=None,
                            global_path=self.global_path_to_goal,
                            global_path_after_goal=self.global_path_after_goal,
                            driven_traj=self.driven_traj,
                            animation_area=50.0,
                            traj_time_begin=self.ego_state.time_step,
                            traj_time_end=self.ego_state.time_step + search_length,
                            ego_traj_time_begin = self.ego_state.time_step,
                            ego_traj_time_end = self.ego_state.time_step + search_length,
                            ego_draw_occ = False,
                        )
                    except Exception as e:
                        print(e)

                    self.interaction_maps = [InteractionMap(bev_map=bev_map,
                                                            anchor_state=self.ego_state,
                                                            size=0.5,
                                                            obs_width=2.0,
                                                            update_length=int(self.frenet_parameters["t_list"][0] // self.frenet_parameters["dt"]) - 1,
                                                            resolution_mode=self.planning_config.get("resolution_mode", "bands"),
                                                            linear_growth_rate=self.planning_config.get("linear_growth_rate", 0.03),
                                                            ego_velocity=self.ego_state.velocity,
                                                            speed_band_scale=self.planning_config.get("speed_band_scale", 0.02),
                                                            speed_risk_factor=self.planning_config.get("speed_risk_factor", 0.02),
                                                            ) for _ in range(3)]
                    valid_trajectories = []
                    time_a = time.time()

                    self.predictions = self.get_prediction()
                    time_b = time.time()

                    self.traverse(future_step=0, state=self.ego_state, current_d_v=(0,-1), parent_behavior_path=[],
                                  ego_id=self.ego_id, valid_trajs=valid_trajectories, tree_ax=ax)

                    time_c = time.time()
                    fig.savefig(os.path.join("../../saved_fig", self.scenario.benchmark_id, str(self.ego_id) + "_" + str(self.ego_state.time_step) + "_tree.png"),
                                dpi=900)
                    plt.close(fig)

                    # profiler = Profiler()
                    # profiler.start()




                    def traj_risk_calc(traj):
                        traj_risk = 0
                        for i, map in enumerate(self.interaction_maps):
                            traj_risk += map.sum_traj_risk(traj[i * search_length: (i + 1) * search_length])
                        # Overlay BEVPredProb risk: use max probability across
                        # all 3 segments (avoids biasing toward shorter trajectories)
                        if self.bev_prob_loader is not None and self.bev_prob_loader.is_available:
                            self.bev_prob_loader.set_time_step(self.time_step)
                            max_seg_prob = 0.0
                            for i in range(min(3, len(self.interaction_maps))):
                                seg = traj[i * search_length: (i + 1) * search_length]
                                seg_prob = self.bev_prob_loader.compute_segment_bev_risk(seg, t_index=i)
                                if seg_prob > max_seg_prob:
                                    max_seg_prob = seg_prob
                            traj_risk += max_seg_prob
                        return traj_risk

                    best_traj = None
                    worst_traj = None
                    if len(valid_trajectories) > 0:
                        best_traj, best_traj_risk, worst_traj, worst_traj_risk = find_best_traj(valid_trajs=valid_trajectories,
                                       traj_risk=[traj_risk_calc(t) for t in valid_trajectories])



                        time_d = time.time()
                        print(time_b - time_a, time_c - time_b, time_d - time_c)

                        eval_res_search = calc_eval_values(self.scenario, self.ego_id, self.time_step, best_traj, self.reference_spline)

                        # profiler.stop()
                        # print(profiler.output_text(unicode=True, color=True))

                        self.search_traj_logger.record_traj(self.ego_id, self.time_step, best_traj, best_traj_risk,
                                                            eval_res_search, time_b - time_a, time_c - time_b, time_d - time_c)

                        traj_time_step_unit = int(self.frenet_parameters["dt"] / 0.1)
                        gt_states = self.reference_traj.states_in_time_interval(self.time_step + 1, self.time_step + len(best_traj) * traj_time_step_unit)
                        gt_traj = []
                        for i, state in enumerate(gt_states):
                            if i % traj_time_step_unit == 0:
                                c_s, _, c_d, _ = self.reference_spline.cartesian_to_frenet(
                                    state.position, state.velocity, state.orientation
                                )
                                gt_traj.append([state.position[0], state.position[1], c_s, c_d, state.velocity, state.orientation])

                        gt_traj_risk = traj_risk_calc(gt_traj)
                        eval_res_gt = calc_eval_values(self.scenario, self.ego_id, self.time_step, gt_traj, self.reference_spline)

                        self.gt_traj_logger.record_traj(self.ego_id, self.time_step, gt_traj, gt_traj_risk, eval_res_gt, 0, 0, 0)
                        if abs(best_traj_risk-gt_traj_risk) > 5:
                            print("big risk")
                    if best_traj is None:
                        return
                    draw_bev_map(
                        scenario=self.scenario,
                        ego_id=self.ego_id,
                        time_step=self.ego_state.time_step,
                        ego_time_step=self.ego_state.time_step,
                        marked_vehicle=self.ego_id,
                        planning_problem=self.planning_problem,
                        traj=None,
                        global_path=self.global_path_to_goal,
                        global_path_after_goal=self.global_path_after_goal,
                        driven_traj=self.driven_traj,
                        animation_area=50.0,
                        traj_time_begin=self.ego_state.time_step,
                        traj_time_end=self.ego_state.time_step,
                        ego_traj_time_begin=self.ego_state.time_step,
                        ego_traj_time_end=self.ego_state.time_step,
                        ego_draw_occ=False,
                        predictions=self.predictions,
                        save_filename=os.path.join("../../saved_fig", self.scenario.benchmark_id, str(self.ego_id) + "_" + str(self.ego_state.time_step) + "_scenario.png"),

                    )
                    plt.close('all')

                    # draw occupancy for best traj and gt traj
                    draw_bev_map(
                        scenario=self.scenario,
                        ego_id=self.ego_id,
                        time_step=self.ego_state.time_step + len(best_traj),
                        ego_time_step=self.ego_state.time_step,
                        marked_vehicle=self.ego_id,
                        planning_problem=self.planning_problem,
                        traj=best_traj,
                        global_path=self.global_path_to_goal,
                        global_path_after_goal=self.global_path_after_goal,
                        driven_traj=self.driven_traj,
                        animation_area=50.0,
                        traj_time_begin=self.ego_state.time_step + len(best_traj),
                        traj_time_end=self.ego_state.time_step + len(best_traj),
                        ego_traj_time_begin=self.ego_state.time_step + len(best_traj),
                        ego_traj_time_end=self.ego_state.time_step + len(best_traj),
                        ego_draw_occ=True,
                        save_filename=os.path.join("../../saved_fig", self.scenario.benchmark_id,
                                                   str(self.ego_id) + "_" + str(
                                                       self.ego_state.time_step) + "_best_traj_occ.png"),
                    )
                    plt.close('all')

                    draw_bev_map(
                        scenario=self.scenario,
                        ego_id=self.ego_id,
                        time_step=self.ego_state.time_step + len(gt_traj),
                        ego_time_step=self.ego_state.time_step,
                        marked_vehicle=self.ego_id,
                        planning_problem=self.planning_problem,
                        traj=gt_traj,
                        global_path=self.global_path_to_goal,
                        global_path_after_goal=self.global_path_after_goal,
                        driven_traj=self.driven_traj,
                        animation_area=50.0,
                        traj_time_begin=self.ego_state.time_step + len(gt_traj),
                        traj_time_end=self.ego_state.time_step+ len(gt_traj),
                        ego_traj_time_begin=self.ego_state.time_step+ len(gt_traj),
                        ego_traj_time_end=self.ego_state.time_step + len(gt_traj),
                        ego_draw_occ=True,
                        save_filename=os.path.join("../../saved_fig", self.scenario.benchmark_id,
                                                   str(self.ego_id) + "_" + str(
                                                       self.ego_state.time_step) + "_gt_traj_occ.png"),
                    )
                    plt.close('all')

                    self.saved_maps.append(self.interaction_maps)
                    for index, map in enumerate(self.interaction_maps):
                        best_traj_seg = best_traj[search_length * index: search_length * (index + 1)]
                        worst_traj_seg = worst_traj[search_length * index: search_length * (index + 1)]
                        _, fig, ax = draw_bev_map(
                            scenario=self.scenario,
                            ego_id=self.ego_id,
                            time_step=self.ego_state.time_step + search_length * (index + 1),
                            ego_time_step=self.ego_state.time_step,
                            marked_vehicle=self.ego_id,
                            planning_problem=self.planning_problem,
                            traj=None,
                            global_path=self.global_path_to_goal,
                            global_path_after_goal=self.global_path_after_goal,
                            driven_traj=self.driven_traj,
                            animation_area=50.0,
                            traj_time_begin=self.ego_state.time_step + search_length * index,
                            traj_time_end=self.ego_state.time_step + search_length * (index + 1),
                            ego_traj_time_begin=self.ego_state.time_step + search_length * index,
                            ego_traj_time_end=self.ego_state.time_step + search_length * (index + 1),
                            ego_draw_occ=False,
                            draw_grid=False,
                        )
                        map.draw_map(fig, ax, os.path.join(self.scenario.benchmark_id, str(self.ego_id) + "_" + str(self.ego_state.time_step) + "_" + str(index) + ".png"),
                                     best_traj=best_traj_seg, worst_traj=worst_traj_seg)

                        if self.trajectory_collect and self.time_step == self.tc_time_step and self.ego_id == self.tc_ego_id:
                            tc_collecter = [[] for i in range(len(self.tc_target))]
                            self.traverse(future_step=0, state=self.ego_state, current_d_v=(0, -1), ego_id=self.ego_id,
                                          trajectory_collect_tree=self.tc_target,
                                          trajectory_collecter=tc_collecter,
                                          valid_trajs=[])
                            draw_future_situations(scenario=self.scenario,
                                                   ego_id=self.ego_id,
                                                   current_time_step=self.time_step,
                                                   animation_area=50.0,
                                                   driven_traj=self.driven_traj,
                                                   ax=ax,
                                                   show_label=True,
                                                   ego_draw_occ=True,
                                                   future_ft_lists=tc_collecter,
                                                   scenario_path=os.path.join("../../saved_sfig", str(self.scenario.benchmark_id))
                                                   )
                        plt.close('all')

                    # if self.prepare_fig_data and self.time_step % int(self.store_interval / self.frenet_parameters["dt"]) == 0:
                    #     global_path_img, hist_traj_img = get_current_graphs(scenario=self.scenario,
                    #                                                         time_step=self.ego_state.time_step,
                    #                                                         driven_trajectory=self.driven_traj,
                    #                                                         ego_id=self.ego_id,
                    #                                                         color_dict=self.obs_id_color_mapping,
                    #                                                         animation_area=50.0,
                    #                                                         global_path=self.global_path_to_goal,
                    #                                                         )
                    #     self.saved_global_path.append((global_path_img, len(self.saved_global_path)))
                    #     self.saved_hist_traj.append((hist_traj_img, len(self.saved_hist_traj)))
                    plt.close("all")
                    # if self.prepare_label:
                    #     for i, map in enumerate(self.interaction_maps):
            elif self.time_step == self.final_step - self.max_exploration_time / self.frenet_parameters["dt"]:
                    self.search_traj_logger.log_average_traj_data()
                    self.gt_traj_logger.log_average_traj_data()
            self.exec_timer.stop_timer("simulation/total")

            return

        with self.exec_timer.time_with_cm("simulation/get v list"):
            v_list = get_v_list(
                v_min=min_v,
                v_max=max_v,
                v_cur=current_v,
                v_goal_min=self.v_goal_min,
                v_goal_max=self.v_goal_max,
                mode=self.frenet_parameters["v_list_generation_mode"],
                n_samples=self.frenet_parameters["n_v_samples"],
            )

        with self.exec_timer.time_with_cm("simulation/calculate trajectories/total"):
            d_list = self.frenet_parameters["d_list"]
            t_list = self.frenet_parameters["t_list"]

            # calculate all possible frenét trajectories
            ft_list = calc_frenet_trajectories(
                c_s=c_s,
                c_s_d=c_s_d,
                c_s_dd=c_s_dd,
                c_d=c_d,
                c_d_d=c_d_d,
                c_d_dd=c_d_dd,
                d_list=d_list,
                t_list=t_list,
                v_list=v_list,
                dt=self.frenet_parameters["dt"],
                csp=self.reference_spline,
                v_thr=self.frenet_parameters["v_thr"],
                exec_timer=self.exec_timer,
            )

        with self.exec_timer.time_with_cm("simulation/prediction"):
            # Overwrite later
            visible_area = None

            # get visible objects if the prediction is used
            if self.mode == "WaleNet" or self.mode == "risk":
                # get_visible_objects may fail sometimes due to bad lanelets (e.g. DEU_A9-1_1_T-1 at [-73.94, -53.24])
                if self.params_mode["sensor_occlusion_model"]:
                    try:
                        visible_obstacles, visible_area = get_visible_objects(
                            scenario=self.scenario,
                            ego_pos=self.ego_state.position,
                            time_step=self.time_step,
                            sensor_radius=self.sensor_radius,
                        )
                    except Exception as e:  # TopologicalError or AttributeError:
                        # if get_visible_objects fails just get every obstacle in the sensor_radius
                        print(
                            f"Warning: <{getframeinfo(currentframe()).filename} >>> Line {getframeinfo(currentframe()).lineno}>",
                            e,
                        )
                        visible_obstacles = get_obstacles_in_radius(
                            scenario=self.scenario,
                            ego_id=self.ego_id,
                            ego_state=self.ego_state,
                            radius=self.sensor_radius,
                        )
                else:
                    visible_obstacles = get_obstacles_in_radius(
                        scenario=self.scenario,
                        ego_id=self.ego_id,
                        ego_state=self.ego_state,
                        radius=self.sensor_radius,
                    )
                # predictions may fail (e.g. SetBasedPrediction DEU_Ffb-1_2_S-1)
                try:
                    # get dynamic and static visible obstacles since predictor can not handle static obstacles
                    (
                        dyn_visible_obstacles,
                        stat_visible_obstacles,
                    ) = get_dyn_and_stat_obstacles(
                        scenario=self.scenario, obstacle_ids=visible_obstacles
                    )
                    # get prediction for dynamic obstacles
                    predictions = self.predictor.step(
                        time_step=self.ego_state.time_step,
                        obstacle_id_list=dyn_visible_obstacles,
                        scenario=self.scenario,
                    )
                    # create and add prediction of static obstacles
                    predictions = add_static_obstacle_to_prediction(
                        scenario=self.scenario,
                        predictions=predictions,
                        obstacle_id_list=stat_visible_obstacles,
                        pred_horizon=max(t_list) / self.scenario.dt,
                    )
                # if prediction fails use ground truth as prediction
                except Exception as e:
                    print(
                        f"Warning: <{getframeinfo(currentframe()).filename} >>> Line {getframeinfo(currentframe()).lineno}>",
                        e,
                    )
                    predictions = get_ground_truth_prediction(
                        scenario=self.scenario,
                        obstacle_ids=visible_obstacles,
                        time_step=self.ego_state.time_step,
                    )
                # add orientation and dimensions of the obstacles to the prediction
                predictions = get_orientation_velocity_and_shape_of_prediction(
                    predictions=predictions, scenario=self.scenario
                )

                # Assign responsibility to predictions
                predictions = assign_responsibility_by_action_space(
                    self.scenario, self.ego_state, predictions
                )

            else:
                # TODO: Get GT prediction here for responsibility

                predictions = None

        # calculate reachable sets
        if self.responsibilpsi_radity:
            with self.exec_timer.time_with_cm(
                    "simulation/calculate and check reachable sets"
            ):
                self.reach_set.calc_reach_sets(self.ego_state, list(predictions.keys()))

        with self.exec_timer.time_with_cm("simulation/sort trajectories/total"):
            # sorted list (increasing costs)

            ft_list_valid, ft_list_invalid, validity_dict = sort_frenet_trajectories(
                ego_state=self.ego_state,
                fp_list=ft_list,
                global_path=self.global_path,
                predictions=predictions,
                mode=self.mode,
                params=self.params_dict,
                planning_problem=self.planning_problem,
                scenario=self.scenario,
                vehicle_params=self.p,
                ego_id=self.ego_id,
                dt=self.frenet_parameters["dt"],
                sensor_radius=self.sensor_radius,
                road_boundary=self.road_boundary,
                collision_checker=self.collision_checker,
                goal_area=self.goal_area,
                exec_timer=self.exec_timer,
                reach_set=(self.reach_set if self.responsibility else None)
            )

            # Overlay BEV probability risk onto trajectory costs (standard mode)
            # Use max probability along the trajectory instead of sum to avoid
            # biasing toward shorter/faster trajectories.
            if hasattr(self, 'bev_prob_loader') and self.bev_prob_loader is not None and self.bev_prob_loader.is_available:
                self.bev_prob_loader.set_time_step(self.time_step)
                for fp in ft_list_valid:
                    max_bev_prob = 0.0
                    dt_step = self.frenet_parameters["dt"]
                    for idx in range(1, len(fp.x)):
                        future_t = idx * dt_step
                        if future_t <= 1.5:
                            t_idx = 0
                        elif future_t <= 2.5:
                            t_idx = 1
                        elif future_t <= 3.5:
                            t_idx = 2
                        else:
                            t_idx = 2
                        p = self.bev_prob_loader.query_prob(fp.x[idx], fp.y[idx], t_idx)
                        if p > max_bev_prob:
                            max_bev_prob = p
                    fp.cost += max_bev_prob * self.bev_prob_loader.bev_weight

            with self.exec_timer.time_with_cm(
                    "simulation/sort trajectories/sort list by costs"
            ):
                # Sort the list of frenet trajectories (minimum cost first):
                ft_list_valid.sort(key=lambda fp: fp.cost, reverse=False)

            # show details of the frenet trajectories
            # from planner.Frenet.utils.visualization import show_frenet_details
            # show_frenet_details(vehicle_params=self.p, fp_list=ft_list)

            if self.reach_set is not None:
                log_reach_set = self.reach_set.reach_sets[self.time_step]
            else:
                log_reach_set = None

        with self.exec_timer.time_with_cm("log trajectories"):
            self.logger.log_data(
                self.time_step,
                self.time_step * self.frenet_parameters["dt"],
                [d.__dict__ for d in ft_list_valid],
                [d.__dict__ for d in ft_list_invalid],
                predictions,
                0,
                log_reach_set,
            )

        with self.exec_timer.time_with_cm("plot trajectories"):
            if self.params_mode["figures"]["create_figures"] is True:
                if self.mode == "risk":
                    create_risk_files(
                        scenario=self.scenario,
                        time_step=self.ego_state.time_step,
                        destination=os.path.join(os.path.dirname(__file__), "results"),
                        risk_modes=self.params_mode,
                        weights=self.params_weights,
                        marked_vehicle=self.ego_id,
                        planning_problem=self.planning_problem,
                        traj=ft_list_valid,
                        global_path=self.global_path_to_goal,
                        global_path_after_goal=self.global_path_after_goal,
                        driven_traj=self.driven_traj,
                    )

                else:
                    warnings.warn(
                        "Harm diagrams could not be created."
                        "Please select mode risk.",
                        UserWarning,
                    )

            if self.params_mode["risk_dashboard"] is True:
                if self.mode == "risk":
                    risk_dashboard(
                        scenario=self.scenario,
                        time_step=self.ego_state.time_step,
                        destination=os.path.join(
                            os.path.dirname(__file__), "results/risk_plots"
                        ),
                        risk_modes=self.params_mode,
                        weights=self.params_weights,
                        planning_problem=self.planning_problem,
                        traj=(ft_list_valid + ft_list_invalid),
                    )

                else:
                    warnings.warn(
                        "Risk dashboard could not be created."
                        "Please select mode risk.",
                        UserWarning,
                    )

            # print some information about the frenet trajectories
            if self.plot_frenet_trajectories:
                matplotlib.use("Agg")
                print(
                    "Time step: {} | Velocity: {:.2f} km/h | Acceleration: {:.2f} m/s2".format(
                        self.time_step, current_v * 3.6, c_s_dd
                    )
                )
                for lvl, descr in VALIDITY_LEVELS.items():
                    print(f"{descr}: {len(validity_dict[lvl])}", end=" | ")
                print("")

                try:
                    draw_frenet_trajectories(
                        scenario=self.scenario,
                        time_step=self.ego_state.time_step,
                        marked_vehicle=self.ego_id,
                        planning_problem=self.planning_problem,
                        traj=None,
                        all_traj=ft_list,
                        global_path=self.global_path_to_goal,
                        global_path_after_goal=self.global_path_after_goal,
                        driven_traj=self.driven_traj,
                        animation_area=50.0,
                        predictions=predictions,
                        visible_area=visible_area,
                    )
                except Exception as e:
                    print(e)

            # best trajectory
            if len(ft_list_valid) > 0:
                best_trajectory = ft_list_valid[0]
            else:
                best_trajectory = ft_list_invalid[0]
                # raise NoLocalTrajectoryFoundError('Failed. No valid frenét path found')

        self.exec_timer.stop_timer("simulation/total")

        # store the best trajectory
        self._trajectory = {
            "s_loc_m": best_trajectory.s,
            "d_loc_m": best_trajectory.d,
            "d_d_loc_mps": best_trajectory.d_d,
            "d_dd_loc_mps2": best_trajectory.d_dd,
            "x_m": best_trajectory.x,
            "y_m": best_trajectory.y,
            "psi_rad": best_trajectory.yaw,
            "kappa_radpm": best_trajectory.curv,
            "v_mps": best_trajectory.s_d,
            "ax_mps2": best_trajectory.s_dd,
            "time_s": best_trajectory.t,
        }

    def traverse(self,
                 future_step: int,
                 state: State,
                 current_d_v: (float, float),
                 ego_id: int,
                 parent_behavior_path: [(str, str)] = [],
                 frenet_state: [float] = None,
                 trajectory_collect_tree: [[[float]]] = None,
                 trajectory_collecter = None,
                 valid_trajs = None,
                 tree_ax = None,
                 ):


        # exploration depth
        depth = int(future_step / self.search_step)
        lateral_points = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
        # if depth == 0:
        #     d_list = lateral_points
        # else:
        #     index = np.asscalar(np.where(lateral_points == current_d)[0])
        #     left = index - 1 if index > 0 else 0
        #     right = index + 2
        #     d_list = lateral_points[left : right]

        # if depth == 0:
        #     if state.velocity < 5:
        #         d_list = [-0.5, 0, 0.5]
        #     else:
        #         d_list = [-1, 0, 1]
        # else:
        #     if state.velocity < 5:
        #         d_list = [current_d-0.5, current_d, current_d+0.5]
        #     else:
        #         d_list = [current_d-1, current_d, current_d+1]
        #         if current_d < 0:
        #             d_list[-1] = 0
        #         elif current_d > 0:
        #             d_list[0] = 0
        #     for d in d_list:
        #         if abs(d) > 1.5:
        #             d_list.remove(d)
        #             break

        cur_behavior_list, cur_action_list = self.behavior_tree.get_behaviors_in_path(current_d_v, state, parent_behavior_path)

        # get possible trajectories
        collision_ft_list, leaving_road_ft_list, valid_ft_list = self.get_possible_trajectories(global_state=state,
                                                                            frenet_state=frenet_state,
                                                                            ego_id=ego_id,
                                                                            d_v_combination=cur_action_list,
                                                                            behavior_combination=cur_behavior_list,
                                                                            anchor_state=self.interaction_maps[depth].anchor_state)

        if len(collision_ft_list) + len(leaving_road_ft_list) + len(valid_ft_list) == 0:
            # all trajectories cannot meet the need for curvature, definitely a dangerous state
            average_risk = 0
            return average_risk

        collision_risk_sum = 0
        if future_step + self.search_step >= self.max_exploration_time - 0.01:
            # the deepest layer, update interaction map
            for ft in leaving_road_ft_list:
                self.interaction_maps[depth].update_map(trajectory=ft,
                                                        risk_value=0,
                                                        update_range=ft.collision_step - 1,
                                                        # draw_tree_ax=tree_ax,
                                                        )
                if trajectory_collect_tree is not None:
                    for i, (d_tree, v_tree) in enumerate(trajectory_collect_tree):
                        if len(d_tree) > depth:
                            if d_tree[depth] == ft.target_d and abs(v_tree[depth] - ft.target_v) < 0.5:
                                trajectory_collecter[i].append(ft)

            for ft in collision_ft_list:

                ft_risk = self.interaction_maps[depth].update_map(trajectory=ft,
                                                            risk_value=0.999,
                                                            # draw_tree_ax=tree_ax
                                                        )
                collision_risk_sum += ft_risk
                if trajectory_collect_tree is not None:
                    for i, (d_tree, v_tree) in enumerate(trajectory_collect_tree):
                        if len(d_tree) > depth:
                            if d_tree[depth] == ft.target_d and abs(v_tree[depth] - ft.target_v) < 0.5:
                                trajectory_collecter[i].append(ft)

            for ft in valid_ft_list:
                # from pyinstrument import Profiler
                # profiler = Profiler()
                # profiler.start()

                self.interaction_maps[depth].update_map(trajectory=ft,
                                                        risk_value=0,
                                                        # draw_tree_ax=tree_ax,
                                                         )

                # profiler.stop()
                # print(profiler.output_text(unicode=True, color=True))
                if trajectory_collect_tree is not None:
                    for i, (d_tree, v_tree) in enumerate(trajectory_collect_tree):
                        if len(d_tree) > depth:
                            if d_tree[depth] == ft.target_d and abs(v_tree[depth] - ft.target_v) < 0.5:
                                trajectory_collecter[i].append(ft)

                current_traj = [[ft.x[i], ft.y[i], ft.s[i], ft.d[i], ft.v[i], ft.yaw[i]] for i in range(1, len(ft))]
                valid_trajs.append(current_traj)



            # return the average risk of the trajectories in the layer
            average_risk = (collision_risk_sum + len(leaving_road_ft_list) * 0 + len(valid_ft_list) * 0) / \
                           (len(collision_ft_list) + len(leaving_road_ft_list) + len(valid_ft_list))

            return average_risk

        else:
            # if collision happened, update the map directly

            for ft in leaving_road_ft_list:

                self.interaction_maps[depth].update_map(trajectory=ft,
                                                        risk_value=0.999,
                                                        update_range=ft.collision_step - 1,
                                                        draw_tree_ax=tree_ax if depth == 1 else None
                                                                )
                if trajectory_collect_tree is not None:
                    for i, (d_tree, v_tree) in enumerate(trajectory_collect_tree):
                        if len(d_tree) > depth:
                            if d_tree[depth] == ft.target_d and abs(v_tree[depth] - ft.target_v) < 0.5:
                                trajectory_collecter[i].append(ft)

            for ft in collision_ft_list:
                print("coll", parent_behavior_path, ft.target_behavior, ft.s[0], ft.s[-1])
                ft_risk = self.interaction_maps[depth].update_map(trajectory=ft,
                                                            risk_value=0.999,
                                                            draw_tree_ax=tree_ax if depth == 1 else None
                                                                  )
                collision_risk_sum += ft_risk
                if trajectory_collect_tree is not None:
                    for i, (d_tree, v_tree) in enumerate(trajectory_collect_tree):
                        if len(d_tree) > depth:
                            if d_tree[depth] == ft.target_d and abs(v_tree[depth] - ft.target_v) < 0.5:
                                trajectory_collecter[i].append(ft)


            end_index = int(self.search_step / self.frenet_parameters["dt"])
            valid_ft_risk_list = []
            # for i in all valid trajectories:


            for ft in valid_ft_list:
                if depth == 1:
                    print(parent_behavior_path, ft.target_behavior, ft.s[0], ft.s[-1])
                new_tree = []
                new_collecter = []
                j = 0
                old_new_mapping = {}
                if trajectory_collect_tree is not None:
                    for i, (d_tree, v_tree) in enumerate(trajectory_collect_tree):
                        if len(d_tree) > depth:
                            if d_tree[depth] == ft.target_d and abs(v_tree[depth] - ft.target_v) < 0.5:
                                trajectory_collecter[i].append(ft)
                                new_tree.append([d_tree, v_tree])
                                new_collecter.append([])
                                old_new_mapping[i] = j
                                j += 1

                # init current and future valid trajs
                current_traj = [[ft.x[i], ft.y[i], ft.s[i], ft.d[i], ft.v[i], ft.yaw[i]] for i in range(1, len(ft))]
                future_valid_trajs = []

                # get next state
                next_global_state = ft.get_global_state(end_index)
                # current time + traverse future time
                next_global_state.time_step = end_index + state.time_step

                next_frenet_state = ft.get_frenet_state(end_index)
                # traverse in the future, calc the risk in the future
                tmp_path = parent_behavior_path[:]
                tmp_path.append(ft.target_behavior)
                future_risk = self.traverse(future_step=future_step + self.search_step,
                                            ego_id=ego_id,
                                            state=next_global_state,
                                            current_d_v=(ft.target_d, ft.target_v),
                                            frenet_state=next_frenet_state,
                                            parent_behavior_path=tmp_path,
                                            trajectory_collect_tree=new_tree if len(new_tree) > 0 else None,
                                            trajectory_collecter=new_collecter if len(new_collecter) > 0 else None,
                                            valid_trajs=future_valid_trajs,
                                            tree_ax=tree_ax if depth == 0 else None)

                for traj in future_valid_trajs:
                    valid_trajs.append(current_traj + traj)

                if len(old_new_mapping) > 0:
                    for i in old_new_mapping.keys():
                        trajectory_collecter[i].extend(new_collecter[old_new_mapping[i]])

                # the risk of each valid traj is the decayed future risk

                self.interaction_maps[depth].update_map(trajectory=ft,
                                                            risk_value=future_risk * self.decay_rate,
                                                        draw_tree_ax=tree_ax if depth == 1 else None)
                valid_ft_risk_list.append(future_risk * self.decay_rate)


            # return the average risk of the trajectories in this layer
            average_risk = (collision_risk_sum + len(leaving_road_ft_list) * 0 + sum(valid_ft_risk_list)) / \
                           (len(collision_ft_list) + len(leaving_road_ft_list) + len(valid_ft_list))

            return average_risk

    def get_prediction(self):
        # In active_learning mode, use original_scenario for predictions because
        # update_scenario() truncates all agent trajectories, making the truncated
        # scenario useless for future position prediction.
        pred_scenario = self.original_scenario if self.active_learning else self.scenario
        radius = self.sensor_radius
        visible_obstacles = get_obstacles_in_radius(
            scenario=self.scenario,
            ego_id=self.ego_id,
            ego_state=self.ego_state,
            radius=radius,
        )
        while len(visible_obstacles) > 7:
            radius /= 2
            visible_obstacles = get_obstacles_in_radius(
                scenario=self.scenario,
                ego_id=self.ego_id,
                ego_state=self.ego_state,
                radius=radius,
            )
        # predictions may fail (e.g. SetBasedPrediction DEU_Ffb-1_2_S-1)
        try:
            # get dynamic and static visible obstacles since predictor can not handle static obstacles
            (
                dyn_visible_obstacles,
                stat_visible_obstacles,
            ) = get_dyn_and_stat_obstacles(
                scenario=self.scenario, obstacle_ids=visible_obstacles
            )
            # get prediction for dynamic obstacles
            predictions = self.predictor.step(
                time_step=self.ego_state.time_step,
                obstacle_id_list=dyn_visible_obstacles,
                scenario=pred_scenario,
            )
            # create and add prediction of static obstacles
            predictions = add_static_obstacle_to_prediction(
                scenario=self.scenario,
                predictions=predictions,
                obstacle_id_list=stat_visible_obstacles,
                pred_horizon=60,
            )
        # if prediction fails use ground truth as prediction
        except Exception as e:
            print(
                f"Warning: <{getframeinfo(currentframe()).filename} >>> Line {getframeinfo(currentframe()).lineno}>",
                e,
            )
            predictions = get_ground_truth_prediction(
                scenario=pred_scenario,
                obstacle_ids=visible_obstacles,
                time_step=self.ego_state.time_step,
                pred_horizon=60
            )
        # add orientation and dimensions of the obstacles to the prediction
        predictions = get_orientation_velocity_and_shape_of_prediction(
            predictions=predictions, scenario=pred_scenario
        )
        return predictions

    def save_prediction_feature(self):
        radius = self.sensor_radius
        visible_obstacles = get_obstacles_in_radius(
            scenario=self.scenario,
            ego_id=self.ego_id,
            ego_state=self.ego_state,
            radius=radius,
        )
        while len(visible_obstacles) > 7:
            radius /= 2
            visible_obstacles = get_obstacles_in_radius(
                scenario=self.scenario,
                ego_id=self.ego_id,
                ego_state=self.ego_state,
                radius=radius,
            )
        # predictions may fail (e.g. SetBasedPrediction DEU_Ffb-1_2_S-1)
        try:
            # get dynamic and static visible obstacles since predictor can not handle static obstacles
            (
                dyn_visible_obstacles,
                stat_visible_obstacles,
            ) = get_dyn_and_stat_obstacles(
                scenario=self.scenario, obstacle_ids=visible_obstacles
            )
            predictions = self.predictor.step(
                time_step=self.ego_state.time_step,
                obstacle_id_list=dyn_visible_obstacles,
                scenario=self.scenario,
            )
        except Exception as e:
            print(
                f"Warning: <{getframeinfo(currentframe()).filename} >>> Line {getframeinfo(currentframe()).lineno}>",
                e,
            )
        print("1+1")

    def get_possible_trajectories(self,
                                  global_state: State,
                                  ego_id: int,
                                  d_v_combination: [(float, float)],
                                  behavior_combination: [(str, str)],
                                  anchor_state: State = None,
                                  frenet_state: [float] = None):

        # get the end velocities for the frenét paths
        current_v = global_state.velocity
        max_acceleration = self.p.longitudinal.a_max
        t_min = min(self.frenet_parameters["t_list"])
        t_max = max(self.frenet_parameters["t_list"])
        # max_v = min(
        #     current_v + (max_acceleration / 2.0) * t_max, self.p.longitudinal.v_max
        # )
        # min_v = max(0.01, current_v - max_acceleration * t_min)

        if frenet_state is None:
            c_s, c_s_d, c_d, c_d_d = self.reference_spline.cartesian_to_frenet(
                global_state.position, global_state.velocity, global_state.orientation
            )
            c_s_dd = 0
            c_d_dd = 0
        else:
            c_s, c_s_d, c_s_dd, c_d, c_d_d, c_d_dd = frenet_state

        # v_list = get_v_list(
        #     v_min=min_v,
        #     v_max=max_v,
        #     v_cur=current_v,
        #     v_goal_min=self.v_goal_min,
        #     v_goal_max=self.v_goal_max,
        #     mode=self.frenet_parameters["v_list_generation_mode"],
        #     n_samples=self.frenet_parameters["n_v_samples"],
        # )

        t_list = self.frenet_parameters["t_list"]

        # calculate all possible frenét trajectories
        ft_list = calc_frenet_trajectories_combination_based(
            c_s=c_s,
            c_s_d=c_s_d,
            c_s_dd=c_s_dd,
            c_d=c_d,
            c_d_d=c_d_d,
            c_d_dd=c_d_dd,
            d_v_combination=d_v_combination,
            behavior_combination=behavior_combination,
            t_list=t_list,
            dt=self.frenet_parameters["dt"],
            csp=self.reference_spline,
            v_thr=self.frenet_parameters["v_thr"],
        )

        collision_ft_list = []
        valid_ft_list = []
        leaving_road_ft_list = []

        for fp in ft_list:
            # check validity
            fp.valid_level, fp.reason_invalid, fp.collision_step, fp.uncertainty_list = check_validity(
                ft=fp,
                ego_state=global_state,
                scenario=self.scenario,
                vehicle_params=self.p,
                risk_params=self.params_dict['modes'],
                predictions=self.predictions,
                mode='ground_truth',
                road_boundary=self.road_boundary,
                # check_horizon=int(self.search_step / self.frenet_parameters["dt"]) + 1,
                check_horizon=self.traj_length,
                collision_checker=self.collision_checker,
                ego_id=ego_id,
                anchor_state=anchor_state
            )

            if fp.valid_level == 10:
                valid_ft_list.append(fp)
            elif fp.valid_level == 1:
                collision_ft_list.append(fp)
            elif fp.valid_level == 2:
                leaving_road_ft_list.append(fp)

        return collision_ft_list, leaving_road_ft_list, valid_ft_list


if __name__ == "__main__":
    import argparse
    from planner.plannertools.evaluate import ScenarioEvaluator
    from planner.Frenet.plannertools.frenetcreator import FrenetCreator

    parser = argparse.ArgumentParser()
    # parser.add_argument("--scenario", default="CHN_Merging-1_57_T-1.xml")
    # parser.add_argument("--scenario", default="CHN_Roundabout-1_60_T-1.xml")
    parser.add_argument("--scenario", default="ZAM_T-Junction-1_600_T-1.xml")
    #
    parser.add_argument("--time", action="store_true")
    args = parser.parse_args()
    #
    if "commonroad" in args.scenario:
        scenario_path = args.scenario.split("scenarios/")[-1]
    else:
        scenario_path = args.scenario

    # load settings from planning_fast.json
    settings_dict = load_planning_json("planning_fast.json")
    settings_dict["risk_dict"] = risk_dict = load_risk_json()
    if not args.time:
        settings_dict["evaluation_settings"]["show_visualization"] = True
    eval_directory = (
        pathlib.Path(__file__).resolve().parents[0].joinpath("results").joinpath("eval")
    )
    # Create the frenet creator
    frenet_creator = FrenetCreator(settings_dict)

    # Create the scenario evaluator
    evaluator = ScenarioEvaluator(
        planner_creator=frenet_creator,
        vehicle_type=settings_dict["evaluation_settings"]["vehicle_type"],
        path_to_scenarios=pathlib.Path(
            os.path.join(mopl_path, "scenarios/")
        ).resolve(),
        log_path=pathlib.Path("./log/example").resolve(),
        collision_report_path=eval_directory,
        timing_enabled=settings_dict["evaluation_settings"]["timing_enabled"],
        active_learning=settings_dict["active_learning"]["active_learning_enabled"],
        prepare_num_data=settings_dict["active_learning"]["prepare_numerical_data"],
        label_path=settings_dict["active_learning"]["label_path"] if settings_dict["active_learning"]["prepare_label"] else None,
        fig_path=settings_dict["active_learning"]["fig_path"] if settings_dict["active_learning"]["prepare_figure_data"] else None,
    )



    def main():
        """Loop for cProfile."""
        _ = evaluator.eval_scenario(scenario_path)


    if args.time:
        import cProfile

        cProfile.run('main()', "output.dat")
        no_trajectores = settings_dict["frenet_settings"]["frenet_parameters"]["n_v_samples"] * len(
            settings_dict["frenet_settings"]["frenet_parameters"]["d_list"])
        import pstats

        sortby = pstats.SortKey.CUMULATIVE
        with open(f"cProfile/{scenario_path.split('/')[-1]}_{no_trajectores}.txt", "w") as f:
            p = pstats.Stats("output.dat", stream=f).sort_stats(sortby)
            p.sort_stats(sortby).print_stats()
    else:
        main()


# EOF
