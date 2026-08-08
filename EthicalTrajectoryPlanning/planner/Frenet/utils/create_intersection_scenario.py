#!/usr/bin/env python3
"""Generate a north-approach left-turn collision on USA_Intersection-1.

Ego drives south from the northern approach and turns left onto the eastbound
main road. A west-to-east truck turns left to the northern exit and occludes
the east-to-west stream until ego has committed to its turn.
"""

import argparse
import copy
from pathlib import Path

import numpy as np

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.common.util import AngleInterval, Interval
from commonroad.geometry.shape import Rectangle
from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.scenario import Scenario, ScenarioID, Tag
from commonroad.scenario.trajectory import State, Trajectory
from commonroad_helper_functions.sensor_model import get_visible_objects


DT = 0.1
TOTAL_STEPS = 160
IMPACT_VALIDATION_STEP = 27
GOAL_DEADLINE_STEP = 55
EGO_SPEED = 14.0
EGO_START_OFFSET = 10.0
FOLLOWER_START_OFFSET = 0.0
FOLLOWER_SPEED_PROFILE = (
    (0.0, 12.0),
    (2.0, 4.0),
    (6.0, 4.0),
    (8.0, 14.0),
    (10.0, 6.0),
    (12.0, 0.0),
)
TURNING_TRUCK_SPEED = 30.0
TURNING_TRUCK_START_OFFSET = 7.0
TURNING_FRONT_FLOW_SPEED = 30.0
TURNING_FRONT_FLOW_START_OFFSETS = (18.0, 48.0)
ONCOMING_CRITICAL_SPEED = 22.0
ONCOMING_CRITICAL_START_OFFSET = 7.0
ONCOMING_REAR_SPEED = 20.0
ONCOMING_REAR_START_OFFSET = 0.0

CAR_LENGTH, CAR_WIDTH = 4.5, 1.8
TRUCK_LENGTH, TRUCK_WIDTH = 8.0, 2.5


def _extract_centerline(lanelet_network, chain):
    points = []
    for lanelet_id in chain:
        lanelet = lanelet_network.find_lanelet_by_id(lanelet_id)
        for left, right in zip(lanelet.left_vertices, lanelet.right_vertices):
            point = (left + right) / 2
            if not points or np.linalg.norm(point - points[-1]) > 0.01:
                points.append(point)
    return np.asarray(points)


def _arc_lengths(points):
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(lengths)))


def _point_at_arc(points, cumulative_lengths, arc):
    arc = min(max(arc, 0.0), cumulative_lengths[-1])
    index = min(np.searchsorted(cumulative_lengths, arc, side="right") - 1,
                len(points) - 2)
    segment_length = cumulative_lengths[index + 1] - cumulative_lengths[index]
    fraction = (arc - cumulative_lengths[index]) / segment_length
    position = points[index] + fraction * (points[index + 1] - points[index])
    direction = points[index + 1] - points[index]
    return position, float(np.arctan2(direction[1], direction[0]))


def _lane_following_trajectory(points, speed, start_offset):
    cumulative_lengths = _arc_lengths(points)
    positions, headings = [], []
    for step in range(TOTAL_STEPS + 1):
        position, heading = _point_at_arc(
            points, cumulative_lengths, start_offset + step * DT * speed)
        positions.append(position)
        headings.append(heading)
    return positions, headings


def _following_lane_trajectory(points, start_offset, speed_profile):
    cumulative_lengths = _arc_lengths(points)
    profile_times = np.asarray([entry[0] for entry in speed_profile])
    profile_speeds = np.asarray([entry[1] for entry in speed_profile])
    positions, headings, speeds = [], [], []
    distance = 0.0
    previous_speed = float(profile_speeds[0])
    for step in range(TOTAL_STEPS + 1):
        time = step * DT
        speed = float(np.interp(time, profile_times, profile_speeds))
        if step > 0:
            distance += 0.5 * (previous_speed + speed) * DT
        position, heading = _point_at_arc(
            points, cumulative_lengths, start_offset + distance)
        positions.append(position)
        headings.append(heading)
        speeds.append(speed)
        previous_speed = speed
    return positions, headings, speeds


def _trajectory_at_arcs(points, arcs):
    cumulative_lengths = _arc_lengths(points)
    positions, headings = [], []
    for arc in arcs:
        position, heading = _point_at_arc(points, cumulative_lengths, arc)
        positions.append(position)
        headings.append(heading)
    return positions, headings


def _make_state(step, position, speed, heading):
    return State(position=position, velocity=speed,
                 orientation=heading, time_step=step)


def _build_obstacle(obstacle_id, obstacle_type, length, width,
                    initial_state, states):
    shape = Rectangle(length=length, width=width)
    trajectory = Trajectory(initial_time_step=1, state_list=states)
    prediction = TrajectoryPrediction(trajectory=trajectory, shape=shape)
    return DynamicObstacle(obstacle_id=obstacle_id,
                           obstacle_type=obstacle_type,
                           obstacle_shape=shape,
                           initial_state=initial_state,
                           prediction=prediction)


def create_ego_planning_problem(ego_points, planning_problem_id):
    cumulative_lengths = _arc_lengths(ego_points)
    initial_position, initial_heading = _point_at_arc(
        ego_points, cumulative_lengths, EGO_START_OFFSET)
    initial_state = State(position=initial_position, velocity=EGO_SPEED,
                          orientation=initial_heading, time_step=0,
                          yaw_rate=0.0, slip_angle=0.0)

    # One metre before the far eastern exit of the main road.
    goal_position, goal_heading = _point_at_arc(
        ego_points, cumulative_lengths, cumulative_lengths[-1] - 1.0)
    goal_rectangle = Rectangle(length=5.0, width=2.0,
                               center=goal_position,
                               orientation=goal_heading)
    goal_state = State(
        position=goal_rectangle,
        orientation=AngleInterval(goal_heading - 0.3, goal_heading + 0.3),
        velocity=Interval(2.0, 25.0),
        # The deadline makes the reactive planner commit to the turn rather
        # than selecting a standstill trace at the northern approach.
        time_step=Interval(0, GOAL_DEADLINE_STEP),
    )
    return PlanningProblem(planning_problem_id=planning_problem_id,
                           initial_state=initial_state,
                           goal_region=GoalRegion([goal_state]))


def _rectangles_overlap(first, second):
    axes, shapes = [], []
    for _, _, heading, length, width in (first, second):
        longitudinal = np.array([np.cos(heading), np.sin(heading)])
        lateral = np.array([-longitudinal[1], longitudinal[0]])
        axes.extend((longitudinal, lateral))
        shapes.append((longitudinal, lateral, length, width))

    for axis in axes:
        radii = [
            length / 2 * abs(np.dot(longitudinal, axis))
            + width / 2 * abs(np.dot(lateral, axis))
            for longitudinal, lateral, length, width in shapes
        ]
        distance = abs(np.dot(np.asarray(first[:2]) - np.asarray(second[:2]), axis))
        if distance > radii[0] + radii[1]:
            return False
    return True


def _collision_steps(first_positions, first_headings, first_shape,
                     second_positions, second_headings, second_shape,
                     max_step=TOTAL_STEPS):
    collisions = []
    for step in range(max_step + 1):
        first = (*first_positions[step], first_headings[step], *first_shape)
        second = (*second_positions[step], second_headings[step], *second_shape)
        if _rectangles_overlap(first, second):
            collisions.append(step)
    return collisions


def validate_design(ego_positions, ego_headings, traffic):
    ego_shape = (CAR_LENGTH, CAR_WIDTH)
    critical = next(vehicle for vehicle in traffic if vehicle[0] == "critical")
    collision_steps = _collision_steps(
        ego_positions, ego_headings, ego_shape,
        critical[1], critical[2], critical[3], critical[4])
    validation_end_step = (collision_steps[0] if collision_steps
                           else IMPACT_VALIDATION_STEP)
    if collision_steps and (collision_steps[0] < 20 or collision_steps[0] > 55):
        raise ValueError("Collision does not leave a visible approach")
    for first_index, first in enumerate(traffic):
        for second in traffic[first_index + 1:]:
            traffic_collision_steps = _collision_steps(
                first[1], first[2], first[3],
                second[1], second[2], second[3],
                min(first[4], second[4]))
            if any(step <= validation_end_step
                   for step in traffic_collision_steps):
                raise ValueError(
                    f"Traffic vehicles overlap before ego impact: "
                    f"{first[0]} and {second[0]}")
    for vehicle in traffic:
        if vehicle[0] == "critical":
            continue
        collision_steps_with_vehicle = _collision_steps(
            ego_positions, ego_headings, ego_shape,
            vehicle[1], vehicle[2], vehicle[3], vehicle[4])
        if any(step <= validation_end_step
               for step in collision_steps_with_vehicle):
            raise ValueError(
                f"Ego reaches {vehicle[0]} before the critical car")
    return collision_steps[0] if collision_steps else None


def validate_occlusion(scenario, ego_positions, oncoming_id, collision_step):
    end_step = collision_step if collision_step is not None else TOTAL_STEPS
    first_visible = next((
        step for step in range(end_step + 1)
        if oncoming_id in get_visible_objects(
            scenario, step, ego_positions[step], sensor_radius=50)[0]
    ), None)
    if first_visible == 0:
        raise ValueError("Oncoming vehicle is visible before the truck can occlude it")
    return first_visible


def _add_vehicle(scenario, points, speed, start_offset,
                 obstacle_type, length, width, exit_at_lane_end=False):
    positions, headings = _lane_following_trajectory(points, speed, start_offset)
    active_until = TOTAL_STEPS
    if exit_at_lane_end and speed > 0:
        path_length = _arc_lengths(points)[-1]
        active_until = min(
            TOTAL_STEPS,
            int(np.floor((path_length - start_offset) / (DT * speed))),
        )
    obstacle_id = scenario.generate_object_id()
    scenario.add_objects(_build_obstacle(
        obstacle_id, obstacle_type, length, width,
        _make_state(0, positions[0], speed, headings[0]),
        [_make_state(step, positions[step], speed, headings[step])
         for step in range(1, active_until + 1)],
    ))
    return obstacle_id, positions, headings, active_until


def _add_following_vehicle(scenario, points, start_offset, speed_profile,
                           obstacle_type, length, width):
    positions, headings, speeds = _following_lane_trajectory(
        points, start_offset, speed_profile)
    obstacle_id = scenario.generate_object_id()
    scenario.add_objects(_build_obstacle(
        obstacle_id, obstacle_type, length, width,
        _make_state(0, positions[0], speeds[0], headings[0]),
        [_make_state(step, positions[step], speeds[step], headings[step])
         for step in range(1, TOTAL_STEPS + 1)],
    ))
    return obstacle_id, positions, headings, TOTAL_STEPS


def _patch_traffic_sign_id(xml_path):
    content = xml_path.read_text(encoding="utf-8")
    content = content.replace("<trafficSignID></trafficSignID>",
                              "<trafficSignID>R2-1</trafficSignID>")
    xml_path.write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a left-turn/oncoming collision on USA_Intersection.")
    parser.add_argument("--scenario-id", type=int, default=0)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[3] / "scenarios")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    source_path = output_dir / "USA_Intersection-1_0_T-1.xml"
    if not source_path.exists():
        print(f"ERROR: source scenario not found: {source_path}")
        return 1

    benchmark_id = f"USA_Intersection-1_{args.scenario_id}_T-1"
    output_path = output_dir / f"{benchmark_id}.xml"

    source_scenario, _ = CommonRoadFileReader(str(source_path)).open()
    lanelet_network = copy.deepcopy(source_scenario.lanelet_network)

    # Ego starts at the northern approach, turns left to the eastbound exit,
    # and the critical car uses every successive lanelet of its own westbound
    # main-road lane. The truck follows the reciprocal west-to-east left turn.
    EGO_LEFT_TURN_CHAIN = [115, 114, 116, 135, 133, 132, 113, 110, 140]
    TURNING_TRUCK_CHAIN = [151, 150, 118, 117, 119]
    ONCOMING_STRAIGHT_CHAIN = [107, 106, 108, 101, 141, 136, 128,
                               153, 124, 146, 147]
    ego_points = _extract_centerline(lanelet_network, EGO_LEFT_TURN_CHAIN)
    truck_points = _extract_centerline(lanelet_network, TURNING_TRUCK_CHAIN)
    oncoming_points = _extract_centerline(lanelet_network,
                                          ONCOMING_STRAIGHT_CHAIN)
    print(f"Ego north-to-east left-turn route: {EGO_LEFT_TURN_CHAIN}")
    print(f"Truck west-to-north left-turn route: {TURNING_TRUCK_CHAIN}")
    print(f"East-to-west lane-centre route: {ONCOMING_STRAIGHT_CHAIN}")

    scenario = Scenario(
        dt=DT,
        scenario_id=ScenarioID.from_benchmark_id(benchmark_id, "2020a"),
    )
    scenario.add_objects(lanelet_network)

    planning_problem = create_ego_planning_problem(
        ego_points, scenario.generate_object_id())
    planning_problem_set = PlanningProblemSet([planning_problem])
    ego_positions, ego_headings = _lane_following_trajectory(
        ego_points, EGO_SPEED, EGO_START_OFFSET)

    _, follower_positions, follower_headings, follower_active_until = _add_following_vehicle(
        scenario, ego_points, FOLLOWER_START_OFFSET, FOLLOWER_SPEED_PROFILE,
        ObstacleType.CAR, CAR_LENGTH, CAR_WIDTH)
    turning_front_flow = []
    for index, start_offset in enumerate(TURNING_FRONT_FLOW_START_OFFSETS,
                                         start=1):
        _, positions, headings, active_until = _add_vehicle(
            scenario, truck_points, TURNING_FRONT_FLOW_SPEED, start_offset,
            ObstacleType.CAR, CAR_LENGTH, CAR_WIDTH, exit_at_lane_end=True)
        turning_front_flow.append(
            (f"truck-front turning car {index}", positions, headings,
             (CAR_LENGTH, CAR_WIDTH), active_until))
    _, turning_truck_positions, turning_truck_headings, truck_active_until = _add_vehicle(
        scenario, truck_points, TURNING_TRUCK_SPEED,
        TURNING_TRUCK_START_OFFSET, ObstacleType.TRUCK,
        TRUCK_LENGTH, TRUCK_WIDTH)
    oncoming_id, critical_positions, critical_headings, critical_active_until = _add_vehicle(
        scenario, oncoming_points, ONCOMING_CRITICAL_SPEED,
        ONCOMING_CRITICAL_START_OFFSET, ObstacleType.CAR,
        CAR_LENGTH, CAR_WIDTH)
    _, oncoming_rear_positions, oncoming_rear_headings, rear_active_until = _add_vehicle(
        scenario, oncoming_points, ONCOMING_REAR_SPEED,
        ONCOMING_REAR_START_OFFSET, ObstacleType.CAR,
        CAR_LENGTH, CAR_WIDTH)
    traffic = [
        ("north-approach follower", follower_positions, follower_headings,
         (CAR_LENGTH, CAR_WIDTH), follower_active_until),
        *turning_front_flow,
        ("turning truck", turning_truck_positions, turning_truck_headings,
         (TRUCK_LENGTH, TRUCK_WIDTH), truck_active_until),
        ("critical", critical_positions, critical_headings,
         (CAR_LENGTH, CAR_WIDTH), critical_active_until),
        ("oncoming rear", oncoming_rear_positions, oncoming_rear_headings,
         (CAR_LENGTH, CAR_WIDTH), rear_active_until),
    ]
    collision_step = validate_design(ego_positions, ego_headings, traffic)
    first_visible_step = validate_occlusion(
        scenario, ego_positions, oncoming_id, collision_step)

    output_dir.mkdir(parents=True, exist_ok=True)
    writer = CommonRoadFileWriter(
        scenario, planning_problem_set,
        author="Hongen Wang", affiliation="BME",
        source="Synthetic north-approach ego-left-turn collision scenario",
        tags={Tag("urban"), Tag("multi_lane"), Tag("intersection"), Tag("critical")},
    )
    writer.write_to_file(str(output_path), OverwriteExistingFile.ALWAYS)
    _patch_traffic_sign_id(output_path)

    goal = planning_problem.goal.state_list[0].position.center
    print(f"✓ {output_path}")
    print(f"  Goal (east exit): ({goal[0]:.1f}, {goal[1]:.1f})")
    print(f"  Traffic vehicles: {len(traffic)}; occluder is a turning truck")
    print(f"  Critical obstacle {oncoming_id} stays on the westbound lane centre")
    print("  Oncoming vehicle first visible: " + (
        f"t={first_visible_step * DT:.1f}s" if first_visible_step is not None
        else "not within the nominal reference horizon"))
    print("  Nominal reference collision: " + (
        f"t={collision_step * DT:.1f}s" if collision_step is not None
        else "none (actual Frenet trajectory is verified separately)"))


if __name__ == "__main__":
    raise SystemExit(main())
