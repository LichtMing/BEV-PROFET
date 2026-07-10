#!/usr/bin/env python3
"""
Generate a 3-vehicle car-following scenario on the CHN_Merging highway map.

Scenario: 3 cars at high speed in the left lane, heading east.
  Car 1 (lead):  emergency brakes at t=2.0s
  Car 2 (middle): swerves right to center lane at t=2.5s
  Car 3 (ego):   planning problem — must react

Usage:
  python create_car_following_scenario.py
  python create_car_following_scenario.py --scenario-id 400 --output-dir /path/to/scenarios
"""

import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.common.util import Interval, AngleInterval
from commonroad.geometry.shape import Rectangle
from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.scenario import Scenario, ScenarioID, Tag
from commonroad.scenario.trajectory import State, Trajectory

# ── constants ──────────────────────────────────────────────────────────
DT = 0.1                     # time step [s]
TOTAL_STEPS = 80             # 8 seconds — enough to reach the far goal
INITIAL_SPEED = 20.0         # m/s  (~72 km/h)
BRAKE_DECEL = -6.0           # m/s²  emergency braking
CAR_LENGTH = 4.5             # m
CAR_WIDTH = 1.8              # m

# Starting x-positions for each car (spaced ~13-15 m apart, all within
# the lanelet network: lanelet 112 starts at x≈-59.33).
EGO_X0 = -58.0
MID_X0 = -44.0
LEAD_X0 = -30.0

# ── lane centerline extraction ──────────────────────────────────────────

def _trace_chain(start_id: int, lanelet_network) -> list[int]:
    """Trace a lanelet chain by following the first successor at each step."""
    chain = []
    visited = set()
    curr_id = start_id
    while curr_id is not None and curr_id not in visited:
        visited.add(curr_id)
        ll = lanelet_network.find_lanelet_by_id(curr_id)
        chain.append(curr_id)
        succs = ll.successor
        curr_id = succs[0] if succs else None
    return chain


def _extract_lane_centerline(lanelet_network, lanelet_id_chain: list[int]):
    """Return (xs, ys) arrays of lane-centre points for a chain of lanelets."""
    points = []
    for lid in lanelet_id_chain:
        ll = lanelet_network.find_lanelet_by_id(lid)
        lv = ll.left_vertices
        rv = ll.right_vertices
        for i in range(len(lv)):
            cx = (lv[i][0] + rv[i][0]) / 2.0
            cy = (lv[i][1] + rv[i][1]) / 2.0
            points.append((cx, cy))
    # Deduplicate consecutive identical points (at lanelet boundaries)
    deduped = [points[0]]
    for p in points[1:]:
        if abs(p[0] - deduped[-1][0]) > 0.01 or abs(p[1] - deduped[-1][1]) > 0.01:
            deduped.append(p)
    xs = np.array([p[0] for p in deduped], dtype=np.float64)
    ys = np.array([p[1] for p in deduped], dtype=np.float64)
    return xs, ys


def _lane_y(x: float, xs: np.ndarray, ys: np.ndarray) -> float:
    """Linearly interpolate lane-centre y for a given x."""
    return float(np.interp(x, xs, ys))


def _lane_heading(x: float, xs: np.ndarray, ys: np.ndarray) -> float:
    """Compute lane tangent heading at x via finite differences."""
    if x <= xs[0]:
        dx = xs[1] - xs[0]
        dy = ys[1] - ys[0]
    elif x >= xs[-1]:
        dx = xs[-1] - xs[-2]
        dy = ys[-1] - ys[-2]
    else:
        idx = np.searchsorted(xs, x)
        idx = max(1, min(idx, len(xs) - 1))
        dx = xs[idx] - xs[idx - 1]
        dy = ys[idx] - ys[idx - 1]
    return float(np.arctan2(dy, dx))


# ── trajectory helpers ──────────────────────────────────────────────────

def _make_state(time_step: int, x: float, y: float, v: float,
                orientation: float) -> State:
    """Build a single CommonRoad State."""
    return State(
        position=np.array([x, y]),
        velocity=v,
        orientation=orientation,
        time_step=time_step,
    )


def _cubic_hermite(p: float) -> float:
    """Smooth step function: 0 at p=0, 1 at p=1, zero derivative at both ends."""
    return 3.0 * p ** 2 - 2.0 * p ** 3


def _resolve_headings(
    positions: list[tuple[float, float]],
) -> list[float]:
    """Compute heading at each waypoint from the actual path tangent.

    Uses central differences so the heading reflects the true direction of
    motion, not an independently interpolated value.  This avoids the
    "sideways-sliding" artifact.
    """
    n = len(positions)
    headings = [0.0] * n
    for i in range(n):
        if i == 0:
            dx = positions[1][0] - positions[0][0]
            dy = positions[1][1] - positions[0][1]
        elif i == n - 1:
            dx = positions[-1][0] - positions[-2][0]
            dy = positions[-1][1] - positions[-2][1]
        else:
            dx = positions[i + 1][0] - positions[i - 1][0]
            dy = positions[i + 1][1] - positions[i - 1][1]
        headings[i] = float(np.arctan2(dy, dx))
    return headings


def compute_lead_trajectory(
    left_xs: np.ndarray, left_ys: np.ndarray,
) -> tuple[State, list[State]]:
    """Car 1 (lead): follows left-lane centre, emergency braking at t=1.0s.

    Builds positions first, then derives headings from the actual path so
    kinematics are self-consistent.
    """
    BRAKE_START = 1.0  # seconds — brake earlier so the situation is critical sooner

    # Phase 1: compute all positions
    positions: list[tuple[float, float]] = []
    velocities: list[float] = []
    x, v = LEAD_X0, INITIAL_SPEED
    positions.append((x, _lane_y(x, left_xs, left_ys)))
    velocities.append(v)

    for step in range(1, TOTAL_STEPS + 1):
        t = step * DT
        a = 0.0 if t <= BRAKE_START else BRAKE_DECEL
        v = max(0.0, v + a * DT)
        x += v * DT
        positions.append((x, _lane_y(x, left_xs, left_ys)))
        velocities.append(v)

    # Phase 2: derive headings from positions
    headings = _resolve_headings(positions)

    initial = _make_state(0, positions[0][0], positions[0][1],
                          velocities[0], headings[0])
    states = [
        _make_state(i + 1, positions[i + 1][0], positions[i + 1][1],
                    velocities[i + 1], headings[i + 1])
        for i in range(TOTAL_STEPS)
    ]
    return initial, states


def compute_mid_trajectory(
    left_xs: np.ndarray, left_ys: np.ndarray,
    centre_xs: np.ndarray, centre_ys: np.ndarray,
) -> tuple[State, list[State]]:
    """Car 2 (middle): follows left lane, then swerves to centre lane at t=1.5s.

    During the swerve the car also brakes moderately (-3 m/s²) since it is
    reacting to the emergency ahead.  Headings are derived from the actual
    path (central differences), so the car points where it actually goes.
    """
    SWERVE_START = 1.5   # seconds
    SWERVE_END = 3.5     # seconds
    SWERVE_DECEL = -3.0  # m/s²  moderate braking during lane change
    POST_SWERVE_DECEL = -2.0  # m/s²  continued braking after lane change

    # Phase 1: compute all positions
    positions: list[tuple[float, float]] = []
    velocities: list[float] = []
    x, v = MID_X0, INITIAL_SPEED
    positions.append((x, _lane_y(x, left_xs, left_ys)))
    velocities.append(v)

    for step in range(1, TOTAL_STEPS + 1):
        t = step * DT

        # longitudinal acceleration
        if t <= SWERVE_START:
            a_long = 0.0
        elif t <= SWERVE_END:
            a_long = SWERVE_DECEL
        else:
            a_long = POST_SWERVE_DECEL

        v = max(0.0, v + a_long * DT)
        x += v * DT

        # lateral: lane-centre-following with smooth transition
        if t <= SWERVE_START:
            y = _lane_y(x, left_xs, left_ys)
        elif t <= SWERVE_END:
            progress = (t - SWERVE_START) / (SWERVE_END - SWERVE_START)
            smooth = _cubic_hermite(progress)
            y_left = _lane_y(x, left_xs, left_ys)
            y_centre = _lane_y(x, centre_xs, centre_ys)
            y = y_left + (y_centre - y_left) * smooth
        else:
            y = _lane_y(x, centre_xs, centre_ys)

        positions.append((x, y))
        velocities.append(v)

    # Phase 2: derive headings from actual positions
    headings = _resolve_headings(positions)

    initial = _make_state(0, positions[0][0], positions[0][1],
                          velocities[0], headings[0])
    states = [
        _make_state(i + 1, positions[i + 1][0], positions[i + 1][1],
                    velocities[i + 1], headings[i + 1])
        for i in range(TOTAL_STEPS)
    ]
    return initial, states


# ── obstacle builder ────────────────────────────────────────────────────

def create_dynamic_obstacle(
    obs_id: int,
    initial_state: State,
    state_list: list[State],
) -> DynamicObstacle:
    """Build a DynamicObstacle with a full trajectory prediction."""
    shape = Rectangle(length=CAR_LENGTH, width=CAR_WIDTH)
    trajectory = Trajectory(initial_time_step=1, state_list=state_list)
    prediction = TrajectoryPrediction(trajectory=trajectory, shape=shape)

    return DynamicObstacle(
        obstacle_id=obs_id,
        obstacle_type=ObstacleType.CAR,
        obstacle_shape=shape,
        initial_state=initial_state,
        prediction=prediction,
    )


# ── planning-problem builder ────────────────────────────────────────────

def create_ego_planning_problem(
    left_xs: np.ndarray,
    left_ys: np.ndarray,
    right_xs: np.ndarray,
    right_ys: np.ndarray,
    pp_id: int,
) -> PlanningProblem:
    """Car 3 (ego): starts on left lane; goal is on the RIGHT lane."""
    x, v = EGO_X0, INITIAL_SPEED
    y = _lane_y(x, left_xs, left_ys)
    heading = _lane_heading(x, left_xs, left_ys)

    initial_state = State(
        position=np.array([x, y]),
        velocity=v,
        orientation=heading,
        time_step=0,
        yaw_rate=0.0,
        slip_angle=0.0,
    )

    # Goal: as far as possible on the goal lane.
    goal_x = float(right_xs[-1])  # furthest reachable x (~93)
    goal_y = _lane_y(goal_x, right_xs, right_ys)
    goal_heading = _lane_heading(goal_x, right_xs, right_ys)
    goal_rect = Rectangle(
        length=5.0,
        width=2.0,
        center=np.array([goal_x, goal_y]),
        orientation=goal_heading,
    )
    goal_state = State(
        position=goal_rect,
        orientation=AngleInterval(
            goal_heading - 0.3, goal_heading + 0.3,
        ),
        velocity=Interval(10.0, 30.0),
        time_step=Interval(0, TOTAL_STEPS),
    )
    goal_region = GoalRegion([goal_state])

    return PlanningProblem(
        planning_problem_id=pp_id,
        initial_state=initial_state,
        goal_region=goal_region,
    )


# ── main ────────────────────────────────────────────────────────────────

def _patch_traffic_sign_id(xml_path: Path) -> None:
    """Restore the original traffic-sign ID that CommonRoad serialises away.

    ``CommonRoadFileReader`` substitutes ``TrafficSignIDZamunda.UNKNOWN`` for
    unknown country-specific IDs, but ``CommonRoadFileWriter`` serialises
    that as an empty ``<trafficSignID/>`` which cannot be re-read.  We put
    back the original ``274`` value — it triggers a harmless warning and
    loads correctly (same as all existing CHN_Merging scenarios).
    """
    content = xml_path.read_text(encoding="utf-8")
    content = content.replace(
        "<trafficSignID></trafficSignID>",
        "<trafficSignID>274</trafficSignID>",
    )
    xml_path.write_text(content, encoding="utf-8")

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a 3-vehicle car-following scenario on CHN_Merging."
    )
    parser.add_argument(
        "--scenario-id", type=int, default=375,
        help="Scenario ID (default: 375, the next after the last real scenario)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent.parent.parent.parent
        / "scenarios",
        help="Output directory for the generated XML file",
    )
    parser.add_argument(
        "--source-scenario", type=str,
        default="CHN_Merging-1_100_T-1.xml",
        help="Existing scenario to copy the lanelet network from",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    source_path = output_dir / args.source_scenario
    if not source_path.exists():
        print(f"ERROR: source scenario not found: {source_path}")
        return 1

    benchmark_id = f"CHN_Merging-1_{args.scenario_id}_T-1"
    output_path = output_dir / f"{benchmark_id}.xml"

    # ── 1. Load lanelet network from existing scenario ──────────────
    print(f"Loading lanelet network from {source_path} …")
    source_scenario, _ = CommonRoadFileReader(str(source_path)).open()
    lanelet_network = copy.deepcopy(source_scenario.lanelet_network)

    # ── 2. Extract lane centre lines (full chains via successors) ───
    LEFT_CHAIN = _trace_chain(112, source_scenario.lanelet_network)
    CENTRE_CHAIN = _trace_chain(111, source_scenario.lanelet_network)
    RIGHT_CHAIN = _trace_chain(109, source_scenario.lanelet_network)
    print(f"Left lane chain:   {LEFT_CHAIN}")
    print(f"Centre lane chain: {CENTRE_CHAIN}")
    print(f"Right lane chain:  {RIGHT_CHAIN}")
    left_xs, left_ys = _extract_lane_centerline(source_scenario.lanelet_network, LEFT_CHAIN)
    centre_xs, centre_ys = _extract_lane_centerline(source_scenario.lanelet_network, CENTRE_CHAIN)
    right_xs, right_ys = _extract_lane_centerline(source_scenario.lanelet_network, RIGHT_CHAIN)
    print(f"Left lane:   x=[{left_xs[0]:.0f},{left_xs[-1]:.0f}]  y=[{left_ys[0]:.1f},{left_ys[-1]:.1f}]")
    print(f"Centre lane: x=[{centre_xs[0]:.0f},{centre_xs[-1]:.0f}]  y=[{centre_ys[0]:.1f},{centre_ys[-1]:.1f}]")
    print(f"Right lane:  x=[{right_xs[0]:.0f},{right_xs[-1]:.0f}]  y=[{right_ys[0]:.1f},{right_ys[-1]:.1f}]")

    # ── 3. Create new scenario ──────────────────────────────────────
    scenario = Scenario(
        dt=DT,
        scenario_id=ScenarioID.from_benchmark_id(benchmark_id, "2020a"),
    )
    scenario.add_objects(lanelet_network)

    # Scenario tags (passed to the writer, not stored on scenario directly
    # in commonroad 2020a).
    scenario_tags = {
        Tag("highway"),
        Tag("multi_lane"),
        Tag("parallel_lanes"),
        Tag("critical"),
    }

    # ── 4. Build dynamic obstacles ──────────────────────────────────
    print("Generating Car 1 (lead, emergency brake) …")
    lead_init, lead_states = compute_lead_trajectory(left_xs, left_ys)
    lead_obs = create_dynamic_obstacle(
        scenario.generate_object_id(), lead_init, lead_states,
    )
    scenario.add_objects(lead_obs)

    print("Generating Car 2 (middle, swerve) …")
    mid_init, mid_states = compute_mid_trajectory(
        left_xs, left_ys, centre_xs, centre_ys,
    )
    mid_obs = create_dynamic_obstacle(
        scenario.generate_object_id(), mid_init, mid_states,
    )
    scenario.add_objects(mid_obs)

    # ── 5. Build planning problem ───────────────────────────────────
    print("Generating Car 3 (ego, planning problem) …")
    pp = create_ego_planning_problem(
        left_xs, left_ys, left_xs, left_ys,  # goal on LEFT lane, same as Car 1
        pp_id=scenario.generate_object_id(),
    )
    pp_set = PlanningProblemSet([pp])

    # ── 6. Write XML ────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    fw = CommonRoadFileWriter(
        scenario=scenario,
        planning_problem_set=pp_set,
        author="Hongen Wang",
        affiliation="BME",
        source="Synthetic car-following scenario",
        tags=scenario_tags,
    )
    fw.write_to_file(str(output_path), OverwriteExistingFile.ALWAYS)

    # Patch: CommonRoadFileWriter serialises TrafficSignIDChina.UNKNOWN as
    # an empty <trafficSignID/> tag, which CommonRoadFileReader cannot
    # deserialise back.  Replace with a string that round-trips.
    _patch_traffic_sign_id(output_path)

    print(f"✓ Scenario written to {output_path}")
    print(f"  Obstacles: {len(scenario.dynamic_obstacles)}")
    print(f"  Planning problems: {len(pp_set.planning_problem_dict)}")
    print(f"  Lanelets: {len(scenario.lanelet_network.lanelets)}")
    print(f"  Time steps: {TOTAL_STEPS}  ({TOTAL_STEPS * DT:.1f} s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
