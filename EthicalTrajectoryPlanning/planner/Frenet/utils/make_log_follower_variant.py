#!/usr/bin/env python3
"""Replace ID 160 with a lane-centred follower derived from an ego log."""

import argparse
import csv
import json
import tempfile
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Point

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.geometry.shape import Rectangle
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.trajectory import State, Trajectory

from planner.Frenet.utils.create_intersection_scenario import _extract_centerline
from planner.Frenet.utils.make_intersection_variants import obstacle_block


EGO_ROUTE = [115, 114, 116, 135, 133, 132, 113, 110, 140]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--ego-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gap", type=float, default=10.0)
    args = parser.parse_args()

    scenario, planning_problem_set = CommonRoadFileReader(str(args.base)).open()
    points = _extract_centerline(scenario.lanelet_network, EGO_ROUTE)
    centerline = LineString(points)

    with args.ego_log.open(newline="") as log_file:
        rows = list(csv.DictReader(log_file, delimiter=";"))

    ego_arcs = []
    for row in rows:
        best = json.loads(row["valid_trajectories"])[0]
        ego_arcs.append(centerline.project(Point(best["x"][0], best["y"][0])))

    follower_arcs = np.maximum(np.asarray(ego_arcs) - args.gap, 0.0)
    positions = []
    headings = []
    for arc in follower_arcs:
        point = centerline.interpolate(float(arc))
        before = centerline.interpolate(max(float(arc) - 0.05, 0.0))
        after = centerline.interpolate(min(float(arc) + 0.05, centerline.length))
        positions.append(np.asarray(point.coords[0]))
        headings.append(float(np.arctan2(after.y - before.y, after.x - before.x)))

    speeds = [
        float(np.linalg.norm(positions[min(i + 1, len(positions) - 1)] - positions[i]) / scenario.dt)
        for i in range(len(positions))
    ]
    if len(speeds) > 1:
        speeds[-1] = speeds[-2]

    states = [
        State(
            position=positions[i],
            velocity=speeds[i],
            orientation=headings[i],
            time_step=i,
        )
        for i in range(len(positions))
    ]
    follower = DynamicObstacle(
        obstacle_id=160,
        obstacle_type=ObstacleType.CAR,
        obstacle_shape=Rectangle(length=4.5, width=1.8),
        initial_state=states[0],
        prediction=TrajectoryPrediction(
            trajectory=Trajectory(initial_time_step=1, state_list=states[1:]),
            shape=Rectangle(length=4.5, width=1.8),
        ),
    )

    scenario.remove_obstacle(scenario.obstacle_by_id(160))
    scenario.add_objects(follower)
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as temp_file:
        temp_path = Path(temp_file.name)
    try:
        writer = CommonRoadFileWriter(
            scenario,
            planning_problem_set,
            author="Hongen Wang",
            affiliation="BME",
            source="ID 160 lane-centred log follower",
            tags=set(),
        )
        writer.write_to_file(str(temp_path), OverwriteExistingFile.ALWAYS)
        base_xml = args.base.read_text(encoding="utf-8")
        temp_xml = temp_path.read_text(encoding="utf-8")
        base_match = obstacle_block(base_xml, 160)
        follower_match = obstacle_block(temp_xml, 160)
        output_xml = (
            base_xml[:base_match.start()]
            + follower_match.group()
            + base_xml[base_match.end():]
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_xml, encoding="utf-8")
    finally:
        temp_path.unlink(missing_ok=True)

    gaps = np.asarray(ego_arcs) - follower_arcs
    print(
        f"{args.output}: {len(states)} states, "
        f"arc gap {gaps.min():.2f}-{gaps.max():.2f} m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
