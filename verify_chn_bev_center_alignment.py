#!/usr/bin/env python3
"""
Automatically validate CHN_Merging BEV alignment and pick the best BEV center.

Method:
1) Load a CHN_Merging CommonRoad scenario.
2) For each candidate BEV center (cx, cy), project scenario vehicles into a 288x288 mask.
3) Compare against BEVLabel vehicle channel with IoU.
4) Report the best center and top-ranked candidates.

Notes:
- This uses CHN_Merging BEVLabel generated from merging pipeline:
  bev_v2x_transformer/merging/BEVLabel/1_<idx>.npy
- Mapping used here (already observed in this workspace):
  idx_base = 10 * scenario_id + 59
  scenario_time_step increments by 10 for each idx increment.
"""

import argparse
import math
import os
import re
import sys
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ETHICAL_ROOT = os.path.join(BASE_DIR, "EthicalTrajectoryPlanning")
if ETHICAL_ROOT not in sys.path:
    sys.path.insert(0, ETHICAL_ROOT)

from commonroad.common.file_reader import CommonRoadFileReader  # noqa: E402


BEV_SIZE = 288
BEV_RESOLUTION = 0.5  # m/pixel
CHN_X_OFFSET = 1056.0
CHN_Y_OFFSET = 954.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find best BEV center for CHN_Merging by IoU with BEVLabel."
    )
    parser.add_argument(
        "--scenario",
        required=True,
        help="Path to CHN_Merging CommonRoad XML, e.g. EthicalTrajectoryPlanning/scenarios/CHN_Merging-1_15_T-1.xml",
    )
    parser.add_argument(
        "--bev-label-dir",
        default=os.path.join(BASE_DIR, "bev_v2x_transformer", "merging", "BEVLabel"),
        help="Directory of CHN BEVLabel npy files.",
    )
    parser.add_argument(
        "--x-offset",
        type=float,
        default=CHN_X_OFFSET,
        help="CommonRoad->INTERACTION x offset for CHN_Merging.",
    )
    parser.add_argument(
        "--y-offset",
        type=float,
        default=CHN_Y_OFFSET,
        help="CommonRoad->INTERACTION y offset for CHN_Merging.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=8,
        help="How many time anchors (1s apart) to evaluate.",
    )
    parser.add_argument(
        "--start-offset",
        type=int,
        default=0,
        help="Start offset from scenario min time step (in 0.1s step units).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Print top-k candidate centers.",
    )

    # Candidate mode A: explicit centers
    parser.add_argument(
        "--candidate-centers",
        type=str,
        default="",
        help="Explicit centers, format: '1003,995;1056,954.5;...'.",
    )

    # Candidate mode B: auto grid when --candidate-centers is empty
    parser.add_argument("--grid-min-x", type=float, default=980.0)
    parser.add_argument("--grid-max-x", type=float, default=1080.0)
    parser.add_argument("--grid-step-x", type=float, default=5.0)
    parser.add_argument("--grid-min-y", type=float, default=930.0)
    parser.add_argument("--grid-max-y", type=float, default=1010.0)
    parser.add_argument("--grid-step-y", type=float, default=5.0)

    return parser.parse_args()


def parse_scenario_id(benchmark_id: str) -> int:
    m = re.search(r"CHN_Merging-1_(\d+)_T-1", benchmark_id)
    if not m:
        raise ValueError(f"benchmark_id '{benchmark_id}' is not CHN_Merging-1")
    return int(m.group(1))


def build_candidates(args: argparse.Namespace) -> List[Tuple[float, float]]:
    if args.candidate_centers.strip():
        centers: List[Tuple[float, float]] = []
        for item in args.candidate_centers.split(";"):
            item = item.strip()
            if not item:
                continue
            x_str, y_str = item.split(",")
            centers.append((float(x_str), float(y_str)))
        if not centers:
            raise ValueError("No valid center parsed from --candidate-centers")
        return centers

    xs = np.arange(args.grid_min_x, args.grid_max_x + 1e-6, args.grid_step_x)
    ys = np.arange(args.grid_min_y, args.grid_max_y + 1e-6, args.grid_step_y)
    return [(float(x), float(y)) for x in xs for y in ys]


def cr_to_pixel(
    cr_x: float,
    cr_y: float,
    center_x: float,
    center_y: float,
    x_offset: float,
    y_offset: float,
) -> Tuple[float, float]:
    inter_x = cr_x + x_offset
    inter_y = cr_y + y_offset

    # Keep this consistent with BEV loader geometry.
    col_144 = (inter_x - center_x) + 72.0 + 0.5
    row_144 = -(inter_y - center_y) + 72.0 + 0.5

    col_288 = col_144 * 2.0
    row_288 = row_144 * 2.0
    return row_288, col_288


def draw_vehicle_mask_for_time(
    scenario,
    time_step: int,
    center_x: float,
    center_y: float,
    x_offset: float,
    y_offset: float,
) -> np.ndarray:
    mask = np.zeros((BEV_SIZE, BEV_SIZE), dtype=np.uint8)

    for obs in scenario.dynamic_obstacles:
        try:
            state = obs.state_at_time(time_step)
            if state is None:
                continue

            row, col = cr_to_pixel(
                float(state.position[0]),
                float(state.position[1]),
                center_x,
                center_y,
                x_offset,
                y_offset,
            )

            length_px = float(obs.obstacle_shape.length) / BEV_RESOLUTION
            width_px = float(obs.obstacle_shape.width) / BEV_RESOLUTION
            angle_deg = -math.degrees(float(state.orientation))

            rect = ((float(col), float(row)), (float(length_px), float(width_px)), float(angle_deg))
            box = cv2.boxPoints(rect)
            box = np.int32(np.round(box))
            cv2.fillConvexPoly(mask, box, 1)
        except Exception:
            continue

    return mask


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def evaluate_candidate(
    scenario,
    eval_pairs: Sequence[Tuple[int, str]],
    center_x: float,
    center_y: float,
    x_offset: float,
    y_offset: float,
) -> Tuple[float, int]:
    ious: List[float] = []

    for scenario_ts, label_path in eval_pairs:
        label = np.load(label_path)
        if label.ndim != 3 or label.shape[0] < 5:
            continue

        # channel[4] is vehicle occupancy at current anchor time in this dataset.
        gt_vehicle = (label[4] > 0).astype(np.uint8)
        pred_vehicle = draw_vehicle_mask_for_time(
            scenario=scenario,
            time_step=scenario_ts,
            center_x=center_x,
            center_y=center_y,
            x_offset=x_offset,
            y_offset=y_offset,
        )
        ious.append(compute_iou(pred_vehicle, gt_vehicle))

    if not ious:
        return 0.0, 0

    return float(np.mean(ious)), len(ious)


def main() -> None:
    args = parse_args()

    scenario, _ = CommonRoadFileReader(args.scenario).open()
    sid = parse_scenario_id(scenario.benchmark_id)
    idx_base = 10 * sid + 59

    # Build evaluation anchors: scenario_ts increases by 10 (1s), label idx also +10.
    min_ts = min(obs.initial_state.time_step for obs in scenario.dynamic_obstacles)
    eval_pairs: List[Tuple[int, str]] = []
    for i in range(args.samples):
        scenario_ts = int(min_ts + args.start_offset + i * 10)
        idx = int(idx_base + args.start_offset + i * 10)
        label_name = f"1_{idx}.npy"
        label_path = os.path.join(args.bev_label_dir, label_name)
        if os.path.exists(label_path):
            eval_pairs.append((scenario_ts, label_path))

    if not eval_pairs:
        raise RuntimeError(
            f"No valid BEVLabel anchors found for scenario {scenario.benchmark_id} under {args.bev_label_dir}"
        )

    candidates = build_candidates(args)
    print(f"Scenario: {scenario.benchmark_id}")
    print(f"Scenario ID: {sid}, idx_base={idx_base}, eval_pairs={len(eval_pairs)}")
    print(f"Candidates: {len(candidates)}")

    results = []
    for cx, cy in candidates:
        mean_iou, used = evaluate_candidate(
            scenario=scenario,
            eval_pairs=eval_pairs,
            center_x=cx,
            center_y=cy,
            x_offset=args.x_offset,
            y_offset=args.y_offset,
        )
        results.append((mean_iou, used, cx, cy))

    results.sort(key=lambda x: x[0], reverse=True)
    best = results[0]

    print("\nTop candidates by mean IoU:")
    top_k = max(1, min(args.top_k, len(results)))
    for rank, (score, used, cx, cy) in enumerate(results[:top_k], start=1):
        print(f"  {rank:02d}. center=({cx:.3f}, {cy:.3f})  mean_iou={score:.6f}  used={used}")

    print("\nBest center:")
    print(f"  center_x={best[2]:.6f}")
    print(f"  center_y={best[3]:.6f}")
    print(f"  mean_iou={best[0]:.6f}")
    print(f"  used_anchors={best[1]}")


if __name__ == "__main__":
    main()
