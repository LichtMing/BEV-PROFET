#!/usr/bin/env python3
"""
Visualize InteractionMap grid mapping on a scenario or a raster map.

Examples:
  python visualize_grid_mapping.py \
    --scenario EthicalTrajectoryPlanning/scenarios/USA_Intersection-1_3_T-1.xml \
    --ego-id 1 --time-step 0 --output grid.png

  python visualize_grid_mapping.py \
    --map bev_v2x_transformer/BEVPredProb/0_33000/T1.npy \
    --map-type bev --output grid_bev.png

  python visualize_grid_mapping.py \
    --resolution-mode linear --linear-growth-rate 0.03 --output grid_linear.png
"""

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

BASE = os.path.dirname(os.path.abspath(__file__))
SYS_PATH = os.path.join(BASE, "EthicalTrajectoryPlanning")
if SYS_PATH not in sys.path:
    sys.path.insert(0, SYS_PATH)

from planner.Frenet.utils.InteractionMap import InteractionMap


@dataclass
class AnchorState:
    position: np.ndarray
    orientation: float = 0.0


def _parse_res_bands(text: str):
    if not text:
        return None
    bands = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        radius_str, mpp_str = part.split(":")
        bands.append((float(radius_str), float(mpp_str)))
    return bands


def _load_map(path: str) -> np.ndarray:
    if path.lower().endswith(".npy"):
        return np.load(path)
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("PIL is required to load non-npy images") from exc
    img = Image.open(path).convert("L")
    return np.array(img)


def _load_scenario(path: str):
    from commonroad.common.file_reader import CommonRoadFileReader

    scenario, _ = CommonRoadFileReader(path).open()
    return scenario


def _pick_anchor_state(scenario, ego_id: int, time_step: int) -> AnchorState:
    if scenario is None:
        return AnchorState(position=np.array([0.0, 0.0], dtype=float), orientation=0.0)

    if ego_id is None:
        if scenario.dynamic_obstacles:
            ego_id = scenario.dynamic_obstacles[0].obstacle_id
        else:
            return AnchorState(position=np.array([0.0, 0.0], dtype=float), orientation=0.0)

    obs = scenario.obstacle_by_id(ego_id)
    if obs is None:
        # Fallback to the first dynamic obstacle if the requested ID is missing.
        if scenario.dynamic_obstacles:
            obs = scenario.dynamic_obstacles[0]
        else:
            return AnchorState(position=np.array([0.0, 0.0], dtype=float), orientation=0.0)

    state = obs.state_at_time(time_step) if time_step is not None else obs.initial_state
    if state is None:
        state = obs.initial_state

    orientation = getattr(state, "orientation", 0.0) or 0.0
    return AnchorState(position=np.array(state.position, dtype=float), orientation=float(orientation))


def _draw_scenario(ax, scenario, time_step: int, show_label: bool):
    from commonroad.visualization.draw_dispatch_cr import draw_object

    draw_object(
        scenario,
        draw_params={
            "time_begin": int(time_step),
            "dynamic_obstacle": {
                "draw_shape": True,
                "draw_bounding_box": True,
                "draw_icon": False,
                "show_label": bool(show_label),
            },
        },
        ax=ax,
    )


def _draw_grid(ax, imap: InteractionMap, stride: int, edgecolor: str,
               facecolor: str, alpha: float, linewidth: float):
    patches = []
    grid_side = imap.risk_map.shape[0]

    for r in range(0, grid_side, max(1, stride)):
        x_center = imap._pixel_to_coord_1d(int(r)) + imap.ego_center[0]
        mpp_r = imap._mpp_at_pixel_1d(int(r))
        x0 = x_center - mpp_r / 2.0
        for c in range(0, grid_side, max(1, stride)):
            y_center = imap._pixel_to_coord_1d(int(c)) + imap.ego_center[1]
            mpp_c = imap._mpp_at_pixel_1d(int(c))
            y0 = y_center - mpp_c / 2.0
            rect = Rectangle((x0, y0), mpp_r, mpp_c)
            patches.append(rect)

    collection = PatchCollection(
        patches,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        zorder=25,
    )
    ax.add_collection(collection)


def _add_map_background(ax, map_arr: np.ndarray, map_center, map_range_m, alpha: float):
    cx, cy = map_center
    half = map_range_m / 2.0
    extent = [cx - half, cx + half, cy - half, cy + half]
    ax.imshow(map_arr, extent=extent, origin="upper", cmap="gray", alpha=alpha, zorder=1)


def main():
    parser = argparse.ArgumentParser(description="Visualize InteractionMap grid mapping")
    parser.add_argument("--scenario", type=str, default=None, help="CommonRoad XML path")
    parser.add_argument("--map", type=str, default=None, help="Raster map path (.npy or image)")
    parser.add_argument("--map-type", type=str, default="bev", choices=["bev", "custom"],
                        help="Map type to determine default extent")
    parser.add_argument("--map-center-x", type=float, default=None, help="Map center x in meters")
    parser.add_argument("--map-center-y", type=float, default=None, help="Map center y in meters")
    parser.add_argument("--map-range", type=float, default=None, help="Map coverage size in meters")
    parser.add_argument("--map-res", type=float, default=None, help="Map resolution (m/px)")
    parser.add_argument("--map-alpha", type=float, default=0.55, help="Map alpha")

    parser.add_argument("--ego-id", type=int, default=None, help="Ego obstacle id")
    parser.add_argument("--time-step", type=int, default=0, help="Scenario time step")
    parser.add_argument("--show-label", action="store_true", help="Show obstacle labels")

    parser.add_argument("--adaptive", action="store_true", help="Enable adaptive grid")
    parser.add_argument("--uniform", action="store_true", help="Force uniform grid")
    parser.add_argument("--resolution-mode", type=str, default="bands",
                        choices=["bands", "linear", "speed"], help="Adaptive mode")
    parser.add_argument("--size", type=float, default=0.5, help="Base resolution (m/px)")
    parser.add_argument("--res-bands", type=str, default=None,
                        help="Bands like '20:0.5,50:1.5,100:4.0'")
    parser.add_argument("--linear-growth-rate", type=float, default=0.03, help="Linear growth rate")
    parser.add_argument("--ego-velocity", type=float, default=0.0, help="Ego velocity for speed mode")
    parser.add_argument("--speed-band-scale", type=float, default=0.02, help="Speed band scale")
    parser.add_argument("--speed-risk-factor", type=float, default=0.02, help="Speed risk factor")
    parser.add_argument("--max-distance", type=float, default=None, help="Max distance (m)")
    parser.add_argument("--update-length", type=int, default=17, help="Update length (unused)")
    parser.add_argument("--obs-width", type=float, default=2.0, help="Obstacle width (unused)")

    parser.add_argument("--grid-alpha", type=float, default=0.7, help="Grid alpha")
    parser.add_argument("--grid-edge", type=str, default="#2b2b2b", help="Grid edge color")
    parser.add_argument("--grid-face", type=str, default="none", help="Grid face color")
    parser.add_argument("--grid-linewidth", type=float, default=0.25, help="Grid line width")
    parser.add_argument("--grid-stride", type=int, default=1, help="Draw every Nth cell")

    parser.add_argument("--area", type=float, default=None, help="Half-size of view window (m)")
    parser.add_argument("--output", type=str, default="grid_mapping.png", help="Output image path")

    args = parser.parse_args()

    scenario = _load_scenario(args.scenario) if args.scenario else None
    anchor = _pick_anchor_state(scenario, args.ego_id, args.time_step)

    map_arr = _load_map(args.map) if args.map else None

    use_adaptive = args.adaptive or not args.uniform
    res_bands = _parse_res_bands(args.res_bands)

    if not use_adaptive:
        if map_arr is not None:
            bev_map = map_arr
        else:
            if args.max_distance is None:
                raise ValueError("--max-distance is required for uniform grid without --map")
            side = int(round((args.max_distance * 2) / args.size))
            side = max(1, side)
            bev_map = np.zeros((side, side), dtype=np.uint8)
    else:
        bev_map = np.zeros((1, 1), dtype=np.uint8)

    imap = InteractionMap(
        bev_map=bev_map,
        anchor_state=anchor,
        size=args.size,
        obs_width=args.obs_width,
        update_length=args.update_length,
        adaptive_resolution=use_adaptive,
        res_bands=res_bands,
        resolution_mode=args.resolution_mode,
        linear_growth_rate=args.linear_growth_rate,
        ego_velocity=args.ego_velocity,
        speed_band_scale=args.speed_band_scale,
        speed_risk_factor=args.speed_risk_factor,
        max_distance=args.max_distance,
    )

    fig, ax = plt.subplots(figsize=(10, 10))

    if scenario is not None:
        _draw_scenario(ax, scenario, args.time_step, args.show_label)

    if map_arr is not None:
        if args.map_type == "bev":
            default_center = (58.0, 1.5)
            default_range = 144.0
            default_res = 0.5
        else:
            default_center = (0.0, 0.0)
            default_range = None
            default_res = 1.0

        map_res = args.map_res if args.map_res is not None else default_res
        map_range_m = args.map_range
        if map_range_m is None:
            map_range_m = default_range
        if map_range_m is None:
            map_range_m = map_arr.shape[0] * map_res

        center_x = args.map_center_x if args.map_center_x is not None else default_center[0]
        center_y = args.map_center_y if args.map_center_y is not None else default_center[1]
        _add_map_background(ax, map_arr, (center_x, center_y), map_range_m, args.map_alpha)

    _draw_grid(
        ax,
        imap,
        stride=args.grid_stride,
        edgecolor=args.grid_edge,
        facecolor=args.grid_face,
        alpha=args.grid_alpha,
        linewidth=args.grid_linewidth,
    )

    ax.plot(anchor.position[0], anchor.position[1], "+", color="#c0392b", markersize=8, zorder=30)
    ax.set_aspect("equal")

    if args.area is not None:
        area = args.area
        ax.set_xlim(anchor.position[0] - area, anchor.position[0] + area)
        ax.set_ylim(anchor.position[1] - area, anchor.position[1] + area)
    else:
        area = imap.max_distance
        ax.set_xlim(anchor.position[0] - area, anchor.position[0] + area)
        ax.set_ylim(anchor.position[1] - area, anchor.position[1] + area)

    ax.set_title(
        f"grid: mode={args.resolution_mode}, adaptive={use_adaptive}, size={args.size} m/px",
        fontsize=10,
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
