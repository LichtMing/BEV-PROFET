#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualise CommonRoad XML scenarios.

Usage:
    # Single file
    python3 visualize_commonroad.py path/to/scenario.xml

    # All XMLs in a directory (including sub-folders like track_000/)
    python3 visualize_commonroad.py path/to/scenarios_dir

    # Default: process EthicalTrajectoryPlanning/scenarios/ and its sub-folders
    python3 visualize_commonroad.py

Output PNGs are saved to  <script_dir>/scenario_vis/  preserving any
sub-folder structure (e.g. track_000/, track_001/, …).
"""

import glob
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import Normalize

warnings.filterwarnings("ignore")

from commonroad.common.file_reader import CommonRoadFileReader


# ── colour palette ──────────────────────────────────────────────────────────
LANELET_FILL   = "#d9d9d9"
LANELET_EDGE   = "#888888"
BOUND_LEFT     = "#4a90d9"
BOUND_RIGHT    = "#d94a4a"
OBS_CMAP       = plt.cm.Set1       # each obstacle gets a distinct colour
TRAJ_ALPHA     = 0.7
ARROW_SCALE    = 0.6               # velocity arrow length multiplier


def draw_lanelet_network(ax, lanelet_network):
    """Draw lanelets as filled polygons with left/right boundaries."""
    for ll in lanelet_network.lanelets:
        left  = ll.left_vertices
        right = ll.right_vertices

        # filled polygon (left → right reversed → close)
        poly = np.vstack([left, right[::-1]])
        patch = plt.Polygon(poly, closed=True,
                            facecolor=LANELET_FILL, edgecolor=LANELET_EDGE,
                            linewidth=0.6, alpha=0.8, zorder=1)
        ax.add_patch(patch)

        # boundary lines
        ax.plot(left[:, 0],  left[:, 1],  color=BOUND_LEFT,
                linewidth=0.9, linestyle="-", zorder=2)
        ax.plot(right[:, 0], right[:, 1], color=BOUND_RIGHT,
                linewidth=0.9, linestyle="-", zorder=2)

        # lanelet ID label at the centre of the centreline
        centre = ll.center_vertices
        mid = centre[len(centre) // 2]
        ax.text(mid[0], mid[1], str(ll.lanelet_id),
                fontsize=6, ha="center", va="center",
                color="#555555", zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))


def draw_obstacle(ax, obs, colour, time_step=None):
    """Draw one dynamic obstacle: rectangle + trajectory + velocity arrow."""
    # ---- pick state -------------------------------------------------------
    if time_step is None or time_step == 0:
        state = obs.initial_state
    else:
        traj_states = obs.prediction.trajectory.state_list if obs.prediction else []
        matches = [s for s in traj_states if s.time_step == time_step]
        state = matches[0] if matches else obs.initial_state

    x, y   = state.position
    theta  = state.orientation
    v      = state.velocity
    length = obs.obstacle_shape.length
    width  = obs.obstacle_shape.width

    # ---- vehicle rectangle (rotated) --------------------------------------
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # corners relative to centre
    dx = length / 2.0
    dy = width  / 2.0
    corners_local = np.array([
        [-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]
    ])
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    corners_world = corners_local @ rot.T + np.array([x, y])
    rect = plt.Polygon(corners_world, closed=True,
                        facecolor=colour, edgecolor="black",
                        linewidth=0.8, alpha=0.85, zorder=10)
    ax.add_patch(rect)

    # ---- velocity arrow ---------------------------------------------------
    if v > 0.05:
        ax.annotate("", xy=(x + cos_t * v * ARROW_SCALE,
                            y + sin_t * v * ARROW_SCALE),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color=colour,
                                   lw=1.5, mutation_scale=10),
                    zorder=11)

    # ---- full trajectory line ---------------------------------------------
    traj_states = obs.prediction.trajectory.state_list if obs.prediction else []
    if traj_states:
        all_states = [obs.initial_state] + traj_states
        xs = [s.position[0] for s in all_states]
        ys = [s.position[1] for s in all_states]
        ax.plot(xs, ys, color=colour, linewidth=1.2, linestyle="--",
                alpha=TRAJ_ALPHA, zorder=8)
        # mark trajectory end
        ax.plot(xs[-1], ys[-1], "x", color=colour, markersize=6,
                markeredgewidth=1.5, zorder=9)

    # ---- label ------------------------------------------------------------
    ax.text(x, y + width * 0.8, f"id {obs.obstacle_id}",
            fontsize=7, ha="center", va="bottom", color=colour,
            fontweight="bold", zorder=12)


def visualise(xml_path: str, time_step: int = 0, out_path: str = None):
    """Load a CommonRoad XML and produce a single-frame visualisation."""
    scenario, _ = CommonRoadFileReader(xml_path).open()

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_aspect("equal")

    # ---- draw road --------------------------------------------------------
    draw_lanelet_network(ax, scenario.lanelet_network)

    # ---- draw obstacles ---------------------------------------------------
    n_obs = len(scenario.dynamic_obstacles)
    colours = [OBS_CMAP(i / max(n_obs, 1)) for i in range(n_obs)]
    for obs, col in zip(scenario.dynamic_obstacles, colours):
        draw_obstacle(ax, obs, col, time_step=time_step)

    # ---- axes / title -----------------------------------------------------
    ax.set_xlabel("x  [m]")
    ax.set_ylabel("y  [m]")
    ax.set_title(f"{os.path.basename(xml_path)}   (t = {time_step * scenario.dt:.1f} s)",
                 fontsize=12)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.autoscale()
    # add some padding
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    pad = 2.0
    ax.set_xlim(xlim[0] - pad, xlim[1] + pad)
    ax.set_ylim(ylim[0] - pad, ylim[1] + pad)

    # ---- legend -----------------------------------------------------------
    legend_handles = [
        mpatches.Patch(facecolor=LANELET_FILL, edgecolor=LANELET_EDGE, label="Lanelet"),
        plt.Line2D([], [], color=BOUND_LEFT,  linewidth=1, label="Left boundary"),
        plt.Line2D([], [], color=BOUND_RIGHT, linewidth=1, label="Right boundary"),
    ]
    for obs, col in zip(scenario.dynamic_obstacles, colours):
        legend_handles.append(
            mpatches.Patch(facecolor=col, edgecolor="black",
                           label=f"Obstacle {obs.obstacle_id}  "
                                 f"({obs.obstacle_shape.length:.1f}x"
                                 f"{obs.obstacle_shape.width:.1f} m)")
        )
    ax.legend(handles=legend_handles, loc="upper left", fontsize=7,
              framealpha=0.9)

    plt.tight_layout()
    if out_path is None:
        out_path = xml_path.rsplit(".", 1)[0] + "_vis.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")
    return out_path


def batch_visualise(scenario_dir: str, out_root: str, time_step: int = 0):
    """Visualise every .xml file under *scenario_dir* (recursively).

    Output PNGs are written to *out_root*, preserving any sub-folder
    structure relative to *scenario_dir*.
    """
    scenario_dir = os.path.abspath(scenario_dir)
    xml_files = sorted(glob.glob(os.path.join(scenario_dir, "**", "*.xml"), recursive=True))
    if not xml_files:
        print(f"No .xml files found under {scenario_dir}")
        return

    print(f"Found {len(xml_files)} XML scenario(s) under {scenario_dir}")
    for i, xml_path in enumerate(xml_files, 1):
        rel = os.path.relpath(xml_path, scenario_dir)
        out_sub = os.path.join(out_root, os.path.dirname(rel)) if os.path.dirname(rel) else out_root
        os.makedirs(out_sub, exist_ok=True)
        base = os.path.splitext(os.path.basename(xml_path))[0]
        out_path = os.path.join(out_sub, base + "_vis.png")
        try:
            visualise(xml_path, time_step=time_step, out_path=out_path)
            print(f"  [{i}/{len(xml_files)}] {rel}  →  {os.path.relpath(out_path, out_root)}")
        except Exception as e:
            print(f"  [{i}/{len(xml_files)}] FAILED {rel}: {e}")


# ── main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUT_ROOT   = os.path.join(SCRIPT_DIR, "scenario_vis")

    default_dir = os.path.join(SCRIPT_DIR, "EthicalTrajectoryPlanning", "scenarios")
    target = sys.argv[1] if len(sys.argv) > 1 else default_dir

    if os.path.isfile(target) and target.endswith(".xml"):
        # Single file mode
        os.makedirs(OUT_ROOT, exist_ok=True)
        base = os.path.splitext(os.path.basename(target))[0]
        out_path = os.path.join(OUT_ROOT, base + "_vis.png")
        visualise(target, out_path=out_path)
    elif os.path.isdir(target):
        # Directory / batch mode
        os.makedirs(OUT_ROOT, exist_ok=True)
        batch_visualise(target, OUT_ROOT)
    else:
        print(f"Path not found or not a .xml file: {target}")
        sys.exit(1)

    print(f"\nAll outputs saved to: {OUT_ROOT}")
