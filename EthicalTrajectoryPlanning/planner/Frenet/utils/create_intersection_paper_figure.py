#!/usr/bin/env python3
"""Build a publication-ready collision/safe comparison figure."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_ego_series(log_path: Path, append_next: bool = False):
    with log_path.open(newline="") as log_file:
        rows = list(csv.DictReader(log_file, delimiter=";"))
    times, speeds, accelerations = [], [], []
    for row in rows:
        best = json.loads(row["valid_trajectories"])[0]
        times.append(float(row["time"]))
        speeds.append(float(best["v"][0]))
        accelerations.append(float(best["s_dd"][0]))
    if append_next:
        best = json.loads(rows[-1]["valid_trajectories"])[0]
        times.append(times[-1] + 0.1)
        speeds.append(float(best["v"][1]))
        accelerations.append(float(best["s_dd"][1]))
    return times, speeds, accelerations


def add_frame(
    ax, path: Path, title: str, highlight: bool = False,
    highlight_color: str = "#d62728",
):
    ax.imshow(plt.imread(path))
    ax.set_xticks([])
    ax.set_yticks([])
    color = highlight_color if highlight else "#303030"
    width = 3.0 if highlight else 1.2
    for spine in ax.spines.values():
        spine.set_color(color)
        spine.set_linewidth(width)
    ax.set_title(title, fontsize=10, color=color if highlight else "#202020", pad=5)


def add_kinematics(ax, log_path: Path, event_time: float, event_label: str,
                   end_time: float, append_next: bool = False):
    times, speeds, accelerations = load_ego_series(log_path, append_next)
    speed_line = ax.plot(times, speeds, color="#1764ab", linewidth=2.2,
                         label="Speed")[0]
    ax.set_xlim(0.0, end_time)
    ax.set_ylabel("Speed (m/s)", color="#1764ab")
    ax.tick_params(axis="y", labelcolor="#1764ab")
    ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.8)
    ax.axvline(event_time, color="#d62728", linestyle="--", linewidth=1.8)

    accel_ax = ax.twinx()
    accel_line = accel_ax.plot(
        times, accelerations, color="#e07a1f", linewidth=1.8,
        label="Acceleration"
    )[0]
    accel_ax.set_ylabel("Acceleration (m/s²)", color="#e07a1f")
    accel_ax.tick_params(axis="y", labelcolor="#e07a1f")
    ax.set_xlabel("Simulation time (s)")
    ax.legend([speed_line, accel_line], ["Speed", "Acceleration"],
              loc="upper right", frameon=True, ncol=2, fontsize=9)
    align_right = event_time / end_time > 0.75
    ax.annotate(
        event_label,
        xy=(event_time, (0.78 if align_right else 0.97)),
        xycoords=("data", "axes fraction"),
        xytext=((-6 if align_right else 6), -4),
        textcoords="offset points",
        color="#d62728",
        fontsize=9,
        ha=("right" if align_right else "left"),
        va="top",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    collision = args.root / "collision_risk_occluded_30m"
    safe = args.root / "safe_ground_truth_clear_500m"

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titleweight": "semibold",
    })
    fig = plt.figure(figsize=(13.5, 12.5), constrained_layout=False)
    grid = GridSpec(
        4, 4, figure=fig, height_ratios=[1.0, 0.62, 1.0, 0.62],
        hspace=0.43, wspace=0.08, left=0.065, right=0.94,
        top=0.965, bottom=0.065,
    )

    safe_frames = [0, 18, 30, 100]
    safe_titles = [
        "t = 0.0 s  Start",
        "t = 1.8 s  Early response",
        "t = 3.0 s  Yielding",
        "t = 10.0 s  Safe exit",
    ]
    for column, (frame, title) in enumerate(zip(safe_frames, safe_titles)):
        add_frame(
            fig.add_subplot(grid[2, column]),
            safe / "frames" / f"frame_{frame:04d}.png",
            title,
            highlight=frame == 18,
            highlight_color="#2ca02c",
        )
    safe_plot = fig.add_subplot(grid[3, :])
    add_kinematics(
        safe_plot, safe / "planner_log.csv", 1.8,
        "Hidden-risk information available before conflict", 10.0,
    )

    collision_frames = [0, 10, 21, 27]
    collision_titles = [
        "t = 0.0 s  Start",
        "t = 1.0 s  Occluded approach",
        "t = 2.1 s  Late response",
        "t = 2.7 s  Impact with ID 164",
    ]
    for column, (frame, title) in enumerate(zip(collision_frames, collision_titles)):
        add_frame(
            fig.add_subplot(grid[0, column]),
            collision / "frames" / f"frame_{frame:04d}.png",
            title,
            highlight=frame == 27,
        )
    collision_plot = fig.add_subplot(grid[1, :])
    add_kinematics(
        collision_plot, collision / "planner_log.csv", 2.8,
        "Collision; impact state held for visibility", 3.2, append_next=True,
    )

    output_png = args.root / "intersection_collision_comparison.png"
    output_pdf = args.root / "intersection_collision_comparison.pdf"
    fig.savefig(output_png, dpi=300, facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(output_png)
    print(output_pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
