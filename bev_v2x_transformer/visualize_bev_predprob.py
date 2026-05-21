#!/usr/bin/env python3
"""
Visualize BEV prediction probability maps (BEVPredProb / mergingBEVPredProb).

Produces PNG visualizations mirroring the input directory structure with
"_vision" appended to the output root name.

Usage:
    # Per-frame mode (default): T1.png, T2.png, T3.png per scene
    python visualize_bev_predprob.py --pred-dir ./BEVPredProb -n 10

    # Combined mode: single overview PNG per scene with optional labels
    python visualize_bev_predprob.py --pred-dir ./BEVPredProb \\
        --label-dir ./BEVLabel_01 --style combined -n 10

    # Merging dataset
    python visualize_bev_predprob.py --pred-dir ./mergingBEVPredProb \\
        --label-dir ./merging/BEVLabel --style combined
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── tqdm is optional ─────────────────────────────────────────────────────────
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_label(label_dir, scene_name):
    """Return path to label .npy for *scene_name*, or None."""
    if not label_dir or not os.path.isdir(label_dir):
        return None
    path = os.path.join(label_dir, scene_name + ".npy")
    return path if os.path.isfile(path) else None


def _render_single(ax, data, title, cmap, vmin, vmax, add_colorbar):
    """Render a single probability map on an Axes."""
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest", origin="upper")
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    if add_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# ── per-frame mode ───────────────────────────────────────────────────────────

def _per_frame(scene_name, pred_dir, output_dir, label_dir, args):
    """Save T1.png / T2.png / T3.png (and optional _label variants)."""
    out_scene = os.path.join(output_dir, scene_name)
    os.makedirs(out_scene, exist_ok=True)

    label_path = _find_label(label_dir, scene_name)
    label = np.load(label_path) if label_path else None  # (5, 288, 288)

    for ti, t_name in enumerate(("T1", "T2", "T3"), start=2):
        pred = np.load(os.path.join(pred_dir, scene_name, t_name + ".npy"))
        vmax = max(float(pred.max()), 1e-2)

        # -- prediction --
        fig, ax = plt.subplots(figsize=(6, 6))
        _render_single(ax, pred, f"{scene_name}  {t_name}  pred",
                       args.cmap, 0.0, vmax, not args.no_colorbar)
        fig.savefig(os.path.join(out_scene, f"{t_name}.png"),
                    dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

        # -- label (if available) --
        if label is not None and ti < label.shape[0]:
            fig, ax = plt.subplots(figsize=(6, 6))
            _render_single(ax, label[ti].astype(np.float32),
                           f"{scene_name}  {t_name}  label",
                           "gray", 0.0, 1.0, not args.no_colorbar)
            fig.savefig(os.path.join(out_scene, f"{t_name}_label.png"),
                        dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)


# ── combined mode ─────────────────────────────────────────────────────────────

def _combined(scene_name, pred_dir, output_dir, label_dir, args):
    """Save a single vision.png with 2x3 (pred top, label bottom) layout."""
    out_scene = os.path.join(output_dir, scene_name)
    os.makedirs(out_scene, exist_ok=True)

    label_path = _find_label(label_dir, scene_name)
    label = np.load(label_path) if label_path else None
    has_label = label is not None and label.shape[0] >= 5

    nrows = 2 if has_label else 1
    fig, axes = plt.subplots(nrows, 3, figsize=(15, 5 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)  # keep 2-D

    for col, t_name in enumerate(("T1", "T2", "T3")):
        pred = np.load(os.path.join(pred_dir, scene_name, t_name + ".npy"))
        vmax = max(float(pred.max()), 1e-2)

        _render_single(axes[0, col], pred, f"{t_name}  pred",
                       args.cmap, 0.0, vmax, not args.no_colorbar)

        if has_label:
            _render_single(axes[1, col], label[col + 2].astype(np.float32),
                           f"{t_name}  label",
                           "gray", 0.0, 1.0, not args.no_colorbar)

    fig.suptitle(scene_name, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_scene, "vision.png"),
                dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize BEV prediction probability maps")
    parser.add_argument("--pred-dir", required=True,
                        help="Path to prediction directory (BEVPredProb or mergingBEVPredProb)")
    parser.add_argument("--label-dir", default=None,
                        help="Path to label directory (e.g. BEVLabel_01 or merging/BEVLabel)")
    parser.add_argument("--output-dir", default=None,
                        help="Output root directory (default: {pred-dir}_vision)")
    parser.add_argument("--style", choices=("per-frame", "combined"),
                        default="per-frame",
                        help="per-frame = individual T1/T2/T3 PNGs; combined = single overview PNG")
    parser.add_argument("--max-samples", "-n", type=int, default=None,
                        help="Maximum number of scenes to process")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Output image DPI (default: 150)")
    parser.add_argument("--cmap", default="hot",
                        help="Matplotlib colormap for predictions (default: hot)")
    parser.add_argument("--no-colorbar", action="store_true",
                        help="Omit colorbar from images")

    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pred_dir = args.pred_dir if os.path.isabs(args.pred_dir) \
        else os.path.join(script_dir, args.pred_dir)

    if not os.path.isdir(pred_dir):
        sys.exit(f"ERROR: prediction directory not found: {pred_dir}")

    if args.output_dir is None:
        output_dir = pred_dir.rstrip("/\\") + "_vision"
    else:
        output_dir = args.output_dir if os.path.isabs(args.output_dir) \
            else os.path.join(script_dir, args.output_dir)

    label_dir = None
    if args.label_dir:
        label_dir = args.label_dir if os.path.isabs(args.label_dir) \
            else os.path.join(script_dir, args.label_dir)
        if not os.path.isdir(label_dir):
            print(f"WARNING: label directory not found, skipping labels: {label_dir}")
            label_dir = None

    # Discover scenes (subdirs that contain T1.npy)
    scenes = sorted(
        d for d in os.listdir(pred_dir)
        if os.path.isdir(os.path.join(pred_dir, d))
        and os.path.isfile(os.path.join(pred_dir, d, "T1.npy"))
    )

    if not scenes:
        sys.exit(f"ERROR: no scene directories with T1.npy found in {pred_dir}")

    if args.max_samples:
        scenes = scenes[: args.max_samples]

    handler = _per_frame if args.style == "per-frame" else _combined

    print(f"Pred dir : {pred_dir}")
    print(f"Label dir: {label_dir or '(none)'}")
    print(f"Output   : {output_dir}")
    print(f"Style    : {args.style}")
    print(f"Scenes   : {len(scenes)}")
    print()

    for name in tqdm(scenes, desc="Rendering"):
        handler(name, pred_dir, output_dir, label_dir, args)

    print(f"\nDone. Output written to {output_dir}")


if __name__ == "__main__":
    main()
