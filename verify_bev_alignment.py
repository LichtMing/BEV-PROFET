#!/usr/bin/env python3
"""
verify_bev_alignment.py
-----------------------
Overlay CommonRoad XML vehicle rectangles on BEV probability maps
to visually verify coordinate transformation correctness.

Uses BEV sample 0_33000 (k=0, timestamp_now=33000ms).
  T1 → vehicles at 31000ms → CR step 310 → scenario 3, ts 9
  T2 → vehicles at 32000ms → CR step 320 → scenario 3, ts 19
  T3 → vehicles at 33000ms → CR step 330 → scenario 3, ts 29

Output: bev_alignment_check.png (2×3 grid: top=predictions, bottom=GT labels)
"""
import os, sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms

# ── Paths ──
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "EthicalTrajectoryPlanning"))

from commonroad.common.file_reader import CommonRoadFileReader

# ══════════════════════ BEV Geometry ══════════════════════
CENTER_X, CENTER_Y = 1003.0, 995.0   # INTERACTION world center of BEV
AREA = 144.0       # meters covered
SIZE = 288         # pixels (2× upsampled from 144)
RES = AREA / SIZE  # 0.5 m/pixel
X_OFF, Y_OFF = 945.0, 993.5  # CommonRoad → INTERACTION offset


def cr_to_bev(cr_x, cr_y):
    """CommonRoad (x, y) → BEV 288×288 pixel (row, col) as floats."""
    ix = cr_x + X_OFF
    iy = cr_y + Y_OFF
    col_144 = (ix - CENTER_X) + AREA / 2 + 0.5
    row_144 = -(iy - CENTER_Y) + AREA / 2 + 0.5
    return row_144 * 2, col_144 * 2


def get_vehicles(scenario, ts):
    """Return list of vehicle info dicts at time step ts."""
    vehicles = []
    for obs in scenario.dynamic_obstacles:
        try:
            s = obs.state_at_time(ts)
            vehicles.append(dict(
                id=obs.obstacle_id,
                x=s.position[0], y=s.position[1],
                ori=s.orientation,
                L=obs.obstacle_shape.length,
                W=obs.obstacle_shape.width,
            ))
        except Exception:
            pass
    return vehicles


def draw_panel(ax, img, title, vehicles, cmap="hot", is_binary=False):
    """Draw BEV image + cyan rotated rectangles for XML vehicles."""
    vmax = 1.0 if is_binary else max(img.max(), 0.01)
    ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax,
              origin="upper", interpolation="nearest")

    for v in vehicles:
        r, c = cr_to_bev(v["x"], v["y"])
        lp = v["L"] / RES   # length in pixels
        wp = v["W"] / RES   # width in pixels
        # y-axis is flipped (row↓) → image angle = -orientation
        ang = -np.degrees(v["ori"])

        # Rectangle centered at (c, r), long side = lp along heading
        rect = mpatches.Rectangle(
            (c - lp / 2, r - wp / 2), lp, wp,
            linewidth=1.3, edgecolor="cyan", facecolor="none",
        )
        t = (mtransforms.Affine2D().rotate_deg_around(c, r, ang)
             + ax.transData)
        rect.set_transform(t)
        ax.add_patch(rect)

        ax.plot(c, r, "c+", ms=6, mew=1.0)
        ax.annotate(str(v["id"]), (c + 3, r - 3),
                    color="cyan", fontsize=5, fontweight="bold")

    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, SIZE)
    ax.set_ylim(SIZE, 0)
    ax.set_aspect("equal")


# ══════════════════════ Main ══════════════════════
BEV_SAMPLE = "0_33000"
SCENARIO   = os.path.join(
    BASE, "EthicalTrajectoryPlanning", "scenarios",
    "USA_Intersection-1_3_T-1.xml")

# -- Load BEV predictions T1/T2/T3 --
prob_dir = os.path.join(BASE, "bev_v2x_transformer", "BEVPredProb", BEV_SAMPLE)
T = [np.load(os.path.join(prob_dir, f"T{i}.npy")) for i in (1, 2, 3)]
print(f"Predictions loaded from {BEV_SAMPLE}: shapes {[t.shape for t in T]}")

# -- Load BEV label (ground truth) --
lab_path = os.path.join(BASE, "bev_v2x_transformer", "BEVLabel_01",
                        BEV_SAMPLE + ".npy")
has_lab = os.path.exists(lab_path)
if has_lab:
    lab = np.load(lab_path)   # (5, 288, 288) uint8
    print(f"Label loaded: shape={lab.shape}")
else:
    print("Label not found, bottom row will be empty")

# -- Load CommonRoad scenario --
scen, _ = CommonRoadFileReader(SCENARIO).open()
print(f"Scenario: {scen.benchmark_id}, obstacles: {len(scen.dynamic_obstacles)}")

# -- Time mapping (all within scenario 3) --
panels = [
    (0,  9, "T1 pred → 31000 ms (ts 9)"),
    (1, 19, "T2 pred → 32000 ms (ts 19)"),
    (2, 29, "T3 pred → 33000 ms (ts 29)"),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

for col_idx, (tidx, ts, desc) in enumerate(panels):
    vehs = get_vehicles(scen, ts)
    print(f"  ts={ts}: {len(vehs)} vehicles visible")

    # Top row: model predictions (hot colormap)
    draw_panel(axes[0, col_idx], T[tidx], f"Pred — {desc}", vehs)

    # Bottom row: ground truth label channel (gray)
    if has_lab:
        draw_panel(axes[1, col_idx], lab[2 + tidx].astype(float),
                   f"GT label ch{2+tidx} — {desc}",
                   vehs, cmap="gray", is_binary=True)
    else:
        axes[1, col_idx].axis("off")
        axes[1, col_idx].text(0.5, 0.5, "No label",
                              ha="center", va="center",
                              transform=axes[1, col_idx].transAxes)

fig.suptitle(
    f"BEV ↔ CommonRoad Alignment: {BEV_SAMPLE} ↔ {scen.benchmark_id}\n"
    "Cyan rectangles = XML vehicle footprints  |  "
    f"Center=({CENTER_X},{CENTER_Y})  Offset=({X_OFF},{Y_OFF})",
    fontsize=11,
)
plt.tight_layout()
out = os.path.join(BASE, "bev_alignment_check.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close()
print(f"\n✓ Saved → {out}")
