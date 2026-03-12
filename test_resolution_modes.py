#!/usr/bin/env python3
"""Test three InteractionMap resolution modes: bands, linear, speed."""
import sys, numpy as np, math
sys.path.insert(0, 'EthicalTrajectoryPlanning')

import matplotlib
matplotlib.use("Agg")

from commonroad.common.file_reader import CommonRoadFileReader
scen, _ = CommonRoadFileReader(
    'EthicalTrajectoryPlanning/scenarios/USA_Intersection-1_3_T-1.xml').open()
obs = scen.obstacle_by_id(160)
s0 = obs.state_at_time(10)

from planner.Frenet.utils.visualization import draw_bev_map
bev_map, fig, ax = draw_bev_map(
    scenario=scen, ego_id=160, time_step=10+36, ego_time_step=10,
    marked_vehicle=160, planning_problem=None, traj=None,
    global_path=None, global_path_after_goal=None,
    driven_traj=[s0], animation_area=50.0,
    traj_time_begin=10, traj_time_end=28,
    ego_traj_time_begin=10, ego_traj_time_end=28, ego_draw_occ=False,
)
import matplotlib.pyplot as plt
plt.close(fig)

from planner.Frenet.utils.InteractionMap import InteractionMap

ego = s0.position

# ── 1. Bands mode (default) ──
print("=" * 60)
print("MODE: bands (piecewise-linear)")
imap_bands = InteractionMap(
    bev_map=bev_map, anchor_state=s0, size=0.5, obs_width=2.0,
    update_length=17, resolution_mode='bands')
print(f"  Grid: {imap_bands.risk_map.shape} = {imap_bands.risk_map.size:,} px")
print(f"  max_distance: {imap_bands.max_distance} m")

# ── 2. Linear mode ──
print("\n" + "=" * 60)
print("MODE: linear (Δs = a + b·d, logarithmic compression)")
imap_linear = InteractionMap(
    bev_map=bev_map, anchor_state=s0, size=0.5, obs_width=2.0,
    update_length=17, resolution_mode='linear',
    linear_growth_rate=0.03, max_distance=100.0)
print(f"  Grid: {imap_linear.risk_map.shape} = {imap_linear.risk_map.size:,} px")
print(f"  max_distance: {imap_linear.max_distance} m")
print(f"  growth_rate: {imap_linear.linear_growth_rate}")

# ── 3. Speed mode (v=0, should match bands) ──
print("\n" + "=" * 60)
print("MODE: speed (v_ego=0, should match bands)")
imap_speed0 = InteractionMap(
    bev_map=bev_map, anchor_state=s0, size=0.5, obs_width=2.0,
    update_length=17, resolution_mode='speed',
    ego_velocity=0.0, speed_band_scale=0.02)
print(f"  Grid: {imap_speed0.risk_map.shape} = {imap_speed0.risk_map.size:,} px")
assert imap_speed0.risk_map.shape == imap_bands.risk_map.shape, \
    "speed mode at v=0 should match bands grid size"
print("  ✓ Grid matches bands (v=0)")

# ── 4. Speed mode (v=20 m/s) ──
print("\n" + "=" * 60)
print("MODE: speed (v_ego=20 m/s)")
imap_speed20 = InteractionMap(
    bev_map=bev_map, anchor_state=s0, size=0.5, obs_width=2.0,
    update_length=17, resolution_mode='speed',
    ego_velocity=20.0, speed_band_scale=0.02)
print(f"  Grid: {imap_speed20.risk_map.shape} = {imap_speed20.risk_map.size:,} px")
print(f"  max_distance: {imap_speed20.max_distance:.1f} m")
print(f"  Scaled bands: {imap_speed20.res_bands}")
assert imap_speed20.risk_map.shape[0] > imap_bands.risk_map.shape[0], \
    "speed mode at v=20 should have larger grid"
print("  ✓ Grid larger than bands")

# ── Coordinate round-trip for all modes ──
print("\n" + "=" * 60)
print("Coordinate round-trip accuracy (all modes)")
test_offsets = [
    np.array([0, 0]),
    np.array([5, 3]),
    np.array([15, 10]),
    np.array([30, -20]),
    np.array([60, 50]),
    np.array([90, -80]),
]
for name, imap in [("bands", imap_bands), ("linear", imap_linear),
                    ("speed0", imap_speed0), ("speed20", imap_speed20)]:
    errs = []
    for off in test_offsets:
        p = ego + off
        if not imap.in_map_check(p):
            continue
        px = imap.position_to_pixel(p)
        p_back = imap.pixel_to_position(np.array(px))
        errs.append(np.linalg.norm(p - p_back))
    max_err = max(errs) if errs else float('nan')
    print(f"  {name:8s}: max_err = {max_err:.3f} m  (near-ego err = {errs[0]:.3f} m)")

# ── mpp_at_pixel for linear mode ──
print("\n" + "=" * 60)
print("Linear mode: mpp at various distances")
for d in [0, 10, 20, 50, 100]:
    px = imap_linear._coord_to_pixel_1d(d)
    mpp = imap_linear._mpp_at_pixel_1d(px)
    expected = 0.5 + 0.03 * d
    print(f"  d={d:4d}m  px_offset={px - imap_linear._half_pixels:+5d}  "
          f"mpp={mpp:.3f} m/px  (expected≈{expected:.3f})")

# ── Compression ratios ──
print("\n" + "=" * 60)
print("Compression ratios vs uniform 400×400")
uniform_px = 400 * 400
for name, imap in [("bands", imap_bands), ("linear", imap_linear),
                    ("speed0", imap_speed0), ("speed20", imap_speed20)]:
    ratio = uniform_px / imap.risk_map.size
    print(f"  {name:8s}: {imap.risk_map.shape[0]:4d}×{imap.risk_map.shape[1]:<4d} "
          f"= {imap.risk_map.size:>6,} px  ({ratio:.1f}× compression)")

print("\n✓ All tests passed!")
