#!/usr/bin/env python3
"""Test adaptive-resolution InteractionMap."""
import sys, numpy as np, math, time
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

# Adaptive (default)
imap_a = InteractionMap(
    bev_map=bev_map, anchor_state=s0, size=0.5, obs_width=2.0, update_length=17)
# Uniform (original)
imap_u = InteractionMap(
    bev_map=bev_map, anchor_state=s0, size=0.5, obs_width=2.0, update_length=17,
    adaptive_resolution=False)

print("=== Grid size comparison ===")
print(f"Uniform:   {imap_u.risk_map.shape} = {imap_u.risk_map.size:,} pixels")
print(f"Adaptive:  {imap_a.risk_map.shape} = {imap_a.risk_map.size:,} pixels")
print(f"Compression: {imap_u.risk_map.size / imap_a.risk_map.size:.2f}x")

# Coordinate round-trip
ego = s0.position
test_offsets = [
    np.array([0, 0]),
    np.array([5, 3]),
    np.array([15, 10]),
    np.array([25, 20]),
    np.array([40, 35]),
    np.array([80, -70]),
]

print("\n=== position -> pixel -> position round-trip ===")
for off in test_offsets:
    p = ego + off
    d = np.linalg.norm(off)
    px = imap_a.position_to_pixel(p)
    p_back = imap_a.pixel_to_position(np.array(px))
    err = np.linalg.norm(p - p_back)
    in_map = imap_a.in_map_check(p)
    print(f"  d={d:6.1f}m  px={px}  err={err:.2f}m  in_map={in_map}")

# sum_traj_risk
print("\n=== sum_traj_risk ===")
near_pt = ego + np.array([3, 2])
px_a = imap_a.position_to_pixel(near_pt)
px_u = imap_u.position_to_pixel(near_pt)
print(f"Near point pixel: adaptive={px_a}, uniform={px_u}")
imap_a.risk_map[px_a[0], px_a[1]] = 0.75
imap_u.risk_map[px_u[0], px_u[1]] = 0.75
traj = [[near_pt[0], near_pt[1], 0, 0, 5, 0]]
r_a = imap_a.sum_traj_risk(traj)
r_u = imap_u.sum_traj_risk(traj)
print(f"Sum traj risk: adaptive={r_a:.4f}  uniform={r_u:.4f}")

print("\nDone!")
