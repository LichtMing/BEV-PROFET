# Project Guidelines

## Architecture

Three subsystems for autonomous driving trajectory planning:

| Module | Path | Purpose |
|---|---|---|
| **EthicalTrajectoryPlanning** | `EthicalTrajectoryPlanning/` | Frenet-frame sampling planner + risk assessment + interaction maps |
| **BEV V2X Transformer** | `bev_v2x_transformer/` | Transformer-based BEV prediction → probability maps for future 3 timesteps |
| **Dataset Convert** | `dataset_convert/` | INTERACTION/highD/inD → CommonRoad XML conversion |

**Data flow**: INTERACTION dataset → `intersection_bev_generator_*.py` → BEV training data → `training.py` → model → `inference_prob_map.py` → `BEVPredProb/*.npy` → `BEVProbLoader` in planner → overlaid onto `InteractionMap` risk → risk-aware trajectory selection.

**Key inheritance**: `Planner` → `FrenetPlanner`; `Agent` → `PlanningAgent`; `ScenarioHandler` → `ScenarioEvaluator`; `PlannerCreator` → `FrenetCreator`.

## Build and Test

```bash
# Environment
conda activate traj_env  # Python 3.9, PyTorch 2.1

# Run planner on a single scenario (must cd to Frenet dir for relative imports)
cd EthicalTrajectoryPlanning/planner/Frenet
python frenet_planner.py --scenario USA_Intersection-1_1_T-1.xml

# Batch evaluate all scenarios
python plannertools/evaluatefrenet.py --weights ethical --all

# BEV Transformer training/inference (must cd to module dir)
cd bev_v2x_transformer
python training.py          # Train
python inference_prob_map.py  # Generate probability maps
```

No formal test suite exists. Validation is done by running `frenet_planner.py --scenario <name>` and checking output.

### End-to-End Pipeline (BEV → Planner)

```bash
# Step 1: Generate BEV probability maps (requires trained model + GPU)
cd bev_v2x_transformer
python inference_prob_map.py   # Outputs to BEVPredProb/{k}_{timestamp_ms}/T1-T3.npy

# Step 2: Ensure planning_fast.json has correct paths
#   "bev_prob_dir": "/absolute/path/to/bev_v2x_transformer/BEVPredProb"
#   "bev_weight": 1.0            (scale factor for BEV risk contribution)

# Step 3: Create saved_fig directories
cd EthicalTrajectoryPlanning
python create_saved_fig_dirs.py

# Step 4: Run planner (active_learning mode with BEV risk overlay)
cd planner/Frenet
python frenet_planner.py --scenario USA_Intersection-1_3_T-1.xml
# Scenarios 3,5-8,11-13,... (134/155) have BEV data; others run with BEV risk = 0
```

## Project Conventions

- **CWD matters**: Planner must run from `planner/Frenet/` due to `from utils.*` and `from prediction import WaleNet` relative imports. BEV code must run from `bev_v2x_transformer/`.
- **Config loading**: All planner configs are JSON in `planner/Frenet/configs/`, loaded via `load_json.py` functions (`load_planning_json`, `load_risk_json`, `load_weight_json`). The `__main__` block uses `planning_fast.json`.
- **active_learning mode** (`planning_fast.json`): Converts each dynamic obstacle into a planning problem, replays ground truth trajectory while exploring alternatives via `BehaviorTree`. Labels/figures saved to paths in config (use absolute paths for current machine).
- **Scenario format**: CommonRoad XML (`.xml`) in `scenarios/` — contains lanelet network, dynamic obstacles with trajectories, and planning problems.
- **`sys.path.append`**: Used extensively instead of proper packaging. `mopl_path` points to `EthicalTrajectoryPlanning/` root.

## BEV Data ↔ Scenario Mapping

BEVLabel/BEVData files use naming `{k}_{timestamp_ms}.npy`:
- **`k`** (0-7): INTERACTION recording index → `vehicle_tracks_00{k}.csv`
- **`timestamp_ms`**: Raw millisecond timestamp from INTERACTION dataset

**Conversion to CommonRoad scenario**:
- CR time step = `timestamp_ms / 100` (dt=0.1s)
- Each scenario covers 150 CR steps → segment = `(CR_step - 1) // 150`
- Example: `0_31000.npy` → CR step 310 → segment 2 → `USA_Intersection-1_3_T-1.xml`, step 9

| case `k` | Segments | Scenario ID range | XML files |
|-----------|----------|-------------------|----------|
| 0 | 20 | 1 – 20 | 19 (ID 4 missing) |
| 1 | 18 | 21 – 38 | 17 (ID 24 missing) |
| 2 | 20 | 39 – 58 | 19 (ID 42 missing) |
| 3 | 20 | 59 – 78 | 19 (ID 62 missing) |
| 4 | 20 | 79 – 98 | 19 (ID 82 missing) |
| 5 | 20 | 99 – 118 | 19 (ID 102 missing) |
| 6 | 20 | 119 – 138 | 19 (ID 122 missing) |
| 7 | 17 | 139 – 155 | 16 (ID 142 missing) |

Total: 155 segments, 147 XML files. The 8 missing IDs (4, 24, 42, 62, 82, 102, 122, 142) are all segment index 3 of each recording — at CR step 451, no vehicle exists at `time_start_scenario` so `obstacle_start_at_zero=True` filtering discards all vehicles, causing `generate_single_scenario()` to return without writing a file (the ID is still consumed).

**Scenario grouping by recording**: `group_scenarios_by_recording.py` copies XML files into `scenarios/track_000/` ~ `scenarios/track_007/` subfolders based on their source recording.

## BEVPredProb Risk Integration

### Overview

`BEVProbLoader` (`planner/Frenet/utils/bev_prob_loader.py`) loads BEV Transformer prediction probability maps and overlays them as an additive risk factor onto the existing InteractionMap-based risk. This affects trajectory selection in both active_learning mode and standard planning mode.

### Coordinate Transformation (CommonRoad ↔ BEV 288×288)

BEV 288×288 covers 144m×144m at 0.5 m/pixel, centered at INTERACTION world (1003, 995).

INTERACTION ↔ CommonRoad offset for USA_Intersection-1: `CR_x = INTER_x - 945`, `CR_y = INTER_y - 993.5`.

```
CommonRoad (x,y) → BEV pixel (row, col):
  col_288 = 2 * (cr_x + 14.5)
  row_288 = -2 * cr_y + 148
```

### BEVPredProb → Scenario Time Mapping

BEV folder `{k}_{timestamp_ms}` → scenario time step:

```
CR_step = timestamp_ms / 100
segment = (CR_step - 1) // 150
time_step_in_scenario = (CR_step - 1) % 150
scenario_id = ID_OFFSET[k] + segment
  where ID_OFFSET = [1, 21, 39, 59, 79, 99, 119, 139]
```

T1.npy/T2.npy/T3.npy = BEV prediction for +1s/+2s/+3s from that timestamp. 134/155 scenarios have BEV data.

### Risk Overlay Logic

**Active_learning mode** (`traj_risk_calc` in `_step_planner`):
```
total_risk(traj) = Σ InteractionMap[i].sum_traj_risk(segment_i)
                 + max_i{ BEVProbLoader.compute_segment_bev_risk(segment_i, t_index=i) }
```
Each of the 3 InteractionMap segments (0–1.8s, 1.8–3.6s, 3.6–5.4s) uses the corresponding T1/T2/T3 prediction. The BEV contribution is the **max** across segments (not sum) to avoid biasing toward shorter trajectories.

**Standard planning mode** (after `sort_frenet_trajectories`):
```
for each valid trajectory fp:
  max_bev_prob = max over i { BEVProbLoader.query_prob(fp.x[i], fp.y[i], t_index) }
  fp.cost += max_bev_prob * bev_weight
```
Time-to-T mapping: 0–1.5s → T1, 1.5–2.5s → T2, 2.5s+ → T3.

**Note**: Using `max` instead of `sum` ensures: (1) risk stays in [0, 1] × bev_weight, (2) fast trajectories that rush through danger are not rewarded for having fewer sample points, (3) the trajectory's danger is determined by its most hazardous point.

### Configuration (`planning_fast.json`)

```json
"active_learning": {
  "bev_prob_dir": "/absolute/path/to/bev_v2x_transformer/BEVPredProb",
  "bev_weight": 1.0
}
```

- `bev_prob_dir`: Path to BEVPredProb root. Default auto-resolved to `../../bev_v2x_transformer/BEVPredProb` relative to project root.
- `bev_weight`: Scaling factor for BEV risk contribution. Set to 0 to disable.

### Modified/New Files

| File | Changes |
|---|---|
| `planner/Frenet/utils/bev_prob_loader.py` | **New**: `BEVProbLoader` class with timestamp scanning, coordinate transform, prob querying |
| `planner/Frenet/frenet_planner.py` | Import `BEVProbLoader`; init in both active_learning and standard mode; overlay BEV risk in `traj_risk_calc` and after `sort_frenet_trajectories` |
| `planner/Frenet/configs/planning_fast.json` | Added `bev_prob_dir`, `bev_weight` fields |

## Visualization Conditions (active_learning mode)

In `frenet_planner.py` `_step_planner()`, figures are only generated when **all** of these hold:
1. `time_step < final_step - max_exploration_time / 0.1` (leave exploration buffer)
2. `prepare_label == True`
3. `time_step % int(store_interval / 0.1) == 0` (current: `store_interval=0.5` → every 5 steps)
4. `len(valid_trajectories) > 0` after search tree traversal
5. Output directory `saved_fig/<benchmark_id>/` exists (no auto-mkdir; use `create_saved_fig_dirs.py`)

**Pitfall**: Short trajectories (e.g., ~70 steps) with `max_exploration_time=5.4` leave only steps 1–15 available. With `store_interval=2.0` requiring `step % 20 == 0`, visualization **never triggers**. **Applied fix**: `store_interval` reduced to `0.5` (every 5 steps), minimum trajectory length reduced from 74 to 59 steps.

**Coverage statistics** (147 scenarios, 858 obstacles, trajectory lengths: min=1, median=114, max=761):

| `store_interval` | `max_exploration_time` | Min trajectory | Qualifying obstacles | Qualifying scenarios |
|:-:|:-:|:-:|:-:|:-:|
| 2.0 (old) | 5.4 | 74 steps | 569/858 (66%) | 134/147 |
| **0.5 (current)** | **5.4** | **59 steps** | **626/858 (73%)** | **137/147** |
| 0.5 | 2.0 | 25 steps | 775/858 (90%) | 144/147 |
| 0.5 | 1.0 | 15 steps | 811/858 (95%) | 144/147 |

## Known Issues & Workarounds

1. **`commonroad_helper_functions/spacial.py` `all_lanelets_by_merging_predecessors_from_lanelet`**: Original code mutates the lanelet network's successor lists via `j[k].successor.append(...)`, corrupting the network for subsequent agent creation. **Fixed locally** with `copy.deepcopy`.
2. **`Polygon.center` returns 0-d numpy array** wrapping a shapely Point in active_learning mode. `lanelet_based_planner.py` patched to extract `[x, y]` from 0-d arrays.
3. **`reference_traj` shared reference**: In `frenet_planner.py`, must `copy.deepcopy` the trajectory to avoid corruption by `update_scenario()`.
4. **`update_scenario()` signature**: `scenario_handler.py` must not pass `active_learning=` kwarg — `agent_sim` doesn't accept it.
5. **Absolute paths in `planning_fast.json`**: `label_path`/`fig_path` must point to valid local directories.
6. **`saved_fig` subdirectories not auto-created**: `plt.savefig` fails silently if `saved_fig/<benchmark_id>/` doesn't exist. Run `python create_saved_fig_dirs.py` from `EthicalTrajectoryPlanning/` before generating figures.
7. **`store_interval` vs short trajectories**: With default `store_interval=2.0`, the trigger condition `time_step % 20 == 0` may never fire for obstacles with short trajectories (<74 steps). **Fixed**: `store_interval` set to `0.5` in `planning_fast.json` (trigger every 5 steps, min trajectory 59 steps).
8. **Missing scenario IDs (segment 3 gap)**: Every recording's 4th segment (CR steps 451–600) has no vehicle present at the exact start frame. The `obstacle_start_at_zero=True` filter discards all vehicles, so no XML is generated. The 8 missing IDs are: 4, 24, 42, 62, 82, 102, 122, 142.
9. **`transOffset` incompatibility in `commonroad-io==2020.3` + `matplotlib>=3.6`**: The `draw_rectangle` and lanelet fill functions in `commonroad/visualization/scenario.py` create `PolyCollection` with `transOffset=ax.transData`. In matplotlib 3.6+, this causes fill colors not to render (DeprecationWarning; TypeError in 3.8+). **Fixed locally** by removing `transOffset=ax.transData` from all 4 `PolyCollection` calls (lines 834, 838, 843 for lanelet fill; line 1369 for `draw_rectangle`). The 2 `EllipseCollection` calls (lines 855, 863) correctly use `transOffset` with `offsets` and are left unchanged. After modifying, must delete `__pycache__/scenario.cpython-39.pyc` for changes to take effect.
10. **TkAgg backend crashes in headless/batch mode**: `matplotlib.use("TkAgg")` in `frenet_planner.py` (2 locations) causes `TclError` when no display is available. **Fixed**: changed to `matplotlib.use("Agg")`.
11. **`plt.savefig` / `plt.gcf()` figure reference mismatch**: In active_learning mode, multiple figures exist simultaneously (scene, tree, occupancy, interaction maps). `plt.savefig()` saves whichever figure is "current" (`plt.gcf()`), which may differ from the intended figure. **Fixed**: all `plt.savefig(...)` → `fig.savefig(...)` and `FigureCanvasAgg(plt.gcf())` → `FigureCanvasAgg(fig)` in `visualization.py`, `frenet_planner.py`, and `InteractionMap.py`. Also replaced `plt.clf()` → `plt.close(fig)` or `plt.close('all')` to prevent figure accumulation and memory leaks.
12. **`draw_bev_scenario` `time_begin` vs truncated trajectories**: In active_learning mode, `update_scenario()` truncates obstacle trajectories to `[0, ego_time_step]`. The `draw_bev_scenario` function originally passed `time_step` (a future time, e.g. `ego_time + 36` for tree images) as `time_begin` to `draw_object`. Since `occupancy_at_time(future_time)` returns `None` for truncated obstacles, no vehicle rectangles were drawn in tree/occ/interaction images. **Fixed**: changed `"time_begin": time_step` → `"time_begin": ego_time_step` in `visualization.py` `draw_bev_scenario`'s `draw_object` call, so vehicles are drawn at the current time step where they actually exist. The `time_step` parameter is still used for other purposes (e.g. occupancy projection range) but no longer controls vehicle shape rendering.
13. **`traj_evaluate.py` IndexError on short trajectories**: `query_time = time_step + len(traj)` can exceed the ego obstacle's trajectory range, causing `state_at_time_step()` to fail. **Fixed**: added clamping `query_time = min(query_time, max_ts)` where `max_ts` is the ego obstacle's last time step.
14. **InteractionMap zorder too low**: Tree lines and interaction scatter were drawn at default zorder (~2), below lanelet fills (zorder 9.0–9.2) and borders (zorder 10–11). **Fixed**: raised `update_map` tree line plot and `draw_map` scatter to `zorder=25`.
15. **`update_scenario()` truncates ALL agent trajectories → zero predictions/risk in active_learning mode**: In active_learning mode, every dynamic obstacle becomes an agent. `update_scenario()` (from `agent_sim`) replaces each agent's `state_list` and `occupancy_set` with only the states accumulated so far (`[1..ego_time]`). This breaks the entire prediction→collision→risk pipeline:
    - WaleNet `_obstacles_in_scenario()` filters with `final_time_step > time_step` — after truncation `final_time_step == ego_time`, so `5 > 5` is False → obstacle filtered out
    - WaleNet returns empty predictions `{}`
    - Ground truth fallback: `range(time_step, min(pred_horizon, len_pred))` → `range(5, min(60, 5))` = empty
    - `get_orientation_velocity_and_shape_of_prediction` deletes obstacles with empty `pos_list`
    - `collision_checker_prediction` has zero obstacles → all trajectories valid → risk = 0.0
    **Fixed**: Store `self.original_scenario = copy.deepcopy(scenario)` at planner initialization (before any truncation). Use `original_scenario` (with full untruncated trajectories) for: WaleNet `.step()`, ground truth fallback `get_ground_truth_prediction()`, and `get_orientation_velocity_and_shape_of_prediction()`. The truncated `self.scenario` is still used for `get_obstacles_in_radius()` (current-time visibility check) and `get_dyn_and_stat_obstacles()`.
16. **`collision_checker_prediction` IndexError on prediction array slicing**: The formula `traj_length = (end_time_step - start_time_step) / time_interval` computes length mathematically, but the actual numpy slice `pred_traj[:, 0][start:end:step]` gets truncated when `end_time_step > len(pred_traj)`. This causes IndexError in the list comprehension `[[x[i], y[i], pred_orientation[i]] for i in range(int(traj_length))]`. **Fixed**: clamp `end_time_step = min(end_time_step, len(pred_traj))` before slicing, skip if `end <= start`, use `traj_length = len(x)` (actual sliced length) instead of the formula.
17. **`InteractionMap.update_map` uncertainty_list index OOB**: `trajectory.uncertainty_list` (from collision covariance slice) may be shorter than `update_range` (=17). Accessing `uncertainty_list[i-1]` with `i` up to 17 causes IndexError. **Fixed**: clamped index to `min(i-1, len(trajectory.uncertainty_list)-1)`.
18. **`avg_thw` stayed at -1 / unstable after re-enabling THW**:
  - **Root cause (historical)**: In `planner/Frenet/utils/traj_evaluate.py`, `thw` was initialized as `-1` but the legacy THW computation block was commented. In addition, THW logging was gated by `if eval_res["ttc"] != -1`, which could suppress valid THW samples.
  - **Observed issue after first re-enable attempt**: A terminal-time formula (`thw = distance / (traj[-1][4] + 0.01)`) at `query_time = time_step + len(traj)` produced inflated THW in search-tree mode (very small terminal speed causes large values).
  - **Applied fix (minimal and comment-aligned)**: Reused the original commented sampling logic: evaluate every 5th trajectory point (`time_step + i + 1`), compute `thw = distance / (state[4] + 0.01)` and `ttc = distance / (approaching_rate + 0.01)`, clamp with historical bounds (`ttc` to `[-10, 10]`, `thw` to `[0, 30]`), and aggregate by mean.
  - **Logging change**: THW is recorded independently with `if eval_res["thw"] != -1` (no longer blocked by TTC availability).
  - **Validation** (`USA_Intersection-1_7_T-1.xml`):
    - Before stabilization: scenario search-tree `avg_thw` could spike to `73.23`.
    - After stabilization: scenario search-tree `avg_thw` reduced to `8.04` (same scenario rerun), indicating outlier suppression worked.
  - **Important semantic note**: Current THW is still a gap/speed surrogate, not strict "time difference at the same passed point" headway. A strict definition requires point-crossing event matching for ego/leader.

## InteractionMap Adaptive Non-Uniform Resolution

`InteractionMap` supports an optional `adaptive_resolution=True` mode (default on) with three selectable `resolution_mode` strategies. When `adaptive_resolution=False`, the map uses the original uniform 400×400 grid regardless of mode.

### Resolution Modes

#### `'bands'` — Piecewise-linear bands (default)

| Distance from ego | Resolution | Pixel ↔ m | Single-axis coverage |
|---|---|---|---|
| 0 – 20 m | 0.5 m/px | 2 px/m | ±40 px |
| 20 – 50 m | 1.5 m/px | 0.67 px/m | ±20 px |
| 50 – 100 m | 4.0 m/px | 0.25 px/m | ±12.5 px |

Total grid: **148 × 148 = 21,904 pixels** (7.3× compression). Configurable via `res_bands` parameter.

Coordinate transform (piecewise-linear, applied per axis):
```
d = position - ego_center
|d| < 20m  →  pixel_offset = d / 0.5
20 ≤ |d| < 50m  →  pixel_offset = 40 + (|d| - 20) / 1.5
|d| ≥ 50m  →  pixel_offset = 60 + (|d| - 50) / 4.0
```

#### `'linear'` — Continuous logarithmic compression

Grid cell size grows linearly with distance: **Δs(d) = a + b·d** where `a = size` (0.5 m/px) and `b = linear_growth_rate` (default 0.03).

The resulting coordinate transform is logarithmic:
```
pixel(d)  = (1/b) · ln(1 + b·d/a)        # forward
d(pixel)  = (a/b) · (exp(b·pixel) - 1)    # inverse
mpp(d)    = a + b·d                        # metres-per-pixel at distance d
mpp(pixel)= a · exp(b·pixel)              # mpp in pixel space
```

With default parameters (a=0.5, b=0.03, max_d=100m): total grid **132 × 132 = 17,424 pixels** (9.2× compression).

| Distance | Δs (m/px) | Pixel offset |
|---|---|---|
| 0 m | 0.50 | 0 |
| 10 m | 0.80 | ±15 |
| 20 m | 1.10 | ±26 |
| 50 m | 2.00 | ±46 |
| 100 m | 3.50 | ±64 |

Near-ego round-trip error: **0.00m**. Far (100m): ~3.0m.

#### `'speed'` — Speed-coupled bands + risk inflation

Combines band-based grid with ego-speed-dependent scaling:
1. **Band radii scale with ego velocity**: `r_i → r_i × (1 + speed_band_scale × v_ego)`. This expands coverage at higher speeds.
2. **Risk amplification in `update_map`**: `effective_risk = grid_risk × (1 + speed_risk_factor × v_obstacle)`. Faster obstacles deposit more risk.
3. **Wider risk spreading**: `weighted_line` width += `v_obstacle / 5.0` pixels. Faster obstacles paint wider risk corridors.

Grid sizes at various ego speeds (default params: `speed_band_scale=0.02`):

| Ego speed | Scale | Band radii | Grid size | Compression |
|---|---|---|---|---|
| 0 m/s | 1.0× | 20/50/100 m | 148×148 | 7.3× |
| 10 m/s | 1.2× | 24/60/120 m | 176×176 | 5.2× |
| 20 m/s | 1.4× | 28/70/140 m | 206×206 | 3.8× |
| 40 m/s | 1.8× | 36/90/180 m | 262×262 | 2.3× |

### Configuration (`planning_fast.json`)

```json
"active_learning": {
  "resolution_mode": "bands",
  "linear_growth_rate": 0.03,
  "speed_band_scale": 0.02,
  "speed_risk_factor": 0.02
}
```

- `resolution_mode`: `"bands"` | `"linear"` | `"speed"` — selects the coordinate compression strategy.
- `linear_growth_rate`: For `"linear"` mode. Controls how fast cell size grows with distance. Higher → more aggressive compression. Default `0.03`.
- `speed_band_scale`: For `"speed"` mode. How much band radii scale per m/s of ego velocity. Default `0.02`.
- `speed_risk_factor`: For `"speed"` mode. Risk multiplier per m/s of obstacle velocity in `update_map`. Default `0.02`.

### Accuracy (all modes)

Near ego (d < 20m): **0.00m** round-trip error (all modes identical). Far (d ≈ 100m): bands ~2.0m, linear ~3.0m, speed same as bands.

### Usage — Switching Resolution Modes

Only one JSON field needs to change in `planner/Frenet/configs/planning_fast.json` (inside the `"active_learning"` block):

```jsonc
// ── Mode 1: Piecewise-linear bands (default) ──
"resolution_mode": "bands"

// ── Mode 2: Continuous logarithmic compression ──
"resolution_mode": "linear",
"linear_growth_rate": 0.03       // b in Δs = 0.5 + b·d; larger → more aggressive

// ── Mode 3: Speed-coupled bands + risk inflation ──
"resolution_mode": "speed",
"speed_band_scale": 0.02,        // band radii ×(1 + scale·v_ego)
"speed_risk_factor": 0.02        // risk ×(1 + factor·v_obstacle)
```

All other `active_learning` fields remain unchanged. Parameters that are irrelevant to the chosen mode are simply ignored (e.g. `linear_growth_rate` is ignored when `resolution_mode` is `"bands"` or `"speed"`).

To disable adaptive resolution entirely and revert to the original uniform 400×400 grid, set `adaptive_resolution=False` at the code level in `InteractionMap(...)`.

## Visualization Architecture (active_learning mode)

### Image Types and Time Steps

Each visualization step (triggered every `store_interval/0.1` steps) generates multiple image files per ego obstacle. The `time_step` parameter passed to `draw_bev_map` varies by image type:

| Image type | Filename suffix | `time_step` value | Purpose |
|---|---|---|---|
| `_tree.png` | `{ego_id}_{ego_time}_tree.png` | `ego_time + search_length * 2` | Search tree visualization with future projection |
| `_scenario.png` | `{ego_id}_{ego_time}_scenario.png` | `ego_time` | Current scenario state |
| `_best_traj_occ.png` | `{ego_id}_{ego_time}_best_traj_occ.png` | `ego_time + len(traj)` | Best trajectory occupancy footprint |
| `_gt_traj_occ.png` | `{ego_id}_{ego_time}_gt_traj_occ.png` | `ego_time + len(traj)` | Ground truth trajectory occupancy |
| `_0.png`, `_1.png`, `_2.png` | `{ego_id}_{ego_time}_{i}.png` | `ego_time + search_length * (i+1)` | Interaction maps at future timesteps |

Where `search_length = int(search_step / dt) = int(1.8 / 0.1) = 18`.

**Critical insight**: Because `update_scenario()` truncates trajectories to `[0, ego_time]`, the `time_step` parameter must NOT be used as `time_begin` for `draw_object` — vehicles won't exist at future times. The fix uses `ego_time_step` as `time_begin` for vehicle rendering while keeping `time_step` for other non-vehicle purposes.

### Zorder Hierarchy

| Layer | Elements | Zorder |
|---|---|---|
| Lanelet fill (normal/incoming/crossing) | `PolyCollection` | 9.0 / 9.1 / 9.2 |
| Lanelet borders | `LineCollection` | 10 / 10.1 / 11 |
| Vehicle rectangles | `PolyCollection` via `draw_rectangle` | 20 |
| Tree lines & interaction scatter | `ax.plot` / `ax.scatter` | 25 |
| Vehicle labels | `ax.text` | 1000 |

### Pipeline Flow (tree.png)

```
draw_bev_map(time_step=ego+36, ego_time_step=ego)
  └─ draw_bev_scenario(time_step=ego+36, ego_time_step=ego)
       └─ draw_object(scenario, time_begin=ego_time_step)  ← draws vehicles at ego_time
            └─ returns fig, ax with vehicle PolyCollection
  └─ FigureCanvasAgg(fig).draw()  ← renders to BEV binary mask
  └─ returns bev_map, fig, ax

InteractionMap(bev_map) × 3  ← creates interaction maps from BEV mask
get_prediction()              ← WaleNet prediction step
traverse(tree_ax=ax)          ← draws tree lines on ax, updates interaction maps
fig.savefig(tree.png)         ← saves the fig with vehicles + tree lines
plt.close(fig)                ← releases figure memory
```

### Modified Files Summary

| File | Changes |
|---|---|
| `planner/Frenet/frenet_planner.py` | TkAgg→Agg (2 places); `plt.savefig`→`fig.savefig`; `plt.clf()`→`plt.close(fig)`/`plt.close('all')` (5 places); `self.original_scenario = copy.deepcopy(scenario)` at init; `get_prediction()` uses `original_scenario` for WaleNet/ground-truth/orientation; BEVProbLoader init + BEV risk overlay in both planning modes |
| `planner/Frenet/utils/bev_prob_loader.py` | **New**: BEVPredProb loader with scenario mapping, coordinate transform, probability query |
| `planner/Frenet/utils/visualization.py` | `plt.savefig`→`fig.savefig`; `FigureCanvasAgg(plt.gcf())`→`FigureCanvasAgg(fig)`; `time_begin: time_step`→`ego_time_step` |
| `planner/Frenet/utils/InteractionMap.py` | `draw_map` accepts `fig,ax` params; scatter zorder 2→25; tree line zorder→25; `fig.savefig` + `plt.close(fig)`; `uncertainty_list` index clamping; **adaptive non-uniform resolution** (see below) |
| `planner/Frenet/utils/prediction_helpers.py` | `collision_checker_prediction`: clamp `end_time_step` to `len(pred_traj)`, use `len(x)` for `traj_length` |
| `planner/Frenet/utils/traj_evaluate.py` | `query_time` clamping to prevent IndexError |
| `commonroad/visualization/scenario.py` (site-packages) | Removed `transOffset=ax.transData` from 4 `PolyCollection` calls |
| `planner/Frenet/configs/planning_fast.json` | `store_interval` 2.0→0.5; added `bev_prob_dir`, `bev_weight`, `resolution_mode`, `linear_growth_rate`, `speed_band_scale`, `speed_risk_factor` |

## Key Dependencies

- `commonroad-io==2020.3`, `commonroad-drivability-checker==2022.1`, `commonroad-helper-functions==1.0.0`, `commonroad-agent==1.1.0`
- `wale-net==2.0.2` (trajectory prediction)
- `networkx==2.5`, `shapely~=1.7.1`, `numba~=0.50.1`
- `torch>=2.1` + CUDA (BEV Transformer training)
- `alphashape`, `scikit-image` (InteractionMap risk computation)
