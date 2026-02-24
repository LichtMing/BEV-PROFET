# Project Guidelines

## Architecture

Three subsystems for autonomous driving trajectory planning:

| Module | Path | Purpose |
|---|---|---|
| **EthicalTrajectoryPlanning** | `EthicalTrajectoryPlanning/` | Frenet-frame sampling planner + risk assessment + interaction maps |
| **BEV V2X Transformer** | `bev_v2x_transformer/` | Transformer-based BEV prediction → probability maps for future 3 timesteps |
| **Dataset Convert** | `dataset_convert/` | INTERACTION/highD/inD → CommonRoad XML conversion |

**Data flow**: INTERACTION dataset → `intersection_bev_generator_*.py` → BEV training data → `training.py` → model → `inference_prob_map.py` → `BEVPredProb/*.npy` → `InteractionMap` in planner → risk-aware trajectory selection.

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
| `planner/Frenet/frenet_planner.py` | TkAgg→Agg (2 places); `plt.savefig`→`fig.savefig`; `plt.clf()`→`plt.close(fig)`/`plt.close('all')` (5 places); `self.original_scenario = copy.deepcopy(scenario)` at init; `get_prediction()` uses `original_scenario` for WaleNet/ground-truth/orientation |
| `planner/Frenet/utils/visualization.py` | `plt.savefig`→`fig.savefig`; `FigureCanvasAgg(plt.gcf())`→`FigureCanvasAgg(fig)`; `time_begin: time_step`→`ego_time_step` |
| `planner/Frenet/utils/InteractionMap.py` | `draw_map` accepts `fig,ax` params; scatter zorder 2→25; tree line zorder→25; `fig.savefig` + `plt.close(fig)`; `uncertainty_list` index clamping |
| `planner/Frenet/utils/prediction_helpers.py` | `collision_checker_prediction`: clamp `end_time_step` to `len(pred_traj)`, use `len(x)` for `traj_length` |
| `planner/Frenet/utils/traj_evaluate.py` | `query_time` clamping to prevent IndexError |
| `commonroad/visualization/scenario.py` (site-packages) | Removed `transOffset=ax.transData` from 4 `PolyCollection` calls |
| `planner/Frenet/configs/planning_fast.json` | `store_interval` 2.0→0.5 |

## Key Dependencies

- `commonroad-io==2020.3`, `commonroad-drivability-checker==2022.1`, `commonroad-helper-functions==1.0.0`, `commonroad-agent==1.1.0`
- `wale-net==2.0.2` (trajectory prediction)
- `networkx==2.5`, `shapely~=1.7.1`, `numba~=0.50.1`
- `torch>=2.1` + CUDA (BEV Transformer training)
- `alphashape`, `scikit-image` (InteractionMap risk computation)
