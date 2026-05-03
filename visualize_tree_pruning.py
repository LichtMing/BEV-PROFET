#!/usr/bin/env python3
"""Generate a standalone Frenet search-tree visualization with pruning marks.

This script reuses the existing planner logic only for trajectory generation,
validity classification, and BEV-backed risk overlays. It does not modify the
original planner code path.

Example:
    python visualize_tree_pruning.py \
        --scenario EthicalTrajectoryPlanning/scenarios/USA_Intersection-1_3_T-1.xml \
        --ego-id 160 \
        --output tree_pruning.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


BASE_DIR = Path(__file__).resolve().parent
ETP_ROOT = BASE_DIR / "EthicalTrajectoryPlanning"
if str(ETP_ROOT) not in sys.path:
    sys.path.insert(0, str(ETP_ROOT))
FRENET_DIR = ETP_ROOT / "planner" / "Frenet"
if str(FRENET_DIR) not in sys.path:
    sys.path.insert(0, str(FRENET_DIR))

from planner.Frenet.configs.load_json import load_planning_json, load_risk_json, load_weight_json  # type: ignore[reportMissingImports]
from planner.Frenet.frenet_planner import FrenetPlanner  # type: ignore[reportMissingImports]
from planner.Frenet.plannertools.frenetcreator import FrenetCreator  # type: ignore[reportMissingImports]
from planner.Frenet.utils.InteractionMap import InteractionMap  # type: ignore[reportMissingImports]
from planner.Frenet.utils.visualization import draw_bev_map  # type: ignore[reportMissingImports]
from planner.plannertools.evaluate import ScenarioEvaluator  # type: ignore[reportMissingImports]


def _resolve_scenario_path(raw_path: str | None) -> Path:
    scenario_root = ETP_ROOT / "scenarios"
    if raw_path:
        candidate = Path(raw_path).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        if not candidate.is_absolute():
            from_root = (BASE_DIR / candidate).resolve()
            if from_root.is_file():
                return from_root
            from_scenarios = (scenario_root / candidate).resolve()
            if from_scenarios.is_file():
                return from_scenarios
        matches = sorted(scenario_root.glob(f"**/{raw_path}"))
        if matches:
            return matches[0].resolve()
        raise FileNotFoundError(f"Cannot find scenario: {raw_path}")

    xml_files = sorted(scenario_root.glob("**/*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No .xml scenarios found under {scenario_root}")
    return xml_files[0].resolve()


def _choose_ego_id(planner: FrenetPlanner, requested_ego_id: int | None) -> int:
    if requested_ego_id is not None:
        return requested_ego_id

    if getattr(planner, "ego_id", None) is not None:
        return int(planner.ego_id)

    if planner.scenario.dynamic_obstacles:
        return int(planner.scenario.dynamic_obstacles[0].obstacle_id)

    raise RuntimeError("Scenario does not contain a dynamic obstacle to use as ego.")


def _choose_visualization_time_step(planner: FrenetPlanner, ego_id: int, requested_time_step: int | None) -> int:
    if requested_time_step is not None:
        return int(requested_time_step)

    obstacle = planner.scenario.obstacle_by_id(ego_id)
    if obstacle is None:
        return int(planner.ego_state.time_step)

    candidate_steps = []
    if obstacle.prediction is not None and obstacle.prediction.trajectory is not None:
        candidate_steps.extend(state.time_step for state in obstacle.prediction.trajectory.state_list)
    candidate_steps.append(obstacle.initial_state.time_step)
    candidate_steps = sorted(set(int(step) for step in candidate_steps))

    preferred_min_step = max(20, int(planner.search_step / 0.1))

    for step in candidate_steps:
        if step <= obstacle.initial_state.time_step:
            continue
        if step < preferred_min_step:
            continue
        if step >= planner.final_step - int(planner.max_exploration_time / 0.1) - 2:
            continue
        return step

    for step in candidate_steps:
        if step > obstacle.initial_state.time_step:
            return step

    return int(obstacle.initial_state.time_step)


def _obstacle_state_at_time(obstacle, time_step: int):
    if obstacle is None:
        return None
    if obstacle.prediction is not None and obstacle.prediction.trajectory is not None:
        for state in obstacle.prediction.trajectory.state_list:
            if int(state.time_step) == int(time_step):
                return state
    if int(getattr(obstacle.initial_state, "time_step", -1)) == int(time_step):
        return obstacle.initial_state
    return None


def _build_snapshot(planner: FrenetPlanner, ego_id: int, time_step: int):
    obstacle = planner.scenario.obstacle_by_id(ego_id)
    if obstacle is None:
        raise RuntimeError(f"Could not find obstacle {ego_id} in the scenario")

    state = _obstacle_state_at_time(obstacle, time_step)
    if state is None:
        state = obstacle.initial_state

    planner._Planner__time_step = int(time_step)
    planner._Planner__ego_state = state

    next_state = _obstacle_state_at_time(obstacle, time_step + 1)
    if next_state is None:
        next_state = state

    c_s, c_s_d, c_d, c_d_d = planner.reference_spline.cartesian_to_frenet(
        state.position, state.velocity, state.orientation
    )
    c_s_dd = 0
    c_d_dd = 0
    planner._trajectory = {
        "s_loc_m": [0, c_s],
        "d_loc_m": [0, c_d],
        "d_d_loc_mps": [0, c_d_d],
        "d_dd_loc_mps2": [0, c_d_dd],
        "x_m": [state.position[0], next_state.position[0]],
        "y_m": [state.position[1], next_state.position[1]],
        "psi_rad": [state.orientation, next_state.orientation],
        "kappa_radpm": [0, 0],
        "v_mps": [state.velocity, next_state.velocity],
        "ax_mps2": [getattr(state, "acceleration", 0) or 0, getattr(next_state, "acceleration", 0) or 0],
        "time_s": [0, 0.1],
    }

    return state, float(c_d), float(c_d_d)


def _pick_agent_planner(evaluator: ScenarioEvaluator, ego_id: int) -> FrenetPlanner:
    for agent in evaluator.agent_list:
        if agent.agent_id == ego_id:
            return agent.planner
    raise RuntimeError(f"Could not find agent/planner for ego_id={ego_id}")


def _prepare_planner(planner: FrenetPlanner):
    search_length = int(planner.search_step / planner.frenet_parameters["dt"])
    bev_map, fig, ax = draw_bev_map(
        scenario=planner.scenario,
        ego_id=planner.ego_id,
        time_step=planner.ego_state.time_step + search_length * 2,
        ego_time_step=planner.ego_state.time_step,
        marked_vehicle=planner.ego_id,
        planning_problem=planner.planning_problem,
        traj=None,
        global_path=planner.global_path_to_goal,
        global_path_after_goal=planner.global_path_after_goal,
        driven_traj=planner.driven_traj,
        animation_area=50.0,
        traj_time_begin=planner.ego_state.time_step,
        traj_time_end=planner.ego_state.time_step + search_length,
        ego_traj_time_begin=planner.ego_state.time_step,
        ego_traj_time_end=planner.ego_state.time_step + search_length,
        ego_draw_occ=False,
    )

    planner.interaction_maps = [
        InteractionMap(
            bev_map=bev_map,
            anchor_state=planner.ego_state,
            size=0.5,
            obs_width=2.0,
            update_length=int(planner.frenet_parameters["t_list"][0] // planner.frenet_parameters["dt"]) - 1,
            resolution_mode=planner.planning_config.get("resolution_mode", "bands"),
            linear_growth_rate=planner.planning_config.get("linear_growth_rate", 0.03),
            ego_velocity=planner.ego_state.velocity,
            speed_band_scale=planner.planning_config.get("speed_band_scale", 0.02),
            speed_risk_factor=planner.planning_config.get("speed_risk_factor", 0.02),
        )
        for _ in range(3)
    ]

    planner.predictions = planner.get_prediction()
    return fig, ax, search_length


def _branch_style(kind: str, depth: int):
    if kind == "valid":
        return {"color": "#1b9e77", "alpha": max(0.40, 0.78 - depth * 0.10), "linewidth": 2.4}
    if kind == "collision":
        return {"color": "#d73027", "alpha": 0.85, "linewidth": 2.2}
    if kind == "boundary":
        return {"color": "#ff8c00", "alpha": 0.85, "linewidth": 2.2}
    return {"color": "#6a3d9a", "alpha": 0.70, "linewidth": 2.0}


def _draw_leaf(ax, ft, kind: str, depth: int):
    style = _branch_style(kind, depth)
    ax.scatter(
        [ft.x[-1]],
        [ft.y[-1]],
        marker="x" if kind != "valid" else "o",
        s=55 if kind != "valid" else 28,
        c=style["color"],
        linewidths=1.5,
        zorder=80 + depth,
    )
    if kind != "valid":
        ax.text(
            ft.x[-1],
            ft.y[-1],
            f" prune: {ft.reason_invalid}",
            fontsize=6,
            color=style["color"],
            ha="left",
            va="bottom",
            zorder=90 + depth,
        )


def _walk_tree(
    planner: FrenetPlanner,
    ax,
    state,
    current_d_v,
    ego_id: int,
    future_step: int,
    parent_behavior_path,
    summary,
):
    depth = int(future_step / planner.search_step)
    cur_behavior_list, cur_action_list = planner.behavior_tree.get_behaviors_in_path(
        current_d_v, state, parent_behavior_path
    )

    collision_ft_list, leaving_road_ft_list, valid_ft_list = planner.get_possible_trajectories(
        global_state=state,
        frenet_state=None,
        ego_id=ego_id,
        d_v_combination=cur_action_list,
        behavior_combination=cur_behavior_list,
        anchor_state=planner.interaction_maps[depth].anchor_state,
    )

    summary["depths"].setdefault(depth, {"valid": 0, "collision": 0, "boundary": 0})
    summary["depths"][depth]["valid"] += len(valid_ft_list)
    summary["depths"][depth]["collision"] += len(collision_ft_list)
    summary["depths"][depth]["boundary"] += len(leaving_road_ft_list)

    def _plot_segment(ft, kind):
        style = _branch_style(kind, depth)
        ax.plot(
            ft.x,
            ft.y,
            color=style["color"],
            alpha=style["alpha"],
            linewidth=style["linewidth"],
            zorder=35 + depth,
        )

    for ft in collision_ft_list:
        summary["pruned"]["collision"] += 1
        _plot_segment(ft, "collision")
        _draw_leaf(ax, ft, "collision", depth)

    for ft in leaving_road_ft_list:
        summary["pruned"]["boundary"] += 1
        _plot_segment(ft, "boundary")
        _draw_leaf(ax, ft, "boundary", depth)

    is_terminal_layer = future_step + planner.search_step >= planner.max_exploration_time - 0.01
    for ft in valid_ft_list:
        summary["kept"] += 1
        _plot_segment(ft, "valid")
        if is_terminal_layer:
            _draw_leaf(ax, ft, "valid", depth)
            continue

        end_index = int(planner.search_step / planner.frenet_parameters["dt"])
        next_state = ft.get_global_state(end_index)
        next_state.time_step = state.time_step + end_index
        next_path = list(parent_behavior_path)
        next_path.append(ft.target_behavior)

        _walk_tree(
            planner=planner,
            ax=ax,
            state=next_state,
            current_d_v=(ft.target_d, ft.target_v),
            ego_id=ego_id,
            future_step=future_step + planner.search_step,
            parent_behavior_path=next_path,
            summary=summary,
        )


def _add_legend(ax):
    handles = [
        Line2D([0], [0], color="#1b9e77", lw=2.4, label="kept / valid expansion"),
        Line2D([0], [0], color="#d73027", lw=2.2, label="pruned: collision"),
        Line2D([0], [0], color="#ff8c00", lw=2.2, label="pruned: boundary"),
    ]
    ax.legend(handles=handles, loc="upper left", framealpha=0.92, fontsize=8)


def _add_summary_box(fig, summary, planner: FrenetPlanner, ego_id: int):
    lines = [
        f"ego_id = {ego_id}",
        f"time_step = {planner.ego_state.time_step}",
        f"kept branches = {summary['kept']}",
        f"collision pruned = {summary['pruned']['collision']}",
        f"boundary pruned = {summary['pruned']['boundary']}",
        "",
        "per-depth counts:",
    ]
    for depth in sorted(summary["depths"]):
        item = summary["depths"][depth]
        lines.append(
            f"  depth {depth}: valid={item['valid']} collision={item['collision']} boundary={item['boundary']}"
        )

    fig.text(
        0.015,
        0.015,
        "\n".join(lines),
        ha="left",
        va="bottom",
        fontsize=8,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#888888", alpha=0.92),
    )


def _default_output_path(scenario_path: Path) -> Path:
    out_dir = BASE_DIR / "scenario_vis"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{scenario_path.stem}_tree_pruning.png"


def main():
    parser = argparse.ArgumentParser(description="Standalone Frenet tree pruning visualization")
    parser.add_argument("--scenario", type=str, default=None, help="CommonRoad XML path")
    parser.add_argument("--ego-id", type=int, default=None, help="Dynamic obstacle id to visualize")
    parser.add_argument("--time-step", type=int, default=None, help="Snapshot time step to visualize")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path")
    parser.add_argument("--weights", type=str, default="ethical", help="Weight config name")
    parser.add_argument("--settings", type=str, default=None, help="Risk config suffix")
    args = parser.parse_args()

    scenario_path = _resolve_scenario_path(args.scenario)
    output_path = Path(args.output).expanduser().resolve() if args.output else _default_output_path(scenario_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    settings_dict = load_planning_json("planning_fast.json")
    settings_dict["risk_dict"] = load_risk_json() if args.settings is None else load_risk_json(filename=f"risk_{args.settings}.json")
    weights = load_weight_json(filename=f"weights_{args.weights}.json")

    creator = FrenetCreator(settings_dict, weights=weights)
    evaluator = ScenarioEvaluator(
        planner_creator=creator,
        vehicle_type=settings_dict["evaluation_settings"]["vehicle_type"],
        path_to_scenarios=(ETP_ROOT / "scenarios").resolve(),
        log_path=(ETP_ROOT / "console_log").resolve(),
        collision_report_path=(ETP_ROOT / "planner" / "Frenet" / "results" / "eval").resolve(),
        timing_enabled=settings_dict["evaluation_settings"]["timing_enabled"],
        active_learning=settings_dict["active_learning"]["active_learning_enabled"],
        prepare_num_data=False,
        label_path=None,
        fig_path=None,
    )

    evaluator.scenario_path = scenario_path
    evaluator._initialize()

    initial_planner = _pick_agent_planner(evaluator, args.ego_id or int(evaluator.agent_list[0].agent_id))
    ego_id = _choose_ego_id(initial_planner, args.ego_id)
    planner = _pick_agent_planner(evaluator, ego_id)

    viz_time_step = _choose_visualization_time_step(planner, ego_id, args.time_step)
    _, root_d, root_d_v = _build_snapshot(planner, ego_id, viz_time_step)

    fig, ax, _search_length = _prepare_planner(planner)
    summary = {"kept": 0, "pruned": {"collision": 0, "boundary": 0}, "depths": {}}

    ax.plot(
        [planner.ego_state.position[0]],
        [planner.ego_state.position[1]],
        marker="*",
        markersize=12,
        color="#111111",
        zorder=120,
        label="ego root",
    )

    _walk_tree(
        planner=planner,
        ax=ax,
        state=planner.ego_state,
        current_d_v=(root_d, root_d_v),
        ego_id=ego_id,
        future_step=0,
        parent_behavior_path=[],
        summary=summary,
    )

    ax.set_title(
        f"Frenet tree pruning visualization: {scenario_path.name} | ego {ego_id} | t={viz_time_step}",
        fontsize=11,
    )
    _add_legend(ax)
    _add_summary_box(fig, summary, planner, ego_id)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved tree pruning visualization to {output_path}")


if __name__ == "__main__":
    main()