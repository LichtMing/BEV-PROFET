#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


SCENARIO_REGEX = re.compile(r"^USA_Intersection-1_(\d+)_T-1\.xml$")


@dataclass
class ScenarioResult:
    scenario_id: int
    scenario_name: str
    command: str
    status: str
    return_code: Optional[int]
    duration_sec: float
    started_at: str
    finished_at: str
    log_file: str


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_scenarios_dir = (script_dir / "../../scenarios").resolve()
    default_log_root = (script_dir / "batch_logs").resolve()

    parser = argparse.ArgumentParser(
        description="Batch run CommonRoad scenarios using frenet_planner.py"
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=16,
        help="Start scenario id (default: 16, because 1-15 are already done)",
    )
    parser.add_argument(
        "--end-id",
        type=int,
        default=None,
        help="End scenario id (inclusive). Default: run to the maximum available id.",
    )
    parser.add_argument(
        "--scenarios-dir",
        type=Path,
        default=default_scenarios_dir,
        help=f"Scenario directory (default: {default_scenarios_dir})",
    )
    parser.add_argument(
        "--planner-file",
        type=Path,
        default=script_dir / "frenet_planner.py",
        help="Path to frenet_planner.py",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=default_log_root,
        help=f"Root directory for logs and reports (default: {default_log_root})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Timeout (seconds) per scenario. Default: no timeout.",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop batch immediately when one scenario fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned commands without executing.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous state.json in log root (latest run).",
    )
    return parser.parse_args()


def find_scenarios(scenarios_dir: Path) -> List[Tuple[int, str]]:
    if not scenarios_dir.exists():
        raise FileNotFoundError(f"Scenarios dir not found: {scenarios_dir}")

    scenarios: List[Tuple[int, str]] = []
    for entry in scenarios_dir.iterdir():
        if not entry.is_file():
            continue
        match = SCENARIO_REGEX.match(entry.name)
        if match:
            scenario_id = int(match.group(1))
            scenarios.append((scenario_id, entry.name))

    scenarios.sort(key=lambda x: x[0])
    return scenarios


def make_run_dir(log_root: Path, resume: bool) -> Path:
    log_root.mkdir(parents=True, exist_ok=True)
    if resume:
        runs = sorted([p for p in log_root.iterdir() if p.is_dir() and p.name.startswith("run_")])
        if runs:
            return runs[-1]
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = log_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def load_state(state_file: Path) -> dict:
    if not state_file.exists():
        return {"completed_ids": [], "results": []}
    with state_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state_file: Path, state: dict) -> None:
    with state_file.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def write_results_csv(csv_path: Path, results: List[ScenarioResult]) -> None:
    fields = [
        "scenario_id",
        "scenario_name",
        "status",
        "return_code",
        "duration_sec",
        "started_at",
        "finished_at",
        "log_file",
        "command",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in results:
            row = asdict(item)
            writer.writerow(row)


def summarize(results: List[ScenarioResult]) -> dict:
    total = len(results)
    success = sum(1 for r in results if r.status == "success")
    failed = sum(1 for r in results if r.status == "failed")
    timeout = sum(1 for r in results if r.status == "timeout")
    skipped = sum(1 for r in results if r.status == "skipped")
    duration = sum(r.duration_sec for r in results)
    failed_ids = [r.scenario_id for r in results if r.status in {"failed", "timeout"}]

    return {
        "total": total,
        "success": success,
        "failed": failed,
        "timeout": timeout,
        "skipped": skipped,
        "total_duration_sec": round(duration, 3),
        "avg_duration_sec": round(duration / total, 3) if total else 0.0,
        "failed_or_timeout_ids": failed_ids,
    }


def write_summary_md(summary_path: Path, summary: dict, run_dir: Path) -> None:
    lines = [
        "# Scenario Batch Run Report",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Run directory: {run_dir}",
        f"- Total: {summary['total']}",
        f"- Success: {summary['success']}",
        f"- Failed: {summary['failed']}",
        f"- Timeout: {summary['timeout']}",
        f"- Skipped (resume): {summary['skipped']}",
        f"- Total duration (sec): {summary['total_duration_sec']}",
        f"- Average duration (sec): {summary['avg_duration_sec']}",
        "",
    ]
    failed_ids = summary["failed_or_timeout_ids"]
    if failed_ids:
        lines.append(f"- Failed/Timeout scenario IDs: {failed_ids}")
    else:
        lines.append("- Failed/Timeout scenario IDs: none")

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_one_scenario(
    planner_file: Path,
    scenario_name: str,
    log_file: Path,
    timeout: Optional[float],
) -> ScenarioResult:
    cmd = [sys.executable, str(planner_file.name), "--scenario", scenario_name]
    command_for_display = " ".join(shlex.quote(x) for x in cmd)

    started_at = datetime.now().isoformat(timespec="seconds")
    t0 = time.time()

    with log_file.open("w", encoding="utf-8") as lf:
        lf.write(f"[START] {started_at}\n")
        lf.write(f"[CMD] {command_for_display}\n\n")
        lf.flush()

        try:
            completed = subprocess.run(
                cmd,
                cwd=str(planner_file.parent),
                stdout=lf,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            status = "success" if completed.returncode == 0 else "failed"
            return_code = completed.returncode
        except subprocess.TimeoutExpired:
            status = "timeout"
            return_code = None
            lf.write(f"\n[TIMEOUT] exceeded {timeout} seconds\n")

        finished_at = datetime.now().isoformat(timespec="seconds")
        duration = time.time() - t0
        lf.write(f"\n[END] {finished_at}\n")
        lf.write(f"[STATUS] {status}\n")
        lf.flush()

    match = SCENARIO_REGEX.match(scenario_name)
    scenario_id = int(match.group(1)) if match else -1

    return ScenarioResult(
        scenario_id=scenario_id,
        scenario_name=scenario_name,
        command=command_for_display,
        status=status,
        return_code=return_code,
        duration_sec=round(duration, 3),
        started_at=started_at,
        finished_at=finished_at,
        log_file=str(log_file),
    )


def main() -> int:
    args = parse_args()

    all_scenarios = find_scenarios(args.scenarios_dir.resolve())
    if not all_scenarios:
        print("No matched scenarios found.")
        return 1

    max_available_id = max(sid for sid, _ in all_scenarios)
    end_id = args.end_id if args.end_id is not None else max_available_id
    if end_id < args.start_id:
        print(f"Invalid range: start-id={args.start_id}, end-id={end_id}")
        return 1

    selected = [
        (sid, name)
        for sid, name in all_scenarios
        if args.start_id <= sid <= end_id
    ]

    if not selected:
        print("No scenarios selected in requested id range.")
        return 1

    run_dir = make_run_dir(args.log_root.resolve(), args.resume)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    state_file = run_dir / "state.json"

    state = load_state(state_file)
    completed_ids = set(state.get("completed_ids", [])) if args.resume else set()

    print("=" * 80)
    print("Scenario batch runner")
    print(f"Scenarios dir: {args.scenarios_dir.resolve()}")
    print(f"Planner file : {args.planner_file.resolve()}")
    print(f"Run logs dir : {run_dir}")
    print(f"ID range     : {args.start_id} -> {end_id}")
    print(f"Total queued : {len(selected)}")
    print(f"Resume mode  : {args.resume}")
    print("=" * 80)

    results: List[ScenarioResult] = []
    interrupted = False

    try:
        for index, (scenario_id, scenario_name) in enumerate(selected, start=1):
            if scenario_id in completed_ids:
                print(f"[{index}/{len(selected)}] SKIP  {scenario_name} (already completed)")
                skip_item = ScenarioResult(
                    scenario_id=scenario_id,
                    scenario_name=scenario_name,
                    command="",
                    status="skipped",
                    return_code=None,
                    duration_sec=0.0,
                    started_at="",
                    finished_at="",
                    log_file="",
                )
                results.append(skip_item)
                continue

            log_file = logs_dir / f"{scenario_id}_{scenario_name}.log"
            cmd_preview = f"python {args.planner_file.name} --scenario {scenario_name}"
            print(f"[{index}/{len(selected)}] RUN   {cmd_preview}")

            if args.dry_run:
                dry_item = ScenarioResult(
                    scenario_id=scenario_id,
                    scenario_name=scenario_name,
                    command=cmd_preview,
                    status="skipped",
                    return_code=None,
                    duration_sec=0.0,
                    started_at="",
                    finished_at="",
                    log_file=str(log_file),
                )
                results.append(dry_item)
                continue

            result = run_one_scenario(
                planner_file=args.planner_file.resolve(),
                scenario_name=scenario_name,
                log_file=log_file,
                timeout=args.timeout,
            )
            results.append(result)
            completed_ids.add(scenario_id)

            state = {
                "completed_ids": sorted(completed_ids),
                "results": [asdict(r) for r in results],
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            save_state(state_file, state)

            print(
                f"[{index}/{len(selected)}] DONE  {scenario_name} "
                f"status={result.status} time={result.duration_sec:.2f}s"
            )

            if args.stop_on_failure and result.status in {"failed", "timeout"}:
                print("Stop on failure is enabled. Batch stopped.")
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\nKeyboardInterrupt received. Finalizing partial report...")

    summary = summarize(results)
    summary_json_path = run_dir / "summary.json"
    summary_md_path = run_dir / "summary.md"
    results_csv_path = run_dir / "results.csv"

    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    write_summary_md(summary_md_path, summary, run_dir)
    write_results_csv(results_csv_path, results)

    print("\n" + "=" * 80)
    print("Batch completed")
    print(f"Interrupted: {interrupted}")
    print(f"Total   : {summary['total']}")
    print(f"Success : {summary['success']}")
    print(f"Failed  : {summary['failed']}")
    print(f"Timeout : {summary['timeout']}")
    print(f"Skipped : {summary['skipped']}")
    print(f"Avg sec : {summary['avg_duration_sec']}")
    print(f"Report  : {summary_md_path}")
    print(f"JSON    : {summary_json_path}")
    print(f"CSV     : {results_csv_path}")
    print("=" * 80)

    return 130 if interrupted else 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())