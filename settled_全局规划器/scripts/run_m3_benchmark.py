from __future__ import annotations

import argparse
from pathlib import Path

from auv_planner.evaluation import run_suite, write_metrics_csv
from auv_planner.planners.astar import AStarPlanner
from auv_planner.planners.rrt_star import RRTStarPlanner
from auv_planner.scenarios import fixed_scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed M3 A*/RRT* benchmark suite.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/m3"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--resolution-m", type=float, default=0.4)
    parser.add_argument("--timeout-s", type=float, default=3.0)
    parser.add_argument("--rrt-step-m", type=float, default=0.5)
    parser.add_argument("--rrt-iterations", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    planners = (
        AStarPlanner(resolution_m=args.resolution_m, timeout_s=args.timeout_s),
        RRTStarPlanner(
            step_size_m=args.rrt_step_m,
            max_iterations=args.rrt_iterations,
            timeout_s=args.timeout_s,
        ),
    )
    records = run_suite(
        fixed_scenarios(),
        planners,
        args.seeds,
        log_dir=output_dir / "logs",
    )
    metrics_path = write_metrics_csv(records, output_dir / "metrics.csv")
    successes = sum(record.result.success for record in records)
    print(f"wrote {metrics_path}")
    print(f"runs={len(records)} successes={successes} failures={len(records) - successes}")
    for record in records:
        print(
            f"{record.scenario_name:20s} {record.planner_name:16s} seed={record.seed} "
            f"status={record.plan.status.value:10s} success={record.result.success} "
            f"reason={record.result.failure_reason or '-'}"
        )


if __name__ == "__main__":
    main()
