from __future__ import annotations

import argparse
from pathlib import Path

from auv_planner.experiments import run_m4_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the M4.1 REMUS 100 closed-loop baseline suite.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/m4_closed_loop"),
        help="Directory for summary.csv, report.md, and per-case logs.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results = run_m4_suite(args.output_dir, seed=args.seed)
    for result in results:
        metrics = result.metrics
        print(
            f"{result.case_name}: completed={metrics.completed} "
            f"time={metrics.completion_time_s:.2f}s "
            f"h_rmse={metrics.horizontal_rmse_m:.3f}m "
            f"v_rmse={metrics.vertical_rmse_m:.3f}m "
            f"violations={metrics.constraint_violations}"
        )


if __name__ == "__main__":
    main()
