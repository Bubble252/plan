from __future__ import annotations

import argparse
from pathlib import Path

from auv_planner.experiments.m5_validation import run_m5_validation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the M5-B PVS/MSS REMUS 100 model validation suite."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/m5_model_validation"),
        help="Directory for comparison CSV, traces, logs, summary, and report.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    summary = run_m5_validation(args.output_dir, seed=args.seed)
    print(
        "M5-B complete: "
        f"{summary['derivative_cases']} derivative cases, "
        f"{summary['open_loop_cases']} open-loop cases, "
        f"{summary['closed_loop_cases']} closed-loop cases, "
        f"{summary['convergence_rows']} convergence rows"
    )
    print(f"Report: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
