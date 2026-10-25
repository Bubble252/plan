from __future__ import annotations

import argparse
from pathlib import Path

from auv_planner.experiments.m6_recovery import run_m6_recovery


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the M6 finite backup-maneuver and recoverability-certificate experiments."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/m6_recovery"),
        help="Directory for X0/X1 CSV, certificates, summary, and report.",
    )
    args = parser.parse_args()

    summary = run_m6_recovery(args.output_dir)
    print(
        "M6 complete: "
        f"{summary['node_count']} nodes, "
        f"X0 {summary['x0_certified']}/{summary['x0_rows']} certified, "
        f"X1 {summary['x1_safe']}/{summary['x1_total']} safe fault checks"
    )
    print(f"Report: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
