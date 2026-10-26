from __future__ import annotations

import argparse
from pathlib import Path

from auv_planner.experiments.m7_budget import run_m7_budget


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the M7.0 recovery-certificate query cost profiling experiment."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/m7_budget"),
        help="Directory for CSV, summary, hardware manifest, and report.",
    )
    args = parser.parse_args()

    summary = run_m7_budget(args.output_dir)
    p1 = summary["profiles"]["P1_degraded"]
    print(
        "M7.0 complete: "
        f"P1 single median {p1['single_full']['median_s'] * 1000.0:.1f} ms, "
        f"online path gate {summary['p1_online_budget_gate']}, "
        f"prefilter false negatives {summary['prefilter_false_negative_count']}"
    )
    print(f"Report: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
