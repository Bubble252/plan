from __future__ import annotations

import argparse
from pathlib import Path

from auv_planner.experiments.m7_budget import run_m7_budget
from auv_planner.recovery import RecoveryQueryMode


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run M7.0 cost profiling with online early-stop and a parallel full audit."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/m7_budget_opt_v1"),
        help="Directory for optimized CSV, summary, hardware manifest, and report.",
    )
    parser.add_argument(
        "--full-workers",
        type=int,
        default=8,
        help="Process workers for the offline full audit; 1 disables parallelism.",
    )
    args = parser.parse_args()

    summary = run_m7_budget(
        args.output_dir,
        online_mode=RecoveryQueryMode.ONLINE_FAST,
        online_parallel_workers=1,
        full_parallel_workers=args.full_workers,
    )
    p1 = summary["profiles"]["P1_degraded"]
    print(
        "M7.0 optimized complete: "
        f"P1 single median {p1['single_full']['median_s'] * 1000.0:.1f} ms, "
        f"online time gate {summary['p1_online_time_gate']}, "
        f"online coverage gate {summary['p1_online_coverage_gate']}, "
        f"prefilter false negatives {summary['prefilter_false_negative_count']}"
    )
    print(f"Report: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
