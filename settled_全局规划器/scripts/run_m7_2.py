from __future__ import annotations

import argparse
import json
from pathlib import Path

from auv_planner.experiments.m7_2 import M72Config, run_m7_2


def _csv_names(value: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    if not names:
        raise argparse.ArgumentTypeError("expected at least one comma-separated name")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the M7.2 recovery-aware global planner comparison."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/m7_2"))
    parser.add_argument("--cases", type=_csv_names, default=None)
    parser.add_argument("--variants", type=_csv_names, default=None)
    parser.add_argument("--full-workers", type=int, default=8)
    args = parser.parse_args()

    config = M72Config(
        case_names=args.cases,
        variant_names=(
            M72Config().variant_names if args.variants is None else args.variants
        ),
        full_workers=args.full_workers,
    )
    result = run_m7_2(args.output_dir, config)
    summary = json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))
    print(f"M7.2 complete: {summary['case_count']} cases, {len(result.runs)} runs")
    for variant in config.variant_names:
        values = summary["aggregate"][variant]
        print(
            f"  {variant}: success {values['plan_success']}/{values['paths']}, "
            f"complete recovery {values['recovery_complete']}, "
            f"coverage median {values['coverage_ratio_all']['median']:.3f}, "
            f"max-gap median {values['max_uncovered_s_all']['median']:.3f} s"
        )
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
