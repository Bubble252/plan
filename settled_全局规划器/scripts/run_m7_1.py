from __future__ import annotations

import argparse
import json
from pathlib import Path

from auv_planner.experiments.m7_1 import M71Config, run_m7_1


def _csv_names(value: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    if not names:
        raise argparse.ArgumentTypeError("expected at least one comma-separated name")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit P1 recovery certificates on real A*/RRT* planner paths."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/m7_1"),
        help="Directory for path audits, point certificates, repair attempts, and manifests.",
    )
    parser.add_argument("--cases", type=_csv_names, default=None)
    parser.add_argument(
        "--planners",
        type=_csv_names,
        default=("astar", "rrt_star"),
    )
    parser.add_argument("--full-workers", type=int, default=8)
    parser.add_argument("--max-repair-candidates", type=int, default=12)
    args = parser.parse_args()

    config = M71Config(
        case_names=args.cases,
        planner_names=args.planners,
        full_workers=args.full_workers,
        max_repair_candidates=args.max_repair_candidates,
    )
    result = run_m7_1(args.output_dir, config)
    summary = json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))
    baseline = summary["baseline"]
    print(
        "M7.1 complete: "
        f"{baseline['plan_success']}/{baseline['paths']} plans succeeded, "
        f"median certificate coverage {baseline['coverage_ratio']['median']:.3f}, "
        f"accepted repairs {summary['repair_accepted_count']}/{summary['aggregate_case_count']}, "
        f"elapsed {result.elapsed_s:.1f} s"
    )
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
