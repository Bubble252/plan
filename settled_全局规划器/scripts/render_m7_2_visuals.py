from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from auv_planner.contracts import ObstacleMap, ScenarioConfig
from auv_planner.experiments.m7_1 import M71Config, m71_case_specs


VARIANTS = ("astar", "rrt_star", "candidate_on", "candidate_off")
LABELS = {
    "astar": "A* baseline",
    "rrt_star": "RRT* baseline",
    "candidate_on": "candidate, recovery on",
    "candidate_off": "candidate, recovery off",
}
COLORS = {
    "astar": "0.15",
    "rrt_star": "tab:orange",
    "candidate_on": "tab:blue",
    "candidate_off": "tab:cyan",
}
STYLES = {
    "astar": "-",
    "rrt_star": "-",
    "candidate_on": "-",
    "candidate_off": "--",
}
STATUS_COLORS = {"valid": "tab:green", "invalid": "tab:red", "unknown": "tab:orange"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _scenario_map() -> dict[str, ScenarioConfig]:
    return {spec.name: spec.scenario for spec in m71_case_specs(M71Config())}


def _rows_for(
    rows: Iterable[dict[str, str]], *, case: str, variant: str
) -> list[dict[str, str]]:
    return [row for row in rows if row["case"] == case and row["variant"] == variant]


def _draw_obstacles_top(ax: Axes, obstacle_map: ObstacleMap) -> None:
    from matplotlib.patches import Circle, Rectangle

    for obstacle in obstacle_map.obstacles:
        x, y, _ = obstacle.center_ned_m
        if obstacle.kind in {"sphere", "cylinder"}:
            assert obstacle.radius_m is not None
            ax.add_patch(Circle((x, y), obstacle.radius_m, color="0.55", alpha=0.23))
        else:
            assert obstacle.half_extents_m is not None
            hx, hy, _ = obstacle.half_extents_m
            ax.add_patch(
                Rectangle((x - hx, y - hy), 2.0 * hx, 2.0 * hy, color="0.55", alpha=0.23)
            )
    bounds = obstacle_map.bounds_ned_m
    ax.set_xlim(bounds[0, 0], bounds[0, 1])
    ax.set_ylim(bounds[1, 0], bounds[1, 1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("north x [m]")
    ax.set_ylabel("east y [m]")
    ax.grid(alpha=0.2)


def _draw_obstacles_side(ax: Axes, obstacle_map: ObstacleMap) -> None:
    from matplotlib.patches import Circle, Rectangle

    for obstacle in obstacle_map.obstacles:
        x, _, z = obstacle.center_ned_m
        if obstacle.kind == "sphere":
            assert obstacle.radius_m is not None
            ax.add_patch(Circle((x, z), obstacle.radius_m, color="0.55", alpha=0.23))
        elif obstacle.kind == "cylinder":
            assert obstacle.radius_m is not None and obstacle.height_m is not None
            ax.add_patch(
                Rectangle(
                    (x - obstacle.radius_m, z - obstacle.height_m / 2.0),
                    2.0 * obstacle.radius_m,
                    obstacle.height_m,
                    color="0.55",
                    alpha=0.23,
                )
            )
        else:
            assert obstacle.half_extents_m is not None
            hx, _, hz = obstacle.half_extents_m
            ax.add_patch(Rectangle((x - hx, z - hz), 2.0 * hx, 2.0 * hz, color="0.55", alpha=0.23))
    bounds = obstacle_map.bounds_ned_m
    ax.set_xlim(bounds[0, 0], bounds[0, 1])
    ax.set_ylim(bounds[2, 1], bounds[2, 0])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("north x [m]")
    ax.set_ylabel("down z [m]")
    ax.grid(alpha=0.2)


def _trajectory_figure(
    case: str,
    scenario: ScenarioConfig,
    paths_by_variant: dict[str, dict[str, str]],
    trajectories: list[dict[str, str]],
) -> plt.Figure:
    fig, (top, side) = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    _draw_obstacles_top(top, scenario.world.obstacle_map)
    _draw_obstacles_side(side, scenario.world.obstacle_map)
    for variant in VARIANTS:
        path = paths_by_variant.get(variant)
        rows = _rows_for(trajectories, case=case, variant=variant)
        if path is None or not rows:
            continue
        coverage = 100.0 * float(path["coverage_ratio"])
        gap = float(path["max_uncovered_s"])
        label = f"{LABELS[variant]} (cov {coverage:.1f}%, gap {gap:.2f}s)"
        x = [float(row["x_m"]) for row in rows]
        y = [float(row["y_m"]) for row in rows]
        z = [float(row["z_m"]) for row in rows]
        top.plot(x, y, color=COLORS[variant], ls=STYLES[variant], lw=2.1, label=label)
        side.plot(x, z, color=COLORS[variant], ls=STYLES[variant], lw=2.1)
    start = scenario.start.position_ned_m
    goal = scenario.goal_ned_m
    top.scatter(start[0], start[1], marker="*", s=180, color="tab:green", label="start")
    top.scatter(goal[0], goal[1], marker="P", s=100, color="tab:purple", label="goal")
    side.scatter(start[0], start[2], marker="*", s=180, color="tab:green")
    side.scatter(goal[0], goal[2], marker="P", s=100, color="tab:purple")
    top.set_title("Planning trajectories: top view")
    side.set_title("Planning trajectories: side view")
    top.legend(fontsize=7, loc="best")
    fig.suptitle(
        f"M7.2 PLANNING TRAJECTORIES — {case} (same map, seed, current and 10 s budget)",
        fontsize=14,
        fontweight="bold",
    )
    return fig


def _coverage_figure(
    case: str,
    paths_by_variant: dict[str, dict[str, str]],
    points: list[dict[str, str]],
) -> plt.Figure:
    fig, axes = plt.subplots(len(VARIANTS), 1, figsize=(14, 2.15 * len(VARIANTS)), constrained_layout=True)
    for ax, variant in zip(axes, VARIANTS):
        path = paths_by_variant.get(variant)
        rows = sorted(_rows_for(points, case=case, variant=variant), key=lambda row: float(row["t_s"]))
        if path is None:
            ax.set_axis_off()
            continue
        success = path["plan_success"] == "True"
        if rows:
            start = min(float(row["t_s"]) for row in rows)
            end = max(float(row["t_s"]) for row in rows)
            ax.axvspan(start, end, color="tab:red", alpha=0.10)
            for row in rows:
                if row["full_status"] != "valid" or not row["certificate_valid_from_s"]:
                    continue
                low = max(start, float(row["certificate_valid_from_s"]))
                high = min(end, float(row["certificate_valid_until_s"]))
                if high > low:
                    ax.axvspan(low, high, color="tab:green", alpha=0.35)
            for row in rows:
                status = row["full_status"]
                ax.scatter(
                    float(row["t_s"]),
                    0.0,
                    color=STATUS_COLORS.get(status, "0.2"),
                    marker="o" if status == "valid" else "x",
                    s=42,
                    zorder=5,
                )
            ax.set_xlim(start, end)
        else:
            ax.set_xlim(0.0, 1.0)
            ax.text(
                0.5,
                0.5,
                f"no planning trajectory ({path.get('plan_reason') or path.get('plan_status')})",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.set_ylim(-0.65, 0.65)
        ax.set_yticks([])
        reason = "" if success else f" | failed: {path.get('plan_reason') or path.get('plan_status')}"
        gap_text = (
            f"longest gap {float(path['max_uncovered_s']):.2f} s"
            if success
            else "longest gap n/a"
        )
        ax.set_title(
            f"{LABELS[variant]}: coverage {100.0 * float(path['coverage_ratio']):.1f}%, "
            f"{gap_text}{reason}",
            loc="left",
            fontsize=9,
        )
        ax.grid(axis="x", alpha=0.18)
    axes[-1].set_xlabel("trajectory time [s]")
    fig.suptitle(
        f"M7.2 PLANNING TRAJECTORY CERTIFICATE COVERAGE — {case} (P1 degraded)",
        fontsize=14,
        fontweight="bold",
    )
    return fig


def _summary_scatter(paths: list[dict[str, str]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.0), constrained_layout=True)
    for variant in VARIANTS:
        rows = [
            row
            for row in paths
            if row["variant"] == variant and row["plan_success"] == "True"
        ]
        if not rows:
            continue
        ax.scatter(
            [float(row["max_uncovered_s"]) for row in rows],
            [100.0 * float(row["coverage_ratio"]) for row in rows],
            label=LABELS[variant],
            color=COLORS[variant],
            marker="o" if variant != "candidate_on" else "D",
            s=58,
        )
        for row in rows:
            coverage = 100.0 * float(row["coverage_ratio"])
            if coverage > 10.0:
                ax.annotate(
                    row["case"],
                    (float(row["max_uncovered_s"]), coverage),
                    xytext=(4, 3),
                    textcoords="offset points",
                    fontsize=6,
                )
    ax.text(
        0.99,
        0.03,
        "zero-coverage points are retained but not individually labelled",
        transform=ax.transAxes,
        ha="right",
        fontsize=7,
        color="0.35",
    )
    ax.set_xlabel("longest uncovered interval [s] (lower is better)")
    ax.set_ylabel("P1 certificate coverage [%] (higher is better)")
    ax.set_title("M7.2 PLANNING TRAJECTORIES — coverage / uncovered-gap Pareto view")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _summary_success(paths: list[dict[str, str]], output_path: Path) -> None:
    counts = {
        variant: sum(row["variant"] == variant and row["plan_success"] == "True" for row in paths)
        for variant in VARIANTS
    }
    fig, ax = plt.subplots(figsize=(8.5, 4.3), constrained_layout=True)
    ax.bar(list(counts), list(counts.values()), color=[COLORS[name] for name in counts])
    ax.set_ylim(0, max(8, max(counts.values()) + 1))
    ax.set_ylabel("successful nominal plans / 8 aggregate cases")
    ax.set_title("M7.2 planning success; low recovery-on success is explicit start rejection")
    ax.grid(axis="y", alpha=0.22)
    for index, value in enumerate(counts.values()):
        ax.text(index, value + 0.12, str(value), ha="center", va="bottom")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def render(output_dir: Path) -> None:
    paths = _read_csv(output_dir / "paths.csv")
    points = _read_csv(output_dir / "points.csv")
    trajectories = _read_csv(output_dir / "trajectories.csv")
    scenarios = _scenario_map()
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    cases = sorted({row["case"] for row in paths if row["include_in_aggregate"] == "True"})
    rendered = 0
    for case in cases:
        paths_by_variant = {
            row["variant"]: row for row in paths if row["case"] == case
        }
        trajectory_fig = _trajectory_figure(
            case,
            scenarios[case],
            paths_by_variant,
            trajectories,
        )
        trajectory_fig.savefig(figure_dir / f"{case}__planning_trajectories.png", dpi=170)
        plt.close(trajectory_fig)
        coverage_fig = _coverage_figure(case, paths_by_variant, points)
        coverage_fig.savefig(figure_dir / f"{case}__coverage_timeline.png", dpi=170)
        plt.close(coverage_fig)
        rendered += 1
    _summary_scatter(paths, figure_dir / "summary__coverage_gap_pareto.png")
    _summary_success(paths, figure_dir / "summary__plan_success.png")
    manifest = {
        "status": "M7.2 planning-trajectory visual review",
        "rendered_cases": rendered,
        "figures": sorted(path.name for path in figure_dir.glob("*.png")),
        "warning": (
            "Every trajectory is an actual A*, RRT*, recovery-on or recovery-off planner "
            "output sampled from the frozen scenario. Failure panels are not treated as zero-gap successes."
        ),
    }
    (output_dir / "visual_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render M7.2 planning-trajectory visual review.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/m7_2"))
    args = parser.parse_args()
    render(args.output_dir)
    print(f"Visual review: {args.output_dir / 'figures'}")


if __name__ == "__main__":
    main()
