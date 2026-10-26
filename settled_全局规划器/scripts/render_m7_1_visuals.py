from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from auv_planner.contracts import ObstacleMap, ScenarioConfig
from auv_planner.experiments.m7_1 import M71Config, m71_case_specs


REPRESENTATIVE_CASES = (
    "open_water_start",
    "sphere_start",
    "gate_start",
    "cross_current_open_center",
    "m5_multi_obstacle_start",
)
STATUS_COLORS = {
    "valid": "tab:green",
    "invalid": "tab:red",
    "unknown": "tab:orange",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _scenario_map() -> dict[str, ScenarioConfig]:
    return {spec.name: spec.scenario for spec in m71_case_specs(M71Config())}


def _rows_for(
    rows: Iterable[dict[str, str]],
    *,
    case: str,
    planner: str,
    variant: str | None = None,
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row["case"] == case
        and row["planner"] == planner
        and (variant is None or row["variant"] == variant)
    ]


def _path_state_variants(points: list[dict[str, str]], case: str, planner: str) -> list[str]:
    variants = {
        row["variant"]
        for row in points
        if row["case"] == case and row["planner"] == planner
    }
    ordered = [name for name in ("baseline", "repair_attempt", "repair_accepted") if name in variants]
    return ordered


def _draw_obstacles_top(ax: Axes, obstacle_map: ObstacleMap) -> None:
    for obstacle in obstacle_map.obstacles:
        x, y, _ = obstacle.center_ned_m
        color = "0.55"
        if obstacle.kind in {"sphere", "cylinder"}:
            assert obstacle.radius_m is not None
            ax.add_patch(Circle((x, y), obstacle.radius_m, color=color, alpha=0.28))
        elif obstacle.kind == "box":
            assert obstacle.half_extents_m is not None
            hx, hy, _ = obstacle.half_extents_m
            ax.add_patch(
                Rectangle((x - hx, y - hy), 2.0 * hx, 2.0 * hy, color=color, alpha=0.28)
            )
    bounds = obstacle_map.bounds_ned_m
    ax.set_xlim(bounds[0, 0], bounds[0, 1])
    ax.set_ylim(bounds[1, 0], bounds[1, 1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("north x [m]")
    ax.set_ylabel("east y [m]")
    ax.grid(alpha=0.2)


def _draw_obstacles_side(ax: Axes, obstacle_map: ObstacleMap) -> None:
    for obstacle in obstacle_map.obstacles:
        x, _, z = obstacle.center_ned_m
        color = "0.55"
        if obstacle.kind == "sphere":
            assert obstacle.radius_m is not None
            ax.add_patch(Circle((x, z), obstacle.radius_m, color=color, alpha=0.28))
        elif obstacle.kind == "cylinder":
            assert obstacle.radius_m is not None and obstacle.height_m is not None
            ax.add_patch(
                Rectangle(
                    (x - obstacle.radius_m, z - obstacle.height_m / 2.0),
                    2.0 * obstacle.radius_m,
                    obstacle.height_m,
                    color=color,
                    alpha=0.28,
                )
            )
        elif obstacle.kind == "box":
            assert obstacle.half_extents_m is not None
            hx, _, hz = obstacle.half_extents_m
            ax.add_patch(
                Rectangle((x - hx, z - hz), 2.0 * hx, 2.0 * hz, color=color, alpha=0.28)
            )
    bounds = obstacle_map.bounds_ned_m
    ax.set_xlim(bounds[0, 0], bounds[0, 1])
    ax.set_ylim(bounds[2, 1], bounds[2, 0])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("north x [m]")
    ax.set_ylabel("down z [m]")
    ax.grid(alpha=0.2)


def _draw_status_points(
    ax: Axes,
    point_rows: list[dict[str, str]],
    *,
    side_view: bool,
) -> None:
    for row in point_rows:
        status = row["full_status"]
        marker = "o" if status == "valid" else "x"
        vertical = float(row["z_m"]) if side_view else float(row["y_m"])
        ax.scatter(
            float(row["x_m"]),
            vertical,
            color=STATUS_COLORS.get(status, "0.2"),
            marker=marker,
            s=46,
            linewidths=1.2,
            zorder=7,
        )
    if point_rows:
        for row in point_rows[:: max(1, len(point_rows) // 6)]:
            vertical = float(row["z_m"]) if side_view else float(row["y_m"])
            ax.annotate(
                f"{float(row['t_s']):.1f}s",
                (float(row["x_m"]), vertical),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                zorder=8,
            )


def _draw_attempts(ax: Axes, attempt_rows: list[dict[str, str]]) -> None:
    for row in attempt_rows:
        if not row.get("via_x_m"):
            continue
        ax.scatter(
            float(row["via_x_m"]),
            float(row["via_y_m"]),
            marker="^",
            color="tab:orange",
            s=36,
            alpha=0.85,
            zorder=6,
        )


def _draw_3d(
    ax: Axes3D,
    trajectory_rows: list[dict[str, str]],
    point_rows: list[dict[str, str]],
    scenario: ScenarioConfig,
) -> None:
    ax.plot(
        [float(row["x_m"]) for row in trajectory_rows],
        [float(row["y_m"]) for row in trajectory_rows],
        [float(row["z_m"]) for row in trajectory_rows],
        color="0.2",
        lw=2.0,
    )
    for row in point_rows:
        status = row["full_status"]
        ax.scatter(
            float(row["x_m"]),
            float(row["y_m"]),
            float(row["z_m"]),
            color=STATUS_COLORS.get(status, "0.2"),
            marker="o" if status == "valid" else "x",
            s=44,
            zorder=7,
        )
    for obstacle in scenario.world.obstacle_map.obstacles:
        x, y, z = obstacle.center_ned_m
        if obstacle.kind in {"sphere", "cylinder"}:
            assert obstacle.radius_m is not None
            ax.scatter(x, y, z, s=260.0 * obstacle.radius_m, color="0.55", alpha=0.25)
        elif obstacle.kind == "box":
            assert obstacle.half_extents_m is not None
            ax.scatter(x, y, z, s=260.0, marker="s", color="0.55", alpha=0.25)
    bounds = scenario.world.obstacle_map.bounds_ned_m
    ax.set_xlim(bounds[0, 0], bounds[0, 1])
    ax.set_ylim(bounds[1, 0], bounds[1, 1])
    ax.set_zlim(bounds[2, 1], bounds[2, 0])
    ax.set_xlabel("x north [m]")
    ax.set_ylabel("y east [m]")
    ax.set_zlabel("z down [m]")
    ax.view_init(elev=22, azim=-56)


def _path_figure(
    case: str,
    planner: str,
    scenario: ScenarioConfig,
    path_row: dict[str, str],
    trajectory_rows: list[dict[str, str]],
    point_rows: list[dict[str, str]],
    attempt_rows: list[dict[str, str]],
) -> Figure:
    fig = plt.figure(figsize=(18, 5.6), constrained_layout=True)
    top = fig.add_subplot(1, 3, 1)
    side = fig.add_subplot(1, 3, 2)
    spatial = fig.add_subplot(1, 3, 3, projection="3d")
    _draw_obstacles_top(top, scenario.world.obstacle_map)
    _draw_obstacles_side(side, scenario.world.obstacle_map)
    top.plot(
        [float(row["x_m"]) for row in trajectory_rows],
        [float(row["y_m"]) for row in trajectory_rows],
        color="0.2",
        lw=2.0,
        label="planner trajectory",
    )
    side.plot(
        [float(row["x_m"]) for row in trajectory_rows],
        [float(row["z_m"]) for row in trajectory_rows],
        color="0.2",
        lw=2.0,
    )
    _draw_status_points(top, point_rows, side_view=False)
    _draw_status_points(side, point_rows, side_view=True)
    _draw_attempts(top, attempt_rows)
    _draw_3d(spatial, trajectory_rows, point_rows, scenario)
    goal_x, goal_y, goal_z = scenario.goal_ned_m
    start_x, start_y, start_z = scenario.start.position_ned_m
    top.scatter(start_x, start_y, marker="*", s=180, color="tab:blue", label="start", zorder=8)
    top.scatter(goal_x, goal_y, marker="P", s=110, color="tab:purple", label="goal", zorder=8)
    side.scatter(start_x, start_z, marker="*", s=180, color="tab:blue", zorder=8)
    side.scatter(goal_x, goal_z, marker="P", s=110, color="tab:purple", zorder=8)
    coverage = 100.0 * float(path_row["coverage_ratio"])
    max_gap = float(path_row["max_uncovered_s"])
    fig.suptitle(
        f"REAL PLANNER TRAJECTORY — {case} / {planner} "
        f"(P1 coverage {coverage:.1f}%, max uncovered {max_gap:.2f} s)",
        fontsize=14,
        fontweight="bold",
    )
    top.set_title("Top view: certificate samples")
    side.set_title("Side view: depth and x")
    spatial.set_title("3D planner trajectory")
    top.legend(loc="best", fontsize=8)
    return fig


def _coverage_figure(
    case: str,
    planner: str,
    path_row: dict[str, str],
    point_rows: list[dict[str, str]],
) -> Figure:
    fig, ax = plt.subplots(figsize=(13, 3.8), constrained_layout=True)
    start = min(float(row["t_s"]) for row in point_rows)
    end = max(float(row["t_s"]) for row in point_rows)
    ax.axvspan(start, end, color="tab:red", alpha=0.12, label="uncovered")
    for row in point_rows:
        if row["full_status"] != "valid" or not row["certificate_valid_from_s"]:
            continue
        valid_from = max(start, float(row["certificate_valid_from_s"]))
        valid_until = min(end, float(row["certificate_valid_until_s"]))
        if valid_until > valid_from:
            ax.axvspan(valid_from, valid_until, color="tab:green", alpha=0.35)
    for row in point_rows:
        status = row["full_status"]
        ax.axvline(float(row["t_s"]), color="0.8", lw=0.6, zorder=0)
        ax.scatter(
            float(row["t_s"]),
            0.0,
            color=STATUS_COLORS.get(status, "0.2"),
            marker="o" if status == "valid" else "x",
            s=58,
            zorder=5,
        )
    uncovered = path_row.get("uncovered_intervals", "")
    max_gap = float(path_row["max_uncovered_s"])
    ax.set_ylim(-0.75, 0.75)
    ax.set_yticks([])
    ax.set_xlim(start, end)
    ax.set_xlabel("trajectory time [s]")
    ax.set_title(
        f"PLANNER TRAJECTORY CERTIFICATE COVERAGE — {case} / {planner}; "
        f"longest gap {max_gap:.2f} s ({uncovered})"
    )
    handles = [
        plt.Line2D([0], [0], color="tab:green", lw=7, alpha=0.35, label="covered"),
        plt.Line2D([0], [0], color="tab:red", lw=7, alpha=0.18, label="uncovered"),
        plt.Line2D([0], [0], marker="x", color="tab:red", lw=0, label="invalid point"),
    ]
    ax.legend(handles=handles, loc="upper right", ncol=3, fontsize=8)
    return fig


def _repair_figure(
    case: str,
    planner: str,
    scenario: ScenarioConfig,
    baseline_traj: list[dict[str, str]],
    repair_traj: list[dict[str, str]],
    baseline_points: list[dict[str, str]],
    repair_points: list[dict[str, str]],
) -> Figure:
    fig = plt.figure(figsize=(13, 5.2), constrained_layout=True)
    top = fig.add_subplot(1, 2, 1)
    side = fig.add_subplot(1, 2, 2)
    _draw_obstacles_top(top, scenario.world.obstacle_map)
    _draw_obstacles_side(side, scenario.world.obstacle_map)
    for axis, baseline_values, repair_values, side_view in (
        (top, ("x_m", "y_m"), ("x_m", "y_m"), False),
        (side, ("x_m", "z_m"), ("x_m", "z_m"), True),
    ):
        axis.plot(
            [float(row[baseline_values[0]]) for row in baseline_traj],
            [float(row[baseline_values[1]]) for row in baseline_traj],
            color="0.45",
            lw=2.0,
            label="baseline",
        )
        axis.plot(
            [float(row[repair_values[0]]) for row in repair_traj],
            [float(row[repair_values[1]]) for row in repair_traj],
            color="tab:blue",
            lw=2.4,
            label="accepted repair",
        )
        _draw_status_points(axis, baseline_points, side_view=side_view)
        _draw_status_points(axis, repair_points, side_view=side_view)
    top.legend(loc="best", fontsize=8)
    fig.suptitle(
        f"REPAIR ACCEPTED ON REAL PLANNER TRAJECTORIES — {case} / {planner}",
        fontsize=14,
        fontweight="bold",
    )
    return fig


def _animate_path(
    output_path: Path,
    trajectory_rows: list[dict[str, str]],
    point_rows: list[dict[str, str]],
    title: str,
) -> None:
    x = np.array([float(row["x_m"]) for row in trajectory_rows])
    y = np.array([float(row["y_m"]) for row in trajectory_rows])
    z = np.array([float(row["z_m"]) for row in trajectory_rows])
    times = np.array([float(row["t_s"]) for row in trajectory_rows])
    point_times = np.array([float(row["t_s"]) for row in point_rows])
    point_colors = [STATUS_COLORS.get(row["full_status"], "0.2") for row in point_rows]
    frame_count = 90
    frame_times = np.linspace(float(times[0]), float(times[-1]), frame_count)
    frame_x = np.interp(frame_times, times, x)
    frame_y = np.interp(frame_times, times, y)

    fig, ax = plt.subplots(figsize=(8.0, 6.4), constrained_layout=True)
    ax.plot(x, y, color="0.65", lw=1.5)
    for px, py, color, state in zip(
        [float(row["x_m"]) for row in point_rows],
        [float(row["y_m"]) for row in point_rows],
        point_colors,
        [row["full_status"] for row in point_rows],
    ):
        ax.scatter(px, py, color=color, marker="o" if state == "valid" else "x", s=48, zorder=5)
    trail, = ax.plot([], [], color="tab:blue", lw=2.5, zorder=6)
    vehicle = ax.scatter([], [], color="black", marker=">", s=110, zorder=8)
    clock = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=11, va="top")
    ax.set_title(f"PLANNER TRAJECTORY ANIMATION — {title}")
    ax.set_xlabel("north x [m]")
    ax.set_ylabel("east y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)

    def update(index: int):
        stop = int(index)
        trail.set_data(frame_x[: stop + 1], frame_y[: stop + 1])
        vehicle.set_offsets(np.array([[frame_x[stop], frame_y[stop]]]))
        status_index = int(np.searchsorted(point_times, frame_times[stop], side="right") - 1)
        status = "pending"
        if status_index >= 0:
            status = point_rows[status_index]["full_status"]
        clock.set_text(f"t = {frame_times[stop]:.1f} s    certificate = {status}")
        return trail, vehicle, clock

    animation = FuncAnimation(fig, update, frames=range(frame_count), interval=90, blit=False)
    animation.save(output_path, writer=PillowWriter(fps=11))
    plt.close(fig)


def _animate_repair(
    output_path: Path,
    scenario: ScenarioConfig,
    baseline_rows: list[dict[str, str]],
    repair_rows: list[dict[str, str]],
    title: str,
) -> None:
    def arrays(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.array([float(row["t_s"]) for row in rows]),
            np.array([float(row["x_m"]) for row in rows]),
            np.array([float(row["y_m"]) for row in rows]),
        )

    baseline_t, baseline_x, baseline_y = arrays(baseline_rows)
    repair_t, repair_x, repair_y = arrays(repair_rows)
    frame_count = 90
    frame_times = np.linspace(0.0, max(float(baseline_t[-1]), float(repair_t[-1])), frame_count)
    baseline_fx = np.interp(frame_times, baseline_t, baseline_x)
    baseline_fy = np.interp(frame_times, baseline_t, baseline_y)
    repair_fx = np.interp(frame_times, repair_t, repair_x)
    repair_fy = np.interp(frame_times, repair_t, repair_y)

    fig, ax = plt.subplots(figsize=(8.0, 6.4), constrained_layout=True)
    _draw_obstacles_top(ax, scenario.world.obstacle_map)
    ax.plot(baseline_x, baseline_y, color="0.55", lw=1.7, label="baseline")
    ax.plot(repair_x, repair_y, color="tab:blue", lw=1.9, label="accepted repair")
    baseline_trail, = ax.plot([], [], color="0.2", lw=2.7, zorder=6)
    repair_trail, = ax.plot([], [], color="tab:blue", lw=2.7, zorder=7)
    baseline_vehicle = ax.scatter([], [], color="0.2", marker=">", s=105, zorder=8)
    repair_vehicle = ax.scatter([], [], color="tab:blue", marker=">", s=105, zorder=9)
    clock = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=11, va="top")
    ax.set_title(f"REPAIR COMPARISON ON REAL PLANNER TRAJECTORIES — {title}")
    ax.set_xlabel("north x [m]")
    ax.set_ylabel("east y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)

    def update(index: int):
        stop = int(index)
        baseline_trail.set_data(baseline_fx[: stop + 1], baseline_fy[: stop + 1])
        repair_trail.set_data(repair_fx[: stop + 1], repair_fy[: stop + 1])
        baseline_vehicle.set_offsets(np.array([[baseline_fx[stop], baseline_fy[stop]]]))
        repair_vehicle.set_offsets(np.array([[repair_fx[stop], repair_fy[stop]]]))
        clock.set_text(f"comparison clock = {frame_times[stop]:.1f} s")
        return baseline_trail, repair_trail, baseline_vehicle, repair_vehicle, clock

    animation = FuncAnimation(fig, update, frames=range(frame_count), interval=90, blit=False)
    animation.save(output_path, writer=PillowWriter(fps=11))
    plt.close(fig)


def render(output_dir: Path) -> None:
    figure_dir = output_dir / "figures"
    animation_dir = output_dir / "animations"
    figure_dir.mkdir(parents=True, exist_ok=True)
    animation_dir.mkdir(parents=True, exist_ok=True)
    paths = _read_csv(output_dir / "paths.csv")
    points = _read_csv(output_dir / "points.csv")
    trajectories = _read_csv(output_dir / "trajectories.csv")
    attempts = _read_csv(output_dir / "repair_attempts.csv")
    scenarios = _scenario_map()
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    rendered = 0
    for path_row in paths:
        case = path_row["case"]
        planner = path_row["planner"]
        variant = path_row["variant"]
        if variant != "baseline":
            continue
        scenario = scenarios[case]
        prefix = f"{case}__{planner}"
        point_rows = _rows_for(points, case=case, planner=planner, variant="baseline")
        trajectory_rows = _rows_for(trajectories, case=case, planner=planner, variant="baseline")
        if not point_rows or not trajectory_rows:
            continue
        attempt_rows = [row for row in attempts if row["case"] == case and row["planner"] == planner]
        path_fig = _path_figure(
            case,
            planner,
            scenario,
            path_row,
            trajectory_rows,
            point_rows,
            attempt_rows,
        )
        path_fig.savefig(figure_dir / f"{prefix}__path.png", dpi=170)
        plt.close(path_fig)
        coverage_fig = _coverage_figure(case, planner, path_row, point_rows)
        coverage_fig.savefig(figure_dir / f"{prefix}__coverage.png", dpi=170)
        plt.close(coverage_fig)
        if case in REPRESENTATIVE_CASES:
            _animate_path(
                animation_dir / f"{prefix}__path.gif",
                trajectory_rows,
                point_rows,
                f"{case} / {planner}",
            )
        accepted = next(
            (
                row
                for row in paths
                if row["case"] == case
                and row["planner"] == planner
                and row["variant"] == "repair_accepted"
            ),
            None,
        )
        if accepted is not None:
            repair_points = _rows_for(
                points, case=case, planner=planner, variant="repair_accepted"
            )
            repair_trajectory = _rows_for(
                trajectories,
                case=case,
                planner=planner,
                variant="repair_accepted",
            )
            repair_fig = _repair_figure(
                case,
                planner,
                scenario,
                trajectory_rows,
                repair_trajectory,
                point_rows,
                repair_points,
            )
            repair_fig.savefig(figure_dir / f"{prefix}__repair_comparison.png", dpi=170)
            plt.close(repair_fig)
            _animate_repair(
                animation_dir / f"{prefix}__repair_comparison.gif",
                scenario,
                trajectory_rows,
                repair_trajectory,
                f"{case} / {planner}",
            )
        rendered += 1

    figure, ax = plt.subplots(figsize=(12, max(4.0, 0.42 * len(paths))), constrained_layout=True)
    labels = [
        f"{row['case']} / {row['planner']}"
        + ("" if row["variant"] == "baseline" else f" [{row['variant']}]")
        for row in paths
    ]
    values = [100.0 * float(row["coverage_ratio"]) for row in paths]
    colors = [
        ("tab:blue" if row["variant"] == "repair_accepted" else "tab:green")
        if row["plan_success"] == "True"
        else "0.55"
        for row in paths
    ]
    ax.barh(np.arange(len(paths)), values, color=colors)
    ax.set_yticks(np.arange(len(paths)), labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 100.0)
    ax.set_xlabel("P1 certificate coverage [%]")
    ax.set_title("M7.1 REAL PLANNER PATH CERTIFICATE COVERAGE")
    ax.grid(axis="x", alpha=0.25)
    figure.savefig(figure_dir / "summary__coverage_by_path.png", dpi=170)
    plt.close(figure)

    (output_dir / "visual_manifest.json").write_text(
        json.dumps(
            {
                "status": "M7.1 real planner trajectory visual review",
                "rendered_paths": rendered,
                "figures": sorted(path.name for path in figure_dir.glob("*.png")),
                "animations": sorted(path.name for path in animation_dir.glob("*.gif")),
                "summary_repair_accepted": summary["repair_accepted_count"],
                "warning": "All paths are real A*/RRT* planner trajectories sampled at <=1 s.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render M7.1 planner-path visual review.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/m7_1"))
    args = parser.parse_args()
    render(args.output_dir)
    print(f"Visual review: {args.output_dir / 'figures'}")
    print(f"Animations: {args.output_dir / 'animations'}")


if __name__ == "__main__":
    main()
