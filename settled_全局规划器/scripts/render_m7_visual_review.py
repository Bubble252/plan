from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Patch, Rectangle
import numpy as np

from auv_planner.contracts import ObstacleMap, State
from auv_planner.experiments.m7_budget import M7Probe, m7_audit_states, m7_probes
from auv_planner.recovery import (
    P1_DEGRADED,
    RecoverabilityChecker,
    RecoveryAssessment,
    RecoveryConfig,
)


SUCCESS_PROBES = ("open_water_start", "m5_multi_obstacle_start")
FAILURE_PROBES = ("gate_start", "s_bend_start")
PROBE_NOTICE = "FROZEN COST PROBE - NOT A PLANNER TRAJECTORY"


def _load_path_rows(results_dir: Path) -> list[dict[str, str]]:
    path = results_dir / "path_audit.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing path audit CSV: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _rows_for(
    rows: Iterable[dict[str, str]],
    *,
    probe: str,
    profile: str = "P1_degraded",
) -> dict[str, dict[str, str]]:
    return {
        row["mode"]: row
        for row in rows
        if row["probe"] == probe and row["profile"] == profile
    }


def _assess_probe(
    probe: M7Probe,
    *,
    config: RecoveryConfig,
) -> tuple[tuple[State, ...], tuple[RecoveryAssessment, ...]]:
    states = m7_audit_states(probe, config)
    checker = RecoverabilityChecker()
    assessments = tuple(
        checker.assess(
            state,
            probe.scenario.world,
            probe.scenario.vehicle,
            config=config,
            profile=P1_DEGRADED,
            safe_anchor_ned_m=probe.safe_anchor_ned_m,
        )
        for state in states
    )
    return states, assessments


def _merge_intervals(intervals: list[tuple[float, float]], *, start: float, end: float) -> list[tuple[float, float]]:
    clipped = sorted(
        (max(start, lo), min(end, hi))
        for lo, hi in intervals
        if hi > start and lo < end
    )
    if not clipped:
        return []
    merged: list[tuple[float, float]] = [clipped[0]]
    for lo, hi in clipped[1:]:
        previous_lo, previous_hi = merged[-1]
        if lo <= previous_hi:
            merged[-1] = (previous_lo, max(previous_hi, hi))
        else:
            merged.append((lo, hi))
    return merged


def _coverage(
    states: tuple[State, ...],
    assessments: tuple[RecoveryAssessment, ...],
) -> tuple[float, list[tuple[float, float]]]:
    start = float(states[0].t_s)
    end = float(states[-1].t_s)
    intervals = [
        (float(a.certificate.valid_from_s), float(a.certificate.valid_until_s))
        for a in assessments
        if a.valid
        and a.certificate.valid_from_s is not None
        and a.certificate.valid_until_s is not None
    ]
    merged = _merge_intervals(intervals, start=start, end=end)
    covered = sum(hi - lo for lo, hi in merged)
    duration = end - start
    return (0.0 if duration <= 0.0 else min(1.0, covered / duration), merged)


def _draw_obstacles_top(ax: Axes, obstacle_map: ObstacleMap) -> None:
    for obstacle in obstacle_map.obstacles:
        center = obstacle.center_ned_m
        if obstacle.kind == "sphere" and obstacle.radius_m is not None:
            ax.add_patch(Circle((center[0], center[1]), obstacle.radius_m, alpha=0.22))
        elif obstacle.kind == "cylinder" and obstacle.radius_m is not None:
            ax.add_patch(Circle((center[0], center[1]), obstacle.radius_m, alpha=0.22))
        elif obstacle.kind == "box" and obstacle.half_extents_m is not None:
            half = obstacle.half_extents_m
            ax.add_patch(
                Rectangle(
                    (center[0] - half[0], center[1] - half[1]),
                    2.0 * half[0],
                    2.0 * half[1],
                    alpha=0.22,
                )
            )
    bounds = obstacle_map.bounds_ned_m
    ax.set_xlim(bounds[0, 0], bounds[0, 1])
    ax.set_ylim(bounds[1, 0], bounds[1, 1])
    ax.set_aspect("equal", adjustable="box")


def _draw_obstacles_side(ax: Axes, obstacle_map: ObstacleMap) -> None:
    for obstacle in obstacle_map.obstacles:
        center = obstacle.center_ned_m
        if obstacle.kind == "sphere" and obstacle.radius_m is not None:
            ax.add_patch(Circle((center[0], center[2]), obstacle.radius_m, alpha=0.22))
        elif obstacle.kind == "cylinder" and obstacle.radius_m is not None and obstacle.height_m is not None:
            ax.add_patch(
                Rectangle(
                    (center[0] - obstacle.radius_m, center[2] - obstacle.height_m / 2.0),
                    2.0 * obstacle.radius_m,
                    obstacle.height_m,
                    alpha=0.22,
                )
            )
        elif obstacle.kind == "box" and obstacle.half_extents_m is not None:
            half = obstacle.half_extents_m
            ax.add_patch(
                Rectangle(
                    (center[0] - half[0], center[2] - half[2]),
                    2.0 * half[0],
                    2.0 * half[2],
                    alpha=0.22,
                )
            )
    bounds = obstacle_map.bounds_ned_m
    ax.set_xlim(bounds[0, 0], bounds[0, 1])
    ax.set_ylim(bounds[2, 1], bounds[2, 0])
    ax.set_aspect("equal", adjustable="box")


def _plot_status_points(
    ax: Axes,
    states: tuple[State, ...],
    assessments: tuple[RecoveryAssessment, ...],
    *,
    z_axis: bool = False,
) -> None:
    positions = np.vstack([state.position_ned_m for state in states])
    ax.plot(positions[:, 0], positions[:, 1] if not z_axis else positions[:, 2], color="0.25", lw=1.5)
    for state, assessment in zip(states, assessments):
        x, y, z = state.position_ned_m
        ax.scatter(
            x,
            z if z_axis else y,
            color="tab:green" if assessment.valid else "tab:red",
            marker="o" if assessment.valid else "x",
            s=42,
            zorder=5,
        )
        ax.annotate(
            f"{state.t_s:g}s",
            (x, z if z_axis else y),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )


def _plot_coverage_timeline(
    ax: Axes,
    states: tuple[State, ...],
    assessments: tuple[RecoveryAssessment, ...],
    intervals: list[tuple[float, float]],
) -> None:
    start = float(states[0].t_s)
    end = float(states[-1].t_s)
    ax.set_xlim(start, end)
    ax.set_ylim(-0.8, 0.8)
    ax.axvspan(start, end, color="tab:red", alpha=0.12, label="not covered")
    for lo, hi in intervals:
        ax.axvspan(lo, hi, color="tab:green", alpha=0.35, label="valid certificate")
    for state, assessment in zip(states, assessments):
        ax.axvline(state.t_s, color="0.75", lw=0.7, zorder=0)
        ax.scatter(
            state.t_s,
            0.0,
            color="tab:green" if assessment.valid else "tab:red",
            marker="o" if assessment.valid else "x",
            s=45,
            zorder=5,
        )
    ax.set_yticks([])
    ax.set_xlabel("time [s]")
    ax.set_title("Cost-probe certificate coverage timeline")
    handles = [
        Patch(color="tab:green", alpha=0.35, label="covered"),
        Patch(color="tab:red", alpha=0.12, label="not covered"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7)


def _save_case_figure(
    probe: M7Probe,
    states: tuple[State, ...],
    assessments: tuple[RecoveryAssessment, ...],
    output_dir: Path,
) -> Path:
    coverage_ratio, intervals = _coverage(states, assessments)
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 12),
        gridspec_kw={"height_ratios": [1.2, 1.2, 0.7]},
        constrained_layout=True,
    )
    top, side, timeline = axes
    _draw_obstacles_top(top, probe.scenario.world.obstacle_map)
    _plot_status_points(top, states, assessments)
    top.set_xlabel("x [m, NED]")
    top.set_ylabel("y [m, NED]")
    top.set_title(
        f"{probe.name}: brake_to_min_speed cost probe | coverage={coverage_ratio:.2f} | "
        f"valid={sum(a.valid for a in assessments)}/{len(assessments)}"
    )
    top.text(
        0.02,
        0.98,
        PROBE_NOTICE,
        transform=top.transAxes,
        ha="left",
        va="top",
        color="tab:red",
        fontsize=9,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "tab:red", "alpha": 0.9},
    )
    _draw_obstacles_side(side, probe.scenario.world.obstacle_map)
    _plot_status_points(side, states, assessments, z_axis=True)
    side.set_xlabel("x [m, NED]")
    side.set_ylabel("z [m, NED, down positive]")
    side.set_title("Side view: fixed-heading brake probe")
    _plot_coverage_timeline(timeline, states, assessments, intervals)
    output_path = output_dir / f"p1_{probe.name}_full_audit.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _save_case_gif(
    probe: M7Probe,
    states: tuple[State, ...],
    assessments: tuple[RecoveryAssessment, ...],
    output_dir: Path,
) -> Path:
    positions = np.vstack([state.position_ned_m for state in states])
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    _draw_obstacles_top(ax, probe.scenario.world.obstacle_map)
    ax.plot(positions[:, 0], positions[:, 1], color="0.65", lw=1.2, zorder=2)
    path_line, = ax.plot([], [], color="0.15", lw=2.0, zorder=3)
    point = ax.scatter([], [], color="tab:green", s=70, zorder=6)
    title = ax.set_title("")

    def update(index: int) -> tuple[Any, ...]:
        state = states[index]
        assessment = assessments[index]
        path_line.set_data(positions[: index + 1, 0], positions[: index + 1, 1])
        point.set_offsets(np.array([[state.position_ned_m[0], state.position_ned_m[1]]]))
        point.set_color("tab:green" if assessment.valid else "tab:red")
        status = "VALID" if assessment.valid else "INVALID"
        title.set_text(
            f"COST PROBE (not planner) | {probe.name} | t={state.t_s:g}s | {status} | "
            f"checkpoint {index + 1}/{len(states)}"
        )
        return path_line, point, title

    animation = FuncAnimation(fig, update, frames=len(states), interval=450, blit=False)
    output_path = output_dir / f"p1_{probe.name}_walkthrough.gif"
    animation.save(output_path, writer=PillowWriter(fps=2))
    plt.close(fig)
    return output_path


def _save_statistics(
    rows: list[dict[str, str]],
    output_dir: Path,
    *,
    baseline_dir: Path | None,
    results_dir: Path,
) -> Path:
    selected = [
        row
        for row in rows
        if row["profile"] == "P1_degraded"
        and row["mode"] == "online_10s"
        and row["include_in_aggregate"] == "True"
    ]
    labels = [row["probe"] for row in selected]
    online = [float(row["total_s"]) for row in selected]
    coverage = [float(row["coverage_ratio"]) for row in selected]
    early = [int(row["early_stopped_points"]) for row in selected]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    axes[0].bar(labels, online, color="tab:blue")
    axes[0].set_title("P1 online_fast cost-probe audit time")
    axes[0].set_ylabel("seconds")
    axes[0].tick_params(axis="x", rotation=35)
    axes[1].bar(labels, coverage, color=["tab:green" if value >= 1.0 else "tab:orange" for value in coverage])
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("P1 cost-probe certificate coverage")
    axes[1].set_ylabel("covered / path duration")
    axes[1].tick_params(axis="x", rotation=35)
    axes[2].bar(labels, early, color="tab:purple")
    axes[2].set_title("P1 cost-probe early-stopped points")
    axes[2].set_ylabel("count / 11")
    axes[2].tick_params(axis="x", rotation=35)
    output_path = output_dir / "p1_online_time_coverage_earlystop.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    if baseline_dir is not None:
        baseline_summary = json.loads((baseline_dir / "summary.json").read_text(encoding="utf-8"))
        optimized_summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
        profiles = ("P0_nominal", "P1_degraded", "P2_m6_stress")
        baseline_online = [
            baseline_summary["profiles"][profile]["path_online_10s"]["total_s"]["median_s"]
            for profile in profiles
        ]
        optimized_online = [
            optimized_summary["profiles"][profile]["path_online_10s"]["total_s"]["median_s"]
            for profile in profiles
        ]
        baseline_full = [
            baseline_summary["profiles"][profile]["path_full_cold"]["total_s"]["median_s"]
            for profile in profiles
        ]
        optimized_full = [
            optimized_summary["profiles"][profile]["path_full_cold"]["total_s"]["median_s"]
            for profile in profiles
        ]
        x = np.arange(len(profiles))
        width = 0.38
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
        axes[0].bar(x - width / 2, baseline_online, width, label="baseline")
        axes[0].bar(x + width / 2, optimized_online, width, label="optimized")
        axes[0].set_xticks(x, profiles)
        axes[0].set_ylabel("median seconds")
        axes[0].set_title("Cost-probe online audit: baseline vs optimized")
        axes[0].legend()
        axes[1].bar(x - width / 2, baseline_full, width, label="baseline")
        axes[1].bar(x + width / 2, optimized_full, width, label="optimized")
        axes[1].set_xticks(x, profiles)
        axes[1].set_ylabel("median seconds")
        axes[1].set_title("Cost-probe full cold audit: baseline vs optimized")
        axes[1].legend()
        comparison_path = output_dir / "m7_before_after_time.png"
        fig.savefig(comparison_path, dpi=160)
        plt.close(fig)
        return comparison_path
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render M7.0 success/failure review figures.")
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/m7_budget_opt_v1"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("outputs/m7_budget"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/m7_visual_review"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_path_rows(args.results_dir)
    probes = {probe.name: probe for probe in m7_probes()}
    config = RecoveryConfig()
    generated: list[Path] = []
    for name in (*SUCCESS_PROBES, *FAILURE_PROBES):
        probe = probes[name]
        states, assessments = _assess_probe(probe, config=config)
        generated.append(_save_case_figure(probe, states, assessments, args.output_dir))
        generated.append(_save_case_gif(probe, states, assessments, args.output_dir))
    generated.append(
        _save_statistics(
            rows,
            args.output_dir,
            baseline_dir=args.baseline_dir if args.baseline_dir.exists() else None,
            results_dir=args.results_dir,
        )
    )

    readme = [
        "# M7.0 可视化审阅",
        "",
        "> **重要：图中的轨迹是冻结的成本探针 `brake_to_min_speed`，不是 A*/RRT*、PSO 或 M7 候选规划器的输出。**",
        "> 该机动保持初始艏向和俯仰、只把速度降到最低，所以轨迹近似直线是预期现象；它也不会主动绕开障碍。",
        "> 这些图用于审阅“沿固定轨迹逐点查询证书”的成本和证书覆盖情况，不用于评价规划器路径质量。",
        "",
        "由 `scripts/render_m7_visual_review.py` 生成，数据来自 `"
        + str(args.results_dir)
        + "`。",
        "",
        "## 证书部分覆盖案例",
        "- `open_water_start`: 7/11 个点有效，覆盖率约 0.7；后段约 3 秒未被证书覆盖。",
        "- `m5_multi_obstacle_start`: 5/11 个点有效，覆盖率约 0.5；有效区间不连续。",
        "",
        "## 证书拒绝案例",
        "- `gate_start`: 0/11 个点有效，展示门框附近的恢复退路不足。",
        "- `s_bend_start`: 查询很快完成但 0/11 有效，说明“查询完成”不等于“证书覆盖”。",
        "",
        "## 文件",
        "- `p1_*_full_audit.png`: 成本探针的俯视、侧视和证书覆盖时间轴。",
        "- `p1_*_walkthrough.gif`: 沿成本探针逐点播放，绿色有效、红色无效。",
        "- `p1_online_time_coverage_earlystop.png`: 在线耗时、覆盖率和早停点统计。",
        "- `m7_before_after_time.png`: 若存在 baseline，显示优化前后耗时对比。",
        "",
        "绿色表示有效证书，红色表示无效证书；覆盖时间轴中的绿色区间才是真正被证书覆盖的时间。",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print("Generated:")
    for path in generated:
        print(f"- {path}")


if __name__ == "__main__":
    main()
