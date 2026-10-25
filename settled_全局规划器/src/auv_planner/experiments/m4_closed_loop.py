from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import cos, pi, sin
from pathlib import Path

import numpy as np

from ..contracts import (
    FloatArray,
    ObstacleMap,
    ScenarioConfig,
    State,
    Trajectory,
    UniformCurrent,
    WorldSnapshot,
)
from ..control import Remus100TrackingController
from ..dynamics.remus100 import PvsRemus100Backend
from ..evaluation.closed_loop import (
    ClosedLoopResult,
    run_closed_loop,
    write_closed_loop_log,
    write_closed_loop_summary_csv,
)
from ..guidance import ThreeDAlosGuidance
from ..scenarios import remus100_vehicle


@dataclass(frozen=True)
class M4Case:
    name: str
    description: str
    scenario: ScenarioConfig
    trajectory: Trajectory
    max_duration_s: float


def m4_cases() -> tuple[M4Case, ...]:
    return (
        M4Case(
            name="straight",
            description="无流直航基线",
            scenario=_straight_scenario("straight", np.array([-5.0, 0.0, 0.0]), np.zeros(3), 0.35),
            trajectory=_straight_trajectory(),
            max_duration_s=10.0,
        ),
        M4Case(
            name="lateral_offset",
            description="初始横向偏置 0.5 m",
            scenario=_straight_scenario("lateral_offset", np.array([-5.0, 0.5, 0.0]), np.zeros(3), 0.5),
            trajectory=_straight_trajectory(),
            max_duration_s=12.0,
        ),
        M4Case(
            name="depth_offset",
            description="初始垂向偏置 0.5 m",
            scenario=_straight_scenario("depth_offset", np.array([-5.0, 0.0, 0.5]), np.zeros(3), 0.5),
            trajectory=_straight_trajectory(),
            max_duration_s=12.0,
        ),
        M4Case(
            name="single_turn",
            description="30 deg、40 m 半径单拐角",
            scenario=_turn_scenario(),
            trajectory=_turn_trajectory(),
            max_duration_s=30.0,
        ),
        M4Case(
            name="straight_along_current",
            description="顺流 0.3 m/s",
            scenario=_straight_scenario(
                "straight_along_current",
                np.array([-5.0, 0.0, 0.0]),
                np.array([0.3, 0.0, 0.0]),
                0.5,
            ),
            trajectory=_straight_trajectory(),
            max_duration_s=12.0,
        ),
        M4Case(
            name="straight_cross_current",
            description="横向均匀流 0.3 m/s",
            scenario=_straight_scenario(
                "straight_cross_current",
                np.array([-5.0, 0.0, 0.0]),
                np.array([0.0, 0.3, 0.0]),
                0.5,
            ),
            trajectory=_straight_trajectory(),
            max_duration_s=12.0,
        ),
    )


def run_m4_suite(
    output_dir: Path | str,
    *,
    seed: int = 0,
) -> tuple[ClosedLoopResult, ...]:
    root = Path(output_dir)
    logs_dir = root / "logs"
    results: list[ClosedLoopResult] = []
    for case in m4_cases():
        current = case.scenario.world.current_field
        assert isinstance(current, UniformCurrent)
        current_enabled = bool(np.linalg.norm(current.velocity_ned_mps) > 0.0)
        guidance = ThreeDAlosGuidance(
            3.0,
            2.0,
            1.5,
            current_field=current,
            compensate_current=current_enabled,
        )
        controller = Remus100TrackingController(
            case.scenario.vehicle,
            current_field=current,
        )
        backend = PvsRemus100Backend(sample_time_s=0.02)
        result = run_closed_loop(
            case.name,
            case.scenario,
            case.trajectory,
            guidance,
            controller,
            backend,
            max_duration_s=case.max_duration_s,
            seed=seed,
        )
        results.append(result)
        write_closed_loop_log(result, logs_dir / f"{case.name}.json")

    write_closed_loop_summary_csv(results, root / "summary.csv")
    _write_report(results, m4_cases(), root / "report.md")
    return tuple(results)


def _straight_trajectory() -> Trajectory:
    return _trajectory(np.array([[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float64))


def _turn_trajectory() -> Trajectory:
    radius_m = 40.0
    center = np.array([-10.0, radius_m, 0.0], dtype=np.float64)
    angles_rad = np.linspace(-pi / 2.0, -pi / 3.0, 30)
    arc = np.column_stack(
        (
            center[0] + radius_m * np.cos(angles_rad),
            center[1] + radius_m * np.sin(angles_rad),
            np.zeros_like(angles_rad),
        )
    )
    heading_rad = pi / 6.0
    direction = np.array([cos(heading_rad), sin(heading_rad), 0.0], dtype=np.float64)
    extension = np.array(
        [arc[-1] + distance * direction for distance in np.linspace(1.0, 10.0, 10)],
        dtype=np.float64,
    )
    return _trajectory(np.vstack((arc, extension)))


def _trajectory(points_ned_m: FloatArray) -> Trajectory:
    points = np.asarray(points_ned_m, dtype=np.float64)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    time_s = np.concatenate(([0.0], np.cumsum(lengths))) / 1.5
    rows = points.shape[0]
    return Trajectory(
        trajectory_id="m4-closed-loop",
        t_s=time_s,
        position_ned_m=points,
        quat_xyzw=np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (rows, 1)),
        linear_velocity_body_mps=np.tile(np.array([1.5, 0.0, 0.0]), (rows, 1)),
        angular_velocity_body_rps=np.zeros((rows, 3), dtype=np.float64),
        feasibility="unchecked",
    )


def _straight_scenario(
    name: str,
    start_ned_m: FloatArray,
    current_ned_mps: FloatArray,
    goal_radius_m: float,
) -> ScenarioConfig:
    return _scenario(
        name,
        start_ned_m,
        np.array([5.0, 0.0, 0.0], dtype=np.float64),
        current_ned_mps,
        goal_radius_m,
    )


def _turn_scenario() -> ScenarioConfig:
    path = _turn_trajectory().position_ned_m
    return _scenario(
        "single_turn",
        path[0],
        path[-1],
        np.zeros(3, dtype=np.float64),
        0.8,
    )


def _scenario(
    name: str,
    start_ned_m: FloatArray,
    goal_ned_m: FloatArray,
    current_ned_mps: FloatArray,
    goal_radius_m: float,
) -> ScenarioConfig:
    world = WorldSnapshot(
        t_s=0.0,
        obstacle_map=ObstacleMap(
            bounds_ned_m=np.array([[-100.0, 100.0], [-100.0, 100.0], [-20.0, 20.0]])
        ),
        current_field=UniformCurrent(velocity_ned_mps=current_ned_mps),
    )
    start = State(
        t_s=0.0,
        position_ned_m=start_ned_m,
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        linear_velocity_body_mps=np.array([1.0, 0.0, 0.0]),
        angular_velocity_body_rps=np.zeros(3),
    )
    return ScenarioConfig(
        name=name,
        world=world,
        vehicle=remus100_vehicle(),
        start=start,
        goal_ned_m=goal_ned_m,
        goal_radius_m=goal_radius_m,
        constraints={"controller_sample_time_s": 0.02},
    )


def _write_report(
    results: Sequence[ClosedLoopResult],
    cases: tuple[M4Case, ...],
    path: Path,
) -> Path:
    descriptions = {case.name: case.description for case in cases}
    lines = [
        "# M4.1 REMUS 100 闭环基线",
        "",
        "本报告只建立控制基线，不预设论文成功阈值。指标中的横向/垂向误差是相对规划折线路径的有符号投影误差，RMSE 使用全部闭环样本。",
        "",
        "| 场景 | 完成 | 时间 s | 横向 RMSE m | 垂向 RMSE m | 最大舵角 rad | 最大舵速 rad/s | 最大 RPM | 舵饱和 | 约束违反 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        metrics = result.metrics
        lines.append(
            "| {name} | {completed} | {time:.2f} | {hrmse:.3f} | {vrmse:.3f} | "
            "{rudder:.3f} | {rate:.3f} | {rpm:.1f} | {sat:.1%} | {violations} |".format(
                name=f"{result.case_name} ({descriptions[result.case_name]})",
                completed="是" if metrics.completed else "否",
                time=metrics.completion_time_s,
                hrmse=metrics.horizontal_rmse_m,
                vrmse=metrics.vertical_rmse_m,
                rudder=metrics.max_abs_rudder_angle_rad,
                rate=metrics.max_abs_rudder_rate_rad_s,
                rpm=metrics.max_propeller_rpm,
                sat=metrics.rudder_saturation_fraction,
                violations=metrics.constraint_violations,
            )
        )
    lines.append("")
    lines.append("逐样本状态、参考和指令见 `logs/`；汇总数据见 `summary.csv`。")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


__all__ = ["M4Case", "m4_cases", "run_m4_suite"]
