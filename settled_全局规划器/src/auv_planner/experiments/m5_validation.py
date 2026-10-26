from __future__ import annotations

import csv
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import radians
from pathlib import Path
from typing import Any

import numpy as np

from ..attitude import euler_from_quaternion, quaternion_from_euler
from ..contracts import (
    CurrentField,
    FloatArray,
    ObstacleMap,
    Remus100ActuatorCommand,
    ScenarioConfig,
    SimulationSnapshot,
    State,
    Trajectory,
    UniformCurrent,
    WorldSnapshot,
)
from ..control import Remus100TrackingController
from ..current import DepthShearCurrent
from ..dynamics.mss_remus100 import MssRemus100Backend
from ..dynamics.remus100 import PvsRemus100Backend
from ..evaluation.closed_loop import ClosedLoopResult, run_closed_loop
from ..guidance import ThreeDAlosGuidance
from ..interfaces import SimulatorBackend
from ..scenarios import remus100_vehicle
from .m4_closed_loop import m4_cases

CommandPolicy = Callable[[State], Remus100ActuatorCommand]


@dataclass(frozen=True)
class OpenLoopCase:
    name: str
    description: str
    scenario: ScenarioConfig
    duration_s: float
    policy: CommandPolicy
    response_channel: str


@dataclass(frozen=True)
class OpenLoopTrace:
    times_s: FloatArray
    states: tuple[State, ...]
    actual_commands: tuple[Remus100ActuatorCommand, ...]

    def positions(self) -> FloatArray:
        return np.vstack([state.position_ned_m for state in self.states])

    def velocities(self) -> FloatArray:
        return np.vstack([state.linear_velocity_body_mps for state in self.states])

    def angular_velocities(self) -> FloatArray:
        return np.vstack([state.angular_velocity_body_rps for state in self.states])

    def euler_angles(self) -> FloatArray:
        return np.vstack([euler_from_quaternion(state.quat_xyzw) for state in self.states])


@dataclass(frozen=True)
class ClosedLoopCase:
    name: str
    description: str
    scenario: ScenarioConfig
    trajectory: Trajectory
    max_duration_s: float


class _ZigzagPolicy:
    def __init__(self, threshold_rad: float = radians(10.0)) -> None:
        self.threshold_rad = threshold_rad
        self.sign = 1.0

    def __call__(self, state: State) -> Remus100ActuatorCommand:
        yaw = float(euler_from_quaternion(state.quat_xyzw)[2])
        if self.sign > 0.0 and yaw >= self.threshold_rad:
            self.sign = -1.0
        elif self.sign < 0.0 and yaw <= -self.threshold_rad:
            self.sign = 1.0
        return _command(
            stamp_s=state.t_s,
            rudder_rad=self.sign * self.threshold_rad,
            rpm=1000.0,
        )


def run_m5_validation(output_dir: Path | str, *, seed: int = 0) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    derivative_rows = run_derivative_comparison()
    mass_rows = run_mass_density_sensitivity()
    open_loop_rows, traces = run_open_loop_comparison()
    closed_loop_rows, closed_loop_results = run_closed_loop_comparison(seed=seed)
    convergence_rows = run_step_convergence()
    threshold_audit = build_threshold_audit(
        derivative_rows,
        open_loop_rows,
        closed_loop_rows,
        convergence_rows,
    )

    _write_csv(root / "derivative_comparison.csv", derivative_rows)
    _write_csv(root / "mass_density_sensitivity.csv", mass_rows)
    _write_csv(root / "open_loop_comparison.csv", open_loop_rows)
    _write_csv(root / "closed_loop_comparison.csv", closed_loop_rows)
    _write_csv(root / "step_convergence.csv", convergence_rows)
    _write_traces(root / "traces.json", traces)
    _write_closed_loop_logs(root / "closed_loop_logs", closed_loop_results)
    summary = {
        "seed": seed,
        "derivative_cases": len(derivative_rows),
        "open_loop_cases": len(open_loop_rows),
        "closed_loop_cases": len(closed_loop_rows),
        "convergence_rows": len(convergence_rows),
        "model_thresholds": proposed_model_thresholds(),
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "threshold_audit.json").write_text(
        json.dumps(threshold_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(
        root / "report.md",
        derivative_rows,
        mass_rows,
        open_loop_rows,
        closed_loop_rows,
        convergence_rows,
    )
    return summary


def derivative_cases() -> tuple[tuple[str, str, State, Remus100ActuatorCommand, CurrentField], ...]:
    calm = UniformCurrent(np.zeros(3))
    cross = UniformCurrent(np.array([0.0, 0.3, 0.0]))
    shear = DepthShearCurrent(
        reference_velocity_ned_mps=np.array([0.2, 0.0, 0.0]),
        shear_ned_per_m=np.array([0.02, 0.01, 0.0]),
    )
    return (
        (
            "calm_straight",
            "1.5 m/s, zero fins, 1000 RPM",
            _state(),
            _command(rpm=1000.0),
            calm,
        ),
        (
            "calm_rudder",
            "1.5 m/s, +10 deg rudder, 1000 RPM",
            _state(),
            _command(rudder_rad=radians(10.0), rpm=1000.0),
            calm,
        ),
        (
            "calm_stern_plane",
            "1.5 m/s, +10 deg stern plane, 1000 RPM",
            _state(),
            _command(stern_plane_rad=radians(10.0), rpm=1000.0),
            calm,
        ),
        (
            "combined_attitude",
            "roll 0.05 rad, pitch -0.05 rad, yaw 0.3 rad, combined fin command",
            _state(euler=(0.05, -0.05, 0.3)),
            _command(
                rudder_rad=radians(8.0),
                stern_plane_rad=radians(-6.0),
                rpm=1100.0,
            ),
            calm,
        ),
        (
            "cross_current",
            "1.5 m/s with 0.3 m/s beam current",
            _state(),
            _command(rpm=1000.0),
            cross,
        ),
        (
            "depth_shear",
            "10 m depth in a horizontal depth-shear current",
            _state(position=(0.0, 0.0, 10.0)),
            _command(rpm=1000.0),
            shear,
        ),
    )


def run_derivative_comparison() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, description, state, command, current in derivative_cases():
        scenario = _open_loop_scenario(name, state, current)
        pvs = PvsRemus100Backend(allow_sheared_current=True)
        mss = MssRemus100Backend(allow_sheared_current=True)
        try:
            pvs.reset(scenario, seed=0)
            mss.reset(scenario, seed=0)
            pvs_derivative = pvs.derivative_nu_dot(command)
            mss_derivative = mss.derivative_nu_dot(command)
        finally:
            mss.close()
        relative_l2 = float(np.linalg.norm(pvs_derivative - mss_derivative)) / max(
            float(np.linalg.norm(pvs_derivative)), 1e-12
        )
        row: dict[str, object] = {
            "case": name,
            "description": description,
            "relative_l2": relative_l2,
            "max_abs_error": float(np.max(np.abs(pvs_derivative - mss_derivative))),
            "pvs_nu_dot_l2": float(np.linalg.norm(pvs_derivative)),
            "mss_nu_dot_l2": float(np.linalg.norm(mss_derivative)),
        }
        for index, component in enumerate(("u", "v", "w", "p", "q", "r")):
            row[f"pvs_{component}_dot"] = float(pvs_derivative[index])
            row[f"mss_{component}_dot"] = float(mss_derivative[index])
            row[f"delta_{component}_dot"] = float(mss_derivative[index] - pvs_derivative[index])
        rows.append(row)
    return rows


def run_mass_density_sensitivity() -> list[dict[str, object]]:
    scenario = _open_loop_scenario("mass-density", _state(), UniformCurrent(np.zeros(3)))
    pvs = PvsRemus100Backend()
    mss = MssRemus100Backend()
    try:
        pvs.reset(scenario, seed=0)
        mss.reset(scenario, seed=0)
        pvs_mass = pvs.mass_matrix_diagonal()
        mss_mass = mss.mass_matrix_diagonal()
    finally:
        mss.close()
    rows: list[dict[str, object]] = []
    for model, density, derived_mass, diagonal in (
        ("PVS", 1026.0, 31.0294, pvs_mass),
        ("MSS", 1025.0, 31.9316, mss_mass),
    ):
        row: dict[str, object] = {
            "model": model,
            "native_density_kg_m3": density,
            "derived_mass_kg": derived_mass,
            "first_order_density_only_acceleration_ratio": 1.0,
        }
        for index, axis in enumerate(("surge", "sway", "heave", "roll", "pitch", "yaw")):
            row[f"mass_matrix_{axis}"] = float(diagonal[index])
        rows.append(row)
    rows.append(
        {
            "model": "mass_scaling",
            "native_density_kg_m3": "",
            "derived_mass_kg": 31.9316 / 31.0294,
            "first_order_density_only_acceleration_ratio": 1.0,
            "note": "MSS/PVS derived-mass ratio; for equal forces acceleration scales by the inverse ratio",
        }
    )
    return rows


def open_loop_cases() -> tuple[OpenLoopCase, ...]:
    calm = UniformCurrent(np.zeros(3))
    cross = UniformCurrent(np.array([0.0, 0.3, 0.0]))
    return (
        OpenLoopCase(
            "straight",
            "zero current constant RPM",
            _open_loop_scenario("m5_straight", _state(), calm),
            10.0,
            lambda state: _command(stamp_s=state.t_s, rpm=1000.0),
            "surge_mps",
        ),
        OpenLoopCase(
            "rudder_step",
            "+10 deg rudder step",
            _open_loop_scenario("m5_rudder_step", _state(), calm),
            8.0,
            lambda state: _command(stamp_s=state.t_s, rudder_rad=radians(10.0), rpm=1000.0),
            "yaw_rate_rps",
        ),
        OpenLoopCase(
            "stern_plane_step",
            "+10 deg stern-plane step",
            _open_loop_scenario("m5_stern_step", _state(), calm),
            8.0,
            lambda state: _command(stamp_s=state.t_s, stern_plane_rad=radians(10.0), rpm=1000.0),
            "pitch_rate_rps",
        ),
        OpenLoopCase(
            "zigzag_10_10",
            "10 deg / 10 deg heading zigzag",
            _open_loop_scenario("m5_zigzag", _state(), calm),
            12.0,
            _ZigzagPolicy(),
            "yaw_rate_rps",
        ),
        OpenLoopCase(
            "uniform_cross_current",
            "0.3 m/s beam current, constant command",
            _open_loop_scenario("m5_cross_current", _state(), cross),
            10.0,
            lambda state: _command(stamp_s=state.t_s, rpm=1000.0),
            "sway_mps",
        ),
    )


def run_open_loop_comparison() -> tuple[list[dict[str, object]], dict[str, Any]]:
    rows: list[dict[str, object]] = []
    trace_payload: dict[str, Any] = {}
    for case in open_loop_cases():
        pvs_backend = PvsRemus100Backend()
        mss_backend = MssRemus100Backend(integrator="euler")
        try:
            pvs_trace = run_open_loop_trace(pvs_backend, case, sample_time_s=0.02)
            mss_trace = run_open_loop_trace(mss_backend, case, sample_time_s=0.02)
        finally:
            mss_backend.close()
        pvs_metrics = model_trace_metrics(pvs_trace, case.response_channel)
        mss_metrics = model_trace_metrics(mss_trace, case.response_channel)
        cross = compare_traces(pvs_trace, mss_trace)
        row: dict[str, object] = {
            "case": case.name,
            "description": case.description,
            "response_channel": case.response_channel,
            **{f"pvs_{key}": value for key, value in pvs_metrics.items()},
            **{f"mss_{key}": value for key, value in mss_metrics.items()},
            **cross,
        }
        rows.append(row)
        trace_payload[case.name] = {
            "pvs": _trace_payload(pvs_trace),
            "mss": _trace_payload(mss_trace),
        }
    return rows, trace_payload


def run_open_loop_trace(
    backend: SimulatorBackend,
    case: OpenLoopCase,
    *,
    sample_time_s: float,
) -> OpenLoopTrace:
    snapshot = backend.reset(case.scenario, seed=0)
    states = [snapshot.state]
    actual_commands = [_remus_actuator_state(snapshot)]
    end_time_s = snapshot.state.t_s + case.duration_s
    while snapshot.state.t_s < end_time_s - 1e-12:
        command = case.policy(snapshot.state)
        if not isinstance(command, Remus100ActuatorCommand):
            raise TypeError("open-loop policies must return Remus100ActuatorCommand")
        if abs(command.stamp_s - snapshot.state.t_s) > 1e-9:
            raise ValueError("policy command timestamp must match the current state")
        snapshot = backend.step(command)
        states.append(snapshot.state)
        actual_commands.append(_remus_actuator_state(snapshot))
        if not np.isclose(snapshot.state.t_s - states[-2].t_s, sample_time_s, atol=1e-9):
            raise ValueError("backend sample time does not match the validation sample time")
    return OpenLoopTrace(
        times_s=np.array([state.t_s for state in states], dtype=np.float64),
        states=tuple(states),
        actual_commands=tuple(actual_commands),
    )


def model_trace_metrics(trace: OpenLoopTrace, response_channel: str) -> dict[str, float]:
    velocities = trace.velocities()
    angular_velocities = trace.angular_velocities()
    positions = trace.positions()
    euler = trace.euler_angles()
    channel_values = _channel_values(trace, response_channel)
    limits = _actuator_limits()
    actual = np.vstack(
        [
            [command.rudder_angle_rad, command.stern_plane_angle_rad, command.propeller_rpm]
            for command in trace.actual_commands
        ]
    )
    return {
        "terminal_surge_mps": float(velocities[-1, 0]),
        "terminal_sway_mps": float(velocities[-1, 1]),
        "terminal_heave_mps": float(velocities[-1, 2]),
        "terminal_yaw_rate_rps": float(angular_velocities[-1, 2]),
        "terminal_pitch_rate_rps": float(angular_velocities[-1, 1]),
        "terminal_depth_m": float(positions[-1, 2]),
        "terminal_yaw_rad": float(euler[-1, 2]),
        "response_final_value": float(channel_values[-1]),
        "response_peak_abs": float(np.max(np.abs(channel_values))),
        "response_time_90_s": _response_time_90(trace.times_s, channel_values),
        "path_length_m": float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))),
        "max_abs_rudder_rad": float(np.max(np.abs(actual[:, 0]))),
        "max_abs_stern_plane_rad": float(np.max(np.abs(actual[:, 1]))),
        "max_propeller_rpm": float(np.max(actual[:, 2])),
        "rudder_saturation_fraction": float(np.mean(np.abs(actual[:, 0]) >= 0.99 * limits[0])),
        "stern_plane_saturation_fraction": float(np.mean(np.abs(actual[:, 1]) >= 0.99 * limits[1])),
        "propeller_saturation_fraction": float(np.mean(actual[:, 2] >= 0.99 * limits[2])),
        "constraint_violations": float(_constraint_violations(actual, limits)),
    }


def compare_traces(pvs: OpenLoopTrace, mss: OpenLoopTrace) -> dict[str, float]:
    count = min(len(pvs.states), len(mss.states))
    pvs_positions = pvs.positions()[:count]
    mss_positions = mss.positions()[:count]
    pvs_velocities = pvs.velocities()[:count]
    mss_velocities = mss.velocities()[:count]
    pvs_euler = pvs.euler_angles()[:count]
    mss_euler = mss.euler_angles()[:count]
    position_errors = np.linalg.norm(pvs_positions - mss_positions, axis=1)
    pvs_path_length = float(
        np.sum(np.linalg.norm(np.diff(pvs_positions, axis=0), axis=1))
    )
    return {
        "trajectory_rms_position_error_m": float(np.sqrt(np.mean(position_errors**2))),
        "trajectory_max_position_error_m": float(np.max(position_errors)),
        "trajectory_final_position_error_m": float(position_errors[-1]),
        "trajectory_rms_velocity_error_mps": float(
            np.sqrt(np.mean(np.sum((pvs_velocities - mss_velocities) ** 2, axis=1)))
        ),
        "final_surge_error_mps": float(pvs_velocities[-1, 0] - mss_velocities[-1, 0]),
        "final_sway_error_mps": float(pvs_velocities[-1, 1] - mss_velocities[-1, 1]),
        "final_yaw_error_rad": float(pvs_euler[-1, 2] - mss_euler[-1, 2]),
        "final_pitch_error_rad": float(pvs_euler[-1, 1] - mss_euler[-1, 1]),
        "normalized_final_position_error": float(position_errors[-1] / max(pvs_path_length, 1e-12)),
    }


def run_step_convergence() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sample_times = (0.02, 0.01, 0.005, 0.0025)
    selected_cases = tuple(case for case in open_loop_cases() if case.name in {"straight", "rudder_step"})
    for case in selected_cases:
        for model in ("pvs_euler", "mss_euler", "mss_rk4"):
            traces: dict[float, OpenLoopTrace] = {}
            for sample_time in sample_times:
                if model == "pvs_euler":
                    backend: SimulatorBackend = PvsRemus100Backend(sample_time_s=sample_time)
                    close_backend = False
                else:
                    backend = MssRemus100Backend(
                        sample_time_s=sample_time,
                        integrator="euler" if model == "mss_euler" else "rk4",
                    )
                    close_backend = True
                try:
                    traces[sample_time] = run_open_loop_trace(
                        backend, case, sample_time_s=sample_time
                    )
                finally:
                    if close_backend:
                        backend.close()  # type: ignore[attr-defined]
            reference = traces[sample_times[-1]]
            reference_positions = reference.positions()
            reference_speed = reference.velocities()[-1, 0]
            reference_path_length = float(
                np.sum(np.linalg.norm(np.diff(reference_positions, axis=0), axis=1))
            )
            previous_error: float | None = None
            for sample_time in sample_times:
                trace = traces[sample_time]
                position_error = float(
                    np.linalg.norm(trace.positions()[-1] - reference_positions[-1])
                )
                ratio = (
                    previous_error / position_error
                    if previous_error is not None and position_error > 0.0
                    else float("nan")
                )
                rows.append(
                    {
                        "case": case.name,
                        "model": model,
                        "sample_time_s": sample_time,
                        "final_position_error_vs_finest_m": position_error,
                        "normalized_final_position_error": position_error
                        / max(reference_path_length, 1e-12),
                        "final_surge_error_vs_finest_mps": float(
                            trace.velocities()[-1, 0] - reference_speed
                        ),
                        "observed_error_ratio_to_next_finer": ratio,
                        "step_count": len(trace.states) - 1,
                    }
                )
                previous_error = position_error
    return rows


def closed_loop_cases() -> tuple[ClosedLoopCase, ...]:
    selected = {"straight", "lateral_offset", "depth_offset", "single_turn", "straight_cross_current"}
    cases: list[ClosedLoopCase] = []
    for case in m4_cases():
        if case.name not in selected:
            continue
        cases.append(
            ClosedLoopCase(
                case.name,
                case.description,
                case.scenario,
                case.trajectory,
                case.max_duration_s,
            )
        )
    current = DepthShearCurrent(
        reference_velocity_ned_mps=np.array([0.2, 0.0, 0.0]),
        shear_ned_per_m=np.array([0.02, 0.01, 0.0]),
    )
    shear_scenario = _closed_loop_scenario(current)
    cases.append(
        ClosedLoopCase(
            "depth_shear_tracking",
            "depth-shear current with 3D ALOS and the common tracking controller",
            shear_scenario,
            _trajectory(
                np.array(
                    [[-5.0, 0.0, 0.0], [0.0, 0.0, -0.75], [5.0, 0.0, -1.5]],
                    dtype=np.float64,
                )
            ),
            12.0,
        )
    )
    return tuple(cases)


def run_closed_loop_comparison(*, seed: int = 0) -> tuple[list[dict[str, object]], dict[str, ClosedLoopResult]]:
    rows: list[dict[str, object]] = []
    logs: dict[str, ClosedLoopResult] = {}
    for case in closed_loop_cases():
        pvs_result = _run_closed_loop_case(case, seed=seed, model="pvs")
        mss_result = _run_closed_loop_case(case, seed=seed, model="mss")
        pvs_metrics = pvs_result.metrics
        mss_metrics = mss_result.metrics
        common = min(len(pvs_result.samples), len(mss_result.samples))
        position_difference = np.array(
            [
                np.linalg.norm(
                    pvs_result.samples[index].state.position_ned_m
                    - mss_result.samples[index].state.position_ned_m
                )
                for index in range(common)
            ],
            dtype=np.float64,
        )
        rows.append(
            {
                "case": case.name,
                "description": case.description,
                "pvs_completed": pvs_metrics.completed,
                "mss_completed": mss_metrics.completed,
                "pvs_completion_time_s": pvs_metrics.completion_time_s,
                "mss_completion_time_s": mss_metrics.completion_time_s,
                "completion_time_difference_s": mss_metrics.completion_time_s
                - pvs_metrics.completion_time_s,
                "pvs_horizontal_rmse_m": pvs_metrics.horizontal_rmse_m,
                "mss_horizontal_rmse_m": mss_metrics.horizontal_rmse_m,
                "pvs_vertical_rmse_m": pvs_metrics.vertical_rmse_m,
                "mss_vertical_rmse_m": mss_metrics.vertical_rmse_m,
                "pvs_goal_distance_m": pvs_metrics.goal_distance_m,
                "mss_goal_distance_m": mss_metrics.goal_distance_m,
                "common_sample_trajectory_rms_m": float(
                    np.sqrt(np.mean(position_difference**2))
                ),
                "common_sample_trajectory_final_m": float(position_difference[-1]),
                "pvs_max_abs_rudder_rad": pvs_metrics.max_abs_rudder_angle_rad,
                "mss_max_abs_rudder_rad": mss_metrics.max_abs_rudder_angle_rad,
                "pvs_max_propeller_rpm": pvs_metrics.max_propeller_rpm,
                "mss_max_propeller_rpm": mss_metrics.max_propeller_rpm,
                "pvs_rudder_saturation_fraction": pvs_metrics.rudder_saturation_fraction,
                "mss_rudder_saturation_fraction": mss_metrics.rudder_saturation_fraction,
                "pvs_constraint_violations": pvs_metrics.constraint_violations,
                "mss_constraint_violations": mss_metrics.constraint_violations,
            }
        )
        logs[f"{case.name}__pvs"] = pvs_result
        logs[f"{case.name}__mss"] = mss_result
    return rows, logs


def proposed_model_thresholds() -> dict[str, float | str]:
    return {
        "interface_axis_and_sign": "exact within numerical tolerance; zero components below 1e-8",
        "calm_derivative_relative_l2_max": 0.07,
        "cross_current_derivative_relative_l2_max": 0.35,
        "open_loop_final_position_normalized_max": 0.02,
        "closed_loop_completion_time_difference_max_s": 0.5,
        "closed_loop_common_trajectory_rms_max_m": 0.25,
        "constraint_violations_max": 0.0,
        "euler_step_0p02_normalized_error_max": 0.005,
        "status": "frozen by user on 2026-10-25; project acceptance thresholds, not industry standards",
    }


def _audit_entry(measured: float, threshold: float) -> dict[str, float | bool]:
    return {
        "measured": float(measured),
        "pass": bool(measured <= threshold + 1e-12),
        "threshold": float(threshold),
    }


def _row_float(row: dict[str, object], key: str) -> float:
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"{key} must be numeric, got {type(value).__name__}")
    return float(value)


def build_threshold_audit(
    derivative_rows: Sequence[dict[str, object]],
    open_loop_rows: Sequence[dict[str, object]],
    closed_loop_rows: Sequence[dict[str, object]],
    convergence_rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    thresholds = proposed_model_thresholds()
    derivative_by_case = {str(row["case"]): row for row in derivative_rows}
    calm_cases = ("calm_straight", "calm_rudder", "calm_stern_plane", "combined_attitude")
    calm_derivative = max(
        _row_float(derivative_by_case[name], "relative_l2") for name in calm_cases
    )
    depth_shear_derivative = _row_float(derivative_by_case["depth_shear"], "relative_l2")
    cross_current_derivative = _row_float(
        derivative_by_case["cross_current"], "relative_l2"
    )
    open_loop_error = max(
        _row_float(row, "normalized_final_position_error") for row in open_loop_rows
    )
    completion_time_difference = max(
        abs(_row_float(row, "completion_time_difference_s")) for row in closed_loop_rows
    )
    common_trajectory_rms = max(
        _row_float(row, "common_sample_trajectory_rms_m") for row in closed_loop_rows
    )
    constraint_violations = sum(
        _row_float(row, "pvs_constraint_violations")
        + _row_float(row, "mss_constraint_violations")
        for row in closed_loop_rows
    )
    euler_error = max(
        _row_float(row, "normalized_final_position_error")
        for row in convergence_rows
        if str(row["model"]) in {"pvs_euler", "mss_euler"}
        and _row_float(row, "sample_time_s") == 0.02
    )
    entries = {
        "calm_derivative_relative_l2": _audit_entry(
            calm_derivative, float(thresholds["calm_derivative_relative_l2_max"])
        ),
        "depth_shear_derivative_relative_l2": _audit_entry(
            depth_shear_derivative, float(thresholds["calm_derivative_relative_l2_max"])
        ),
        "cross_current_derivative_relative_l2": _audit_entry(
            cross_current_derivative,
            float(thresholds["cross_current_derivative_relative_l2_max"]),
        ),
        "open_loop_normalized_final_position_error": _audit_entry(
            open_loop_error, float(thresholds["open_loop_final_position_normalized_max"])
        ),
        "closed_loop_completion_time_difference_s": _audit_entry(
            completion_time_difference,
            float(thresholds["closed_loop_completion_time_difference_max_s"]),
        ),
        "closed_loop_common_trajectory_rms_m": _audit_entry(
            common_trajectory_rms,
            float(thresholds["closed_loop_common_trajectory_rms_max_m"]),
        ),
        "constraint_violations": _audit_entry(
            constraint_violations, float(thresholds["constraint_violations_max"])
        ),
        "euler_0p02_normalized_error": _audit_entry(
            euler_error, float(thresholds["euler_step_0p02_normalized_error_max"])
        ),
    }
    return {
        "all_pass": all(bool(entry["pass"]) for entry in entries.values()),
        **entries,
    }


def _run_closed_loop_case(case: ClosedLoopCase, *, seed: int, model: str) -> ClosedLoopResult:
    current = case.scenario.world.current_field
    current_magnitude = float(
        np.linalg.norm(current.sample(case.scenario.start.position_ned_m, 0.0))
    )
    guidance = ThreeDAlosGuidance(
        3.0,
        2.0,
        1.5,
        current_field=current,
        compensate_current=current_magnitude > 1e-9,
    )
    controller = Remus100TrackingController(case.scenario.vehicle, current_field=current)
    if model == "pvs":
        backend: SimulatorBackend = PvsRemus100Backend(sample_time_s=0.02, allow_sheared_current=True)
        close_backend = False
    elif model == "mss":
        backend = MssRemus100Backend(
            sample_time_s=0.02,
            allow_sheared_current=True,
            integrator="euler",
        )
        close_backend = True
    else:
        raise ValueError("model must be 'pvs' or 'mss'")
    try:
        return run_closed_loop(
            f"{case.name}__{model}",
            case.scenario,
            case.trajectory,
            guidance,
            controller,
            backend,
            max_duration_s=case.max_duration_s,
            seed=seed,
        )
    finally:
        if close_backend:
            backend.close()  # type: ignore[attr-defined]


def _state(
    *,
    velocity: tuple[float, float, float] = (1.5, 0.0, 0.0),
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> State:
    return State(
        t_s=0.0,
        position_ned_m=np.asarray(position, dtype=np.float64),
        quat_xyzw=quaternion_from_euler(*euler),
        linear_velocity_body_mps=np.asarray(velocity, dtype=np.float64),
        angular_velocity_body_rps=np.asarray(angular_velocity, dtype=np.float64),
    )


def _command(
    *,
    stamp_s: float = 0.0,
    rudder_rad: float = 0.0,
    stern_plane_rad: float = 0.0,
    rpm: float = 1000.0,
) -> Remus100ActuatorCommand:
    return Remus100ActuatorCommand(
        stamp_s=stamp_s,
        rudder_angle_rad=rudder_rad,
        stern_plane_angle_rad=stern_plane_rad,
        propeller_rpm=rpm,
    )


def _open_loop_scenario(name: str, state: State, current: CurrentField) -> ScenarioConfig:
    return ScenarioConfig(
        name=name,
        world=WorldSnapshot(
            t_s=state.t_s,
            obstacle_map=ObstacleMap(
                bounds_ned_m=np.array([[-100.0, 100.0], [-100.0, 100.0], [-100.0, 100.0]])
            ),
            current_field=current,
        ),
        vehicle=remus100_vehicle(),
        start=state,
        goal_ned_m=np.array([100.0, 0.0, 0.0]),
        goal_radius_m=0.5,
    )


def _closed_loop_scenario(current: CurrentField) -> ScenarioConfig:
    return ScenarioConfig(
        name="m5_depth_shear_tracking",
        world=WorldSnapshot(
            t_s=0.0,
            obstacle_map=ObstacleMap(
                bounds_ned_m=np.array([[-100.0, 100.0], [-100.0, 100.0], [-20.0, 20.0]])
            ),
            current_field=current,
        ),
        vehicle=remus100_vehicle(),
        start=State(
            t_s=0.0,
            position_ned_m=np.array([-5.0, 0.0, 0.0]),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            linear_velocity_body_mps=np.array([1.0, 0.0, 0.0]),
            angular_velocity_body_rps=np.zeros(3),
        ),
        goal_ned_m=np.array([5.0, 0.0, -1.5]),
        goal_radius_m=0.5,
        constraints={"controller_sample_time_s": 0.02},
    )


def _trajectory(points_ned_m: FloatArray) -> Trajectory:
    points = np.asarray(points_ned_m, dtype=np.float64)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    time_s = np.concatenate(([0.0], np.cumsum(lengths))) / 1.5
    rows = points.shape[0]
    return Trajectory(
        trajectory_id="m5-depth-shear",
        t_s=time_s,
        position_ned_m=points,
        quat_xyzw=np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (rows, 1)),
        linear_velocity_body_mps=np.tile(np.array([1.5, 0.0, 0.0]), (rows, 1)),
        angular_velocity_body_rps=np.zeros((rows, 3), dtype=np.float64),
        feasibility="unchecked",
    )


def _remus_actuator_state(snapshot: SimulationSnapshot) -> Remus100ActuatorCommand:
    command = snapshot.actuator_state
    if not isinstance(command, Remus100ActuatorCommand):
        raise TypeError("REMUS validation backends must report Remus100ActuatorCommand")
    return command


def _channel_values(trace: OpenLoopTrace, channel: str) -> FloatArray:
    velocities = trace.velocities()
    angular_velocities = trace.angular_velocities()
    positions = trace.positions()
    euler = trace.euler_angles()
    channels = {
        "surge_mps": velocities[:, 0],
        "sway_mps": velocities[:, 1],
        "heave_mps": velocities[:, 2],
        "yaw_rate_rps": angular_velocities[:, 2],
        "pitch_rate_rps": angular_velocities[:, 1],
        "depth_m": positions[:, 2],
        "yaw_rad": euler[:, 2],
    }
    try:
        return np.asarray(channels[channel], dtype=np.float64)
    except KeyError as error:
        raise ValueError(f"unknown response channel: {channel}") from error


def _response_time_90(times_s: FloatArray, values: FloatArray) -> float:
    change = float(values[-1] - values[0])
    if abs(change) <= 1e-12:
        return float("nan")
    indices = np.nonzero(np.abs(values - values[-1]) <= 0.1 * abs(change))[0]
    return float(times_s[int(indices[0])]) if indices.size else float("nan")


def _actuator_limits() -> tuple[float, float, float]:
    vehicle = remus100_vehicle()
    assert vehicle.max_rudder_angle_rad is not None
    assert vehicle.max_stern_plane_angle_rad is not None
    assert vehicle.max_propeller_rpm is not None
    return (
        vehicle.max_rudder_angle_rad,
        vehicle.max_stern_plane_angle_rad,
        vehicle.max_propeller_rpm,
    )


def _constraint_violations(actual: FloatArray, limits: tuple[float, float, float]) -> int:
    tolerance = 1e-9
    return int(
        np.count_nonzero(np.abs(actual[:, 0]) > limits[0] + tolerance)
        + np.count_nonzero(np.abs(actual[:, 1]) > limits[1] + tolerance)
        + np.count_nonzero((actual[:, 2] < -tolerance) | (actual[:, 2] > limits[2] + tolerance))
    )


def _trace_payload(trace: OpenLoopTrace) -> dict[str, object]:
    return {
        "times_s": trace.times_s.tolist(),
        "positions_ned_m": trace.positions().tolist(),
        "linear_velocity_body_mps": trace.velocities().tolist(),
        "angular_velocity_body_rps": trace.angular_velocities().tolist(),
        "actual_commands": [
            {
                "stamp_s": command.stamp_s,
                "rudder_angle_rad": command.rudder_angle_rad,
                "stern_plane_angle_rad": command.stern_plane_angle_rad,
                "propeller_rpm": command.propeller_rpm,
            }
            for command in trace.actual_commands
        ],
    }


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_traces(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_closed_loop_logs(path: Path, results: dict[str, ClosedLoopResult]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name, result in results.items():
        (path / f"{name}.json").write_text(
            json.dumps(result.log_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _write_report(
    path: Path,
    derivative_rows: Sequence[dict[str, Any]],
    mass_rows: Sequence[dict[str, Any]],
    open_loop_rows: Sequence[dict[str, Any]],
    closed_loop_rows: Sequence[dict[str, Any]],
    convergence_rows: Sequence[dict[str, Any]],
) -> None:
    derivative_by_case = {str(row["case"]): row for row in derivative_rows}
    calm_max = max(
        _row_float(derivative_by_case[name], "relative_l2")
        for name in ("calm_straight", "calm_rudder", "calm_stern_plane", "combined_attitude")
    )
    depth_shear_max = _row_float(derivative_by_case["depth_shear"], "relative_l2")
    lines = [
        "# M5-B REMUS 100 模型可信性与场景冻结",
        "",
        "本报告比较未修改的 PVS REMUS 100 与 MSS 2.0 REMUS 100。两边统一 15° 舵限，"
        "保留各自 native 质量与密度；轨迹对照把 MSS 积分器切到与 PVS 相同的 Euler，"
        "另以 RK4/Euler 步长收敛检查量化积分差异。",
        "",
        "## 已冻结口径",
        "",
        "- 论文物理标称质量：31.9 kg。",
        "- PVS native 派生质量约 31.0294 kg；MSS native 派生质量约 31.9316 kg；模型对照不强行归一化。",
        "- 公共舵角上限：15°；MSS 的 20° 只是供应商软件默认，不进入本项目控制口径。",
        "- M5 主对照最高航速不超过 2.5 m/s；项目 `VehicleSpec` 已改为 31.9 kg 和 2.5 m/s。",
        "",
        "## 同状态导数对照",
        "",
        "| 场景 | 相对 L2 | 最大绝对差 | PVS u_dot | MSS u_dot | PVS r_dot | MSS r_dot |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in derivative_rows:
        lines.append(
            "| {case} | {relative_l2:.3%} | {max_abs_error:.6g} | {pu:.6g} | {mu:.6g} | {pr:.6g} | {mr:.6g} |".format(
                case=row["case"],
                relative_l2=float(row["relative_l2"]),
                max_abs_error=float(row["max_abs_error"]),
                pu=float(row["pvs_u_dot"]),
                mu=float(row["mss_u_dot"]),
                pr=float(row["pvs_r_dot"]),
                mr=float(row["mss_r_dot"]),
            )
        )
    lines.extend(
        [
            "",
            "## 质量与密度",
            "",
            "两套模型都由密度和椭球几何派生质量，因此仅把密度从 1025 改到 1026、并让质量和浮力同步缩放时，"
            "一阶上不会改变加速度。真正可观测的差异主要来自 MSS 将椭球半轴同比放大 1.0096 倍。"
            "下表保留两侧 native 质量矩阵对角线，供后续敏感性分析使用。",
            "",
            "| 模型 | 密度 kg/m3 | 派生质量 kg | M11 | M22 | M33 | M44 | M55 | M66 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in mass_rows:
        if row["model"] == "mass_scaling":
            continue
        lines.append(
            "| {model} | {rho:.1f} | {mass:.4f} | {m11:.6g} | {m22:.6g} | {m33:.6g} | {m44:.6g} | {m55:.6g} | {m66:.6g} |".format(
                model=row["model"],
                rho=float(row["native_density_kg_m3"]),
                mass=float(row["derived_mass_kg"]),
                m11=float(row["mass_matrix_surge"]),
                m22=float(row["mass_matrix_sway"]),
                m33=float(row["mass_matrix_heave"]),
                m44=float(row["mass_matrix_roll"]),
                m55=float(row["mass_matrix_pitch"]),
                m66=float(row["mass_matrix_yaw"]),
            )
        )
    lines.extend(
        [
            "",
            "## 开环操纵对照",
            "",
            "| 场景 | 响应通道 | 90% 响应时间 PVS/MSS s | 峰值 PVS/MSS | 稳态值 PVS/MSS |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in open_loop_rows:
        lines.append(
            "| {case} | {channel} | {ptr:.3f}/{mtr:.3f} | {ppeak:.5g}/{mpeak:.5g} | {pfinal:.5g}/{mfinal:.5g} |".format(
                case=row["case"],
                channel=row["response_channel"],
                ptr=float(row["pvs_response_time_90_s"]),
                mtr=float(row["mss_response_time_90_s"]),
                ppeak=float(row["pvs_response_peak_abs"]),
                mpeak=float(row["mss_response_peak_abs"]),
                pfinal=float(row["pvs_response_final_value"]),
                mfinal=float(row["mss_response_final_value"]),
            )
        )
    lines.extend(
        [
            "",
            "| 场景 | 轨迹 RMS m | 末端位置差 m | 归一化末端差 | 末端 surge 差 m/s | 约束违反 PVS/MSS |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in open_loop_rows:
        lines.append(
            "| {case} | {rms:.4f} | {final:.4f} | {norm:.3%} | {surge:.4f} | {pv}/{mv} |".format(
                case=row["case"],
                rms=float(row["trajectory_rms_position_error_m"]),
                final=float(row["trajectory_final_position_error_m"]),
                norm=float(row["normalized_final_position_error"]),
                surge=float(row["final_surge_error_mps"]),
                pv=int(float(row["pvs_constraint_violations"])),
                mv=int(float(row["mss_constraint_violations"])),
            )
        )
    lines.extend(
        [
            "",
            "## 闭环跟踪对照",
            "",
            "| 场景 | PVS/MSS 完成 | 完成时间差 s | 横向 RMSE PVS/MSS m | 垂向 RMSE PVS/MSS m | 共同样本轨迹 RMS m | 约束违反 PVS/MSS |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in closed_loop_rows:
        lines.append(
            "| {case} | {pc}/{mc} | {dt:.3f} | {ph:.4f}/{mh:.4f} | {pv:.4f}/{mv:.4f} | {rms:.4f} | {pvl}/{mvl} |".format(
                case=row["case"],
                pc="是" if row["pvs_completed"] else "否",
                mc="是" if row["mss_completed"] else "否",
                dt=float(row["completion_time_difference_s"]),
                ph=float(row["pvs_horizontal_rmse_m"]),
                mh=float(row["mss_horizontal_rmse_m"]),
                pv=float(row["pvs_vertical_rmse_m"]),
                mv=float(row["mss_vertical_rmse_m"]),
                rms=float(row["common_sample_trajectory_rms_m"]),
                pvl=int(float(row["pvs_constraint_violations"])),
                mvl=int(float(row["mss_constraint_violations"])),
            )
        )
    lines.extend(
        [
            "",
            "## 步长收敛",
            "",
            "误差以 0.0025 s 的结果为细网格参考。Euler 模型在 0.02 s 的归一化末端误差用于判断当前项目步长是否足够；"
            "RK4 行提供 MSS native 积分的对照。",
            "",
            "| 场景 | 模型 | dt s | 归一化位置误差 | surge 误差 m/s | 误差下降比 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in convergence_rows:
        ratio = float(row["observed_error_ratio_to_next_finer"])
        ratio_text = "—" if not np.isfinite(ratio) else f"{ratio:.2f}"
        lines.append(
            "| {case} | {model} | {dt:.4f} | {err:.4%} | {surge:.6g} | {ratio} |".format(
                case=row["case"],
                model=row["model"],
                dt=float(row["sample_time_s"]),
                err=float(row["normalized_final_position_error"]),
                surge=float(row["final_surge_error_vs_finest_mps"]),
                ratio=ratio_text,
            )
        )
    lines.extend(
        [
            "",
            "## 已冻结模型验收阈值（2026-10-25）",
            "",
            "- 坐标、单位、状态顺序和舵面符号：轴与符号一致；数值零分量保持在 1e-8 内。",
            f"- 平静水动力导数：PVS/MSS 相对 L2 误差 ≤ 7%（本次最大 {calm_max:.3%}）。",
            f"- 深度剪切流导数：采用同一 7% 门槛并单独报告（本次最大 {depth_shear_max:.3%}）。",
            "- 横向均匀流导数：允许 ≤ 35%，因为两侧横流模型分别为 Hoerner 与 DNV cylinder。",
            "- 开环末端轨迹：归一化位置差 ≤ 2%。",
            "- 闭环：完成时间差 ≤ 0.5 s，共同样本轨迹 RMS ≤ 0.25 m，约束违反必须为 0。",
            "- Euler 在 0.02 s 的归一化末端位置收敛误差 ≤ 0.5%。",
            "",
            "这些阈值只用于判断模型和接口是否足以支撑规划比较，不是论文中的算法成功率阈值。",
            "",
            "## 未解释差异",
            "",
            "- MSS 缩放椭球几何、DNV 横流模型与 PVS Hoerner 模型导致横向流和复合机动存在系统差异。",
            "- PVS 的 surge/sway 高速阻尼同时衰减；MSS 只衰减 surge，因此高航速转向不宜期待逐点重合。",
            "- MSS 的供应商 20° 默认限幅已由桥接层统一改为 15°；20° 仅保留为软件默认记录。",
            "",
            "## 复现",
            "",
            "```bash",
            "cd '/home/bubble/水下机器人/settled_全局规划器'",
            "../.venv/bin/python scripts/run_m5_model_validation.py",
            "../.venv/bin/pytest -q",
            "../.venv/bin/mypy",
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "ClosedLoopCase",
    "OpenLoopCase",
    "OpenLoopTrace",
    "build_threshold_audit",
    "closed_loop_cases",
    "compare_traces",
    "derivative_cases",
    "model_trace_metrics",
    "open_loop_cases",
    "proposed_model_thresholds",
    "run_closed_loop_comparison",
    "run_derivative_comparison",
    "run_m5_validation",
    "run_mass_density_sensitivity",
    "run_open_loop_comparison",
    "run_open_loop_trace",
    "run_step_convergence",
]
