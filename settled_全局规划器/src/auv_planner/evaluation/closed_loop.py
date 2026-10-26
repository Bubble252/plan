from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import hypot
from pathlib import Path
from typing import Protocol

import numpy as np

from ..contracts import (
    FloatArray,
    Reference,
    Remus100ActuatorCommand,
    ScenarioConfig,
    SimulationSnapshot,
    State,
    Trajectory,
)
from ..interfaces import Guidance


class Remus100Controller(Protocol):
    def control(self, state: State, reference: Reference) -> Remus100ActuatorCommand: ...


class Remus100Backend(Protocol):
    def reset(self, scenario: ScenarioConfig, seed: int) -> SimulationSnapshot: ...

    def step(self, command: Remus100ActuatorCommand) -> SimulationSnapshot: ...


@dataclass(frozen=True)
class PathProjection:
    segment_index: int
    progress_m: float
    horizontal_error_m: float
    vertical_error_m: float


@dataclass(frozen=True)
class ClosedLoopSample:
    t_s: float
    state: State
    reference: Reference
    command: Remus100ActuatorCommand
    path_progress_m: float
    horizontal_error_m: float
    vertical_error_m: float

    def log_dict(self) -> dict[str, object]:
        return {
            "t_s": self.t_s,
            "state": _state_log_dict(self.state),
            "reference": _reference_log_dict(self.reference),
            "command": _command_log_dict(self.command),
            "path_progress_m": self.path_progress_m,
            "horizontal_error_m": self.horizontal_error_m,
            "vertical_error_m": self.vertical_error_m,
        }


@dataclass(frozen=True)
class ClosedLoopMetrics:
    completed: bool
    completion_time_s: float
    sample_count: int
    goal_distance_m: float
    horizontal_rmse_m: float
    vertical_rmse_m: float
    horizontal_max_abs_m: float
    vertical_max_abs_m: float
    terminal_horizontal_error_m: float
    terminal_vertical_error_m: float
    mean_speed_mps: float
    max_abs_rudder_angle_rad: float
    mean_abs_rudder_angle_rad: float
    max_abs_rudder_rate_rad_s: float
    rudder_saturation_fraction: float
    max_abs_stern_plane_angle_rad: float
    mean_abs_stern_plane_angle_rad: float
    max_abs_stern_plane_rate_rad_s: float
    stern_plane_saturation_fraction: float
    max_propeller_rpm: float
    mean_propeller_rpm: float
    propeller_saturation_fraction: float
    constraint_violations: int
    failure_reason: str | None

    def summary_dict(self) -> dict[str, object]:
        return {field_name: getattr(self, field_name) for field_name in self.__dataclass_fields__}


@dataclass(frozen=True)
class ClosedLoopResult:
    case_name: str
    initial_state: State
    metrics: ClosedLoopMetrics
    samples: tuple[ClosedLoopSample, ...]

    def log_dict(self) -> dict[str, object]:
        return {
            "case_name": self.case_name,
            "initial_state": _state_log_dict(self.initial_state),
            "metrics": self.metrics.summary_dict(),
            "samples": [sample.log_dict() for sample in self.samples],
        }


def project_onto_path(position_ned_m: FloatArray, points_ned_m: FloatArray) -> PathProjection:
    position = np.asarray(position_ned_m, dtype=np.float64)
    points = np.asarray(points_ned_m, dtype=np.float64)
    if position.shape != (3,):
        raise ValueError("position_ned_m must have shape (3,)")
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 3:
        raise ValueError("points_ned_m must have shape (N, 3) with N >= 2")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(points)):
        raise ValueError("positions must contain only finite values")

    segments = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)
    if np.any(segment_lengths <= 1e-12):
        raise ValueError("path segments must have positive length")
    tangents = segments / segment_lengths[:, None]
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))

    best_index = 0
    best_distance_sq = float("inf")
    best_along_m = 0.0
    best_projection = points[0]
    for index, (start, tangent, length) in enumerate(
        zip(points[:-1], tangents, segment_lengths, strict=True)
    ):
        along_m = float(np.clip(np.dot(position - start, tangent), 0.0, length))
        projection = start + along_m * tangent
        distance_sq = float(np.dot(position - projection, position - projection))
        if distance_sq < best_distance_sq - 1e-12:
            best_index = index
            best_distance_sq = distance_sq
            best_along_m = along_m
            best_projection = projection

    tangent = tangents[best_index]
    horizontal_norm = float(hypot(float(tangent[0]), float(tangent[1])))
    if horizontal_norm > 1e-12:
        lateral_axis = np.array(
            [-tangent[1] / horizontal_norm, tangent[0] / horizontal_norm, 0.0],
            dtype=np.float64,
        )
    else:
        lateral_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    vertical_axis = np.cross(tangent, lateral_axis)
    vertical_axis /= np.linalg.norm(vertical_axis)
    offset = position - best_projection
    return PathProjection(
        segment_index=best_index,
        progress_m=float(cumulative[best_index] + best_along_m),
        horizontal_error_m=float(np.dot(offset, lateral_axis)),
        vertical_error_m=float(np.dot(offset, vertical_axis)),
    )


def run_closed_loop(
    case_name: str,
    scenario: ScenarioConfig,
    trajectory: Trajectory,
    guidance: Guidance,
    controller: Remus100Controller,
    backend: Remus100Backend,
    *,
    max_duration_s: float,
    seed: int = 0,
    saturation_fraction_threshold: float = 0.99,
) -> ClosedLoopResult:
    if not case_name:
        raise ValueError("case_name must not be empty")
    if not isinstance(scenario, ScenarioConfig):
        raise TypeError("scenario must be a ScenarioConfig")
    if not isinstance(trajectory, Trajectory):
        raise TypeError("trajectory must be a Trajectory")
    if not np.isfinite(max_duration_s) or max_duration_s <= 0.0:
        raise ValueError("max_duration_s must be finite and positive")
    if not 0.0 < saturation_fraction_threshold <= 1.0:
        raise ValueError("saturation_fraction_threshold must be in (0, 1]")

    snapshot = backend.reset(scenario, seed)
    initial_state = snapshot.state
    end_time_s = initial_state.t_s + float(max_duration_s)
    path_points = np.asarray(trajectory.position_ned_m, dtype=np.float64)
    samples: list[ClosedLoopSample] = []
    completed = False

    while snapshot.state.t_s < end_time_s - 1e-12:
        reference = guidance.guide(snapshot.state, trajectory, snapshot.state.t_s)
        command = controller.control(snapshot.state, reference)
        if not isinstance(command, Remus100ActuatorCommand):
            raise TypeError("controller must return a Remus100ActuatorCommand")
        if abs(command.stamp_s - snapshot.state.t_s) > 1e-9:
            raise ValueError("command timestamp must match the current state timestamp")

        next_snapshot = backend.step(command)
        if next_snapshot.state.t_s <= snapshot.state.t_s:
            raise ValueError("backend state time must increase after step()")
        projection = project_onto_path(next_snapshot.state.position_ned_m, path_points)
        samples.append(
            ClosedLoopSample(
                t_s=next_snapshot.state.t_s,
                state=next_snapshot.state,
                reference=reference,
                command=command,
                path_progress_m=projection.progress_m,
                horizontal_error_m=projection.horizontal_error_m,
                vertical_error_m=projection.vertical_error_m,
            )
        )

        snapshot = next_snapshot
        goal_distance = float(np.linalg.norm(snapshot.state.position_ned_m - scenario.goal_ned_m))
        if goal_distance <= scenario.goal_radius_m:
            completed = True
            break

    metrics = _compute_metrics(
        scenario=scenario,
        initial_state=initial_state,
        final_state=snapshot.state,
        samples=samples,
        completed=completed,
        saturation_fraction_threshold=saturation_fraction_threshold,
    )
    return ClosedLoopResult(
        case_name=case_name,
        initial_state=initial_state,
        metrics=metrics,
        samples=tuple(samples),
    )


def write_closed_loop_log(result: ClosedLoopResult, path: Path | str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result.log_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def write_closed_loop_summary_csv(results: Iterable[ClosedLoopResult], path: Path | str) -> Path:
    rows = list(results)
    if not rows:
        raise ValueError("results must not be empty")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case_name", *rows[0].metrics.summary_dict().keys()]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in rows:
            writer.writerow({"case_name": result.case_name, **result.metrics.summary_dict()})
    return output_path


def _compute_metrics(
    *,
    scenario: ScenarioConfig,
    initial_state: State,
    final_state: State,
    samples: Sequence[ClosedLoopSample],
    completed: bool,
    saturation_fraction_threshold: float,
) -> ClosedLoopMetrics:
    if not samples:
        raise ValueError("closed loop must produce at least one sample")
    horizontal = np.array([sample.horizontal_error_m for sample in samples], dtype=np.float64)
    vertical = np.array([sample.vertical_error_m for sample in samples], dtype=np.float64)
    times = np.array([sample.t_s for sample in samples], dtype=np.float64)
    commands = [sample.command for sample in samples]
    rudder = np.array([command.rudder_angle_rad for command in commands], dtype=np.float64)
    stern_plane = np.array(
        [command.stern_plane_angle_rad for command in commands], dtype=np.float64
    )
    rpm = np.array([command.propeller_rpm for command in commands], dtype=np.float64)
    reference_speed = np.array(
        [sample.reference.linear_velocity_body_mps[0] for sample in samples], dtype=np.float64
    )
    speeds = np.array(
        [sample.state.linear_velocity_body_mps[0] for sample in samples], dtype=np.float64
    )

    dt = np.diff(np.concatenate(([initial_state.t_s], times)))
    rudder_rate = np.divide(np.diff(np.concatenate(([0.0], rudder))), dt)
    stern_plane_rate = np.divide(np.diff(np.concatenate(([0.0], stern_plane))), dt)
    angle_limit = float(scenario.vehicle.max_rudder_angle_rad or 0.0)
    stern_limit = float(scenario.vehicle.max_stern_plane_angle_rad or 0.0)
    rpm_limit = float(scenario.vehicle.max_propeller_rpm or 0.0)
    rate_limit = float(scenario.vehicle.max_rudder_rate_rad_s or 0.0)
    stern_rate_limit = float(scenario.vehicle.max_stern_plane_rate_rad_s or 0.0)
    tolerance = 1e-9

    violations = 0
    violations += int(np.count_nonzero(np.abs(rudder) > angle_limit + tolerance))
    violations += int(np.count_nonzero(np.abs(stern_plane) > stern_limit + tolerance))
    violations += int(np.count_nonzero((rpm < -tolerance) | (rpm > rpm_limit + tolerance)))
    violations += int(np.count_nonzero(np.abs(rudder_rate) > rate_limit + tolerance))
    violations += int(np.count_nonzero(np.abs(stern_plane_rate) > stern_rate_limit + tolerance))
    violations += int(
        np.count_nonzero(
            (reference_speed < scenario.vehicle.min_speed_mps - tolerance)
            | (reference_speed > scenario.vehicle.max_speed_mps + tolerance)
        )
    )

    goal_distance = float(np.linalg.norm(final_state.position_ned_m - scenario.goal_ned_m))
    failure_reason = None
    if not completed:
        failure_reason = "timeout"
    if violations:
        failure_reason = (
            "constraint_violation" if failure_reason is None else "timeout,constraint_violation"
        )

    return ClosedLoopMetrics(
        completed=completed,
        completion_time_s=float(final_state.t_s - initial_state.t_s),
        sample_count=len(samples),
        goal_distance_m=goal_distance,
        horizontal_rmse_m=float(np.sqrt(np.mean(np.square(horizontal)))),
        vertical_rmse_m=float(np.sqrt(np.mean(np.square(vertical)))),
        horizontal_max_abs_m=float(np.max(np.abs(horizontal))),
        vertical_max_abs_m=float(np.max(np.abs(vertical))),
        terminal_horizontal_error_m=float(horizontal[-1]),
        terminal_vertical_error_m=float(vertical[-1]),
        mean_speed_mps=float(np.mean(speeds)),
        max_abs_rudder_angle_rad=float(np.max(np.abs(rudder))),
        mean_abs_rudder_angle_rad=float(np.mean(np.abs(rudder))),
        max_abs_rudder_rate_rad_s=float(np.max(np.abs(rudder_rate))),
        rudder_saturation_fraction=float(
            np.mean(np.abs(rudder) >= saturation_fraction_threshold * angle_limit)
        ),
        max_abs_stern_plane_angle_rad=float(np.max(np.abs(stern_plane))),
        mean_abs_stern_plane_angle_rad=float(np.mean(np.abs(stern_plane))),
        max_abs_stern_plane_rate_rad_s=float(np.max(np.abs(stern_plane_rate))),
        stern_plane_saturation_fraction=float(
            np.mean(np.abs(stern_plane) >= saturation_fraction_threshold * stern_limit)
        ),
        max_propeller_rpm=float(np.max(rpm)),
        mean_propeller_rpm=float(np.mean(rpm)),
        propeller_saturation_fraction=float(
            np.mean(rpm >= saturation_fraction_threshold * rpm_limit)
        ),
        constraint_violations=violations,
        failure_reason=failure_reason,
    )


def _state_log_dict(state: State) -> dict[str, object]:
    return {
        "t_s": state.t_s,
        "position_ned_m": state.position_ned_m.tolist(),
        "quat_xyzw": state.quat_xyzw.tolist(),
        "linear_velocity_body_mps": state.linear_velocity_body_mps.tolist(),
        "angular_velocity_body_rps": state.angular_velocity_body_rps.tolist(),
        "frame_id": state.frame_id,
    }


def _reference_log_dict(reference: Reference) -> dict[str, object]:
    return {
        "stamp_s": reference.stamp_s,
        "position_ned_m": reference.position_ned_m.tolist(),
        "quat_xyzw": reference.quat_xyzw.tolist(),
        "linear_velocity_body_mps": reference.linear_velocity_body_mps.tolist(),
        "angular_velocity_body_rps": reference.angular_velocity_body_rps.tolist(),
        "feedforward_accel_body_mps2": (
            None
            if reference.feedforward_accel_body_mps2 is None
            else reference.feedforward_accel_body_mps2.tolist()
        ),
        "valid_until_s": reference.valid_until_s,
    }


def _command_log_dict(command: Remus100ActuatorCommand) -> dict[str, object]:
    return {
        "stamp_s": command.stamp_s,
        "rudder_angle_rad": command.rudder_angle_rad,
        "stern_plane_angle_rad": command.stern_plane_angle_rad,
        "propeller_rpm": command.propeller_rpm,
        "valid_until_s": command.valid_until_s,
    }


__all__ = [
    "ClosedLoopMetrics",
    "ClosedLoopResult",
    "ClosedLoopSample",
    "PathProjection",
    "project_onto_path",
    "run_closed_loop",
    "write_closed_loop_log",
    "write_closed_loop_summary_csv",
]
