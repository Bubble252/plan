from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .contracts import EvalResult, ObstacleMap, Trajectory, VehicleSpec
from .geometry import trajectory_sweep


@dataclass(frozen=True)
class PathMetrics:
    path_length_m: float
    duration_s: float
    mean_speed_mps: float
    min_clearance_m: float
    min_clearance_time_s: float
    collision: bool
    out_of_bounds: bool
    failure_reason: str | None
    sampled_poses: int

    def __post_init__(self) -> None:
        for name in ("path_length_m", "duration_s", "mean_speed_mps", "min_clearance_time_s"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not np.isfinite(self.min_clearance_m):
            raise ValueError("min_clearance_m must be finite")
        if self.sampled_poses < 1:
            raise ValueError("sampled_poses must be positive")
        if self.is_safe and self.failure_reason is not None:
            raise ValueError("safe paths must not define failure_reason")
        if not self.is_safe and not self.failure_reason:
            raise ValueError("unsafe paths require failure_reason")

    @property
    def is_safe(self) -> bool:
        return not self.collision and not self.out_of_bounds

    def to_eval_result(
        self,
        *,
        success: bool,
        energy_j: float = 0.0,
        tracking_rmse: float = 0.0,
        constraint_violations: int = 0,
        recovery_success: bool | None = None,
        task_failure_reason: str | None = None,
        metrics: Mapping[str, float] | None = None,
    ) -> EvalResult:
        if success and not self.is_safe:
            raise ValueError("successful evaluations require a safe geometric path")
        reasons = [reason for reason in (self.failure_reason, task_failure_reason) if reason]
        failure_reason = ",".join(dict.fromkeys(reasons)) if reasons else None
        return EvalResult(
            success=success,
            collision=self.collision,
            out_of_bounds=self.out_of_bounds,
            min_clearance_m=self.min_clearance_m,
            path_length_m=self.path_length_m,
            duration_s=self.duration_s,
            energy_j=energy_j,
            tracking_rmse=tracking_rmse,
            constraint_violations=constraint_violations,
            recovery_success=recovery_success,
            failure_reason=failure_reason,
            metrics=metrics or {},
        )


def compute_path_metrics(
    trajectory: Trajectory,
    obstacle_map: ObstacleMap,
    vehicle: VehicleSpec,
    *,
    sphere_count: int | None = None,
    max_translation_step_m: float = 0.1,
    max_rotation_step_rad: float = 0.1,
) -> PathMetrics:
    if not isinstance(trajectory, Trajectory):
        raise TypeError("trajectory must be a Trajectory")
    if not isinstance(obstacle_map, ObstacleMap):
        raise TypeError("obstacle_map must be an ObstacleMap")
    if not isinstance(vehicle, VehicleSpec):
        raise TypeError("vehicle must be a VehicleSpec")

    path_length = float(np.sum(np.linalg.norm(np.diff(trajectory.position_ned_m, axis=0), axis=1)))
    duration = float(trajectory.t_s[-1] - trajectory.t_s[0])
    mean_speed = path_length / duration
    sweep = trajectory_sweep(
        trajectory,
        obstacle_map,
        vehicle,
        sphere_count=sphere_count,
        max_translation_step_m=max_translation_step_m,
        max_rotation_step_rad=max_rotation_step_rad,
    )
    return PathMetrics(
        path_length_m=path_length,
        duration_s=duration,
        mean_speed_mps=mean_speed,
        min_clearance_m=sweep.min_clearance_m,
        min_clearance_time_s=sweep.min_clearance_time_s,
        collision=sweep.collision,
        out_of_bounds=sweep.out_of_bounds,
        failure_reason=sweep.failure_reason,
        sampled_poses=sweep.sampled_poses,
    )


__all__ = ["PathMetrics", "compute_path_metrics"]
