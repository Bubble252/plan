from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
from numpy.typing import NDArray

from ..attitude import forward_quaternion, quaternion_angle
from ..contracts import FloatArray, ObstacleMap, PlanRequest, Trajectory, VehicleSpec
from ..geometry import (
    HullSpec,
    SweepResult,
    hull_clearance_details_from_offsets,
    hull_from_vehicle,
    rotated_hull_offsets,
    trajectory_sweep,
)


def _point(value: object, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _unit_direction(direction: FloatArray) -> FloatArray | None:
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return None
    return direction / norm


@dataclass(frozen=True)
class SegmentValidator:
    obstacle_map: ObstacleMap
    vehicle: VehicleSpec
    max_step_m: float = 0.05
    hull: HullSpec | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.obstacle_map, ObstacleMap):
            raise TypeError("obstacle_map must be an ObstacleMap")
        if not isinstance(self.vehicle, VehicleSpec):
            raise TypeError("vehicle must be a VehicleSpec")
        if not np.isfinite(self.max_step_m) or self.max_step_m <= 0.0:
            raise ValueError("max_step_m must be finite and positive")
        if self.hull is None:
            object.__setattr__(self, "hull", hull_from_vehicle(self.vehicle))
        elif not isinstance(self.hull, HullSpec):
            raise TypeError("hull must be a HullSpec")

    def state_is_safe(self, position_ned_m: FloatArray, quat_xyzw: FloatArray) -> bool:
        assert self.hull is not None
        details = hull_clearance_details_from_offsets(
            self.obstacle_map,
            _point(position_ned_m, "position_ned_m"),
            rotated_hull_offsets(self.hull, quat_xyzw),
            self.hull.sphere_radius_m,
        )
        return details.is_safe

    def __call__(self, start_ned_m: FloatArray, end_ned_m: FloatArray) -> bool:
        start = _point(start_ned_m, "start_ned_m")
        end = _point(end_ned_m, "end_ned_m")
        direction = _unit_direction(end - start)
        if direction is None:
            return self.state_is_safe(start, np.array([0.0, 0.0, 0.0, 1.0]))
        quat = forward_quaternion(direction)
        distance = float(np.linalg.norm(end - start))
        samples = max(1, int(ceil(distance / self.max_step_m)))
        assert self.hull is not None
        rotated_offsets = rotated_hull_offsets(self.hull, quat)
        for index in range(samples + 1):
            fraction = index / samples
            position = start + fraction * (end - start)
            details = hull_clearance_details_from_offsets(
                self.obstacle_map,
                position,
                rotated_offsets,
                self.hull.sphere_radius_m,
            )
            if not details.is_safe:
                return False
        return True


def _clean_waypoints(waypoints: NDArray[np.float64]) -> NDArray[np.float64]:
    points = np.asarray(waypoints, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 1:
        raise ValueError("waypoints must have shape (N, 3) with N >= 1")
    if not np.all(np.isfinite(points)):
        raise ValueError("waypoints must contain only finite values")
    cleaned = [points[0]]
    for point in points[1:]:
        if float(np.linalg.norm(point - cleaned[-1])) > 1e-12:
            cleaned.append(point)
    return np.asarray(cleaned, dtype=np.float64)


def shortcut_waypoints(
    waypoints: NDArray[np.float64],
    validator: SegmentValidator,
) -> NDArray[np.float64]:
    """Deterministically keep the farthest fully-collision-free shortcut."""

    points = _clean_waypoints(waypoints)
    if points.shape[0] <= 2:
        return points
    kept = [0]
    current = 0
    last = points.shape[0] - 1
    while current < last:
        candidate = last
        while candidate > current + 1 and not validator(points[current], points[candidate]):
            candidate -= 1
        kept.append(candidate)
        current = candidate
    return points[kept]


def _rotation_duration(quat_a: FloatArray, quat_b: FloatArray, vehicle: VehicleSpec) -> float:
    angle = quaternion_angle(quat_a, quat_b)
    rate = min(vehicle.max_yaw_rate_rps, vehicle.max_pitch_rate_rps)
    return max(angle / max(rate, 1e-6), 0.05)


def build_trajectory(
    waypoints: NDArray[np.float64],
    request: PlanRequest,
    trajectory_id: str,
) -> Trajectory:
    points = _clean_waypoints(waypoints)
    speed = max(request.vehicle.min_speed_mps, 1e-6)
    positions: list[FloatArray] = []
    quaternions: list[FloatArray] = []
    times: list[float] = []
    linear_velocities: list[FloatArray] = []
    angular_velocities: list[FloatArray] = []

    def append(position: FloatArray, quat: FloatArray, time_s: float, velocity: FloatArray) -> None:
        if times and time_s <= times[-1]:
            time_s = times[-1] + 0.05
        positions.append(np.asarray(position, dtype=np.float64).copy())
        quaternions.append(np.asarray(quat, dtype=np.float64).copy())
        times.append(float(time_s))
        linear_velocities.append(np.asarray(velocity, dtype=np.float64).copy())
        angular_velocities.append(np.zeros(3, dtype=np.float64))

    time_s = 0.0
    current_quat = np.asarray(request.start.quat_xyzw, dtype=np.float64).copy()
    append(points[0], current_quat, time_s, request.start.linear_velocity_body_mps)
    if points.shape[0] == 1:
        append(points[0], current_quat, time_s + 0.1, np.zeros(3))
    for index in range(points.shape[0] - 1):
        direction = points[index + 1] - points[index]
        segment_quat = forward_quaternion(direction)
        if quaternion_angle(current_quat, segment_quat) > 1e-9:
            time_s += _rotation_duration(current_quat, segment_quat, request.vehicle)
            append(points[index], segment_quat, time_s, np.zeros(3))
            current_quat = segment_quat
        length = float(np.linalg.norm(direction))
        time_s += length / speed
        append(points[index + 1], segment_quat, time_s, np.array([speed, 0.0, 0.0]))

    return Trajectory(
        trajectory_id=trajectory_id,
        t_s=np.asarray(times, dtype=np.float64),
        position_ned_m=np.vstack(positions),
        quat_xyzw=np.vstack(quaternions),
        linear_velocity_body_mps=np.vstack(linear_velocities),
        angular_velocity_body_rps=np.vstack(angular_velocities),
        feasibility="unchecked",
    )


@dataclass(frozen=True)
class FinalizedPath:
    trajectory: Trajectory
    path_length_m: float
    sweep: SweepResult


def finalize_path(
    waypoints: NDArray[np.float64],
    request: PlanRequest,
    validator: SegmentValidator,
    *,
    trajectory_id: str,
    max_rotation_step_rad: float = 0.05,
) -> FinalizedPath:
    simplified = shortcut_waypoints(waypoints, validator)
    trajectory = build_trajectory(simplified, request, trajectory_id)
    sweep = trajectory_sweep(
        trajectory,
        request.world.obstacle_map,
        request.vehicle,
        max_translation_step_m=validator.max_step_m,
        max_rotation_step_rad=max_rotation_step_rad,
    )
    path_length = float(np.sum(np.linalg.norm(np.diff(trajectory.position_ned_m, axis=0), axis=1)))
    return FinalizedPath(trajectory=trajectory, path_length_m=path_length, sweep=sweep)


__all__ = [
    "FinalizedPath",
    "SegmentValidator",
    "build_trajectory",
    "finalize_path",
    "forward_quaternion",
    "quaternion_angle",
    "shortcut_waypoints",
]
