from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
from numpy.typing import NDArray

from .contracts import FloatArray, Obstacle, ObstacleMap, State, Trajectory, VehicleSpec


def _point(value: object, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _points(value: object, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] < 1:
        raise ValueError(f"{name} must have shape (N, 3) with N >= 1")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _quaternion(value: object) -> FloatArray:
    quat = np.asarray(value, dtype=np.float64)
    if quat.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {quat.shape}")
    if not np.all(np.isfinite(quat)):
        raise ValueError("quaternion must contain only finite values")
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0:
        raise ValueError("quaternion must have non-zero norm")
    return quat / norm


def _slerp(quat_a: FloatArray, quat_b: FloatArray, fraction: float) -> FloatArray:
    qa = _quaternion(quat_a)
    qb = _quaternion(quat_b)
    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb = -qb
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = qa + fraction * (qb - qa)
        return result / np.linalg.norm(result)
    theta_0 = float(np.arccos(dot))
    sin_theta_0 = float(np.sin(theta_0))
    weight_a = float(np.sin((1.0 - fraction) * theta_0) / sin_theta_0)
    weight_b = float(np.sin(fraction * theta_0) / sin_theta_0)
    return weight_a * qa + weight_b * qb


@dataclass(frozen=True)
class HullSpec:
    """Multi-sphere approximation of a capsule-shaped hull aligned with body +x."""

    sphere_radius_m: float
    offsets_body_m: FloatArray

    def __post_init__(self) -> None:
        if not np.isfinite(self.sphere_radius_m) or self.sphere_radius_m <= 0.0:
            raise ValueError("sphere_radius_m must be finite and positive")
        offsets = np.asarray(self.offsets_body_m, dtype=np.float64)
        if offsets.ndim != 2 or offsets.shape[0] < 2 or offsets.shape[1] != 3:
            raise ValueError("offsets_body_m must have shape (N, 3) with N >= 2")
        if not np.all(np.isfinite(offsets)):
            raise ValueError("offsets_body_m must contain only finite values")
        if not np.allclose(offsets[:, 1:], 0.0):
            raise ValueError("hull sphere offsets must lie on the body x axis")
        object.__setattr__(self, "offsets_body_m", offsets)

    @property
    def sphere_count(self) -> int:
        return int(self.offsets_body_m.shape[0])


def hull_from_vehicle(vehicle: VehicleSpec, sphere_count: int | None = None) -> HullSpec:
    if not isinstance(vehicle, VehicleSpec):
        raise TypeError("vehicle must be a VehicleSpec")
    if sphere_count is None:
        sphere_count = max(2, int(ceil(vehicle.length_m / vehicle.diameter_m)))
    if sphere_count < 2:
        raise ValueError("hull sphere_count must be at least 2; use point clearance for a point model")
    radius = vehicle.diameter_m / 2.0
    segment_half_length = max(vehicle.length_m / 2.0 - radius, 0.0)
    offsets = np.zeros((sphere_count, 3), dtype=np.float64)
    offsets[:, 0] = np.linspace(-segment_half_length, segment_half_length, sphere_count)
    return HullSpec(sphere_radius_m=radius, offsets_body_m=offsets)


def hull_sphere_centers(state: State, hull: HullSpec) -> FloatArray:
    if not isinstance(state, State):
        raise TypeError("state must be a State")
    if not isinstance(hull, HullSpec):
        raise TypeError("hull must be a HullSpec")
    return hull_sphere_centers_from_pose(state.position_ned_m, state.quat_xyzw, hull)


def rotated_hull_offsets(hull: HullSpec, quat_xyzw: FloatArray) -> FloatArray:
    if not isinstance(hull, HullSpec):
        raise TypeError("hull must be a HullSpec")
    x, y, z, w = _quaternion(quat_xyzw)
    rotation = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return hull.offsets_body_m @ rotation.T


def hull_sphere_centers_from_pose(
    position_ned_m: FloatArray,
    quat_xyzw: FloatArray,
    hull: HullSpec,
) -> FloatArray:
    position = _point(position_ned_m, "position_ned_m")
    return rotated_hull_offsets(hull, quat_xyzw) + position.reshape(1, 3)


def obstacle_signed_distance_array(points_ned_m: FloatArray, obstacle: Obstacle) -> FloatArray:
    if not isinstance(obstacle, Obstacle):
        raise TypeError("obstacle must be an Obstacle")
    points = _points(points_ned_m, "points_ned_m")
    local = points - obstacle.center_ned_m.reshape(1, 3)
    if obstacle.kind == "sphere":
        assert obstacle.radius_m is not None
        return np.linalg.norm(local, axis=1) - obstacle.radius_m
    if obstacle.kind == "box":
        assert obstacle.half_extents_m is not None
        delta = np.abs(local) - obstacle.half_extents_m
        outside = np.linalg.norm(np.maximum(delta, 0.0), axis=1)
        inside = np.minimum(np.max(delta, axis=1), 0.0)
        return outside + inside
    assert obstacle.kind == "cylinder"
    assert obstacle.radius_m is not None and obstacle.height_m is not None
    radial = np.hypot(local[:, 0], local[:, 1]) - obstacle.radius_m
    axial = np.abs(local[:, 2]) - obstacle.height_m / 2.0
    outside = np.hypot(np.maximum(radial, 0.0), np.maximum(axial, 0.0))
    inside = np.minimum(np.maximum(radial, axial), 0.0)
    return outside + inside


def obstacle_signed_distance(point_ned_m: FloatArray, obstacle: Obstacle) -> float:
    point = _point(point_ned_m, "point_ned_m")
    return float(obstacle_signed_distance_array(point.reshape(1, 3), obstacle)[0])


def _bounds_signed_distance_array(points_ned_m: FloatArray, bounds_ned_m: FloatArray) -> FloatArray:
    points = _points(points_ned_m, "points_ned_m")
    lower = bounds_ned_m[:, 0]
    upper = bounds_ned_m[:, 1]
    inside_distance = np.minimum(points - lower, upper - points)
    all_inside = np.all(inside_distance >= 0.0, axis=1)
    outside_offset = np.maximum(lower - points, 0.0) + np.maximum(points - upper, 0.0)
    outside_distance = -np.linalg.norm(outside_offset, axis=1)
    return np.where(all_inside, np.min(inside_distance, axis=1), outside_distance)


def _bounds_signed_distance(point_ned_m: FloatArray, bounds_ned_m: FloatArray) -> float:
    point = _point(point_ned_m, "point_ned_m")
    return float(_bounds_signed_distance_array(point.reshape(1, 3), bounds_ned_m)[0])


@dataclass(frozen=True)
class ClearanceDetails:
    clearance_m: float
    obstacle_clearance_m: float
    boundary_clearance_m: float
    collision: bool
    out_of_bounds: bool

    @property
    def is_safe(self) -> bool:
        return not self.collision and not self.out_of_bounds


def _clearance_from_centers(
    centers_ned_m: FloatArray,
    radius_m: float,
    obstacle_map: ObstacleMap,
) -> ClearanceDetails:
    obstacle_clearance = float("inf")
    for obstacle in obstacle_map.obstacles:
        obstacle_clearance = min(
            obstacle_clearance,
            float(np.min(obstacle_signed_distance_array(centers_ned_m, obstacle))),
        )
    obstacle_clearance -= radius_m
    boundary_clearance = float(
        np.min(_bounds_signed_distance_array(centers_ned_m, obstacle_map.bounds_ned_m))
    ) - radius_m
    clearance = min(obstacle_clearance, boundary_clearance)
    return ClearanceDetails(
        clearance_m=float(clearance),
        obstacle_clearance_m=float(obstacle_clearance),
        boundary_clearance_m=float(boundary_clearance),
        collision=bool(obstacle_clearance < 0.0),
        out_of_bounds=bool(boundary_clearance < 0.0),
    )


def hull_clearance_details_from_offsets(
    obstacle_map: ObstacleMap,
    position_ned_m: FloatArray,
    rotated_offsets_ned_m: FloatArray,
    radius_m: float,
) -> ClearanceDetails:
    if not isinstance(obstacle_map, ObstacleMap):
        raise TypeError("obstacle_map must be an ObstacleMap")
    position = _point(position_ned_m, "position_ned_m")
    offsets = np.asarray(rotated_offsets_ned_m, dtype=np.float64)
    if offsets.ndim != 2 or offsets.shape[0] < 1 or offsets.shape[1] != 3:
        raise ValueError("rotated_offsets_ned_m must have shape (N, 3) with N >= 1")
    if not np.all(np.isfinite(offsets)):
        raise ValueError("rotated_offsets_ned_m must contain only finite values")
    if not np.isfinite(radius_m) or radius_m < 0.0:
        raise ValueError("radius_m must be finite and non-negative")
    return _clearance_from_centers(offsets + position.reshape(1, 3), radius_m, obstacle_map)


def point_clearance_details(
    obstacle_map: ObstacleMap,
    position_ned_m: FloatArray,
    radius_m: float = 0.0,
) -> ClearanceDetails:
    if not isinstance(obstacle_map, ObstacleMap):
        raise TypeError("obstacle_map must be an ObstacleMap")
    if not np.isfinite(radius_m) or radius_m < 0.0:
        raise ValueError("radius_m must be finite and non-negative")
    point = _point(position_ned_m, "position_ned_m")
    return _clearance_from_centers(point.reshape(1, 3), radius_m, obstacle_map)


def point_clearance(obstacle_map: ObstacleMap, position_ned_m: FloatArray) -> float:
    return point_clearance_details(obstacle_map, position_ned_m).clearance_m


def hull_clearance_details(obstacle_map: ObstacleMap, state: State, hull: HullSpec) -> ClearanceDetails:
    if not isinstance(obstacle_map, ObstacleMap):
        raise TypeError("obstacle_map must be an ObstacleMap")
    centers = hull_sphere_centers(state, hull)
    return _clearance_from_centers(centers, hull.sphere_radius_m, obstacle_map)


def hull_clearance(obstacle_map: ObstacleMap, state: State, hull: HullSpec) -> float:
    return hull_clearance_details(obstacle_map, state, hull).clearance_m


@dataclass(frozen=True)
class SweepResult:
    collision: bool
    out_of_bounds: bool
    min_clearance_m: float
    min_clearance_time_s: float
    min_clearance_position_ned_m: FloatArray
    min_clearance_sample_index: int
    sampled_poses: int
    failure_reason: str | None

    def __post_init__(self) -> None:
        if not np.isfinite(self.min_clearance_m):
            raise ValueError("min_clearance_m must be finite")
        if self.sampled_poses < 1:
            raise ValueError("sampled_poses must be positive")
        if not self.collision and not self.out_of_bounds and self.failure_reason is not None:
            raise ValueError("safe sweeps must not define failure_reason")
        if (self.collision or self.out_of_bounds) and not self.failure_reason:
            raise ValueError("unsafe sweeps require failure_reason")
        object.__setattr__(
            self,
            "min_clearance_position_ned_m",
            _point(self.min_clearance_position_ned_m, "min_clearance_position_ned_m"),
        )

    @property
    def is_safe(self) -> bool:
        return not self.collision and not self.out_of_bounds


def trajectory_sweep(
    trajectory: Trajectory,
    obstacle_map: ObstacleMap,
    vehicle: VehicleSpec,
    *,
    sphere_count: int | None = None,
    max_translation_step_m: float = 0.1,
    max_rotation_step_rad: float = 0.1,
) -> SweepResult:
    if not isinstance(trajectory, Trajectory):
        raise TypeError("trajectory must be a Trajectory")
    if not isinstance(obstacle_map, ObstacleMap):
        raise TypeError("obstacle_map must be an ObstacleMap")
    if not np.isfinite(max_translation_step_m) or max_translation_step_m <= 0.0:
        raise ValueError("max_translation_step_m must be finite and positive")
    if not np.isfinite(max_rotation_step_rad) or max_rotation_step_rad <= 0.0:
        raise ValueError("max_rotation_step_rad must be finite and positive")

    hull = hull_from_vehicle(vehicle, sphere_count)
    best_clearance = float("inf")
    best_time = float(trajectory.t_s[0])
    best_position = trajectory.position_ned_m[0].copy()
    best_index = 0
    collision = False
    out_of_bounds = False
    sampled_poses = 0

    def inspect(time_s: float, position: FloatArray, quat: FloatArray) -> None:
        nonlocal best_clearance, best_time, best_position, best_index
        nonlocal collision, out_of_bounds, sampled_poses
        state = State(
            t_s=float(time_s),
            position_ned_m=position,
            quat_xyzw=quat,
            linear_velocity_body_mps=np.zeros(3),
            angular_velocity_body_rps=np.zeros(3),
        )
        details = hull_clearance_details(obstacle_map, state, hull)
        collision = collision or details.collision
        out_of_bounds = out_of_bounds or details.out_of_bounds
        if details.clearance_m < best_clearance:
            best_clearance = details.clearance_m
            best_time = float(time_s)
            best_position = np.asarray(position, dtype=np.float64).copy()
            best_index = sampled_poses
        sampled_poses += 1

    inspect(trajectory.t_s[0], trajectory.position_ned_m[0], trajectory.quat_xyzw[0])
    for index in range(trajectory.t_s.shape[0] - 1):
        position_a = trajectory.position_ned_m[index]
        position_b = trajectory.position_ned_m[index + 1]
        quat_a = trajectory.quat_xyzw[index]
        quat_b = trajectory.quat_xyzw[index + 1]
        translation = float(np.linalg.norm(position_b - position_a))
        rotation = float(2.0 * np.arccos(np.clip(abs(float(np.dot(quat_a, quat_b))), -1.0, 1.0)))
        steps = max(
            1,
            int(ceil(translation / max_translation_step_m)),
            int(ceil(rotation / max_rotation_step_rad)),
        )
        for step in range(1, steps + 1):
            fraction = step / steps
            position = position_a + fraction * (position_b - position_a)
            quat = _slerp(quat_a, quat_b, fraction)
            time_s = trajectory.t_s[index] + fraction * (trajectory.t_s[index + 1] - trajectory.t_s[index])
            inspect(time_s, position, quat)

    reasons: list[str] = []
    if collision:
        reasons.append("collision")
    if out_of_bounds:
        reasons.append("out_of_bounds")
    return SweepResult(
        collision=collision,
        out_of_bounds=out_of_bounds,
        min_clearance_m=float(best_clearance),
        min_clearance_time_s=best_time,
        min_clearance_position_ned_m=best_position,
        min_clearance_sample_index=best_index,
        sampled_poses=sampled_poses,
        failure_reason=",".join(reasons) if reasons else None,
    )


@dataclass(frozen=True)
class OccupancyGrid:
    """Small dense occupancy grid; clearance uses occupied-voxel centers."""

    origin_ned_m: FloatArray
    resolution_m: float
    occupied: NDArray[np.bool_]

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin_ned_m", _point(self.origin_ned_m, "origin_ned_m"))
        if not np.isfinite(self.resolution_m) or self.resolution_m <= 0.0:
            raise ValueError("resolution_m must be finite and positive")
        occupied = np.asarray(self.occupied, dtype=bool)
        if occupied.ndim != 3 or 0 in occupied.shape:
            raise ValueError("occupied must be a non-empty 3D array")
        object.__setattr__(self, "occupied", occupied)

    @property
    def shape(self) -> tuple[int, int, int]:
        rows, cols, depths = self.occupied.shape
        return int(rows), int(cols), int(depths)

    @property
    def bounds_ned_m(self) -> FloatArray:
        upper = self.origin_ned_m + self.resolution_m * np.asarray(self.shape, dtype=np.float64)
        return np.column_stack((self.origin_ned_m, upper))

    def contains(self, position_ned_m: FloatArray) -> bool:
        point = _point(position_ned_m, "position_ned_m")
        return bool(np.all(point >= self.origin_ned_m) and np.all(point < self.bounds_ned_m[:, 1]))

    def index_of(self, position_ned_m: FloatArray) -> tuple[int, int, int]:
        point = _point(position_ned_m, "position_ned_m")
        if not self.contains(point):
            raise ValueError("position_ned_m lies outside the occupancy grid")
        index = np.floor((point - self.origin_ned_m) / self.resolution_m).astype(int)
        index = np.minimum(index, np.asarray(self.shape) - 1)
        return int(index[0]), int(index[1]), int(index[2])

    def is_occupied(self, position_ned_m: FloatArray) -> bool:
        return bool(self.occupied[self.index_of(position_ned_m)])

    def point_clearance(self, position_ned_m: FloatArray) -> float:
        point = _point(position_ned_m, "position_ned_m")
        boundary = _bounds_signed_distance(point, self.bounds_ned_m)
        if not self.contains(point):
            return boundary
        if self.is_occupied(point):
            return min(boundary, -0.5 * self.resolution_m)
        occupied_indices = np.argwhere(self.occupied)
        if occupied_indices.size == 0:
            return boundary
        centers = self.origin_ned_m + (occupied_indices + 0.5) * self.resolution_m
        nearest = float(np.min(np.linalg.norm(centers - point, axis=1)))
        return min(boundary, nearest - 0.5 * self.resolution_m)
