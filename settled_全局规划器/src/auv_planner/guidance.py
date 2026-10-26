from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin

import numpy as np

from .attitude import forward_quaternion
from .contracts import CurrentField, FloatArray, Reference, State, Trajectory


def _finite_positive(value: float, name: str) -> float:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _finite_non_negative(value: float, name: str) -> float:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


def _vector(value: object, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _unique_path_points(trajectory: Trajectory) -> FloatArray:
    points = np.asarray(trajectory.position_ned_m, dtype=np.float64)
    unique = [points[0].copy()]
    for point in points[1:]:
        if float(np.linalg.norm(point - unique[-1])) > 1e-9:
            unique.append(point.copy())
    if len(unique) < 2:
        raise ValueError("trajectory must contain at least two distinct positions")
    return np.vstack(unique)


def _point_at_arclength(
    points: FloatArray,
    cumulative_s_m: FloatArray,
    s_m: float,
) -> FloatArray:
    total_s = float(cumulative_s_m[-1])
    target_s = float(np.clip(s_m, 0.0, total_s))
    for index in range(points.shape[0] - 1):
        if target_s <= cumulative_s_m[index + 1] or index == points.shape[0] - 2:
            segment_length = float(cumulative_s_m[index + 1] - cumulative_s_m[index])
            fraction = (target_s - cumulative_s_m[index]) / segment_length
            return points[index] + fraction * (points[index + 1] - points[index])
    return points[-1].copy()


def _wrap_angle(angle_rad: float) -> float:
    return float((angle_rad + pi) % (2.0 * pi) - pi)


@dataclass(frozen=True)
class AlosSolution:
    """Diagnostic view of one 3D ALOS guidance solve."""

    segment_index: int
    projection_s_m: float
    lookahead_s_m: float
    horizontal_error_m: float
    vertical_error_m: float
    horizontal_lookahead_m: float
    vertical_lookahead_m: float
    desired_heading_rad: float
    desired_pitch_rad: float
    desired_body_direction_ned: FloatArray
    command_speed_mps: float
    reference: Reference


class ThreeDAlosGuidance:
    """Kinematic 3D adaptive line-of-sight guidance over a planned path.

    The implementation follows the horizontal/vertical lookahead structure of
    Yuan et al. (JMSE, 2025) but does not include DDPG, PSO, an adaptive
    sideslip observer, or a closed-loop stability guarantee. When current
    compensation is enabled, the body axis and commanded water-relative speed
    are chosen so that the instantaneous ground velocity equals
    ``speed_mps`` along the ALOS direction. A downstream controller remains
    responsible for actuator and vehicle limits.
    """

    def __init__(
        self,
        horizontal_lookahead_m: float,
        vertical_lookahead_m: float,
        speed_mps: float,
        *,
        min_lookahead_m: float = 0.5,
        max_lookahead_m: float = 20.0,
        horizontal_gain: float = 1.0,
        vertical_gain: float = 1.0,
        lookahead_error_gain: float = 1.0,
        current_field: CurrentField | None = None,
        compensate_current: bool = True,
        validity_period_s: float = 0.5,
    ) -> None:
        self.horizontal_lookahead_m = _finite_positive(
            horizontal_lookahead_m, "horizontal_lookahead_m"
        )
        self.vertical_lookahead_m = _finite_positive(vertical_lookahead_m, "vertical_lookahead_m")
        self.speed_mps = _finite_positive(speed_mps, "speed_mps")
        self.min_lookahead_m = _finite_positive(min_lookahead_m, "min_lookahead_m")
        self.max_lookahead_m = _finite_positive(max_lookahead_m, "max_lookahead_m")
        if self.min_lookahead_m > self.max_lookahead_m:
            raise ValueError("min_lookahead_m must not exceed max_lookahead_m")
        if not self.min_lookahead_m <= self.horizontal_lookahead_m <= self.max_lookahead_m:
            raise ValueError("horizontal_lookahead_m must lie within the lookahead bounds")
        if not self.min_lookahead_m <= self.vertical_lookahead_m <= self.max_lookahead_m:
            raise ValueError("vertical_lookahead_m must lie within the lookahead bounds")
        self.horizontal_gain = _finite_non_negative(horizontal_gain, "horizontal_gain")
        self.vertical_gain = _finite_non_negative(vertical_gain, "vertical_gain")
        self.lookahead_error_gain = _finite_non_negative(
            lookahead_error_gain, "lookahead_error_gain"
        )
        self.validity_period_s = _finite_positive(validity_period_s, "validity_period_s")
        if not isinstance(compensate_current, bool):
            raise TypeError("compensate_current must be a bool")
        self.compensate_current = compensate_current
        if current_field is not None and not isinstance(current_field, CurrentField):
            raise TypeError("current_field must implement the CurrentField protocol")
        self.current_field = current_field

    def solve(self, state: State, trajectory: Trajectory, t_s: float) -> AlosSolution:
        if not isinstance(state, State):
            raise TypeError("state must be a State")
        if not isinstance(trajectory, Trajectory):
            raise TypeError("trajectory must be a Trajectory")
        if not np.isfinite(t_s) or t_s < 0.0:
            raise ValueError("t_s must be finite and non-negative")

        points = _unique_path_points(trajectory)
        segments = np.diff(points, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        tangents = segments / lengths[:, None]
        cumulative_s_m = np.concatenate(([0.0], np.cumsum(lengths)))

        position = state.position_ned_m
        best_index = 0
        best_distance_sq = float("inf")
        best_projection_s = 0.0
        best_projection_point = points[0].copy()
        for index, (start, tangent, length) in enumerate(
            zip(points[:-1], tangents, lengths, strict=True)
        ):
            along_m = float(np.clip(np.dot(position - start, tangent), 0.0, length))
            projection_point = start + along_m * tangent
            distance_sq = float(np.dot(position - projection_point, position - projection_point))
            projection_s = float(cumulative_s_m[index] + along_m)
            tied = abs(distance_sq - best_distance_sq) <= 1e-12
            if distance_sq < best_distance_sq - 1e-12 or (tied and projection_s > best_projection_s):
                best_index = index
                best_distance_sq = distance_sq
                best_projection_s = projection_s
                best_projection_point = projection_point

        tangent = tangents[best_index]
        horizontal_norm = float(hypot(tangent[0], tangent[1]))
        if horizontal_norm > 1e-12:
            lateral_axis = np.array(
                [-tangent[1] / horizontal_norm, tangent[0] / horizontal_norm, 0.0],
                dtype=np.float64,
            )
        else:
            lateral_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        vertical_axis = np.cross(tangent, lateral_axis)
        vertical_axis /= np.linalg.norm(vertical_axis)

        offset = position - best_projection_point
        horizontal_error_m = float(np.dot(offset, lateral_axis))
        vertical_error_m = float(np.dot(offset, vertical_axis))
        effective_horizontal_lookahead_m = float(
            np.clip(
                self.horizontal_lookahead_m
                + self.lookahead_error_gain * abs(horizontal_error_m),
                self.min_lookahead_m,
                self.max_lookahead_m,
            )
        )
        effective_vertical_lookahead_m = float(
            np.clip(
                self.vertical_lookahead_m + self.lookahead_error_gain * abs(vertical_error_m),
                self.min_lookahead_m,
                self.max_lookahead_m,
            )
        )

        path_heading_rad = atan2(tangent[1], tangent[0])
        heading_correction_rad = atan2(horizontal_error_m, effective_horizontal_lookahead_m)
        desired_heading_rad = _wrap_angle(
            path_heading_rad - self.horizontal_gain * heading_correction_rad
        )
        path_pitch_rad = atan2(-tangent[2], horizontal_norm)
        pitch_correction_rad = atan2(vertical_error_m, effective_vertical_lookahead_m)
        desired_pitch_rad = float(
            np.clip(
                path_pitch_rad + self.vertical_gain * pitch_correction_rad,
                -pi / 2.0,
                pi / 2.0,
            )
        )
        desired_ground_direction = np.array(
            [
                cos(desired_pitch_rad) * cos(desired_heading_rad),
                cos(desired_pitch_rad) * sin(desired_heading_rad),
                -sin(desired_pitch_rad),
            ],
            dtype=np.float64,
        )

        commanded_water_speed_mps = self.speed_mps
        desired_body_direction = desired_ground_direction.copy()
        if self.compensate_current and self.current_field is not None:
            current = _vector(
                self.current_field.sample(position, t_s),
                "current velocity",
            )
            water_relative_velocity = self.speed_mps * desired_ground_direction - current
            commanded_water_speed_mps = float(np.linalg.norm(water_relative_velocity))
            if commanded_water_speed_mps <= 1e-9:
                raise ValueError("current exactly cancels the requested ground velocity")
            desired_body_direction = water_relative_velocity / commanded_water_speed_mps

        lookahead_distance_m = max(
            effective_horizontal_lookahead_m,
            effective_vertical_lookahead_m,
        )
        lookahead_s_m = min(best_projection_s + lookahead_distance_m, float(cumulative_s_m[-1]))
        lookahead_point_ned_m = _point_at_arclength(points, cumulative_s_m, lookahead_s_m)
        reference = Reference(
            stamp_s=float(t_s),
            position_ned_m=lookahead_point_ned_m,
            quat_xyzw=forward_quaternion(desired_body_direction),
            linear_velocity_body_mps=np.array([commanded_water_speed_mps, 0.0, 0.0]),
            angular_velocity_body_rps=np.zeros(3, dtype=np.float64),
            valid_until_s=float(t_s + self.validity_period_s),
        )
        return AlosSolution(
            segment_index=best_index,
            projection_s_m=best_projection_s,
            lookahead_s_m=lookahead_s_m,
            horizontal_error_m=horizontal_error_m,
            vertical_error_m=vertical_error_m,
            horizontal_lookahead_m=effective_horizontal_lookahead_m,
            vertical_lookahead_m=effective_vertical_lookahead_m,
            desired_heading_rad=desired_heading_rad,
            desired_pitch_rad=desired_pitch_rad,
            desired_body_direction_ned=desired_body_direction,
            command_speed_mps=commanded_water_speed_mps,
            reference=reference,
        )

    def guide(self, state: State, trajectory: Trajectory, t_s: float) -> Reference:
        return self.solve(state, trajectory, t_s).reference


__all__ = ["AlosSolution", "ThreeDAlosGuidance"]
