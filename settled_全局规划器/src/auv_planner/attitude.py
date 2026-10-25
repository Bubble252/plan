from __future__ import annotations

from math import atan2, cos, sin

import numpy as np

from .contracts import FloatArray


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


def forward_quaternion(direction_ned: FloatArray) -> FloatArray:
    """Return the shortest-arc quaternion that aligns body +x with a direction."""

    direction = _unit_direction(_point(direction_ned, "direction_ned"))
    if direction is None:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    dot = float(np.clip(direction[0], -1.0, 1.0))
    if dot > 1.0 - 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if dot < -1.0 + 1e-12:
        return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    axis = np.cross(np.array([1.0, 0.0, 0.0]), direction)
    axis_norm = float(np.linalg.norm(axis))
    axis /= axis_norm
    half_angle = float(np.arccos(dot) / 2.0)
    return np.concatenate((axis * np.sin(half_angle), [np.cos(half_angle)])).astype(np.float64)


def quaternion_from_euler(roll_rad: float, pitch_rad: float, yaw_rad: float) -> FloatArray:
    """Return an xyzw quaternion from intrinsic ZYX Euler angles."""

    cr = cos(roll_rad / 2.0)
    sr = sin(roll_rad / 2.0)
    cp = cos(pitch_rad / 2.0)
    sp = sin(pitch_rad / 2.0)
    cy = cos(yaw_rad / 2.0)
    sy = sin(yaw_rad / 2.0)
    quat = np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float64,
    )
    return quat / np.linalg.norm(quat)


def euler_from_quaternion(quat_xyzw: FloatArray) -> FloatArray:
    """Return intrinsic ZYX roll, pitch and yaw angles in radians."""

    quat = np.asarray(quat_xyzw, dtype=np.float64)
    if quat.shape != (4,):
        raise ValueError("quat_xyzw must have shape (4,)")
    if not np.all(np.isfinite(quat)):
        raise ValueError("quat_xyzw must contain only finite values")
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0:
        raise ValueError("quat_xyzw must have non-zero norm")
    x, y, z, w = quat / norm
    roll = atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_argument = float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    pitch = float(np.arcsin(pitch_argument))
    yaw = atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=np.float64)


def quaternion_angle(quat_a: FloatArray, quat_b: FloatArray) -> float:
    first = np.asarray(quat_a, dtype=np.float64)
    second = np.asarray(quat_b, dtype=np.float64)
    if first.shape != (4,) or second.shape != (4,):
        raise ValueError("quaternions must have shape (4,)")
    if np.linalg.norm(first) <= 0.0 or np.linalg.norm(second) <= 0.0:
        raise ValueError("quaternions must have non-zero norm")
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    return float(2.0 * np.arccos(np.clip(abs(float(np.dot(first, second))), -1.0, 1.0)))


__all__ = [
    "euler_from_quaternion",
    "forward_quaternion",
    "quaternion_angle",
    "quaternion_from_euler",
]
