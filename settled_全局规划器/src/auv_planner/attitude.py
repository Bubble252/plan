from __future__ import annotations

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


__all__ = ["forward_quaternion", "quaternion_angle"]
