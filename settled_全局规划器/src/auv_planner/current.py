from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import FloatArray, UniformCurrent


def _vector(value: object, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _time(value: float) -> float:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("t_s must be finite and non-negative")
    return float(value)


@dataclass(frozen=True)
class DepthShearCurrent:
    """Linear current shear in NED coordinates, with z positive downwards."""

    reference_velocity_ned_mps: FloatArray
    shear_ned_per_m: FloatArray
    reference_depth_m: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reference_velocity_ned_mps",
            _vector(self.reference_velocity_ned_mps, "reference_velocity_ned_mps"),
        )
        object.__setattr__(self, "shear_ned_per_m", _vector(self.shear_ned_per_m, "shear_ned_per_m"))
        if not np.isfinite(self.reference_depth_m):
            raise ValueError("reference_depth_m must be finite")

    def sample(self, position_ned_m: FloatArray, t_s: float) -> FloatArray:
        position = _vector(position_ned_m, "position_ned_m")
        _time(t_s)
        depth_offset = position[2] - self.reference_depth_m
        return (self.reference_velocity_ned_mps + self.shear_ned_per_m * depth_offset).copy()


__all__ = ["DepthShearCurrent", "UniformCurrent"]
