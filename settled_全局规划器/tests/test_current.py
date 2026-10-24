from __future__ import annotations

import numpy as np
import pytest

from auv_planner.contracts import CurrentField, UniformCurrent
from auv_planner.current import DepthShearCurrent


def test_uniform_current_is_position_independent_and_returns_copy() -> None:
    current = UniformCurrent(velocity_ned_mps=np.array([0.3, -0.1, 0.0]))

    first = current.sample(np.array([1.0, 2.0, 3.0]), 0.0)
    second = current.sample(np.array([-1.0, -2.0, -3.0]), 10.0)
    first[0] = 99.0

    assert isinstance(current, CurrentField)
    assert np.allclose(first, [99.0, -0.1, 0.0])
    assert np.allclose(second, [0.3, -0.1, 0.0])
    assert np.allclose(current.velocity_ned_mps, [0.3, -0.1, 0.0])


def test_depth_shear_uses_ned_depth_positive_down() -> None:
    current = DepthShearCurrent(
        reference_velocity_ned_mps=np.array([1.0, 0.0, 0.0]),
        shear_ned_per_m=np.array([0.1, -0.02, 0.0]),
        reference_depth_m=0.0,
    )

    shallow = current.sample(np.array([0.0, 0.0, -5.0]), 0.0)
    reference = current.sample(np.array([0.0, 0.0, 0.0]), 0.0)
    deep = current.sample(np.array([0.0, 0.0, 10.0]), 0.0)

    assert isinstance(current, CurrentField)
    assert np.allclose(shallow, [0.5, 0.1, 0.0])
    assert np.allclose(reference, [1.0, 0.0, 0.0])
    assert np.allclose(deep, [2.0, -0.2, 0.0])


def test_depth_shear_validates_inputs_and_returns_copy() -> None:
    current = DepthShearCurrent(
        reference_velocity_ned_mps=np.array([0.2, 0.0, 0.0]),
        shear_ned_per_m=np.zeros(3),
    )
    sample = current.sample(np.zeros(3), 1.0)
    sample[0] = -1.0

    assert np.allclose(current.reference_velocity_ned_mps, [0.2, 0.0, 0.0])
    with pytest.raises(ValueError, match="shape"):
        DepthShearCurrent(reference_velocity_ned_mps=np.zeros(2), shear_ned_per_m=np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        current.sample(np.zeros(3), -1.0)
