from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from auv_planner.contracts import State, Trajectory, UniformCurrent
from auv_planner.guidance import ThreeDAlosGuidance
from auv_planner.interfaces import Guidance


def make_state(position_ned_m: np.ndarray) -> State:
    return State(
        t_s=0.0,
        position_ned_m=position_ned_m,
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        linear_velocity_body_mps=np.array([1.5, 0.0, 0.0]),
        angular_velocity_body_rps=np.zeros(3),
    )


def make_trajectory(*positions: tuple[float, float, float]) -> Trajectory:
    points = np.asarray(positions, dtype=np.float64)
    rows = points.shape[0]
    return Trajectory(
        trajectory_id="alos-test",
        t_s=np.arange(rows, dtype=np.float64),
        position_ned_m=points,
        quat_xyzw=np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (rows, 1)),
        linear_velocity_body_mps=np.tile(np.array([1.5, 0.0, 0.0]), (rows, 1)),
        angular_velocity_body_rps=np.zeros((rows, 3)),
    )


def body_x_direction(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = quat_xyzw
    return np.array(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
        ]
    )


def make_guidance(**overrides: Any) -> ThreeDAlosGuidance:
    parameters: dict[str, Any] = {
        "horizontal_lookahead_m": 4.0,
        "vertical_lookahead_m": 2.0,
        "speed_mps": 1.5,
        "min_lookahead_m": 0.5,
        "max_lookahead_m": 20.0,
        "horizontal_gain": 1.0,
        "vertical_gain": 1.0,
        "lookahead_error_gain": 0.5,
        "validity_period_s": 0.25,
    }
    parameters.update(overrides)
    return ThreeDAlosGuidance(**parameters)


def test_guidance_satisfies_protocol_and_corrects_lateral_error() -> None:
    guidance = make_guidance()
    trajectory = make_trajectory((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))

    solution = guidance.solve(make_state(np.array([0.0, 2.0, 0.0])), trajectory, 1.0)
    reference = guidance.guide(make_state(np.array([0.0, 2.0, 0.0])), trajectory, 1.0)

    assert isinstance(guidance, Guidance)
    assert solution.horizontal_error_m == pytest.approx(2.0)
    assert solution.horizontal_lookahead_m == pytest.approx(5.0)
    assert solution.desired_heading_rad < 0.0
    assert solution.desired_body_direction_ned[1] < 0.0
    assert body_x_direction(reference.quat_xyzw)[1] < 0.0
    assert reference.position_ned_m[0] > 0.0
    assert reference.position_ned_m[1] == pytest.approx(0.0)
    assert reference.linear_velocity_body_mps == pytest.approx([1.5, 0.0, 0.0])
    assert reference.stamp_s == pytest.approx(1.0)
    assert reference.valid_until_s == pytest.approx(1.25)


def test_guidance_corrects_depth_toward_horizontal_path() -> None:
    guidance = make_guidance()
    trajectory = make_trajectory((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))

    solution = guidance.solve(make_state(np.array([0.0, 0.0, 2.0])), trajectory, 0.0)

    assert solution.vertical_error_m == pytest.approx(2.0)
    assert solution.vertical_lookahead_m == pytest.approx(3.0)
    assert solution.desired_pitch_rad > 0.0
    assert solution.desired_body_direction_ned[2] < 0.0
    assert body_x_direction(solution.reference.quat_xyzw)[2] < 0.0


def test_adaptive_lookahead_increases_with_error_and_respects_bounds() -> None:
    guidance = make_guidance(lookahead_error_gain=2.0, max_lookahead_m=6.0)
    trajectory = make_trajectory((0.0, 0.0, 0.0), (20.0, 0.0, 0.0))

    centered = guidance.solve(make_state(np.zeros(3)), trajectory, 0.0)
    offset = guidance.solve(make_state(np.array([0.0, 5.0, 0.0])), trajectory, 0.0)

    assert centered.horizontal_lookahead_m == pytest.approx(4.0)
    assert offset.horizontal_lookahead_m == pytest.approx(6.0)
    assert offset.lookahead_s_m >= offset.projection_s_m


def test_lookahead_progress_does_not_move_backward_at_a_corner() -> None:
    guidance = make_guidance(
        horizontal_lookahead_m=1.0,
        vertical_lookahead_m=1.0,
        lookahead_error_gain=0.0,
    )
    trajectory = make_trajectory((0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (5.0, 5.0, 0.0))

    before_corner = guidance.solve(make_state(np.array([4.8, 0.1, 0.0])), trajectory, 0.0)
    after_corner = guidance.solve(make_state(np.array([5.0, 0.1, 0.0])), trajectory, 0.0)

    assert after_corner.projection_s_m >= before_corner.projection_s_m
    assert after_corner.lookahead_s_m >= before_corner.lookahead_s_m
    assert after_corner.reference.position_ned_m[0] == pytest.approx(5.0)
    assert after_corner.reference.position_ned_m[1] > 0.0


def test_current_compensation_changes_heading_and_preserves_ground_velocity() -> None:
    trajectory = make_trajectory((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))
    current = UniformCurrent(velocity_ned_mps=np.array([0.0, 0.5, 0.0]))
    compensated = make_guidance(
        current_field=current,
        compensate_current=True,
        lookahead_error_gain=0.0,
    )
    uncompensated = make_guidance(
        current_field=current,
        compensate_current=False,
        lookahead_error_gain=0.0,
    )
    state = make_state(np.zeros(3))

    compensated_solution = compensated.solve(state, trajectory, 0.0)
    uncompensated_solution = uncompensated.solve(state, trajectory, 0.0)

    assert uncompensated_solution.desired_body_direction_ned == pytest.approx([1.0, 0.0, 0.0])
    assert compensated_solution.desired_body_direction_ned[1] < 0.0
    assert compensated_solution.command_speed_mps > uncompensated_solution.command_speed_mps
    reference = compensated_solution.reference
    ground_velocity = (
        body_x_direction(reference.quat_xyzw) * reference.linear_velocity_body_mps[0]
        + current.velocity_ned_mps
    )
    assert ground_velocity == pytest.approx([1.5, 0.0, 0.0], abs=1e-12)


def test_guidance_is_deterministic_and_returns_contract_arrays() -> None:
    guidance = make_guidance()
    trajectory = make_trajectory((0.0, 0.0, 0.0), (4.0, 0.0, 1.0), (8.0, 3.0, 1.0))
    state = make_state(np.array([2.0, 1.0, 0.5]))

    first = guidance.solve(state, trajectory, 3.0)
    second = guidance.solve(state, trajectory, 3.0)

    assert np.allclose(first.desired_body_direction_ned, second.desired_body_direction_ned)
    assert np.allclose(first.reference.position_ned_m, second.reference.position_ned_m)
    assert np.allclose(first.reference.quat_xyzw, second.reference.quat_xyzw)
    assert np.allclose(first.reference.linear_velocity_body_mps, second.reference.linear_velocity_body_mps)
    assert np.allclose(first.reference.angular_velocity_body_rps, np.zeros(3))
    assert np.linalg.norm(first.reference.quat_xyzw) == pytest.approx(1.0)


def test_guidance_validates_inputs_and_configuration() -> None:
    trajectory = make_trajectory((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))
    state = make_state(np.zeros(3))
    guidance = make_guidance()

    with pytest.raises(ValueError, match="t_s"):
        guidance.guide(state, trajectory, -1.0)
    with pytest.raises(ValueError, match="t_s"):
        guidance.guide(state, trajectory, float("nan"))
    with pytest.raises(TypeError, match="State"):
        guidance.guide(np.zeros(3), trajectory, 0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="distinct"):
        guidance.guide(state, make_trajectory((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)), 0.0)
    with pytest.raises(ValueError, match="speed_mps"):
        make_guidance(speed_mps=0.0)
    with pytest.raises(ValueError, match="must not exceed"):
        make_guidance(min_lookahead_m=2.0, max_lookahead_m=1.0)
    with pytest.raises(ValueError, match="within the lookahead bounds"):
        make_guidance(horizontal_lookahead_m=30.0)
    with pytest.raises(ValueError, match="horizontal_gain"):
        make_guidance(horizontal_gain=-1.0)
    with pytest.raises(TypeError, match="compensate_current"):
        make_guidance(compensate_current=1)


def test_current_compensation_rejects_cancellation_and_bad_samples() -> None:
    trajectory = make_trajectory((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))
    state = make_state(np.zeros(3))
    cancelling = make_guidance(
        speed_mps=1.0,
        current_field=UniformCurrent(velocity_ned_mps=np.array([1.0, 0.0, 0.0])),
        lookahead_error_gain=0.0,
    )
    with pytest.raises(ValueError, match="exactly cancels"):
        cancelling.guide(state, trajectory, 0.0)

    class BadCurrent:
        def sample(self, position_ned_m: np.ndarray, t_s: float) -> np.ndarray:
            return np.zeros(2)

    bad_sample = make_guidance(current_field=BadCurrent())
    with pytest.raises(ValueError, match="shape"):
        bad_sample.guide(state, trajectory, 0.0)
