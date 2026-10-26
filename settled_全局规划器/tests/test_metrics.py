from __future__ import annotations

import numpy as np
import pytest

from auv_planner.contracts import EvalResult, Obstacle, ObstacleMap, Trajectory, VehicleSpec
from auv_planner.metrics import compute_path_metrics


def make_vehicle() -> VehicleSpec:
    return VehicleSpec(
        name="test-auv",
        length_m=1.6,
        diameter_m=0.19,
        mass_kg=31.0,
        min_speed_mps=0.1,
        max_speed_mps=4.0,
        max_accel_mps2=1.0,
        max_yaw_rate_rps=0.5,
        max_pitch_rate_rps=0.5,
    )


def make_trajectory(points: np.ndarray, t_s: np.ndarray) -> Trajectory:
    positions = np.asarray(points, dtype=np.float64)
    rows = positions.shape[0]
    return Trajectory(
        trajectory_id="metrics-test",
        t_s=t_s,
        position_ned_m=positions,
        quat_xyzw=np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (rows, 1)),
        linear_velocity_body_mps=np.zeros((rows, 3)),
        angular_velocity_body_rps=np.zeros((rows, 3)),
    )


def make_map(obstacles: tuple[Obstacle, ...] = (), half_size_m: float = 5.0) -> ObstacleMap:
    bounds = np.array(
        [
            [-half_size_m, half_size_m],
            [-half_size_m, half_size_m],
            [-half_size_m, half_size_m],
        ],
        dtype=np.float64,
    )
    return ObstacleMap(bounds_ned_m=bounds, obstacles=obstacles)


def test_compute_path_metrics_reports_length_time_and_clearance() -> None:
    trajectory = make_trajectory(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        np.array([0.0, 1.0, 3.0]),
    )

    metrics = compute_path_metrics(trajectory, make_map(), make_vehicle())

    assert metrics.path_length_m == pytest.approx(3.0)
    assert metrics.duration_s == pytest.approx(3.0)
    assert metrics.mean_speed_mps == pytest.approx(1.0)
    assert metrics.is_safe
    assert metrics.failure_reason is None
    assert metrics.min_clearance_m == pytest.approx(1.2)


def test_compute_path_metrics_reports_collision_reason() -> None:
    obstacle = Obstacle("sphere", "sphere", np.zeros(3), radius_m=0.4)
    trajectory = make_trajectory(
        np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        np.array([0.0, 1.0]),
    )

    metrics = compute_path_metrics(
        trajectory, make_map((obstacle,), half_size_m=4.0), make_vehicle()
    )

    assert metrics.collision
    assert not metrics.out_of_bounds
    assert metrics.failure_reason == "collision"
    assert metrics.min_clearance_m < 0.0


def test_compute_path_metrics_reports_out_of_bounds_reason() -> None:
    trajectory = make_trajectory(
        np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        np.array([0.0, 1.0]),
    )

    metrics = compute_path_metrics(trajectory, make_map(half_size_m=1.0), make_vehicle())

    assert metrics.out_of_bounds
    assert metrics.failure_reason == "out_of_bounds"


def test_path_metrics_convert_to_eval_result() -> None:
    trajectory = make_trajectory(
        np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        np.array([0.0, 2.0]),
    )
    metrics = compute_path_metrics(trajectory, make_map(), make_vehicle())

    result = metrics.to_eval_result(success=True, energy_j=4.0, metrics={"seed": 7.0})

    assert isinstance(result, EvalResult)
    assert result.success
    assert result.min_clearance_m == metrics.min_clearance_m
    assert result.metrics == {"seed": 7.0}


def test_failed_eval_result_preserves_negative_clearance_and_reason() -> None:
    obstacle = Obstacle("sphere", "sphere", np.zeros(3), radius_m=0.4)
    trajectory = make_trajectory(
        np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        np.array([0.0, 1.0]),
    )
    metrics = compute_path_metrics(
        trajectory, make_map((obstacle,), half_size_m=4.0), make_vehicle()
    )

    result = metrics.to_eval_result(success=False, task_failure_reason="goal_not_reached")

    assert not result.success
    assert result.collision
    assert result.min_clearance_m < 0.0
    assert result.failure_reason == "collision,goal_not_reached"
    with pytest.raises(ValueError, match="safe geometric path"):
        metrics.to_eval_result(success=True)
