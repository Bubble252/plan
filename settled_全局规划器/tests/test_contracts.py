from __future__ import annotations

import numpy as np
import pytest

from auv_planner.contracts import (
    CapabilitySet,
    EvalResult,
    Obstacle,
    ObstacleMap,
    PlanRequest,
    PlanResult,
    PlanStatus,
    RecoveryCertificate,
    State,
    TaskSpec,
    Trajectory,
    UniformCurrent,
    VehicleSpec,
    WorldSnapshot,
)
from auv_planner.interfaces import GlobalPlanner, LocalPlanner, SafetyFilter
from auv_planner.noop import NoLocalPlanner, NoSafetyFilter
from auv_planner.planners.straight_line import StraightLinePlanner


def make_request() -> PlanRequest:
    world = WorldSnapshot(
        t_s=0.0,
        obstacle_map=ObstacleMap(bounds_ned_m=np.array([[-10.0, 10.0], [-10.0, 10.0], [-10.0, 10.0]])),
        current_field=UniformCurrent(velocity_ned_mps=np.zeros(3)),
    )
    return PlanRequest(
        request_id="smoke",
        start=State(
            t_s=0.0,
            position_ned_m=np.array([-5.0, 0.0, 0.0]),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            linear_velocity_body_mps=np.array([1.0, 0.0, 0.0]),
            angular_velocity_body_rps=np.zeros(3),
        ),
        task=TaskSpec(goal_ned_m=np.array([5.0, 0.0, 0.0]), goal_radius_m=1.0),
        world=world,
        vehicle=VehicleSpec(
            name="REMUS100",
            length_m=1.6,
            diameter_m=0.19,
            mass_kg=31.0,
            min_speed_mps=1.0,
            max_speed_mps=4.0,
            max_accel_mps2=1.0,
            max_yaw_rate_rps=0.5,
            max_pitch_rate_rps=0.5,
        ),
        capability=CapabilitySet(available_actuators=("propeller", "yaw_fin", "pitch_fin")),
    )


def test_straight_line_planner_is_deterministic_and_returns_contract() -> None:
    request = make_request()
    planner = StraightLinePlanner(samples=11)

    first = planner.plan(request)
    second = planner.plan(request)

    assert isinstance(planner, GlobalPlanner)
    assert first.status is PlanStatus.OK
    assert first.trajectory is not None
    assert second.trajectory is not None
    assert first.cost == pytest.approx(10.0)
    assert first.trajectory.feasibility == "unchecked"
    assert np.allclose(first.trajectory.position_ned_m[0], [-5.0, 0.0, 0.0])
    assert np.allclose(first.trajectory.position_ned_m[-1], [5.0, 0.0, 0.0])
    assert np.allclose(first.trajectory.position_ned_m, second.trajectory.position_ned_m)
    assert np.all(first.trajectory.t_s[1:] > first.trajectory.t_s[:-1])


def test_trajectory_rejects_non_monotonic_time() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        Trajectory(
            trajectory_id="bad-time",
            t_s=np.array([0.0, 1.0, 1.0]),
            position_ned_m=np.zeros((3, 3)),
            quat_xyzw=np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (3, 1)),
            linear_velocity_body_mps=np.zeros((3, 3)),
            angular_velocity_body_rps=np.zeros((3, 3)),
        )


def test_invalid_certificate_requires_reason() -> None:
    with pytest.raises(ValueError, match="require a reason"):
        RecoveryCertificate(valid=False, assumptions=(), fault_set=(), backup_maneuvers=())


def test_ok_plan_requires_trajectory() -> None:
    with pytest.raises(ValueError, match="require a trajectory"):
        PlanResult(
            status=PlanStatus.OK,
            trajectory=None,
            cost=0.0,
            certificate=None,
            planning_time_s=0.0,
            plan_id="bad-plan",
        )


def test_noop_adapters_satisfy_protocols() -> None:
    assert isinstance(NoLocalPlanner(), LocalPlanner)
    assert isinstance(NoSafetyFilter(), SafetyFilter)


def test_cylinder_obstacle_requires_finite_positive_dimensions() -> None:
    cylinder = Obstacle(
        "pillar",
        "cylinder",
        np.zeros(3),
        radius_m=0.5,
        height_m=2.0,
    )

    assert cylinder.radius_m == 0.5
    assert cylinder.height_m == 2.0
    with pytest.raises(ValueError, match="height_m must be finite and positive"):
        Obstacle("bad", "cylinder", np.zeros(3), radius_m=0.5, height_m=float("nan"))
    with pytest.raises(ValueError, match="radius_m must be finite and positive"):
        Obstacle("bad", "cylinder", np.zeros(3), radius_m=-1.0, height_m=1.0)


def test_eval_result_allows_negative_clearance_but_requires_failure_reason() -> None:
    result = EvalResult(
        success=False,
        collision=True,
        min_clearance_m=-0.1,
        path_length_m=1.0,
        duration_s=1.0,
        energy_j=0.0,
        tracking_rmse=0.0,
        constraint_violations=0,
        failure_reason="collision",
    )

    assert result.min_clearance_m == -0.1
    with pytest.raises(ValueError, match="failure_reason"):
        EvalResult(
            success=False,
            collision=True,
            min_clearance_m=-0.1,
            path_length_m=1.0,
            duration_s=1.0,
            energy_j=0.0,
            tracking_rmse=0.0,
            constraint_violations=0,
        )
