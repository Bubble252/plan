from __future__ import annotations

import numpy as np
import pytest

from auv_planner.contracts import (
    ActuatorCommand,
    CapabilitySet,
    EvalResult,
    Obstacle,
    ObstacleMap,
    PlanRequest,
    PlanResult,
    PlanStatus,
    RecoveryCertificate,
    Reference,
    Remus100ActuatorCommand,
    SimulationSnapshot,
    ScenarioConfig,
    State,
    TaskSpec,
    Trajectory,
    UniformCurrent,
    VehicleSpec,
    WorldSnapshot,
)
from auv_planner.interfaces import GlobalPlanner, LocalPlanner, SafetyFilter, SimulatorBackend, TrackingController
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


def test_valid_certificate_requires_window_assumptions_and_backup() -> None:
    with pytest.raises(ValueError, match="validity window"):
        RecoveryCertificate(
            valid=True,
            assumptions=("finite horizon",),
            fault_set=("nominal@0s",),
            backup_maneuvers=("brake",),
        )
    with pytest.raises(ValueError, match="backup maneuver"):
        RecoveryCertificate(
            valid=True,
            assumptions=("finite horizon",),
            fault_set=("nominal@0s",),
            backup_maneuvers=(),
            valid_from_s=0.0,
            valid_until_s=1.0,
        )

    certificate = RecoveryCertificate(
        valid=True,
        assumptions=("finite horizon",),
        fault_set=("nominal@0s",),
        backup_maneuvers=("brake",),
        valid_from_s=0.0,
        valid_until_s=1.0,
        margin_m=0.1,
    )
    assert certificate.valid_from_s == 0.0
    assert certificate.margin_m == 0.1


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


def test_actuator_command_validates_time_and_finite_values() -> None:
    command = Remus100ActuatorCommand(
        stamp_s=1.0,
        rudder_angle_rad=0.1,
        stern_plane_angle_rad=-0.05,
        propeller_rpm=500.0,
        valid_until_s=1.2,
    )

    assert command.valid_until_s == 1.2
    assert isinstance(command, ActuatorCommand)
    with pytest.raises(ValueError, match="finite"):
        Remus100ActuatorCommand(0.0, float("nan"), 0.0, 0.0)
    with pytest.raises(ValueError, match="precede"):
        Remus100ActuatorCommand(1.0, 0.0, 0.0, 0.0, valid_until_s=0.5)


def test_reference_rejects_expired_validity_window() -> None:
    with pytest.raises(ValueError, match="precede"):
        Reference(
            stamp_s=1.0,
            position_ned_m=np.zeros(3),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            linear_velocity_body_mps=np.array([1.0, 0.0, 0.0]),
            angular_velocity_body_rps=np.zeros(3),
            valid_until_s=0.5,
        )


def test_simulation_snapshot_requires_synchronized_state_and_world() -> None:
    request = make_request()
    snapshot = SimulationSnapshot(request.start, request.world)

    assert snapshot.state.t_s == snapshot.world.t_s
    later_state = State(
        t_s=1.0,
        position_ned_m=request.start.position_ned_m,
        quat_xyzw=request.start.quat_xyzw,
        linear_velocity_body_mps=request.start.linear_velocity_body_mps,
        angular_velocity_body_rps=request.start.angular_velocity_body_rps,
    )
    with pytest.raises(ValueError, match="timestamps"):
        SimulationSnapshot(later_state, request.world)
    with pytest.raises(ValueError, match="actuator_state"):
        SimulationSnapshot(
            request.start,
            request.world,
            actuator_state=Remus100ActuatorCommand(1.0, 0.0, 0.0, 0.0),
        )


def test_vehicle_spec_validates_optional_actuator_limits() -> None:
    vehicle = make_request().vehicle
    bounded = VehicleSpec(
        name=vehicle.name,
        length_m=vehicle.length_m,
        diameter_m=vehicle.diameter_m,
        mass_kg=vehicle.mass_kg,
        min_speed_mps=vehicle.min_speed_mps,
        max_speed_mps=vehicle.max_speed_mps,
        max_accel_mps2=vehicle.max_accel_mps2,
        max_yaw_rate_rps=vehicle.max_yaw_rate_rps,
        max_pitch_rate_rps=vehicle.max_pitch_rate_rps,
        max_rudder_angle_rad=0.2,
        max_stern_plane_angle_rad=0.2,
        max_propeller_rpm=1200.0,
    )

    assert bounded.max_rudder_angle_rad == pytest.approx(0.2)
    with pytest.raises(ValueError, match="max_rudder_angle_rad"):
        VehicleSpec(
            name=vehicle.name,
            length_m=vehicle.length_m,
            diameter_m=vehicle.diameter_m,
            mass_kg=vehicle.mass_kg,
            min_speed_mps=vehicle.min_speed_mps,
            max_speed_mps=vehicle.max_speed_mps,
            max_accel_mps2=vehicle.max_accel_mps2,
            max_yaw_rate_rps=vehicle.max_yaw_rate_rps,
            max_pitch_rate_rps=vehicle.max_pitch_rate_rps,
            max_rudder_angle_rad=0.0,
        )


def test_tracking_and_simulator_protocol_conformance() -> None:
    class DummyController:
        def control(self, state: State, reference: Reference) -> Remus100ActuatorCommand:
            del state, reference
            return Remus100ActuatorCommand(0.0, 0.0, 0.0, 0.0)

    class DummyBackend:
        def reset(self, scenario: ScenarioConfig, seed: int) -> SimulationSnapshot:
            del seed
            return SimulationSnapshot(scenario.start, scenario.world)

        def step(self, command: ActuatorCommand) -> SimulationSnapshot:
            del command
            raise NotImplementedError

    controller: TrackingController = DummyController()
    backend: SimulatorBackend = DummyBackend()
    assert isinstance(controller, TrackingController)
    assert isinstance(backend, SimulatorBackend)


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
        out_of_bounds=False,
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
            out_of_bounds=False,
            min_clearance_m=-0.1,
            path_length_m=1.0,
            duration_s=1.0,
            energy_j=0.0,
            tracking_rmse=0.0,
            constraint_violations=0,
        )
