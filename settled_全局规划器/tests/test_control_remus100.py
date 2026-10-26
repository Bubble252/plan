from __future__ import annotations

from dataclasses import replace
from math import cos, sin

import numpy as np
import pytest

from auv_planner.contracts import (
    Reference,
    Remus100ActuatorCommand,
    SimulationSnapshot,
    State,
    UniformCurrent,
)
from auv_planner.control import Remus100TrackingController
from auv_planner.dynamics.remus100 import PvsRemus100Backend
from auv_planner.evaluation.closed_loop import (
    project_onto_path,
    run_closed_loop,
    write_closed_loop_log,
)
from auv_planner.experiments.m4_closed_loop import m4_cases
from auv_planner.guidance import ThreeDAlosGuidance
from auv_planner.interfaces import Guidance, TrackingController
from auv_planner.scenarios import remus100_vehicle


def _state(t_s: float = 0.0) -> State:
    return State(
        t_s=t_s,
        position_ned_m=np.zeros(3),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        linear_velocity_body_mps=np.array([1.5, 0.0, 0.0]),
        angular_velocity_body_rps=np.zeros(3),
    )


def _reference(
    yaw_rad: float = 0.0,
    *,
    stamp_s: float = 0.0,
    valid_until_s: float | None = 1.0,
    speed_mps: float = 1.5,
) -> Reference:
    return Reference(
        stamp_s=stamp_s,
        position_ned_m=np.array([5.0, 0.0, 0.0]),
        quat_xyzw=np.array([0.0, 0.0, sin(yaw_rad / 2.0), cos(yaw_rad / 2.0)]),
        linear_velocity_body_mps=np.array([speed_mps, 0.0, 0.0]),
        angular_velocity_body_rps=np.zeros(3),
        valid_until_s=valid_until_s,
    )


def test_controller_satisfies_protocol_and_enforces_fin_limits() -> None:
    vehicle = remus100_vehicle()
    max_rudder_angle = vehicle.max_rudder_angle_rad
    max_stern_plane_angle = vehicle.max_stern_plane_angle_rad
    max_propeller_rpm = vehicle.max_propeller_rpm
    max_rudder_rate = vehicle.max_rudder_rate_rad_s
    max_stern_plane_rate = vehicle.max_stern_plane_rate_rad_s
    assert max_rudder_angle is not None
    assert max_stern_plane_angle is not None
    assert max_propeller_rpm is not None
    assert max_rudder_rate is not None
    assert max_stern_plane_rate is not None

    controller = Remus100TrackingController(vehicle, nominal_step_s=0.02)
    reference = _reference(1.0)

    first = controller.control(_state(0.0), reference)
    second = controller.control(_state(0.02), reference)

    assert isinstance(controller, TrackingController)
    assert isinstance(first, Remus100ActuatorCommand)
    assert abs(first.rudder_angle_rad) <= max_rudder_angle
    assert abs(first.stern_plane_angle_rad) <= max_stern_plane_angle
    assert 0.0 <= first.propeller_rpm <= max_propeller_rpm
    assert abs(second.rudder_angle_rad - first.rudder_angle_rad) <= (max_rudder_rate * 0.02 + 1e-12)
    assert abs(second.stern_plane_angle_rad - first.stern_plane_angle_rad) <= (
        max_stern_plane_rate * 0.02 + 1e-12
    )
    assert second.valid_until_s == pytest.approx(0.04)
    assert controller.last_diagnostics is not None
    assert controller.last_diagnostics.heading_error_rad > 0.0


def test_controller_rejects_expired_future_and_out_of_range_references() -> None:
    controller = Remus100TrackingController(remus100_vehicle())

    with pytest.raises(ValueError, match="expired"):
        controller.control(_state(2.0), _reference(stamp_s=0.0, valid_until_s=1.0))
    with pytest.raises(ValueError, match="future"):
        controller.control(_state(0.0), _reference(stamp_s=1.0, valid_until_s=2.0))

    controller.reset()
    with pytest.raises(ValueError, match="below"):
        controller.control(_state(0.0), _reference(speed_mps=0.5))
    controller.reset()
    with pytest.raises(ValueError, match="exceeds"):
        controller.control(_state(0.0), _reference(speed_mps=5.0))


def test_controller_requires_increasing_state_time() -> None:
    controller = Remus100TrackingController(remus100_vehicle())
    controller.control(_state(0.0), _reference())
    with pytest.raises(ValueError, match="increase"):
        controller.control(_state(0.0), _reference())


def test_controller_uses_water_relative_surge_with_current() -> None:
    current = UniformCurrent(velocity_ned_mps=np.array([0.3, 0.0, 0.0]))
    controller = Remus100TrackingController(remus100_vehicle(), current_field=current)
    state = _state(0.0)
    reference = _reference(speed_mps=1.2)

    command = controller.control(state, reference)

    assert controller.last_diagnostics is not None
    assert controller.last_diagnostics.relative_surge_speed_mps == pytest.approx(1.2)
    assert controller.last_diagnostics.speed_error_mps == pytest.approx(0.0)
    assert command.propeller_rpm == pytest.approx(720.0)


def test_project_onto_path_reports_horizontal_and_vertical_offsets() -> None:
    projection = project_onto_path(
        np.array([2.0, 0.5, -0.25]),
        np.array([[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
    )

    assert projection.progress_m == pytest.approx(7.0)
    assert projection.horizontal_error_m == pytest.approx(0.5)
    assert projection.vertical_error_m == pytest.approx(-0.25)


class _FakeRemusBackend:
    def __init__(self, sample_time_s: float = 0.02) -> None:
        self.sample_time_s = sample_time_s
        self._snapshot: SimulationSnapshot | None = None

    def reset(self, scenario, seed: int) -> SimulationSnapshot:
        del seed
        self._snapshot = SimulationSnapshot(scenario.start, scenario.world)
        return self._snapshot

    def step(self, command: Remus100ActuatorCommand) -> SimulationSnapshot:
        assert self._snapshot is not None
        state = self._snapshot.state
        assert command.stamp_s == pytest.approx(state.t_s)
        next_state = State(
            t_s=state.t_s + self.sample_time_s,
            position_ned_m=state.position_ned_m,
            quat_xyzw=state.quat_xyzw,
            linear_velocity_body_mps=state.linear_velocity_body_mps,
            angular_velocity_body_rps=state.angular_velocity_body_rps,
        )
        self._snapshot = SimulationSnapshot(
            next_state,
            replace(self._snapshot.world, t_s=next_state.t_s),
            actuator_state=Remus100ActuatorCommand(
                next_state.t_s,
                command.rudder_angle_rad,
                command.stern_plane_angle_rad,
                command.propeller_rpm,
            ),
        )
        return self._snapshot


def test_fake_backend_closed_loop_preserves_time_reference_and_limits(tmp_path) -> None:
    case = m4_cases()[0]
    guidance = ThreeDAlosGuidance(
        3.0,
        2.0,
        1.5,
        current_field=case.scenario.world.current_field,
        compensate_current=False,
    )
    controller = Remus100TrackingController(case.scenario.vehicle)
    result = run_closed_loop(
        case.name,
        case.scenario,
        case.trajectory,
        guidance,
        controller,
        _FakeRemusBackend(),
        max_duration_s=0.1,
    )

    assert isinstance(guidance, Guidance)
    assert result.metrics.sample_count == 5
    assert result.metrics.constraint_violations == 0
    assert all(sample.command.stamp_s < sample.t_s for sample in result.samples)
    log_path = write_closed_loop_log(result, tmp_path / "fake.json")
    assert log_path.is_file()


def test_pvs_straight_closed_loop_smoke() -> None:
    case = m4_cases()[0]
    guidance = ThreeDAlosGuidance(
        3.0,
        2.0,
        1.5,
        current_field=case.scenario.world.current_field,
        compensate_current=False,
    )
    result = run_closed_loop(
        case.name,
        case.scenario,
        case.trajectory,
        guidance,
        Remus100TrackingController(case.scenario.vehicle),
        PvsRemus100Backend(sample_time_s=0.02),
        max_duration_s=case.max_duration_s,
    )

    assert result.metrics.completed
    assert result.metrics.horizontal_rmse_m < 0.05
    assert result.metrics.vertical_rmse_m < 0.1
    assert result.metrics.constraint_violations == 0
    assert np.isfinite(result.metrics.max_propeller_rpm)
