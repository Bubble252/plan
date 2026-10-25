from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from auv_planner.contracts import (
    ObstacleMap,
    Remus100ActuatorCommand,
    ScenarioConfig,
    SimulationSnapshot,
    State,
    UniformCurrent,
    WorldSnapshot,
)
from auv_planner.current import DepthShearCurrent
from auv_planner.dynamics.remus100 import PvsRemus100Backend
from auv_planner.interfaces import SimulatorBackend
from auv_planner.scenarios import remus100_vehicle


def make_state(
    roll_rad: float = 0.08,
    pitch_rad: float = -0.05,
    yaw_rad: float = 0.3,
) -> State:
    cr = np.cos(roll_rad / 2.0)
    sr = np.sin(roll_rad / 2.0)
    cp = np.cos(pitch_rad / 2.0)
    sp = np.sin(pitch_rad / 2.0)
    cy = np.cos(yaw_rad / 2.0)
    sy = np.sin(yaw_rad / 2.0)
    quat = np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ]
    )
    return State(
        t_s=2.0,
        position_ned_m=np.array([1.0, -2.0, 10.0]),
        quat_xyzw=quat,
        linear_velocity_body_mps=np.array([1.2, 0.0, 0.0]),
        angular_velocity_body_rps=np.zeros(3),
    )


def make_scenario(current=UniformCurrent(velocity_ned_mps=np.zeros(3))) -> ScenarioConfig:
    world = WorldSnapshot(
        t_s=2.0,
        obstacle_map=ObstacleMap(
            bounds_ned_m=np.array([[-100.0, 100.0], [-100.0, 100.0], [0.0, 100.0]])
        ),
        current_field=current,
    )
    return ScenarioConfig(
        name="pvs-smoke",
        world=world,
        vehicle=remus100_vehicle(),
        start=make_state(),
        goal_ned_m=np.array([30.0, 10.0, 10.0]),
        goal_radius_m=1.0,
    )


def test_pvs_backend_resets_and_steps_with_synchronized_state() -> None:
    backend = PvsRemus100Backend(sample_time_s=0.02)
    scenario = make_scenario()
    initial = backend.reset(scenario, seed=7)

    assert isinstance(backend, SimulatorBackend)
    assert isinstance(initial, SimulationSnapshot)
    assert initial.state.t_s == pytest.approx(2.0)
    assert initial.world.t_s == pytest.approx(2.0)
    assert initial.state.position_ned_m == pytest.approx(scenario.start.position_ned_m)
    assert abs(float(np.dot(initial.state.quat_xyzw, scenario.start.quat_xyzw))) == pytest.approx(1.0)
    assert initial.actuator_state is not None
    assert isinstance(initial.actuator_state, Remus100ActuatorCommand)
    assert initial.actuator_state.propeller_rpm == pytest.approx(0.0)

    command = Remus100ActuatorCommand(
        stamp_s=initial.state.t_s,
        rudder_angle_rad=0.05,
        stern_plane_angle_rad=0.03,
        propeller_rpm=500.0,
    )
    stepped = backend.step(command)

    assert stepped.state.t_s == pytest.approx(2.02)
    assert stepped.world.t_s == pytest.approx(2.02)
    assert np.linalg.norm(stepped.state.quat_xyzw) == pytest.approx(1.0)
    assert stepped.actuator_state is not None
    assert isinstance(stepped.actuator_state, Remus100ActuatorCommand)
    assert stepped.actuator_state.rudder_angle_rad == pytest.approx(0.01)
    assert stepped.actuator_state.stern_plane_angle_rad == pytest.approx(0.006)
    assert stepped.actuator_state.propeller_rpm == pytest.approx(10.0)
    assert stepped.diagnostics["sample_time_s"] == pytest.approx(0.02)


def test_pvs_backend_is_deterministic_for_identical_inputs() -> None:
    scenario = make_scenario()
    command = Remus100ActuatorCommand(2.0, -0.08, 0.04, 800.0)
    first_backend = PvsRemus100Backend()
    second_backend = PvsRemus100Backend()
    first_backend.reset(scenario, seed=1)
    second_backend.reset(scenario, seed=1)

    first = first_backend.step(command)
    second = second_backend.step(command)

    assert np.allclose(first.state.position_ned_m, second.state.position_ned_m)
    assert np.allclose(first.state.quat_xyzw, second.state.quat_xyzw)
    assert np.allclose(first.state.linear_velocity_body_mps, second.state.linear_velocity_body_mps)
    assert np.allclose(first.state.angular_velocity_body_rps, second.state.angular_velocity_body_rps)


def test_pvs_backend_reports_horizontal_current_mapping() -> None:
    current = UniformCurrent(velocity_ned_mps=np.array([0.3, 0.4, 0.0]))
    snapshot = PvsRemus100Backend().reset(make_scenario(current), seed=0)

    assert snapshot.diagnostics["current_speed_mps"] == pytest.approx(0.5)
    assert snapshot.diagnostics["current_direction_rad"] == pytest.approx(np.arctan2(0.4, 0.3))


def test_pvs_backend_rejects_out_of_limit_stale_and_future_commands() -> None:
    backend = PvsRemus100Backend()
    initial = backend.reset(make_scenario(), seed=0)
    limit = 0.1

    with pytest.raises(ValueError, match="rudder"):
        backend.step(Remus100ActuatorCommand(initial.state.t_s, limit + 1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="stern_plane"):
        backend.step(Remus100ActuatorCommand(initial.state.t_s, 0.0, limit + 1.0, 0.0))
    with pytest.raises(ValueError, match="propeller_rpm"):
        backend.step(Remus100ActuatorCommand(initial.state.t_s, 0.0, 0.0, 2000.0))
    with pytest.raises(ValueError, match="precedes"):
        backend.step(Remus100ActuatorCommand(initial.state.t_s - 0.1, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="beyond"):
        backend.step(Remus100ActuatorCommand(initial.state.t_s + 0.1, 0.0, 0.0, 0.0))


def test_pvs_backend_rejects_unsupported_current_and_missing_limits() -> None:
    vertical_current = UniformCurrent(velocity_ned_mps=np.array([0.1, 0.0, 0.05]))
    with pytest.raises(ValueError, match="horizontal current"):
        PvsRemus100Backend().reset(make_scenario(vertical_current), seed=0)

    shear = DepthShearCurrent(
        reference_velocity_ned_mps=np.array([0.1, 0.0, 0.0]),
        shear_ned_per_m=np.zeros(3),
    )
    with pytest.raises(ValueError, match="only UniformCurrent"):
        PvsRemus100Backend().reset(make_scenario(shear), seed=0)

    scenario = make_scenario()
    missing_limits = replace(
        scenario.vehicle,
        max_rudder_angle_rad=None,
        max_stern_plane_angle_rad=None,
        max_propeller_rpm=None,
    )
    with pytest.raises(ValueError, match="actuator limits"):
        PvsRemus100Backend().reset(replace(scenario, vehicle=missing_limits), seed=0)
