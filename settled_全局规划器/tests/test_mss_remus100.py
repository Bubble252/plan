from __future__ import annotations

from math import radians
from pathlib import Path
import shutil

import numpy as np
import pytest

from auv_planner.attitude import quaternion_from_euler
from auv_planner.contracts import (
    CurrentField,
    ObstacleMap,
    Remus100ActuatorCommand,
    ScenarioConfig,
    State,
    UniformCurrent,
    WorldSnapshot,
)
from auv_planner.current import DepthShearCurrent
from auv_planner.dynamics.mss_remus100 import (
    MssRemus100Backend,
    _default_daemon_script,
    _default_mss_source_dir,
)
from auv_planner.dynamics.remus100 import PvsRemus100Backend
from auv_planner.scenarios import remus100_vehicle


OCTAVE_AVAILABLE = shutil.which("octave") is not None
MSS_AVAILABLE = _default_mss_source_dir().is_dir() and _default_daemon_script().is_file()
MSS_REQUIRED = pytest.mark.skipif(
    not (OCTAVE_AVAILABLE and MSS_AVAILABLE),
    reason="Octave and the local MSS tree are required for the MSS adapter test",
)


def _state(
    *,
    velocity=(1.5, 0.0, 0.0),
    angular_velocity=(0.0, 0.0, 0.0),
    position=(0.0, 0.0, 0.0),
    euler=(0.0, 0.0, 0.0),
) -> State:
    return State(
        t_s=0.0,
        position_ned_m=np.asarray(position, dtype=np.float64),
        quat_xyzw=quaternion_from_euler(*euler),
        linear_velocity_body_mps=np.asarray(velocity, dtype=np.float64),
        angular_velocity_body_rps=np.asarray(angular_velocity, dtype=np.float64),
    )


def _scenario(
    state: State | None = None,
    current: CurrentField | None = None,
) -> ScenarioConfig:
    return ScenarioConfig(
        name="mss-smoke",
        world=WorldSnapshot(
            t_s=0.0,
            obstacle_map=ObstacleMap(
                bounds_ned_m=np.array([[-100.0, 100.0], [-100.0, 100.0], [-100.0, 100.0]])
            ),
            current_field=UniformCurrent(velocity_ned_mps=np.zeros(3)) if current is None else current,
        ),
        vehicle=remus100_vehicle(),
        start=_state() if state is None else state,
        goal_ned_m=np.array([10.0, 0.0, 0.0]),
        goal_radius_m=0.5,
    )


def _command(
    *,
    rudder_rad: float = 0.0,
    stern_plane_rad: float = 0.0,
    rpm: float = 1000.0,
    stamp_s: float = 0.0,
) -> Remus100ActuatorCommand:
    return Remus100ActuatorCommand(
        stamp_s=stamp_s,
        rudder_angle_rad=rudder_rad,
        stern_plane_angle_rad=stern_plane_rad,
        propeller_rpm=rpm,
    )


@MSS_REQUIRED
def test_mss_backend_reset_step_and_actuator_lag() -> None:
    backend = MssRemus100Backend(sample_time_s=0.02)
    try:
        initial = backend.reset(_scenario(), seed=3)
        assert initial.state.t_s == pytest.approx(0.0)
        assert initial.state.position_ned_m == pytest.approx(np.zeros(3))
        assert initial.actuator_state == _command(rpm=0.0)

        stepped = backend.step(_command())
        assert stepped.state.t_s == pytest.approx(0.02)
        assert isinstance(stepped.actuator_state, Remus100ActuatorCommand)
        assert stepped.actuator_state.rudder_angle_rad == pytest.approx(0.0)
        assert stepped.actuator_state.propeller_rpm == pytest.approx(20.0)
        assert np.all(np.isfinite(stepped.state.linear_velocity_body_mps))
        assert stepped.state.linear_velocity_body_mps[0] != pytest.approx(1.5)
    finally:
        backend.close()


@MSS_REQUIRED
def test_mss_derivative_does_not_advance_and_matches_axis_signs() -> None:
    backend = MssRemus100Backend()
    scenario = _scenario()
    try:
        initial = backend.reset(scenario, seed=0)
        rudder_derivative = backend.derivative_nu_dot(
            _command(rudder_rad=radians(10.0))
        )
        stern_derivative = backend.derivative_nu_dot(
            _command(stern_plane_rad=radians(10.0))
        )
        snapshot = backend._snapshot()  # noqa: SLF001 - state must not change
        assert snapshot.state.t_s == pytest.approx(initial.state.t_s)
        assert snapshot.state.linear_velocity_body_mps == pytest.approx(
            initial.state.linear_velocity_body_mps
        )
        assert rudder_derivative[1] < 0.0
        assert rudder_derivative[5] > 0.0
        assert stern_derivative[2] < 0.0
        assert stern_derivative[4] < 0.0
    finally:
        backend.close()


@MSS_REQUIRED
def test_pvs_and_mss_agree_on_frame_units_and_fin_signs() -> None:
    derivative_cases = (
        ("straight", _state(), _command(), UniformCurrent(np.zeros(3)), 0.07),
        (
            "rudder",
            _state(),
            _command(rudder_rad=radians(10.0)),
            UniformCurrent(np.zeros(3)),
            0.07,
        ),
        (
            "stern",
            _state(),
            _command(stern_plane_rad=radians(10.0)),
            UniformCurrent(np.zeros(3)),
            0.07,
        ),
        (
            "combined",
            _state(euler=(0.05, -0.05, 0.3)),
            _command(rudder_rad=radians(8.0), stern_plane_rad=radians(-6.0), rpm=1100.0),
            UniformCurrent(np.zeros(3)),
            0.07,
        ),
        (
            "cross-current",
            _state(),
            _command(),
            UniformCurrent(np.array([0.0, 0.3, 0.0])),
            0.35,
        ),
    )
    for name, state, command, current, tolerance in derivative_cases:
        pvs = PvsRemus100Backend()
        mss = MssRemus100Backend()
        try:
            scenario = _scenario(state, current)
            pvs.reset(scenario, seed=0)
            mss.reset(scenario, seed=0)
            pvs_derivative = pvs.derivative_nu_dot(command)
            mss_derivative = mss.derivative_nu_dot(command)
        finally:
            mss.close()
        relative_error = float(np.linalg.norm(pvs_derivative - mss_derivative)) / max(
            float(np.linalg.norm(pvs_derivative)), 1e-12
        )
        assert relative_error < tolerance, (name, pvs_derivative, mss_derivative)
        for axis in (2, 5):
            if abs(pvs_derivative[axis]) < 1e-9:
                assert abs(mss_derivative[axis]) < 1e-8


@MSS_REQUIRED
def test_mss_backend_accepts_explicit_depth_shear_and_refreshes_it() -> None:
    current = DepthShearCurrent(
        reference_velocity_ned_mps=np.array([0.2, 0.0, 0.0]),
        shear_ned_per_m=np.array([0.1, 0.0, 0.0]),
    )
    state = _state(position=(0.0, 0.0, 10.0))
    backend = MssRemus100Backend(allow_sheared_current=True)
    try:
        snapshot = backend.reset(_scenario(state, current), seed=0)
        assert snapshot.diagnostics["current_speed_mps"] == pytest.approx(1.2)
        assert snapshot.diagnostics["current_direction_rad"] == pytest.approx(0.0)
    finally:
        backend.close()


@MSS_REQUIRED
def test_mss_backend_rejects_unsupported_current_and_limits() -> None:
    shear = DepthShearCurrent(
        reference_velocity_ned_mps=np.array([0.1, 0.0, 0.0]),
        shear_ned_per_m=np.zeros(3),
    )
    backend = MssRemus100Backend()
    try:
        with pytest.raises(ValueError, match="only UniformCurrent"):
            backend.reset(_scenario(current=shear), seed=0)
        backend.reset(_scenario(), seed=0)
        with pytest.raises(ValueError, match="rudder"):
            backend.step(_command(rudder_rad=radians(16.0)))
        with pytest.raises(ValueError, match="stern_plane"):
            backend.step(_command(stern_plane_rad=radians(16.0)))
        with pytest.raises(ValueError, match="propeller_rpm"):
            backend.step(_command(rpm=1600.0))
    finally:
        backend.close()


def test_mss_adapter_defaults_are_outside_planner_core() -> None:
    source = _default_mss_source_dir()
    script = _default_daemon_script()
    assert isinstance(source, Path)
    assert isinstance(script, Path)
    assert script.name == "mss_remus100_daemon.m"
