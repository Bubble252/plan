from __future__ import annotations

import os
import sys
from functools import lru_cache
from math import atan2, hypot
from pathlib import Path
from typing import Any

import numpy as np

from ..attitude import euler_from_quaternion, quaternion_from_euler
from ..contracts import (
    WORLD_FRAME,
    ActuatorCommand,
    FloatArray,
    Remus100ActuatorCommand,
    ScenarioConfig,
    SimulationSnapshot,
    State,
    UniformCurrent,
    VehicleSpec,
    WorldSnapshot,
)


@lru_cache(maxsize=4)
def _load_pvs(source_dir_text: str) -> tuple[Any, Any]:
    source_dir = Path(source_dir_text).expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"PVS source directory does not exist: {source_dir}")
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    from python_vehicle_simulator.lib.gnc import attitudeEuler
    from python_vehicle_simulator.vehicles.remus100 import remus100

    return remus100, attitudeEuler


def _default_pvs_source_dir() -> Path:
    configured = os.environ.get("PVS_SOURCE_DIR")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[4] / "python" / "PythonVehicleSimulator-master" / "src"


def _require_actuator_limits(vehicle: VehicleSpec) -> tuple[float, float, float]:
    max_rudder = vehicle.max_rudder_angle_rad
    max_stern_plane = vehicle.max_stern_plane_angle_rad
    max_propeller_rpm = vehicle.max_propeller_rpm
    required = {
        "max_rudder_angle_rad": max_rudder,
        "max_stern_plane_angle_rad": max_stern_plane,
        "max_propeller_rpm": max_propeller_rpm,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"PVS REMUS 100 requires actuator limits: {', '.join(missing)}")
    assert max_rudder is not None
    assert max_stern_plane is not None
    assert max_propeller_rpm is not None
    return float(max_rudder), float(max_stern_plane), float(max_propeller_rpm)


class PvsRemus100Backend:
    """Stateful adapter around the Python Vehicle Simulator REMUS 100 model.

    The backend accepts a fully allocated ActuatorCommand, advances the plant by
    one fixed sample, and returns the synchronized State and WorldSnapshot.
    PVS is imported lazily only when reset() is called.
    """

    def __init__(
        self,
        sample_time_s: float = 0.02,
        pvs_source_dir: str | Path | None = None,
        allow_sheared_current: bool = False,
    ) -> None:
        if not np.isfinite(sample_time_s) or sample_time_s <= 0.0:
            raise ValueError("sample_time_s must be finite and positive")
        self.sample_time_s = float(sample_time_s)
        self.pvs_source_dir = Path(pvs_source_dir) if pvs_source_dir is not None else _default_pvs_source_dir()
        self.allow_sheared_current = bool(allow_sheared_current)
        self._model: Any = None
        self._attitude_euler: Any = None
        self._eta: FloatArray | None = None
        self._nu: FloatArray | None = None
        self._u_actual: FloatArray | None = None
        self._time_s: float | None = None
        self._scenario: ScenarioConfig | None = None
        self._max_rudder_rad = 0.0
        self._max_stern_plane_rad = 0.0
        self._max_propeller_rpm = 0.0
        self._current_speed_mps = 0.0
        self._current_direction_rad = 0.0

    def reset(self, scenario: ScenarioConfig, seed: int) -> SimulationSnapshot:
        if not isinstance(scenario, ScenarioConfig):
            raise TypeError("scenario must be a ScenarioConfig")
        if not isinstance(seed, int):
            raise TypeError("seed must be an int")
        if scenario.start.frame_id != WORLD_FRAME:
            raise ValueError(f"start state must use frame {WORLD_FRAME}")
        if scenario.vehicle.name.upper() != "REMUS100":
            raise ValueError("PvsRemus100Backend requires a REMUS100 VehicleSpec")

        self._max_rudder_rad, self._max_stern_plane_rad, self._max_propeller_rpm = (
            _require_actuator_limits(scenario.vehicle)
        )
        model_type, attitude_euler = _load_pvs(str(self.pvs_source_dir))
        self._model = model_type(
            controlSystem="stepInput",
            V_current=0.0,
            beta_current=0.0,
        )
        self._attitude_euler = attitude_euler
        euler = euler_from_quaternion(scenario.start.quat_xyzw)
        self._eta = np.concatenate((scenario.start.position_ned_m, euler)).astype(np.float64)
        self._nu = np.concatenate(
            (
                scenario.start.linear_velocity_body_mps,
                scenario.start.angular_velocity_body_rps,
            )
        ).astype(np.float64)
        self._u_actual = np.zeros(3, dtype=np.float64)
        self._time_s = float(scenario.start.t_s)
        self._scenario = scenario
        self._model.nu = self._nu.copy()
        self._model.u_actual = self._u_actual.copy()
        self._refresh_current(self._eta[0:3], self._time_s)
        return self._snapshot()

    def step(self, command: ActuatorCommand) -> SimulationSnapshot:
        if self._model is None or self._eta is None or self._nu is None or self._u_actual is None:
            raise RuntimeError("reset() must be called before step()")
        if self._attitude_euler is None or self._time_s is None or self._scenario is None:
            raise RuntimeError("reset() must be called before step()")
        if not isinstance(command, Remus100ActuatorCommand):
            raise TypeError("command must be a Remus100ActuatorCommand")
        if command.stamp_s < self._time_s - 1e-9:
            raise ValueError("command stamp precedes the current simulation time")
        if command.stamp_s > self._time_s + self.sample_time_s + 1e-9:
            raise ValueError("command stamp is beyond the next simulation step")
        self._validate_command(command)
        self._refresh_current(self._eta[0:3], self._time_s)

        control = np.array(
            [command.rudder_angle_rad, command.stern_plane_angle_rad, command.propeller_rpm],
            dtype=np.float64,
        )
        nu_next, u_actual_next = self._model.dynamics(
            self._eta.copy(),
            self._nu.copy(),
            self._u_actual.copy(),
            control,
            self.sample_time_s,
        )
        eta_next = self._attitude_euler(self._eta.copy(), nu_next.copy(), self.sample_time_s)
        self._eta = np.asarray(eta_next, dtype=np.float64).copy()
        self._nu = np.asarray(nu_next, dtype=np.float64).copy()
        self._u_actual = np.asarray(u_actual_next, dtype=np.float64).copy()
        self._model.nu = self._nu.copy()
        self._model.u_actual = self._u_actual.copy()
        self._time_s += self.sample_time_s
        return self._snapshot()

    def derivative_nu_dot(
        self,
        actual_command: Remus100ActuatorCommand | None = None,
    ) -> FloatArray:
        """Return d(nu)/dt at the current state without advancing the plant."""

        if self._model is None or self._eta is None or self._nu is None:
            raise RuntimeError("reset() must be called before derivative_nu_dot()")
        if self._u_actual is None or self._time_s is None or self._scenario is None:
            raise RuntimeError("reset() must be called before derivative_nu_dot()")
        if actual_command is None:
            actual_command = Remus100ActuatorCommand(
                stamp_s=self._time_s,
                rudder_angle_rad=float(self._u_actual[0]),
                stern_plane_angle_rad=float(self._u_actual[1]),
                propeller_rpm=float(self._u_actual[2]),
            )
        elif not isinstance(actual_command, Remus100ActuatorCommand):
            raise TypeError("actual_command must be a Remus100ActuatorCommand")
        self._validate_command(actual_command)
        self._refresh_current(self._eta[0:3], self._time_s)
        actual = np.array(
            [
                actual_command.rudder_angle_rad,
                actual_command.stern_plane_angle_rad,
                actual_command.propeller_rpm,
            ],
            dtype=np.float64,
        )
        nu_trial = self._nu.copy()
        self._model.dynamics(self._eta.copy(), nu_trial, actual.copy(), actual.copy(), 1.0)
        return np.asarray(nu_trial - self._nu, dtype=np.float64)

    def mass_matrix_diagonal(self) -> FloatArray:
        if self._model is None:
            raise RuntimeError("reset() must be called before mass_matrix_diagonal()")
        return np.asarray(np.diag(self._model.M), dtype=np.float64).copy()

    def _validate_command(self, command: Remus100ActuatorCommand) -> None:
        if abs(command.rudder_angle_rad) > self._max_rudder_rad + 1e-12:
            raise ValueError("rudder_angle_rad exceeds the vehicle limit")
        if abs(command.stern_plane_angle_rad) > self._max_stern_plane_rad + 1e-12:
            raise ValueError("stern_plane_angle_rad exceeds the vehicle limit")
        if abs(command.propeller_rpm) > self._max_propeller_rpm + 1e-12:
            raise ValueError("propeller_rpm exceeds the vehicle limit")

    def _snapshot(self) -> SimulationSnapshot:
        assert self._scenario is not None
        assert self._eta is not None
        assert self._nu is not None
        assert self._u_actual is not None
        assert self._time_s is not None
        state = State(
            t_s=self._time_s,
            position_ned_m=self._eta[0:3].copy(),
            quat_xyzw=quaternion_from_euler(*self._eta[3:6]),
            linear_velocity_body_mps=self._nu[0:3].copy(),
            angular_velocity_body_rps=self._nu[3:6].copy(),
            frame_id=WORLD_FRAME,
        )
        world = WorldSnapshot(
            t_s=self._time_s,
            obstacle_map=self._scenario.world.obstacle_map,
            current_field=self._scenario.world.current_field,
        )
        actuator_state = Remus100ActuatorCommand(
            stamp_s=self._time_s,
            rudder_angle_rad=float(self._u_actual[0]),
            stern_plane_angle_rad=float(self._u_actual[1]),
            propeller_rpm=float(self._u_actual[2]),
        )
        return SimulationSnapshot(
            state=state,
            world=world,
            actuator_state=actuator_state,
            diagnostics={
                "sample_time_s": self.sample_time_s,
                "current_speed_mps": self._current_speed_mps,
                "current_direction_rad": self._current_direction_rad,
            },
        )

    def _refresh_current(self, position_ned_m: FloatArray, t_s: float) -> None:
        assert self._scenario is not None
        assert self._model is not None
        field = self._scenario.world.current_field
        if not isinstance(field, UniformCurrent) and not self.allow_sheared_current:
            raise ValueError("PVS REMUS 100 backend supports only UniformCurrent")
        velocity = field.sample(position_ned_m, t_s)
        if abs(float(velocity[2])) > 1e-9:
            raise ValueError("PVS REMUS 100 supports only horizontal current; vertical component is not modeled")
        speed = float(hypot(velocity[0], velocity[1]))
        direction_rad = float(atan2(float(velocity[1]), float(velocity[0]))) if speed > 1e-12 else 0.0
        self._current_speed_mps = speed
        self._current_direction_rad = direction_rad
        self._model.V_c = speed
        self._model.beta_c = direction_rad


__all__ = ["PvsRemus100Backend"]
