from __future__ import annotations

from math import atan2, hypot
import os
from pathlib import Path
import select
import subprocess
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


def _default_mss_source_dir() -> Path:
    configured = os.environ.get("MSS_SOURCE_DIR")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[4] / "simulator" / "MSS-master"


def _default_daemon_script() -> Path:
    return Path(__file__).resolve().parents[3] / "scripts" / "octave" / "mss_remus100_daemon.m"


def _require_actuator_limits(vehicle: VehicleSpec) -> tuple[float, float, float]:
    if vehicle.max_rudder_angle_rad is None:
        raise ValueError("MSS REMUS 100 requires max_rudder_angle_rad")
    if vehicle.max_stern_plane_angle_rad is None:
        raise ValueError("MSS REMUS 100 requires max_stern_plane_angle_rad")
    if vehicle.max_propeller_rpm is None:
        raise ValueError("MSS REMUS 100 requires max_propeller_rpm")
    return (
        float(vehicle.max_rudder_angle_rad),
        float(vehicle.max_stern_plane_angle_rad),
        float(vehicle.max_propeller_rpm),
    )


class _MssRemus100Process:
    """Line-oriented subprocess bridge to the unmodified MSS REMUS model."""

    def __init__(
        self,
        mss_source_dir: Path,
        daemon_script: Path,
        *,
        octave_binary: str = "octave",
        command_timeout_s: float = 10.0,
    ) -> None:
        self.mss_source_dir = mss_source_dir
        self.daemon_script = daemon_script
        self.octave_binary = octave_binary
        self.command_timeout_s = float(command_timeout_s)
        self._process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        if self._process is not None:
            return
        if not self.mss_source_dir.is_dir():
            raise FileNotFoundError(f"MSS source directory does not exist: {self.mss_source_dir}")
        if not self.daemon_script.is_file():
            raise FileNotFoundError(f"MSS daemon script does not exist: {self.daemon_script}")
        environment = os.environ.copy()
        environment["MSS_ROOT"] = str(self.mss_source_dir)
        self._process = subprocess.Popen(
            [
                self.octave_binary,
                "--no-gui",
                "--quiet",
                str(self.daemon_script),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=environment,
        )
        response = self._read_line(timeout_s=max(self.command_timeout_s, 30.0))
        if response != "READY":
            self.close()
            raise RuntimeError(f"MSS daemon did not become ready: {response!r}")

    def reset(self, state_12: FloatArray) -> None:
        state = self._state_vector(state_12)
        self._send("INIT " + " ".join(f"{value:.17g}" for value in state))
        self._send("ACT 0 0 0")

    def set_integrator(self, integrator: str) -> None:
        if integrator not in {"rk4", "euler"}:
            raise ValueError("integrator must be 'rk4' or 'euler'")
        self._send(f"INTEG {integrator}")

    def step(
        self,
        rudder_angle_rad: float,
        stern_plane_angle_rad: float,
        propeller_rpm: float,
        current_speed_mps: float,
        current_direction_rad: float,
        vertical_current_mps: float,
        sample_time_s: float,
    ) -> tuple[FloatArray, float, FloatArray]:
        values = (
            rudder_angle_rad,
            stern_plane_angle_rad,
            propeller_rpm,
            current_speed_mps,
            current_direction_rad,
            vertical_current_mps,
            sample_time_s,
        )
        result = self._send("STEP " + " ".join(f"{value:.17g}" for value in values))
        return self._unpack_result(result)

    def derivative(
        self,
        actual_command: Remus100ActuatorCommand,
        current_speed_mps: float,
        current_direction_rad: float,
        vertical_current_mps: float,
    ) -> FloatArray:
        self._send(
            "ACT "
            + " ".join(
                f"{value:.17g}"
                for value in (
                    actual_command.rudder_angle_rad,
                    actual_command.stern_plane_angle_rad,
                    actual_command.propeller_rpm,
                )
            )
        )
        values = (current_speed_mps, current_direction_rad, vertical_current_mps)
        result = self._send("DERIV " + " ".join(f"{value:.17g}" for value in values))
        state, _speed, _actual = self._unpack_result(result)
        return state[0:6]

    def mass_matrix_diagonal(self) -> FloatArray:
        result = self._send("MASS")
        if result is None:
            raise RuntimeError("MSS daemon returned no mass matrix")
        values = np.fromstring(result.removeprefix("MASS "), sep=" ", dtype=np.float64)
        if values.shape != (6,):
            raise RuntimeError(f"MSS daemon returned {values.size} mass values, expected 6")
        return values[0:6].copy()

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            if process.poll() is None and process.stdin is not None:
                process.stdin.write("QUIT\n")
                process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()

    def _send(self, line: str) -> str | None:
        self.start()
        assert self._process is not None
        assert self._process.stdin is not None
        self._process.stdin.write(line + "\n")
        self._process.stdin.flush()
        response = self._read_line(self.command_timeout_s)
        if response == "OK":
            return None
        if response.startswith("RESULT ") or response.startswith("MASS "):
            return response
        if response.startswith("error:"):
            raise RuntimeError(f"MSS daemon error: {response}")
        raise RuntimeError(f"Unexpected MSS daemon response: {response!r}")

    def _read_line(self, timeout_s: float) -> str:
        assert self._process is not None
        assert self._process.stdout is not None
        ready, _, _ = select.select([self._process.stdout], [], [], timeout_s)
        if not ready:
            return_code = self._process.poll()
            stderr = ""
            if return_code is not None and self._process.stderr is not None:
                stderr = self._process.stderr.read().strip()
            raise TimeoutError(
                f"MSS daemon did not respond within {timeout_s:.1f} s; "
                f"return_code={return_code}; stderr={stderr!r}"
            )
        line = self._process.stdout.readline()
        if line == "":
            return_code = self._process.poll()
            stderr = ""
            if self._process.stderr is not None:
                stderr = self._process.stderr.read().strip()
            raise RuntimeError(
                f"MSS daemon closed its output; return_code={return_code}; stderr={stderr!r}"
            )
        return line.strip()

    @staticmethod
    def _state_vector(state_12: FloatArray) -> FloatArray:
        state = np.asarray(state_12, dtype=np.float64)
        if state.shape != (12,):
            raise ValueError(f"state_12 must have shape (12,), got {state.shape}")
        if not np.all(np.isfinite(state)):
            raise ValueError("state_12 must contain only finite values")
        return state

    @classmethod
    def _unpack_result(cls, response: str | None) -> tuple[FloatArray, float, FloatArray]:
        if response is None:
            raise RuntimeError("MSS daemon returned no result vector")
        values = np.fromstring(response.removeprefix("RESULT "), sep=" ", dtype=np.float64)
        if values.shape != (16,):
            raise RuntimeError(f"MSS daemon returned {values.size} values, expected 16")
        if not np.all(np.isfinite(values)):
            raise RuntimeError("MSS daemon returned non-finite values")
        return values[0:12].copy(), float(values[12]), values[13:16].copy()


class MssRemus100Backend:
    """Stateful adapter around MSS 2.0's REMUS 100 derivative function.

    MSS supplies a state derivative rather than an integrated plant. The daemon
    integrates it with RK4 and applies the same external first-order fin and
    propeller model used by PVS. This class keeps Octave and MSS paths out of
    the planner core and exposes the standard simulator contract.
    """

    def __init__(
        self,
        sample_time_s: float = 0.02,
        mss_source_dir: str | Path | None = None,
        daemon_script: str | Path | None = None,
        *,
        octave_binary: str = "octave",
        command_timeout_s: float = 10.0,
        allow_sheared_current: bool = False,
        integrator: str = "rk4",
    ) -> None:
        if not np.isfinite(sample_time_s) or sample_time_s <= 0.0:
            raise ValueError("sample_time_s must be finite and positive")
        if not np.isfinite(command_timeout_s) or command_timeout_s <= 0.0:
            raise ValueError("command_timeout_s must be finite and positive")
        if integrator not in {"rk4", "euler"}:
            raise ValueError("integrator must be 'rk4' or 'euler'")
        self.sample_time_s = float(sample_time_s)
        self.mss_source_dir = (
            Path(mss_source_dir) if mss_source_dir is not None else _default_mss_source_dir()
        )
        script = Path(daemon_script) if daemon_script is not None else _default_daemon_script()
        self.allow_sheared_current = bool(allow_sheared_current)
        self.integrator = integrator
        self._process = _MssRemus100Process(
            self.mss_source_dir,
            script,
            octave_binary=octave_binary,
            command_timeout_s=command_timeout_s,
        )
        self._state_12: FloatArray | None = None
        self._actuator_state: FloatArray | None = None
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
            raise ValueError("MssRemus100Backend requires a REMUS100 VehicleSpec")

        self._max_rudder_rad, self._max_stern_plane_rad, self._max_propeller_rpm = (
            _require_actuator_limits(scenario.vehicle)
        )
        euler = euler_from_quaternion(scenario.start.quat_xyzw)
        self._state_12 = np.concatenate(
            (
                scenario.start.linear_velocity_body_mps,
                scenario.start.angular_velocity_body_rps,
                scenario.start.position_ned_m,
                euler,
            )
        ).astype(np.float64)
        self._actuator_state = np.zeros(3, dtype=np.float64)
        self._time_s = float(scenario.start.t_s)
        self._scenario = scenario
        self._process.reset(self._state_12)
        self._process.set_integrator(self.integrator)
        self._refresh_current(self._state_12[6:9], self._time_s)
        return self._snapshot()

    def step(self, command: ActuatorCommand) -> SimulationSnapshot:
        if self._state_12 is None or self._actuator_state is None:
            raise RuntimeError("reset() must be called before step()")
        if self._time_s is None or self._scenario is None:
            raise RuntimeError("reset() must be called before step()")
        if not isinstance(command, Remus100ActuatorCommand):
            raise TypeError("command must be a Remus100ActuatorCommand")
        if command.stamp_s < self._time_s - 1e-9:
            raise ValueError("command stamp precedes the current simulation time")
        if command.stamp_s > self._time_s + self.sample_time_s + 1e-9:
            raise ValueError("command stamp is beyond the next simulation step")
        self._validate_command(command)
        self._refresh_current(self._state_12[6:9], self._time_s)
        state, _speed, actuator_state = self._process.step(
            command.rudder_angle_rad,
            command.stern_plane_angle_rad,
            command.propeller_rpm,
            self._current_speed_mps,
            self._current_direction_rad,
            0.0,
            self.sample_time_s,
        )
        self._state_12 = state
        self._actuator_state = actuator_state
        self._time_s += self.sample_time_s
        return self._snapshot()

    def derivative_nu_dot(
        self,
        actual_command: Remus100ActuatorCommand | None = None,
    ) -> FloatArray:
        """Return d(nu)/dt at the current state without advancing the plant."""

        if self._state_12 is None or self._time_s is None or self._scenario is None:
            raise RuntimeError("reset() must be called before derivative_nu_dot()")
        if actual_command is None:
            if self._actuator_state is None:
                raise RuntimeError("reset() must be called before derivative_nu_dot()")
            actual_command = Remus100ActuatorCommand(
                stamp_s=self._time_s,
                rudder_angle_rad=float(self._actuator_state[0]),
                stern_plane_angle_rad=float(self._actuator_state[1]),
                propeller_rpm=float(self._actuator_state[2]),
            )
        elif not isinstance(actual_command, Remus100ActuatorCommand):
            raise TypeError("actual_command must be a Remus100ActuatorCommand")
        self._validate_command(actual_command)
        self._refresh_current(self._state_12[6:9], self._time_s)
        return self._process.derivative(
            actual_command,
            self._current_speed_mps,
            self._current_direction_rad,
            0.0,
        )

    def mass_matrix_diagonal(self) -> FloatArray:
        if self._scenario is None:
            raise RuntimeError("reset() must be called before mass_matrix_diagonal()")
        return self._process.mass_matrix_diagonal()

    def close(self) -> None:
        self._process.close()

    def __enter__(self) -> MssRemus100Backend:
        self._process.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _validate_command(self, command: Remus100ActuatorCommand) -> None:
        if abs(command.rudder_angle_rad) > self._max_rudder_rad + 1e-12:
            raise ValueError("rudder_angle_rad exceeds the vehicle limit")
        if abs(command.stern_plane_angle_rad) > self._max_stern_plane_rad + 1e-12:
            raise ValueError("stern_plane_angle_rad exceeds the vehicle limit")
        if abs(command.propeller_rpm) > self._max_propeller_rpm + 1e-12:
            raise ValueError("propeller_rpm exceeds the vehicle limit")

    def _refresh_current(self, position_ned_m: FloatArray, t_s: float) -> None:
        assert self._scenario is not None
        field = self._scenario.world.current_field
        if not isinstance(field, UniformCurrent) and not self.allow_sheared_current:
            raise ValueError("MSS REMUS 100 backend supports only UniformCurrent by default")
        velocity = np.asarray(field.sample(position_ned_m, t_s), dtype=np.float64)
        if velocity.shape != (3,) or not np.all(np.isfinite(velocity)):
            raise ValueError("current field must return a finite shape-(3,) velocity")
        if abs(float(velocity[2])) > 1e-9:
            raise ValueError("M5 MSS comparison supports horizontal current only")
        self._current_speed_mps = float(hypot(float(velocity[0]), float(velocity[1])))
        self._current_direction_rad = (
            float(atan2(float(velocity[1]), float(velocity[0])))
            if self._current_speed_mps > 1e-12
            else 0.0
        )

    def _snapshot(self) -> SimulationSnapshot:
        assert self._scenario is not None
        assert self._state_12 is not None
        assert self._actuator_state is not None
        assert self._time_s is not None
        state = State(
            t_s=self._time_s,
            position_ned_m=self._state_12[6:9].copy(),
            quat_xyzw=quaternion_from_euler(*self._state_12[9:12]),
            linear_velocity_body_mps=self._state_12[0:3].copy(),
            angular_velocity_body_rps=self._state_12[3:6].copy(),
            frame_id=WORLD_FRAME,
        )
        world = WorldSnapshot(
            t_s=self._time_s,
            obstacle_map=self._scenario.world.obstacle_map,
            current_field=self._scenario.world.current_field,
        )
        actuator_state = Remus100ActuatorCommand(
            stamp_s=self._time_s,
            rudder_angle_rad=float(self._actuator_state[0]),
            stern_plane_angle_rad=float(self._actuator_state[1]),
            propeller_rpm=float(self._actuator_state[2]),
        )
        return SimulationSnapshot(
            state=state,
            world=world,
            actuator_state=actuator_state,
            diagnostics={
                "sample_time_s": self.sample_time_s,
                "current_speed_mps": self._current_speed_mps,
                "current_direction_rad": self._current_direction_rad,
                "ground_speed_mps": float(np.linalg.norm(self._state_12[0:3])),
            },
        )


__all__ = ["MssRemus100Backend"]
