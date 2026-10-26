from __future__ import annotations

from dataclasses import dataclass
from math import atan2, hypot, pi

import numpy as np

from ..contracts import (
    WORLD_FRAME,
    CurrentField,
    FloatArray,
    Reference,
    Remus100ActuatorCommand,
    State,
    VehicleSpec,
)


def _finite_non_negative(value: float, name: str) -> float:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


def _finite_positive(value: float, name: str) -> float:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _wrap_angle(angle_rad: float) -> float:
    return float((angle_rad + pi) % (2.0 * pi) - pi)


def _body_x_direction(quat_xyzw: FloatArray) -> FloatArray:
    x, y, z, w = quat_xyzw
    return np.array(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
        ],
        dtype=np.float64,
    )


def _yaw_pitch(quat_xyzw: FloatArray) -> tuple[float, float]:
    direction = _body_x_direction(quat_xyzw)
    horizontal = hypot(float(direction[0]), float(direction[1]))
    yaw = atan2(float(direction[1]), float(direction[0]))
    pitch = atan2(-float(direction[2]), horizontal)
    return yaw, pitch


def _world_to_body(vector_ned: FloatArray, quat_xyzw: FloatArray) -> FloatArray:
    x, y, z, w = quat_xyzw
    rotation = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return rotation.T @ vector_ned


def _required_limit(value: float | None, name: str) -> float:
    if value is None:
        raise ValueError(f"REMUS100 tracking requires {name}")
    return float(value)


@dataclass(frozen=True)
class Remus100ControlDiagnostics:
    dt_s: float
    desired_water_speed_mps: float
    relative_surge_speed_mps: float
    speed_error_mps: float
    heading_error_rad: float
    pitch_error_rad: float
    unsaturated_rudder_angle_rad: float
    unsaturated_stern_plane_angle_rad: float
    unsaturated_propeller_rpm: float
    speed_integral_rpm: float


class Remus100TrackingController:
    """Cascaded tracking controller for the REMUS 100 PVS plant.

    Heading and pitch use PD feedback on the reference body axis. The propeller
    command uses relative-surge feedforward plus PI feedback. Commands are
    clipped to vehicle limits and fin commands are additionally slew-limited by
    the elapsed state time. The controller never silently uses a stale or future
    reference and never requests a speed outside the VehicleSpec range.
    """

    def __init__(
        self,
        vehicle: VehicleSpec,
        *,
        current_field: CurrentField | None = None,
        nominal_step_s: float = 0.02,
        heading_gain: float = 2.0,
        heading_rate_gain: float = 0.6,
        pitch_gain: float = 2.0,
        pitch_rate_gain: float = 1.0,
        speed_feedforward_rpm_per_mps: float = 600.0,
        speed_gain_rpm_per_mps: float = 80.0,
        speed_integral_gain_rpm_per_mps_s: float = 10.0,
        max_speed_integral_rpm: float = 400.0,
    ) -> None:
        if not isinstance(vehicle, VehicleSpec):
            raise TypeError("vehicle must be a VehicleSpec")
        if current_field is not None and not isinstance(current_field, CurrentField):
            raise TypeError("current_field must implement the CurrentField protocol")

        self.vehicle = vehicle
        self.current_field = current_field
        self.nominal_step_s = _finite_positive(nominal_step_s, "nominal_step_s")
        self.heading_gain = _finite_non_negative(heading_gain, "heading_gain")
        self.heading_rate_gain = _finite_non_negative(heading_rate_gain, "heading_rate_gain")
        self.pitch_gain = _finite_non_negative(pitch_gain, "pitch_gain")
        self.pitch_rate_gain = _finite_non_negative(pitch_rate_gain, "pitch_rate_gain")
        self.speed_feedforward_rpm_per_mps = _finite_non_negative(
            speed_feedforward_rpm_per_mps, "speed_feedforward_rpm_per_mps"
        )
        self.speed_gain_rpm_per_mps = _finite_non_negative(
            speed_gain_rpm_per_mps, "speed_gain_rpm_per_mps"
        )
        self.speed_integral_gain_rpm_per_mps_s = _finite_non_negative(
            speed_integral_gain_rpm_per_mps_s, "speed_integral_gain_rpm_per_mps_s"
        )
        self.max_speed_integral_rpm = _finite_non_negative(
            max_speed_integral_rpm, "max_speed_integral_rpm"
        )

        self.max_rudder_angle_rad = _required_limit(
            vehicle.max_rudder_angle_rad, "max_rudder_angle_rad"
        )
        self.max_rudder_rate_rad_s = _required_limit(
            vehicle.max_rudder_rate_rad_s, "max_rudder_rate_rad_s"
        )
        self.max_stern_plane_angle_rad = _required_limit(
            vehicle.max_stern_plane_angle_rad, "max_stern_plane_angle_rad"
        )
        self.max_stern_plane_rate_rad_s = _required_limit(
            vehicle.max_stern_plane_rate_rad_s, "max_stern_plane_rate_rad_s"
        )
        self.max_propeller_rpm = _required_limit(vehicle.max_propeller_rpm, "max_propeller_rpm")

        self.last_diagnostics: Remus100ControlDiagnostics | None = None
        self.reset()

    def reset(self) -> None:
        self._last_state_t_s: float | None = None
        self._last_rudder_angle_rad = 0.0
        self._last_stern_plane_angle_rad = 0.0
        self._speed_integral_rpm = 0.0
        self.last_diagnostics = None

    def control(self, state: State, reference: Reference) -> Remus100ActuatorCommand:
        if not isinstance(state, State):
            raise TypeError("state must be a State")
        if not isinstance(reference, Reference):
            raise TypeError("reference must be a Reference")
        if state.frame_id != WORLD_FRAME:
            raise ValueError(f"state must use frame {WORLD_FRAME}")
        if reference.stamp_s > state.t_s + 1e-9:
            raise ValueError("reference stamp is in the future")
        if reference.valid_until_s is not None and state.t_s > reference.valid_until_s + 1e-9:
            raise ValueError("reference has expired")

        desired_water_speed = float(reference.linear_velocity_body_mps[0])
        if desired_water_speed < self.vehicle.min_speed_mps - 1e-9:
            raise ValueError("reference speed is below vehicle min_speed_mps")
        if desired_water_speed > self.vehicle.max_speed_mps + 1e-9:
            raise ValueError("reference speed exceeds vehicle max_speed_mps")

        dt_s = self._time_step(state.t_s)
        desired_yaw, desired_pitch = _yaw_pitch(reference.quat_xyzw)
        actual_yaw, actual_pitch = _yaw_pitch(state.quat_xyzw)
        heading_error = _wrap_angle(desired_yaw - actual_yaw)
        pitch_error = desired_pitch - actual_pitch

        yaw_rate = float(state.angular_velocity_body_rps[2])
        pitch_rate = float(state.angular_velocity_body_rps[1])
        unsaturated_rudder = self.heading_gain * heading_error + self.heading_rate_gain * (
            float(reference.angular_velocity_body_rps[2]) - yaw_rate
        )
        unsaturated_stern_plane = -self.pitch_gain * pitch_error + self.pitch_rate_gain * (
            float(reference.angular_velocity_body_rps[1]) - pitch_rate
        )
        rudder = float(
            np.clip(
                unsaturated_rudder,
                -self.max_rudder_angle_rad,
                self.max_rudder_angle_rad,
            )
        )
        stern_plane = float(
            np.clip(
                unsaturated_stern_plane,
                -self.max_stern_plane_angle_rad,
                self.max_stern_plane_angle_rad,
            )
        )

        max_rudder_step = self.max_rudder_rate_rad_s * dt_s
        max_stern_plane_step = self.max_stern_plane_rate_rad_s * dt_s
        rudder = self._last_rudder_angle_rad + float(
            np.clip(rudder - self._last_rudder_angle_rad, -max_rudder_step, max_rudder_step)
        )
        stern_plane = self._last_stern_plane_angle_rad + float(
            np.clip(
                stern_plane - self._last_stern_plane_angle_rad,
                -max_stern_plane_step,
                max_stern_plane_step,
            )
        )

        current_ned = (
            np.zeros(3, dtype=np.float64)
            if self.current_field is None
            else np.asarray(
                self.current_field.sample(state.position_ned_m, state.t_s),
                dtype=np.float64,
            )
        )
        if current_ned.shape != (3,) or not np.all(np.isfinite(current_ned)):
            raise ValueError("current field must return a finite shape-(3,) velocity")
        current_body = _world_to_body(current_ned, state.quat_xyzw)
        relative_surge_speed = float(state.linear_velocity_body_mps[0] - current_body[0])
        speed_error = desired_water_speed - relative_surge_speed
        unsaturated_rpm = (
            self.speed_feedforward_rpm_per_mps * desired_water_speed
            + self._speed_integral_rpm
            + self.speed_gain_rpm_per_mps * speed_error
        )
        propeller_rpm = float(np.clip(unsaturated_rpm, 0.0, self.max_propeller_rpm))

        saturated_high = unsaturated_rpm > self.max_propeller_rpm and speed_error > 0.0
        saturated_low = unsaturated_rpm < 0.0 and speed_error < 0.0
        if not saturated_high and not saturated_low:
            self._speed_integral_rpm += self.speed_integral_gain_rpm_per_mps_s * speed_error * dt_s
            self._speed_integral_rpm = float(
                np.clip(
                    self._speed_integral_rpm,
                    -self.max_speed_integral_rpm,
                    self.max_speed_integral_rpm,
                )
            )

        self._last_rudder_angle_rad = rudder
        self._last_stern_plane_angle_rad = stern_plane
        self.last_diagnostics = Remus100ControlDiagnostics(
            dt_s=dt_s,
            desired_water_speed_mps=desired_water_speed,
            relative_surge_speed_mps=relative_surge_speed,
            speed_error_mps=speed_error,
            heading_error_rad=heading_error,
            pitch_error_rad=pitch_error,
            unsaturated_rudder_angle_rad=unsaturated_rudder,
            unsaturated_stern_plane_angle_rad=unsaturated_stern_plane,
            unsaturated_propeller_rpm=unsaturated_rpm,
            speed_integral_rpm=self._speed_integral_rpm,
        )
        return Remus100ActuatorCommand(
            stamp_s=state.t_s,
            rudder_angle_rad=rudder,
            stern_plane_angle_rad=stern_plane,
            propeller_rpm=propeller_rpm,
            valid_until_s=state.t_s + dt_s,
        )

    def _time_step(self, t_s: float) -> float:
        if self._last_state_t_s is None:
            dt_s = self.nominal_step_s
        else:
            dt_s = float(t_s - self._last_state_t_s)
            if not np.isfinite(dt_s) or dt_s <= 0.0:
                raise ValueError("state time must increase between controller calls")
        self._last_state_t_s = float(t_s)
        return dt_s


__all__ = ["Remus100ControlDiagnostics", "Remus100TrackingController"]
