from __future__ import annotations

from dataclasses import dataclass
from math import atan2, ceil, cos, hypot, pi, sin

import numpy as np

from ..attitude import euler_from_quaternion, quaternion_from_euler
from ..contracts import FloatArray, State, Trajectory, VehicleSpec, WorldSnapshot
from ..geometry import hull_clearance_details, hull_from_vehicle, point_clearance_details
from .models import (
    ClearanceModel,
    ConstraintMode,
    FaultSpec,
    ManeuverCandidate,
    ManeuverKind,
    RecoveryConfig,
)


def _wrap_angle(angle_rad: float) -> float:
    return float((angle_rad + pi) % (2.0 * pi) - pi)


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


def _forward_direction(yaw_rad: float, pitch_rad: float) -> FloatArray:
    return np.array(
        [
            cos(pitch_rad) * cos(yaw_rad),
            cos(pitch_rad) * sin(yaw_rad),
            -sin(pitch_rad),
        ],
        dtype=np.float64,
    )


def _water_relative_surge_mps(state: State, current_ned: FloatArray) -> float:
    current_body = _world_to_body(current_ned, state.quat_xyzw)
    return max(0.0, float(state.linear_velocity_body_mps[0] - current_body[0]))


def _effective_speed_range(vehicle: VehicleSpec, fault: FaultSpec) -> tuple[float, float]:
    effectiveness = fault.propeller_effectiveness
    return vehicle.min_speed_mps * effectiveness, vehicle.max_speed_mps * effectiveness


def _effective_accel(vehicle: VehicleSpec, fault: FaultSpec) -> float:
    effectiveness = fault.propeller_effectiveness
    return float(vehicle.max_accel_mps2 * max(effectiveness, 0.25))


@dataclass(frozen=True)
class _Profile:
    target_speed_mps: float
    target_yaw_rad: float
    target_pitch_rad: float


def _simulate_profile(
    state: State,
    world: WorldSnapshot,
    vehicle: VehicleSpec,
    fault: FaultSpec,
    config: RecoveryConfig,
    *,
    duration_s: float,
    profile: _Profile,
    trajectory_id: str,
    yaw_limit_override: float | None = None,
) -> tuple[Trajectory, tuple[str, ...]]:
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")

    euler = euler_from_quaternion(state.quat_xyzw)
    yaw = float(euler[2])
    pitch = float(euler[1])
    position = state.position_ned_m.copy()
    time_s = float(state.t_s)
    initial_current = world.current_field.sample(position, time_s)
    speed = _water_relative_surge_mps(state, initial_current)
    min_speed, max_speed = _effective_speed_range(vehicle, fault)
    max_accel = _effective_accel(vehicle, fault)
    max_yaw_rate = vehicle.max_yaw_rate_rps
    max_pitch_rate = vehicle.max_pitch_rate_rps
    if config.constraint_mode is ConstraintMode.UNBOUNDED_TURN_DIAGNOSTIC:
        yaw_limit = vehicle.max_yaw_rate_rps * config.unbounded_yaw_rate_factor
    else:
        yaw_limit = vehicle.max_yaw_rate_rps
    if yaw_limit_override is not None:
        yaw_limit = float(yaw_limit_override)
    forced_yaw_rate = vehicle.max_yaw_rate_rps * fault.forced_yaw_rate_fraction
    forced_pitch_rate = vehicle.max_pitch_rate_rps * fault.forced_pitch_rate_fraction
    pitch_limit = pi / 2.0 - 1e-3

    times: list[float] = [time_s]
    positions: list[FloatArray] = [position.copy()]
    quats: list[FloatArray] = [quaternion_from_euler(0.0, pitch, yaw)]
    velocities: list[FloatArray] = [np.array([speed, 0.0, 0.0], dtype=np.float64)]
    angular_velocities: list[FloatArray] = [np.zeros(3, dtype=np.float64)]
    controls: list[FloatArray] = [np.array([speed, 0.0, 0.0], dtype=np.float64)]
    exceedances: set[str] = set()
    max_accel_observed = 0.0

    step_count = max(1, int(ceil(duration_s / config.sample_time_s)))
    for _ in range(step_count):
        remaining = duration_s - (time_s - float(state.t_s))
        if remaining <= 1e-12:
            break
        dt_s = min(config.sample_time_s, remaining)

        target_speed = float(np.clip(profile.target_speed_mps, min_speed, max_speed))
        speed_step = float(np.clip(target_speed - speed, -max_accel * dt_s, max_accel * dt_s))
        speed += speed_step
        max_accel_observed = max(max_accel_observed, abs(speed_step) / dt_s)

        yaw_error = _wrap_angle(profile.target_yaw_rad - yaw)
        if abs(yaw_error) <= 1e-6:
            commanded_yaw_rate = 0.0
        else:
            commanded_yaw_rate = float(np.sign(yaw_error) * yaw_limit)
        pitch_error = profile.target_pitch_rad - pitch
        if abs(pitch_error) <= 1e-6:
            commanded_pitch_rate = 0.0
        else:
            commanded_pitch_rate = float(np.sign(pitch_error) * max_pitch_rate)

        yaw_rate = forced_yaw_rate if forced_yaw_rate != 0.0 else commanded_yaw_rate
        pitch_rate = forced_pitch_rate if forced_pitch_rate != 0.0 else commanded_pitch_rate
        if pitch >= pitch_limit and pitch_rate > 0.0:
            pitch_rate = 0.0
        if pitch <= -pitch_limit and pitch_rate < 0.0:
            pitch_rate = 0.0

        if abs(yaw_rate) > max_yaw_rate + 1e-12:
            exceedances.add("vehicle_yaw_rate")
        if abs(pitch_rate) > max_pitch_rate + 1e-12:
            exceedances.add("vehicle_pitch_rate")
        if target_speed > vehicle.max_speed_mps + 1e-12:
            exceedances.add("vehicle_speed")

        current_ned = world.current_field.sample(position, time_s)
        ground_velocity = speed * _forward_direction(yaw, pitch) + current_ned
        position = position + ground_velocity * dt_s
        yaw = _wrap_angle(yaw + yaw_rate * dt_s)
        next_pitch = pitch + pitch_rate * dt_s
        if forced_pitch_rate == 0.0:
            if commanded_pitch_rate > 0.0:
                next_pitch = min(next_pitch, profile.target_pitch_rad)
            elif commanded_pitch_rate < 0.0:
                next_pitch = max(next_pitch, profile.target_pitch_rad)
        pitch = float(np.clip(next_pitch, -pitch_limit, pitch_limit))
        time_s += dt_s

        times.append(time_s)
        positions.append(position.copy())
        quats.append(quaternion_from_euler(0.0, pitch, yaw))
        velocities.append(np.array([speed, 0.0, 0.0], dtype=np.float64))
        angular_velocities.append(np.array([0.0, pitch_rate, yaw_rate], dtype=np.float64))
        controls.append(
            np.array([target_speed, commanded_yaw_rate, commanded_pitch_rate], dtype=np.float64)
        )

    controls_array = np.vstack(controls)
    if max_accel_observed > max_accel * 1.05 + 1e-9:
        exceedances.add("vehicle_acceleration")

    trajectory = Trajectory(
        trajectory_id=trajectory_id,
        t_s=np.asarray(times, dtype=np.float64),
        position_ned_m=np.vstack(positions),
        quat_xyzw=np.vstack(quats),
        linear_velocity_body_mps=np.vstack(velocities),
        angular_velocity_body_rps=np.vstack(angular_velocities),
        controls=controls_array,
        feasibility="kinodynamically_feasible" if not exceedances else "unchecked",
    )
    return trajectory, tuple(sorted(exceedances))


def coast_trajectory(
    state: State,
    world: WorldSnapshot,
    vehicle: VehicleSpec,
    fault: FaultSpec,
    config: RecoveryConfig,
    *,
    duration_s: float,
) -> tuple[Trajectory, tuple[str, ...]]:
    """Model the undiagnosed fault period as constant commanded heading/speed."""

    current = world.current_field.sample(state.position_ned_m, state.t_s)
    water_speed = _water_relative_surge_mps(state, current)
    euler = euler_from_quaternion(state.quat_xyzw)
    profile = _Profile(
        target_speed_mps=water_speed,
        target_yaw_rad=float(euler[2]),
        target_pitch_rad=float(euler[1]),
    )
    return _simulate_profile(
        state,
        world,
        vehicle,
        fault,
        config,
        duration_s=duration_s,
        profile=profile,
        trajectory_id=f"coast_{fault.name}_{duration_s:g}s",
    )


def _best_corridor_target(
    state: State,
    world: WorldSnapshot,
    vehicle: VehicleSpec,
    fault: FaultSpec,
    config: RecoveryConfig,
) -> FloatArray:
    min_speed, _ = _effective_speed_range(vehicle, fault)
    max_distance = max(1.0, min(4.0, config.horizon_s * min_speed))
    best_score = float("-inf")
    best_target = state.position_ned_m.copy()
    hull = hull_from_vehicle(vehicle)
    for azimuth_index in range(16):
        azimuth = 2.0 * pi * azimuth_index / 16.0
        for elevation in (-config.max_pitch_angle_rad, 0.0, config.max_pitch_angle_rad):
            direction = _forward_direction(azimuth, elevation)
            target = state.position_ned_m + max_distance * direction
            target_euler_yaw = azimuth
            target_euler_pitch = elevation
            target_state = State(
                t_s=state.t_s,
                position_ned_m=target,
                quat_xyzw=quaternion_from_euler(0.0, target_euler_pitch, target_euler_yaw),
                linear_velocity_body_mps=np.zeros(3),
                angular_velocity_body_rps=np.zeros(3),
            )
            if config.clearance_model is ClearanceModel.FULL_HULL:
                details = hull_clearance_details(world.obstacle_map, target_state, hull)
            else:
                details = point_clearance_details(world.obstacle_map, target, 0.0)
            score = details.clearance_m
            if score > best_score + 1e-12:
                best_score = score
                best_target = target.copy()
    return best_target


def _target_profile(
    state: State,
    target_ned_m: FloatArray,
    *,
    target_speed_mps: float,
    max_pitch_angle_rad: float,
) -> _Profile:
    delta = target_ned_m - state.position_ned_m
    horizontal = hypot(float(delta[0]), float(delta[1]))
    yaw = (
        atan2(float(delta[1]), float(delta[0]))
        if horizontal > 1e-9
        else float(euler_from_quaternion(state.quat_xyzw)[2])
    )
    pitch = atan2(-float(delta[2]), horizontal) if horizontal > 0.0 else 0.0
    if abs(float(delta[2])) > 1e-9 and horizontal <= 1e-9:
        pitch = pi / 2.0 if delta[2] < 0.0 else -pi / 2.0
    return _Profile(
        target_speed_mps=target_speed_mps,
        target_yaw_rad=yaw,
        target_pitch_rad=float(np.clip(pitch, -max_pitch_angle_rad, max_pitch_angle_rad)),
    )


def _exit_current_profile(
    state: State,
    world: WorldSnapshot,
    vehicle: VehicleSpec,
    fault: FaultSpec,
    config: RecoveryConfig,
) -> tuple[_Profile, tuple[str, ...]]:
    """Best-effort current escape: seek a lower-current depth, else run downstream."""

    min_speed, max_speed = _effective_speed_range(vehicle, fault)
    position = state.position_ned_m
    current = world.current_field.sample(position, state.t_s)
    up_position = position.copy()
    up_position[2] -= 2.0
    down_position = position.copy()
    down_position[2] += 2.0
    up_speed = float(np.linalg.norm(world.current_field.sample(up_position, state.t_s)))
    down_speed = float(np.linalg.norm(world.current_field.sample(down_position, state.t_s)))
    if abs(up_speed - down_speed) > 1e-6:
        target_pitch = (
            config.max_pitch_angle_rad if up_speed < down_speed else -config.max_pitch_angle_rad
        )
        yaw = float(euler_from_quaternion(state.quat_xyzw)[2])
        return _Profile(max_speed, yaw, target_pitch), ("propeller", "pitch_fin")
    horizontal_speed = hypot(float(current[0]), float(current[1]))
    if horizontal_speed > 1e-6:
        yaw = atan2(float(current[1]), float(current[0]))
        return _Profile(max_speed, yaw, 0.0), ("propeller", "yaw_fin")
    yaw = float(euler_from_quaternion(state.quat_xyzw)[2])
    return _Profile(max_speed, yaw, 0.0), ("propeller",)


def build_maneuvers(
    state: State,
    world: WorldSnapshot,
    vehicle: VehicleSpec,
    fault: FaultSpec,
    config: RecoveryConfig,
    *,
    duration_s: float,
    safe_anchor_ned_m: FloatArray | None = None,
) -> tuple[ManeuverCandidate, ...]:
    if not isinstance(state, State):
        raise TypeError("state must be a State")
    if not isinstance(world, WorldSnapshot):
        raise TypeError("world must be a WorldSnapshot")
    if not isinstance(vehicle, VehicleSpec):
        raise TypeError("vehicle must be a VehicleSpec")
    if not isinstance(fault, FaultSpec):
        raise TypeError("fault must be a FaultSpec")
    if not isinstance(config, RecoveryConfig):
        raise TypeError("config must be a RecoveryConfig")
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")

    euler = euler_from_quaternion(state.quat_xyzw)
    yaw = float(euler[2])
    min_speed, max_speed = _effective_speed_range(vehicle, fault)
    candidates: list[ManeuverCandidate] = []

    def add(
        name: str,
        kind: ManeuverKind,
        profile: _Profile,
        requires: tuple[str, ...],
        yaw_limit_override: float | None = None,
    ) -> None:
        trajectory, exceedances = _simulate_profile(
            state,
            world,
            vehicle,
            fault,
            config,
            duration_s=duration_s,
            profile=profile,
            trajectory_id=name,
            yaw_limit_override=yaw_limit_override,
        )
        candidates.append(
            ManeuverCandidate(
                kind=kind,
                name=name,
                trajectory=trajectory,
                requires_actuators=requires,
                vehicle_limit_exceedances=exceedances,
            )
        )

    def add_yaw_maneuver(
        name: str,
        kind: ManeuverKind,
        profile: _Profile,
        requires: tuple[str, ...],
    ) -> None:
        add(name, kind, profile, requires)
        if config.constraint_mode is ConstraintMode.UNBOUNDED_TURN_DIAGNOSTIC:
            add(
                name + "_vehicle_rate",
                kind,
                profile,
                requires,
                yaw_limit_override=vehicle.max_yaw_rate_rps,
            )

    add(
        "brake_to_min_speed",
        ManeuverKind.BRAKE,
        _Profile(min_speed, yaw, 0.0),
        ("propeller",),
    )
    add(
        "ascend",
        ManeuverKind.VERTICAL_ESCAPE,
        _Profile(min_speed, yaw, config.max_pitch_angle_rad),
        ("propeller", "pitch_fin"),
    )
    add(
        "descend",
        ManeuverKind.VERTICAL_ESCAPE,
        _Profile(min_speed, yaw, -config.max_pitch_angle_rad),
        ("propeller", "pitch_fin"),
    )
    add_yaw_maneuver(
        "minimum_radius_turn_left",
        ManeuverKind.MINIMUM_RADIUS_TURN,
        _Profile(min_speed, _wrap_angle(yaw + pi - 1e-6), 0.0),
        ("propeller", "yaw_fin"),
    )
    add_yaw_maneuver(
        "minimum_radius_turn_right",
        ManeuverKind.MINIMUM_RADIUS_TURN,
        _Profile(min_speed, _wrap_angle(yaw - pi + 1e-6), 0.0),
        ("propeller", "yaw_fin"),
    )
    exit_profile, exit_actuators = _exit_current_profile(state, world, vehicle, fault, config)
    if "yaw_fin" in exit_actuators:
        add_yaw_maneuver("exit_current", ManeuverKind.EXIT_CURRENT, exit_profile, exit_actuators)
    else:
        add("exit_current", ManeuverKind.EXIT_CURRENT, exit_profile, exit_actuators)

    target = (
        np.asarray(safe_anchor_ned_m, dtype=np.float64)
        if safe_anchor_ned_m is not None
        else _best_corridor_target(state, world, vehicle, fault, config)
    )
    add_yaw_maneuver(
        "enter_safe_corridor",
        ManeuverKind.SAFE_CORRIDOR,
        _target_profile(
            state,
            target,
            target_speed_mps=min_speed,
            max_pitch_angle_rad=config.max_pitch_angle_rad,
        ),
        ("propeller", "yaw_fin", "pitch_fin"),
    )
    assert max_speed >= min_speed
    return tuple(candidates)


__all__ = ["build_maneuvers", "coast_trajectory"]
