from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

WORLD_FRAME = "world_ned"
BODY_FRAME = "body_frd"


def _vector(value: Any, size: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _rows(value: Any, rows: int, cols: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (rows, cols):
        raise ValueError(f"{name} must have shape ({rows}, {cols}), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _row_vector(value: Any, rows: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (rows,):
        raise ValueError(f"{name} must have shape ({rows},), got {array.shape}")
    return array


def _time_array(value: Any) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.shape[0] < 2:
        raise ValueError("t_s must be a one-dimensional array with at least two samples")
    if not np.all(np.isfinite(array)):
        raise ValueError("t_s must contain only finite values")
    if np.any(np.diff(array) <= 0.0):
        raise ValueError("t_s must be strictly increasing")
    return array


def _quat_norm(value: FloatArray, name: str) -> FloatArray:
    norms = np.linalg.norm(value, axis=-1)
    if np.any(norms <= 0.0) or np.any(np.abs(norms - 1.0) > 1e-6):
        raise ValueError(f"{name} must contain unit quaternions")
    return value


def _non_negative(value: float, name: str) -> float:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


def _positive(value: float, name: str) -> float:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _optional_positive(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    return _positive(value, name)


@dataclass(frozen=True)
class VehicleSpec:
    name: str
    length_m: float
    diameter_m: float
    mass_kg: float
    min_speed_mps: float
    max_speed_mps: float
    max_accel_mps2: float
    max_yaw_rate_rps: float
    max_pitch_rate_rps: float
    max_rudder_angle_rad: float | None = None
    max_rudder_rate_rad_s: float | None = None
    max_stern_plane_angle_rad: float | None = None
    max_stern_plane_rate_rad_s: float | None = None
    max_propeller_rpm: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        for name in (
            "length_m",
            "diameter_m",
            "mass_kg",
            "min_speed_mps",
            "max_speed_mps",
            "max_accel_mps2",
            "max_yaw_rate_rps",
            "max_pitch_rate_rps",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.min_speed_mps > self.max_speed_mps:
            raise ValueError("min_speed_mps must not exceed max_speed_mps")
        if self.diameter_m > self.length_m:
            raise ValueError("diameter_m must not exceed length_m")
        for name in (
            "max_rudder_angle_rad",
            "max_rudder_rate_rad_s",
            "max_stern_plane_angle_rad",
            "max_stern_plane_rate_rad_s",
            "max_propeller_rpm",
        ):
            object.__setattr__(self, name, _optional_positive(getattr(self, name), name))


@dataclass(frozen=True)
class State:
    t_s: float
    position_ned_m: FloatArray
    quat_xyzw: FloatArray
    linear_velocity_body_mps: FloatArray
    angular_velocity_body_rps: FloatArray
    frame_id: str = WORLD_FRAME

    def __post_init__(self) -> None:
        _non_negative(self.t_s, "t_s")
        if not self.frame_id:
            raise ValueError("frame_id must not be empty")
        object.__setattr__(self, "position_ned_m", _vector(self.position_ned_m, 3, "position_ned_m"))
        object.__setattr__(self, "quat_xyzw", _quat_norm(_vector(self.quat_xyzw, 4, "quat_xyzw"), "quat_xyzw"))
        object.__setattr__(
            self,
            "linear_velocity_body_mps",
            _vector(self.linear_velocity_body_mps, 3, "linear_velocity_body_mps"),
        )
        object.__setattr__(
            self,
            "angular_velocity_body_rps",
            _vector(self.angular_velocity_body_rps, 3, "angular_velocity_body_rps"),
        )


@dataclass(frozen=True)
class Obstacle:
    """Analytic obstacle; cylinders are axis-aligned and vertical in NED z."""

    obstacle_id: str
    kind: str
    center_ned_m: FloatArray
    radius_m: float | None = None
    half_extents_m: FloatArray | None = None
    height_m: float | None = None

    def __post_init__(self) -> None:
        if not self.obstacle_id:
            raise ValueError("obstacle_id must not be empty")
        if self.kind not in {"sphere", "box", "cylinder"}:
            raise ValueError("kind must be 'sphere', 'box', or 'cylinder'")
        object.__setattr__(self, "center_ned_m", _vector(self.center_ned_m, 3, "center_ned_m"))
        if self.kind == "sphere":
            if self.radius_m is None:
                raise ValueError("sphere obstacles require a positive radius_m")
            object.__setattr__(self, "radius_m", _positive(self.radius_m, "radius_m"))
            if self.half_extents_m is not None:
                raise ValueError("sphere obstacles must not define half_extents_m")
            if self.height_m is not None:
                raise ValueError("sphere obstacles must not define height_m")
        elif self.kind == "box":
            if self.half_extents_m is None:
                raise ValueError("box obstacles require half_extents_m")
            object.__setattr__(
                self,
                "half_extents_m",
                _vector(self.half_extents_m, 3, "half_extents_m"),
            )
            if np.any(self.half_extents_m <= 0.0):
                raise ValueError("half_extents_m must be positive")
            if self.radius_m is not None:
                raise ValueError("box obstacles must not define radius_m")
            if self.height_m is not None:
                raise ValueError("box obstacles must not define height_m")
        else:
            if self.radius_m is None:
                raise ValueError("cylinder obstacles require a positive radius_m")
            object.__setattr__(self, "radius_m", _positive(self.radius_m, "radius_m"))
            if self.height_m is None:
                raise ValueError("cylinder obstacles require a positive height_m")
            object.__setattr__(self, "height_m", _positive(self.height_m, "height_m"))
            if self.half_extents_m is not None:
                raise ValueError("cylinder obstacles must not define half_extents_m")


@dataclass(frozen=True)
class ObstacleMap:
    bounds_ned_m: FloatArray
    obstacles: tuple[Obstacle, ...] = ()

    def __post_init__(self) -> None:
        bounds = np.asarray(self.bounds_ned_m, dtype=np.float64)
        if bounds.shape != (3, 2):
            raise ValueError("bounds_ned_m must have shape (3, 2)")
        if not np.all(np.isfinite(bounds)):
            raise ValueError("bounds_ned_m must contain only finite values")
        if np.any(bounds[:, 1] <= bounds[:, 0]):
            raise ValueError("each bounds pair must have upper > lower")
        object.__setattr__(self, "bounds_ned_m", bounds)
        object.__setattr__(self, "obstacles", tuple(self.obstacles))


@runtime_checkable
class CurrentField(Protocol):
    def sample(self, position_ned_m: FloatArray, t_s: float) -> FloatArray:
        ...


@dataclass(frozen=True)
class UniformCurrent:
    velocity_ned_mps: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "velocity_ned_mps", _vector(self.velocity_ned_mps, 3, "velocity_ned_mps"))

    def sample(self, position_ned_m: FloatArray, t_s: float) -> FloatArray:
        _vector(position_ned_m, 3, "position_ned_m")
        _non_negative(t_s, "t_s")
        return self.velocity_ned_mps.copy()


@dataclass(frozen=True)
class WorldSnapshot:
    t_s: float
    obstacle_map: ObstacleMap
    current_field: CurrentField

    def __post_init__(self) -> None:
        _non_negative(self.t_s, "t_s")
        if not isinstance(self.obstacle_map, ObstacleMap):
            raise TypeError("obstacle_map must be an ObstacleMap")
        if not isinstance(self.current_field, CurrentField):
            raise TypeError("current_field must implement the CurrentField protocol")


@dataclass(frozen=True)
class CapabilitySet:
    available_actuators: tuple[str, ...]
    effectiveness: Mapping[str, float] = field(default_factory=dict)
    diagnosis_delay_s: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "available_actuators", tuple(self.available_actuators))
        object.__setattr__(self, "effectiveness", dict(self.effectiveness))
        _non_negative(self.diagnosis_delay_s, "diagnosis_delay_s")
        for actuator, value in self.effectiveness.items():
            if not actuator:
                raise ValueError("effectiveness keys must not be empty")
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("effectiveness values must be in [0, 1]")


@dataclass(frozen=True)
class TaskSpec:
    goal_ned_m: FloatArray
    goal_radius_m: float
    risk_budget: float = 0.0
    constraints: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_ned_m", _vector(self.goal_ned_m, 3, "goal_ned_m"))
        if not np.isfinite(self.goal_radius_m) or self.goal_radius_m <= 0.0:
            raise ValueError("goal_radius_m must be finite and positive")
        if not np.isfinite(self.risk_budget) or not 0.0 <= self.risk_budget <= 1.0:
            raise ValueError("risk_budget must be in [0, 1]")
        object.__setattr__(self, "constraints", dict(self.constraints))


class PlanReason(str, Enum):
    INITIAL = "initial"
    NEW_GOAL = "new_goal"
    OBSTACLE = "obstacle"
    DEVIATION = "deviation"
    CAPABILITY_CHANGE = "capability_change"
    RECOVERY = "recovery"


@dataclass(frozen=True)
class PlanRequest:
    request_id: str
    start: State
    task: TaskSpec
    world: WorldSnapshot
    vehicle: VehicleSpec
    capability: CapabilitySet
    seed: int = 0
    reason: PlanReason = PlanReason.INITIAL
    parent_plan_id: str | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if not isinstance(self.start, State):
            raise TypeError("start must be a State")
        if not isinstance(self.task, TaskSpec):
            raise TypeError("task must be a TaskSpec")
        if not isinstance(self.world, WorldSnapshot):
            raise TypeError("world must be a WorldSnapshot")
        if not isinstance(self.vehicle, VehicleSpec):
            raise TypeError("vehicle must be a VehicleSpec")
        if not isinstance(self.capability, CapabilitySet):
            raise TypeError("capability must be a CapabilitySet")
        if not isinstance(self.reason, PlanReason):
            raise TypeError("reason must be a PlanReason")


class PlanStatus(str, Enum):
    OK = "ok"
    INFEASIBLE = "infeasible"
    TIMEOUT = "timeout"
    DEGRADED = "degraded"
    ERROR = "error"


@dataclass(frozen=True)
class RecoveryCertificate:
    valid: bool
    assumptions: tuple[str, ...]
    fault_set: tuple[str, ...]
    backup_maneuvers: tuple[str, ...]
    valid_until_s: float | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        object.__setattr__(self, "fault_set", tuple(self.fault_set))
        object.__setattr__(self, "backup_maneuvers", tuple(self.backup_maneuvers))
        if self.valid_until_s is not None:
            _non_negative(self.valid_until_s, "valid_until_s")
        if not self.valid and not self.reason:
            raise ValueError("invalid certificates require a reason")


@dataclass(frozen=True)
class Trajectory:
    trajectory_id: str
    t_s: FloatArray
    position_ned_m: FloatArray
    quat_xyzw: FloatArray
    linear_velocity_body_mps: FloatArray
    angular_velocity_body_rps: FloatArray
    controls: FloatArray | None = None
    clearance_m: FloatArray | None = None
    feasibility: str = "unchecked"
    frame_id: str = WORLD_FRAME

    def __post_init__(self) -> None:
        if not self.trajectory_id:
            raise ValueError("trajectory_id must not be empty")
        if not self.frame_id:
            raise ValueError("frame_id must not be empty")
        if self.feasibility not in {"unchecked", "tracked", "kinodynamically_feasible"}:
            raise ValueError("feasibility must be 'unchecked', 'tracked', or 'kinodynamically_feasible'")
        t_s = _time_array(self.t_s)
        rows = t_s.shape[0]
        object.__setattr__(self, "t_s", t_s)
        object.__setattr__(self, "position_ned_m", _rows(self.position_ned_m, rows, 3, "position_ned_m"))
        object.__setattr__(self, "quat_xyzw", _quat_norm(_rows(self.quat_xyzw, rows, 4, "quat_xyzw"), "quat_xyzw"))
        object.__setattr__(
            self,
            "linear_velocity_body_mps",
            _rows(self.linear_velocity_body_mps, rows, 3, "linear_velocity_body_mps"),
        )
        object.__setattr__(
            self,
            "angular_velocity_body_rps",
            _rows(self.angular_velocity_body_rps, rows, 3, "angular_velocity_body_rps"),
        )
        if self.controls is not None:
            controls = np.asarray(self.controls, dtype=np.float64)
            if controls.ndim != 2 or controls.shape[0] != rows or controls.shape[1] < 1:
                raise ValueError("controls must have shape (N, M) with M >= 1")
            if not np.all(np.isfinite(controls)):
                raise ValueError("controls must contain only finite values")
            object.__setattr__(self, "controls", controls)
        if self.clearance_m is not None:
            object.__setattr__(self, "clearance_m", _row_vector(self.clearance_m, rows, "clearance_m"))


@dataclass(frozen=True)
class PlanResult:
    status: PlanStatus
    trajectory: Trajectory | None
    cost: float
    certificate: RecoveryCertificate | None
    planning_time_s: float
    plan_id: str
    parent_plan_id: str | None = None
    valid_until_s: float | None = None
    reason: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.status, PlanStatus):
            raise TypeError("status must be a PlanStatus")
        if not self.plan_id:
            raise ValueError("plan_id must not be empty")
        _non_negative(self.cost, "cost")
        _non_negative(self.planning_time_s, "planning_time_s")
        if self.valid_until_s is not None:
            _non_negative(self.valid_until_s, "valid_until_s")
        if self.status == PlanStatus.OK and self.trajectory is None:
            raise ValueError("OK plans require a trajectory")
        if self.status != PlanStatus.OK and not self.reason:
            raise ValueError("non-OK plans require a reason")
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


@dataclass(frozen=True)
class Reference:
    stamp_s: float
    position_ned_m: FloatArray
    quat_xyzw: FloatArray
    linear_velocity_body_mps: FloatArray
    angular_velocity_body_rps: FloatArray
    feedforward_accel_body_mps2: FloatArray | None = None
    valid_until_s: float | None = None

    def __post_init__(self) -> None:
        _non_negative(self.stamp_s, "stamp_s")
        object.__setattr__(self, "position_ned_m", _vector(self.position_ned_m, 3, "position_ned_m"))
        object.__setattr__(self, "quat_xyzw", _quat_norm(_vector(self.quat_xyzw, 4, "quat_xyzw"), "quat_xyzw"))
        object.__setattr__(
            self,
            "linear_velocity_body_mps",
            _vector(self.linear_velocity_body_mps, 3, "linear_velocity_body_mps"),
        )
        object.__setattr__(
            self,
            "angular_velocity_body_rps",
            _vector(self.angular_velocity_body_rps, 3, "angular_velocity_body_rps"),
        )
        if self.feedforward_accel_body_mps2 is not None:
            object.__setattr__(
                self,
                "feedforward_accel_body_mps2",
                _vector(self.feedforward_accel_body_mps2, 3, "feedforward_accel_body_mps2"),
            )
        if self.valid_until_s is not None:
            _non_negative(self.valid_until_s, "valid_until_s")
            if self.valid_until_s < self.stamp_s:
                raise ValueError("valid_until_s must not precede stamp_s")


@dataclass(frozen=True)
class LocalObservation:
    stamp_s: float
    state: State
    sonar_ranges_m: FloatArray | None = None
    local_obstacles: tuple[Obstacle, ...] = ()
    health: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _non_negative(self.stamp_s, "stamp_s")
        if not isinstance(self.state, State):
            raise TypeError("state must be a State")
        if self.sonar_ranges_m is not None:
            sonar = np.asarray(self.sonar_ranges_m, dtype=np.float64)
            if sonar.ndim != 1 or sonar.shape[0] == 0:
                raise ValueError("sonar_ranges_m must be a non-empty one-dimensional array")
            if np.any(~np.isfinite(sonar)) or np.any(sonar < 0.0):
                raise ValueError("sonar_ranges_m must be finite and non-negative")
            object.__setattr__(self, "sonar_ranges_m", sonar)
        object.__setattr__(self, "local_obstacles", tuple(self.local_obstacles))
        object.__setattr__(self, "health", dict(self.health))


@dataclass(frozen=True)
class ControlProposal:
    """High-level body-frame motion request before actuator allocation."""

    stamp_s: float
    linear_accel_body_mps2: FloatArray
    angular_velocity_body_rps: FloatArray
    valid_until_s: float | None = None

    def __post_init__(self) -> None:
        _non_negative(self.stamp_s, "stamp_s")
        object.__setattr__(
            self,
            "linear_accel_body_mps2",
            _vector(self.linear_accel_body_mps2, 3, "linear_accel_body_mps2"),
        )
        object.__setattr__(
            self,
            "angular_velocity_body_rps",
            _vector(self.angular_velocity_body_rps, 3, "angular_velocity_body_rps"),
        )
        if self.valid_until_s is not None:
            _non_negative(self.valid_until_s, "valid_until_s")
            if self.valid_until_s < self.stamp_s:
                raise ValueError("valid_until_s must not precede stamp_s")


@runtime_checkable
class ActuatorCommand(Protocol):
    """Platform-neutral command interface consumed by a simulator backend."""

    @property
    def stamp_s(self) -> float:
        ...

    @property
    def valid_until_s(self) -> float | None:
        ...


@dataclass(frozen=True)
class Remus100ActuatorCommand:
    """REMUS 100 actuator command after tracking control and safety filtering."""

    stamp_s: float
    rudder_angle_rad: float
    stern_plane_angle_rad: float
    propeller_rpm: float
    valid_until_s: float | None = None

    def __post_init__(self) -> None:
        _non_negative(self.stamp_s, "stamp_s")
        for name in ("rudder_angle_rad", "stern_plane_angle_rad", "propeller_rpm"):
            value = getattr(self, name)
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, float(value))
        if self.valid_until_s is not None:
            _non_negative(self.valid_until_s, "valid_until_s")
            if self.valid_until_s < self.stamp_s:
                raise ValueError("valid_until_s must not precede stamp_s")


@dataclass(frozen=True)
class SimulationSnapshot:
    """Synchronized plant state, world model, and applied actuator state."""

    state: State
    world: WorldSnapshot
    actuator_state: ActuatorCommand | None = None
    diagnostics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.state, State):
            raise TypeError("state must be a State")
        if not isinstance(self.world, WorldSnapshot):
            raise TypeError("world must be a WorldSnapshot")
        if abs(self.state.t_s - self.world.t_s) > 1e-9:
            raise ValueError("state and world timestamps must match")
        if self.actuator_state is not None and not isinstance(self.actuator_state, ActuatorCommand):
            raise TypeError("actuator_state must be an ActuatorCommand")
        if self.actuator_state is not None and abs(self.actuator_state.stamp_s - self.state.t_s) > 1e-9:
            raise ValueError("actuator_state and state timestamps must match")
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


@dataclass(frozen=True)
class SafetyContext:
    state: State
    local_obstacles: tuple[Obstacle, ...]
    capability: CapabilitySet
    certificate: RecoveryCertificate | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, State):
            raise TypeError("state must be a State")
        if not isinstance(self.capability, CapabilitySet):
            raise TypeError("capability must be a CapabilitySet")
        object.__setattr__(self, "local_obstacles", tuple(self.local_obstacles))


@dataclass(frozen=True)
class SafeCommand:
    command: ControlProposal
    allowed: bool
    reason: str | None = None
    valid_until_s: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.command, ControlProposal):
            raise TypeError("command must be a ControlProposal")
        if not self.allowed and not self.reason:
            raise ValueError("rejected commands require a reason")
        if self.valid_until_s is not None:
            _non_negative(self.valid_until_s, "valid_until_s")


@dataclass(frozen=True)
class EvalResult:
    success: bool
    collision: bool
    out_of_bounds: bool
    min_clearance_m: float
    path_length_m: float
    duration_s: float
    energy_j: float
    tracking_rmse: float
    constraint_violations: int
    recovery_success: bool | None = None
    failure_reason: str | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not np.isfinite(self.min_clearance_m):
            raise ValueError("min_clearance_m must be finite")
        for name in ("path_length_m", "duration_s", "energy_j", "tracking_rmse"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.constraint_violations < 0:
            raise ValueError("constraint_violations must be non-negative")
        if not self.success and not self.failure_reason:
            raise ValueError("unsuccessful evaluations require a failure_reason")
        object.__setattr__(self, "metrics", dict(self.metrics))


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    world: WorldSnapshot
    vehicle: VehicleSpec
    start: State
    goal_ned_m: FloatArray
    goal_radius_m: float
    constraints: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        if not isinstance(self.world, WorldSnapshot):
            raise TypeError("world must be a WorldSnapshot")
        if not isinstance(self.vehicle, VehicleSpec):
            raise TypeError("vehicle must be a VehicleSpec")
        if not isinstance(self.start, State):
            raise TypeError("start must be a State")
        object.__setattr__(self, "goal_ned_m", _vector(self.goal_ned_m, 3, "goal_ned_m"))
        if not np.isfinite(self.goal_radius_m) or self.goal_radius_m <= 0.0:
            raise ValueError("goal_radius_m must be finite and positive")
        object.__setattr__(self, "constraints", dict(self.constraints))
