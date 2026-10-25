from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import pi, radians, sin

import numpy as np

from ..contracts import CapabilitySet, RecoveryCertificate, Trajectory, VehicleSpec


class FaultKind(str, Enum):
    NOMINAL = "nominal"
    THRUSTER_EFFICIENCY_LOSS = "thruster_efficiency_loss"
    YAW_FIN_STUCK = "yaw_fin_stuck"
    PITCH_FIN_STUCK = "pitch_fin_stuck"


class ManeuverKind(str, Enum):
    BRAKE = "brake"
    VERTICAL_ESCAPE = "vertical_escape"
    MINIMUM_RADIUS_TURN = "minimum_radius_turn"
    EXIT_CURRENT = "exit_current"
    SAFE_CORRIDOR = "safe_corridor"


class ConstraintMode(str, Enum):
    VEHICLE = "vehicle"
    UNBOUNDED_TURN_DIAGNOSTIC = "unbounded_turn_diagnostic"


class ClearanceModel(str, Enum):
    FULL_HULL = "full_hull"
    POINT_DIAGNOSTIC = "point_diagnostic"


@dataclass(frozen=True)
class RecoveryErrorBounds:
    """Error bounds used to inflate the frozen 0.25 m clearance threshold.

    These are M6 design assumptions, not identified sensor or current-model
    uncertainties. They are deliberately recorded in every valid certificate.
    """

    position_m: float = 0.02
    heading_rad: float = 0.01
    current_mps: float = 0.005

    def __post_init__(self) -> None:
        for name in ("position_m", "heading_rad", "current_mps"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.heading_rad > pi:
            raise ValueError("heading_rad must not exceed pi")


@dataclass(frozen=True)
class RecoveryConfig:
    horizon_s: float = 10.0
    required_clearance_m: float = 0.25
    certificate_validity_s: float = 1.0
    sample_time_s: float = 0.1
    max_pitch_angle_rad: float = radians(30.0)
    max_translation_step_m: float = 0.05
    max_rotation_step_rad: float = 0.05
    error_bounds: RecoveryErrorBounds = field(default_factory=RecoveryErrorBounds)
    constraint_mode: ConstraintMode = ConstraintMode.VEHICLE
    clearance_model: ClearanceModel = ClearanceModel.FULL_HULL
    unbounded_yaw_rate_factor: float = 4.0

    def __post_init__(self) -> None:
        for name in (
            "horizon_s",
            "required_clearance_m",
            "certificate_validity_s",
            "sample_time_s",
            "max_pitch_angle_rad",
            "max_translation_step_m",
            "max_rotation_step_rad",
            "unbounded_yaw_rate_factor",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.certificate_validity_s > self.horizon_s:
            raise ValueError("certificate_validity_s must not exceed horizon_s")
        if self.sample_time_s > self.horizon_s / 10.0:
            raise ValueError("sample_time_s must provide at least ten horizon samples")
        if self.max_pitch_angle_rad >= pi / 2.0:
            raise ValueError("max_pitch_angle_rad must be less than pi/2")
        if not isinstance(self.error_bounds, RecoveryErrorBounds):
            raise TypeError("error_bounds must be RecoveryErrorBounds")
        if not isinstance(self.constraint_mode, ConstraintMode):
            raise TypeError("constraint_mode must be a ConstraintMode")
        if not isinstance(self.clearance_model, ClearanceModel):
            raise TypeError("clearance_model must be a ClearanceModel")
        if (
            self.constraint_mode is ConstraintMode.UNBOUNDED_TURN_DIAGNOSTIC
            and self.unbounded_yaw_rate_factor <= 1.0
        ):
            raise ValueError("unbounded_yaw_rate_factor must exceed 1 in diagnostic mode")

    def error_clearance_inflation_m(self, vehicle: VehicleSpec) -> float:
        if not isinstance(vehicle, VehicleSpec):
            raise TypeError("vehicle must be a VehicleSpec")
        bounds = self.error_bounds
        heading_inflation = 0.5 * vehicle.length_m * sin(bounds.heading_rad)
        current_drift = bounds.current_mps * self.horizon_s
        return float(bounds.position_m + heading_inflation + current_drift)

    def required_inflated_clearance_m(self, vehicle: VehicleSpec) -> float:
        return float(self.required_clearance_m + self.error_clearance_inflation_m(vehicle))


@dataclass(frozen=True)
class FaultSpec:
    name: str
    kind: FaultKind
    effectiveness: float = 1.0
    stuck_sign: int = 0
    residual_rate_fraction: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("fault name must not be empty")
        if not isinstance(self.kind, FaultKind):
            raise TypeError("kind must be a FaultKind")
        if not np.isfinite(self.effectiveness) or not 0.0 <= self.effectiveness <= 1.0:
            raise ValueError("effectiveness must be in [0, 1]")
        if not np.isfinite(self.residual_rate_fraction) or not 0.0 <= self.residual_rate_fraction <= 1.0:
            raise ValueError("residual_rate_fraction must be in [0, 1]")
        if self.kind is FaultKind.NOMINAL:
            if self.stuck_sign != 0 or self.effectiveness != 1.0 or self.residual_rate_fraction != 1.0:
                raise ValueError("nominal faults require full effectiveness and no stuck sign")
        elif self.kind is FaultKind.THRUSTER_EFFICIENCY_LOSS:
            if self.stuck_sign != 0 or self.effectiveness >= 1.0:
                raise ValueError("thruster loss requires a stuck sign of zero and effectiveness below one")
        else:
            if self.stuck_sign not in {-1, 1}:
                raise ValueError("stuck fin faults require stuck_sign of -1 or +1")
            if self.effectiveness != 1.0:
                raise ValueError("stuck fin faults must not additionally scale propeller effectiveness")

    def capability(self, diagnosis_delay_s: float) -> CapabilitySet:
        if not np.isfinite(diagnosis_delay_s) or diagnosis_delay_s < 0.0:
            raise ValueError("diagnosis_delay_s must be finite and non-negative")
        if self.kind is FaultKind.NOMINAL:
            return CapabilitySet(
                available_actuators=("propeller", "yaw_fin", "pitch_fin"),
                effectiveness={"propeller": 1.0, "yaw_fin": 1.0, "pitch_fin": 1.0},
                diagnosis_delay_s=diagnosis_delay_s,
            )
        if self.kind is FaultKind.THRUSTER_EFFICIENCY_LOSS:
            return CapabilitySet(
                available_actuators=("propeller", "yaw_fin", "pitch_fin"),
                effectiveness={
                    "propeller": self.effectiveness,
                    "yaw_fin": 1.0,
                    "pitch_fin": 1.0,
                },
                diagnosis_delay_s=diagnosis_delay_s,
            )
        if self.kind is FaultKind.YAW_FIN_STUCK:
            return CapabilitySet(
                available_actuators=("propeller", "pitch_fin"),
                effectiveness={"propeller": 1.0, "yaw_fin": 0.0, "pitch_fin": 1.0},
                diagnosis_delay_s=diagnosis_delay_s,
            )
        return CapabilitySet(
            available_actuators=("propeller", "yaw_fin"),
            effectiveness={"propeller": 1.0, "yaw_fin": 1.0, "pitch_fin": 0.0},
            diagnosis_delay_s=diagnosis_delay_s,
        )

    @property
    def propeller_effectiveness(self) -> float:
        return self.effectiveness if self.kind is FaultKind.THRUSTER_EFFICIENCY_LOSS else 1.0

    @property
    def forced_yaw_rate_fraction(self) -> float:
        if self.kind is FaultKind.YAW_FIN_STUCK:
            return float(self.stuck_sign * self.residual_rate_fraction)
        return 0.0

    @property
    def forced_pitch_rate_fraction(self) -> float:
        if self.kind is FaultKind.PITCH_FIN_STUCK:
            return float(self.stuck_sign * self.residual_rate_fraction)
        return 0.0


@dataclass(frozen=True)
class ManeuverCandidate:
    kind: ManeuverKind
    name: str
    trajectory: Trajectory
    requires_actuators: tuple[str, ...]
    vehicle_limit_exceedances: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ManeuverKind):
            raise TypeError("kind must be a ManeuverKind")
        if not self.name:
            raise ValueError("maneuver name must not be empty")
        if not isinstance(self.trajectory, Trajectory):
            raise TypeError("trajectory must be a Trajectory")
        object.__setattr__(self, "requires_actuators", tuple(self.requires_actuators))
        object.__setattr__(self, "vehicle_limit_exceedances", tuple(self.vehicle_limit_exceedances))

    @property
    def duration_s(self) -> float:
        return float(self.trajectory.t_s[-1] - self.trajectory.t_s[0])


@dataclass(frozen=True)
class ManeuverCheck:
    candidate: ManeuverCandidate
    safe: bool
    min_clearance_m: float
    required_clearance_m: float
    margin_m: float
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.min_clearance_m):
            raise ValueError("min_clearance_m must be finite")
        if not np.isfinite(self.required_clearance_m):
            raise ValueError("required_clearance_m must be finite")
        if not np.isfinite(self.margin_m):
            raise ValueError("margin_m must be finite")
        if self.safe and self.failure_reason is not None:
            raise ValueError("safe maneuver checks must not define failure_reason")
        if not self.safe and not self.failure_reason:
            raise ValueError("unsafe maneuver checks require failure_reason")


@dataclass(frozen=True)
class FaultCheck:
    fault: FaultSpec
    diagnosis_delay_s: float
    capability: CapabilitySet
    safe: bool
    coast_min_clearance_m: float | None
    selected: ManeuverCheck | None
    checks: tuple[ManeuverCheck, ...]
    unavailable_maneuvers: tuple[str, ...]
    margin_m: float | None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.fault, FaultSpec):
            raise TypeError("fault must be a FaultSpec")
        if not np.isfinite(self.diagnosis_delay_s) or self.diagnosis_delay_s < 0.0:
            raise ValueError("diagnosis_delay_s must be finite and non-negative")
        if not isinstance(self.capability, CapabilitySet):
            raise TypeError("capability must be a CapabilitySet")
        if self.coast_min_clearance_m is not None and not np.isfinite(self.coast_min_clearance_m):
            raise ValueError("coast_min_clearance_m must be finite when defined")
        if self.margin_m is not None and not np.isfinite(self.margin_m):
            raise ValueError("margin_m must be finite when defined")
        object.__setattr__(self, "checks", tuple(self.checks))
        object.__setattr__(self, "unavailable_maneuvers", tuple(self.unavailable_maneuvers))
        if self.safe:
            if self.failure_reason is not None:
                raise ValueError("safe fault checks must not define failure_reason")
            if self.selected is None:
                raise ValueError("safe fault checks require a selected maneuver")
        elif not self.failure_reason:
            raise ValueError("unsafe fault checks require failure_reason")

    @property
    def label(self) -> str:
        return f"{self.fault.name}@{self.diagnosis_delay_s:g}s"


@dataclass(frozen=True)
class RecoveryAssessment:
    certificate: RecoveryCertificate
    fault_checks: tuple[FaultCheck, ...]
    computation_time_s: float
    current_clearance_m: float

    def __post_init__(self) -> None:
        if not isinstance(self.certificate, RecoveryCertificate):
            raise TypeError("certificate must be a RecoveryCertificate")
        object.__setattr__(self, "fault_checks", tuple(self.fault_checks))
        if not np.isfinite(self.computation_time_s) or self.computation_time_s < 0.0:
            raise ValueError("computation_time_s must be finite and non-negative")
        if not np.isfinite(self.current_clearance_m):
            raise ValueError("current_clearance_m must be finite")

    @property
    def valid(self) -> bool:
        return self.certificate.valid


DEFAULT_DIAGNOSIS_DELAYS_S = (0.0, 1.0, 3.0, 5.0)

DEFAULT_FAULTS = (
    FaultSpec("nominal", FaultKind.NOMINAL),
    FaultSpec(
        "thruster_efficiency_loss_0.50",
        FaultKind.THRUSTER_EFFICIENCY_LOSS,
        effectiveness=0.5,
    ),
    FaultSpec(
        "yaw_fin_stuck_positive",
        FaultKind.YAW_FIN_STUCK,
        stuck_sign=1,
        residual_rate_fraction=1.0,
    ),
    FaultSpec(
        "yaw_fin_stuck_negative",
        FaultKind.YAW_FIN_STUCK,
        stuck_sign=-1,
        residual_rate_fraction=1.0,
    ),
    FaultSpec(
        "pitch_fin_stuck_positive",
        FaultKind.PITCH_FIN_STUCK,
        stuck_sign=1,
        residual_rate_fraction=1.0,
    ),
    FaultSpec(
        "pitch_fin_stuck_negative",
        FaultKind.PITCH_FIN_STUCK,
        stuck_sign=-1,
        residual_rate_fraction=1.0,
    ),
)


__all__ = [
    "ClearanceModel",
    "ConstraintMode",
    "DEFAULT_DIAGNOSIS_DELAYS_S",
    "DEFAULT_FAULTS",
    "FaultCheck",
    "FaultKind",
    "FaultSpec",
    "ManeuverCandidate",
    "ManeuverCheck",
    "ManeuverKind",
    "RecoveryAssessment",
    "RecoveryConfig",
    "RecoveryErrorBounds",
]
