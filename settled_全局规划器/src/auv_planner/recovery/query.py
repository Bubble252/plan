from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
from time import perf_counter
from typing import Any

import numpy as np

from ..contracts import FloatArray, State, UniformCurrent, VehicleSpec, WorldSnapshot
from ..current import DepthShearCurrent
from ..geometry import (
    ClearanceDetails,
    hull_clearance_details,
    hull_from_vehicle,
    point_clearance_details,
)
from .checker import RecoverabilityChecker
from .models import (
    ClearanceModel,
    FaultProfile,
    RecoveryAssessment,
    RecoveryConfig,
)


CHECKER_CACHE_SCHEMA = "m7-recovery-query-v2"


class RecoveryQueryStatus(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    UNKNOWN = "unknown"


class RecoveryQueryMode(str, Enum):
    FULL = "full"
    ONLINE_FAST = "online_fast"


@dataclass(frozen=True)
class StatePrefilterResult:
    rejected: bool
    current_clearance_m: float
    reason: str | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.current_clearance_m):
            raise ValueError("current_clearance_m must be finite")
        if self.rejected and not self.reason:
            raise ValueError("rejected prefilter results require a reason")
        if not self.rejected and self.reason is not None:
            raise ValueError("accepted prefilter results must not define a reason")


@dataclass(frozen=True)
class RecoveryQueryPolicy:
    budget_s: float | None = 10.0
    use_cache: bool = True
    use_prefilter: bool = True
    mode: RecoveryQueryMode = RecoveryQueryMode.FULL
    parallel_workers: int = 1

    def __post_init__(self) -> None:
        if self.budget_s is not None:
            if not np.isfinite(self.budget_s) or self.budget_s <= 0.0:
                raise ValueError("budget_s must be finite and positive when defined")
        if not isinstance(self.mode, RecoveryQueryMode):
            raise TypeError("mode must be a RecoveryQueryMode")
        if not isinstance(self.parallel_workers, int) or self.parallel_workers < 1:
            raise ValueError("parallel_workers must be a positive integer")
        if (
            self.mode is RecoveryQueryMode.ONLINE_FAST
            and self.parallel_workers > 1
        ):
            raise ValueError(
                "online_fast mode is sequential; parallel_workers > 1 is not supported"
            )


@dataclass(frozen=True)
class RecoveryQueryResult:
    status: RecoveryQueryStatus
    profile_name: str
    profile_digest: str
    elapsed_s: float
    mode: RecoveryQueryMode = RecoveryQueryMode.FULL
    assessment: RecoveryAssessment | None = None
    cache_hit: bool = False
    budget_exhausted: bool = False
    prefilter_reason: str | None = None
    query_key: str | None = None
    early_stopped: bool = False
    evaluated_checks: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.status, RecoveryQueryStatus):
            raise TypeError("status must be a RecoveryQueryStatus")
        if not self.profile_name:
            raise ValueError("profile_name must not be empty")
        if not self.profile_digest:
            raise ValueError("profile_digest must not be empty")
        if not np.isfinite(self.elapsed_s) or self.elapsed_s < 0.0:
            raise ValueError("elapsed_s must be finite and non-negative")
        if not isinstance(self.mode, RecoveryQueryMode):
            raise TypeError("mode must be a RecoveryQueryMode")
        if self.assessment is not None and not isinstance(self.assessment, RecoveryAssessment):
            raise TypeError("assessment must be a RecoveryAssessment when defined")
        if self.status is RecoveryQueryStatus.UNKNOWN and not self.budget_exhausted:
            if self.prefilter_reason is None:
                raise ValueError("unknown query results require a reason")
        if self.prefilter_reason is not None and self.status is not RecoveryQueryStatus.INVALID:
            raise ValueError("prefilter reasons apply only to invalid results")
        if not isinstance(self.early_stopped, bool):
            raise TypeError("early_stopped must be a bool")
        if not isinstance(self.evaluated_checks, int) or self.evaluated_checks < 0:
            raise ValueError("evaluated_checks must be a non-negative integer")
        if self.early_stopped and self.status is RecoveryQueryStatus.VALID:
            raise ValueError("early-stopped queries cannot return a valid result")
        if self.early_stopped and self.evaluated_checks < 1:
            raise ValueError("early-stopped queries must evaluate at least one check")

    @property
    def reason(self) -> str | None:
        if self.assessment is not None:
            return self.assessment.certificate.reason
        return self.prefilter_reason


@dataclass
class RecoveryQueryStats:
    queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    prefilter_rejections: int = 0
    full_assessments: int = 0
    valid: int = 0
    invalid: int = 0
    unknown: int = 0
    budget_exhausted: int = 0
    early_stops: int = 0
    total_s: float = 0.0
    full_assessment_s: float = 0.0

    @property
    def cache_hit_rate(self) -> float:
        cacheable_queries = self.cache_hits + self.cache_misses
        return 0.0 if cacheable_queries == 0 else self.cache_hits / cacheable_queries

    def as_dict(self) -> dict[str, int | float]:
        return {
            "queries": self.queries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hit_rate,
            "prefilter_rejections": self.prefilter_rejections,
            "full_assessments": self.full_assessments,
            "valid": self.valid,
            "invalid": self.invalid,
            "unknown": self.unknown,
            "budget_exhausted": self.budget_exhausted,
            "early_stops": self.early_stops,
            "total_s": self.total_s,
            "full_assessment_s": self.full_assessment_s,
        }


class RecoveryQuerySession:
    """Online recovery-certificate query session with a shared wall-clock budget."""

    def __init__(
        self,
        *,
        checker: RecoverabilityChecker | None = None,
        policy: RecoveryQueryPolicy | None = None,
        checker_version: str = CHECKER_CACHE_SCHEMA,
    ) -> None:
        self._checker = RecoverabilityChecker() if checker is None else checker
        self._policy = RecoveryQueryPolicy() if policy is None else policy
        if not isinstance(self._checker, RecoverabilityChecker):
            raise TypeError("checker must be a RecoverabilityChecker")
        if not isinstance(self._policy, RecoveryQueryPolicy):
            raise TypeError("policy must be a RecoveryQueryPolicy")
        if not checker_version:
            raise ValueError("checker_version must not be empty")
        self._checker_version = checker_version
        self._cache: dict[str, RecoveryQueryResult] = {}
        self._started_at = perf_counter()
        self._deadline = (
            None
            if self._policy.budget_s is None
            else self._started_at + self._policy.budget_s
        )
        self.stats = RecoveryQueryStats()

    @property
    def remaining_s(self) -> float | None:
        if self._deadline is None:
            return None
        return max(0.0, self._deadline - perf_counter())

    def query(
        self,
        state: State,
        world: WorldSnapshot,
        vehicle: VehicleSpec,
        *,
        profile: FaultProfile,
        config: RecoveryConfig | None = None,
        safe_anchor_ned_m: FloatArray | None = None,
    ) -> RecoveryQueryResult:
        if not isinstance(profile, FaultProfile):
            raise TypeError("profile must be a FaultProfile")
        if not isinstance(state, State):
            raise TypeError("state must be a State")
        if not isinstance(world, WorldSnapshot):
            raise TypeError("world must be a WorldSnapshot")
        if not isinstance(vehicle, VehicleSpec):
            raise TypeError("vehicle must be a VehicleSpec")
        active_config = RecoveryConfig() if config is None else config
        if not isinstance(active_config, RecoveryConfig):
            raise TypeError("config must be a RecoveryConfig")

        started = perf_counter()
        query_key = (
            _query_key(
                state,
                world,
                vehicle,
                profile,
                active_config,
                safe_anchor_ned_m,
                self._checker_version,
                self._policy.mode,
            )
            if self._policy.use_cache
            else None
        )
        self.stats.queries += 1
        if query_key is not None and query_key in self._cache:
            cached = self._cache[query_key]
            result = replace(cached, cache_hit=True, elapsed_s=perf_counter() - started)
            self.stats.cache_hits += 1
            self._record_result(result)
            return result

        if query_key is not None:
            self.stats.cache_misses += 1
        if self.remaining_s == 0.0:
            result = RecoveryQueryResult(
                status=RecoveryQueryStatus.UNKNOWN,
                profile_name=profile.name,
                profile_digest=profile.digest,
                elapsed_s=perf_counter() - started,
                mode=self._policy.mode,
                budget_exhausted=True,
                query_key=query_key,
            )
            self._record_result(result)
            return result

        if self._policy.use_prefilter:
            prefilter = prefilter_state(state, world, vehicle, active_config)
            if prefilter.rejected:
                result = RecoveryQueryResult(
                    status=RecoveryQueryStatus.INVALID,
                    profile_name=profile.name,
                    profile_digest=profile.digest,
                    elapsed_s=perf_counter() - started,
                    mode=self._policy.mode,
                    prefilter_reason=f"prefilter:{prefilter.reason}",
                    query_key=query_key,
                )
                self.stats.prefilter_rejections += 1
                if query_key is not None:
                    self._cache[query_key] = result
                self._record_result(result)
                return result

        assessment = self._checker.assess(
            state,
            world,
            vehicle,
            config=active_config,
            profile=profile,
            safe_anchor_ned_m=safe_anchor_ned_m,
            early_stop=self._policy.mode is RecoveryQueryMode.ONLINE_FAST,
            parallel_workers=self._policy.parallel_workers,
        )
        self.stats.full_assessments += 1
        self.stats.full_assessment_s += assessment.computation_time_s
        elapsed = perf_counter() - started
        budget_exhausted = self._deadline is not None and perf_counter() > self._deadline
        if budget_exhausted:
            status = RecoveryQueryStatus.UNKNOWN
        else:
            status = (
                RecoveryQueryStatus.VALID
                if assessment.valid
                else RecoveryQueryStatus.INVALID
            )
        result = RecoveryQueryResult(
            status=status,
            profile_name=profile.name,
            profile_digest=profile.digest,
            elapsed_s=elapsed,
            mode=self._policy.mode,
            assessment=assessment,
            budget_exhausted=budget_exhausted,
            query_key=query_key,
            early_stopped=assessment.early_stopped,
            evaluated_checks=len(assessment.fault_checks),
        )
        if query_key is not None and not budget_exhausted:
            self._cache[query_key] = result
        self._record_result(result)
        return result

    def _record_result(self, result: RecoveryQueryResult) -> None:
        self.stats.total_s += result.elapsed_s
        if result.status is RecoveryQueryStatus.VALID:
            self.stats.valid += 1
        elif result.status is RecoveryQueryStatus.INVALID:
            self.stats.invalid += 1
        else:
            self.stats.unknown += 1
        if result.budget_exhausted:
            self.stats.budget_exhausted += 1
        if result.early_stopped:
            self.stats.early_stops += 1


def prefilter_state(
    state: State,
    world: WorldSnapshot,
    vehicle: VehicleSpec,
    config: RecoveryConfig,
) -> StatePrefilterResult:
    details = _state_clearance_details(state, world, vehicle, config)
    reasons: list[str] = []
    if details.collision:
        reasons.append("collision")
    if details.out_of_bounds:
        reasons.append("out_of_bounds")
    if details.clearance_m < config.required_inflated_clearance_m(vehicle):
        reasons.append("clearance_below_required")
    reason = ",".join(reasons) if reasons else None
    return StatePrefilterResult(
        rejected=bool(reasons),
        current_clearance_m=float(details.clearance_m),
        reason=reason,
    )


def _state_clearance_details(
    state: State,
    world: WorldSnapshot,
    vehicle: VehicleSpec,
    config: RecoveryConfig,
) -> ClearanceDetails:
    if config.clearance_model is ClearanceModel.POINT_DIAGNOSTIC:
        return point_clearance_details(world.obstacle_map, state.position_ned_m)
    return hull_clearance_details(world.obstacle_map, state, hull_from_vehicle(vehicle))


def _query_key(
    state: State,
    world: WorldSnapshot,
    vehicle: VehicleSpec,
    profile: FaultProfile,
    config: RecoveryConfig,
    safe_anchor_ned_m: FloatArray | None,
    checker_version: str,
    mode: RecoveryQueryMode,
) -> str | None:
    current = _current_fingerprint(world)
    if current is None:
        return None
    payload = {
        "checker": checker_version,
        "query_mode": mode.value,
        "profile": {"name": profile.name, "digest": profile.digest},
        "state": _state_payload(state),
        "world": {
            "t_s": _float(world.t_s),
            "obstacle_map": _obstacle_map_payload(world),
            "current": current,
        },
        "vehicle": _vehicle_payload(vehicle),
        "config": _config_payload(config),
        "safe_anchor_ned_m": (
            None if safe_anchor_ned_m is None else _array(safe_anchor_ned_m)
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _state_payload(state: State) -> dict[str, Any]:
    return {
        "t_s": _float(state.t_s),
        "position_ned_m": _array(state.position_ned_m),
        "quat_xyzw": _array(state.quat_xyzw),
        "linear_velocity_body_mps": _array(state.linear_velocity_body_mps),
        "angular_velocity_body_rps": _array(state.angular_velocity_body_rps),
        "frame_id": state.frame_id,
    }


def _obstacle_map_payload(world: WorldSnapshot) -> dict[str, Any]:
    obstacle_map = world.obstacle_map
    return {
        "bounds_ned_m": _array(obstacle_map.bounds_ned_m),
        "obstacles": [
            {
                "obstacle_id": obstacle.obstacle_id,
                "kind": obstacle.kind,
                "center_ned_m": _array(obstacle.center_ned_m),
                "radius_m": None if obstacle.radius_m is None else _float(obstacle.radius_m),
                "half_extents_m": (
                    None
                    if obstacle.half_extents_m is None
                    else _array(obstacle.half_extents_m)
                ),
                "height_m": None if obstacle.height_m is None else _float(obstacle.height_m),
            }
            for obstacle in obstacle_map.obstacles
        ],
    }


def _current_fingerprint(world: WorldSnapshot) -> dict[str, Any] | None:
    current = world.current_field
    if isinstance(current, UniformCurrent):
        return {
            "kind": "uniform",
            "velocity_ned_mps": _array(current.velocity_ned_mps),
        }
    if isinstance(current, DepthShearCurrent):
        return {
            "kind": "depth_shear",
            "reference_velocity_ned_mps": _array(current.reference_velocity_ned_mps),
            "shear_ned_per_m": _array(current.shear_ned_per_m),
            "reference_depth_m": _float(current.reference_depth_m),
        }
    return None


def _vehicle_payload(vehicle: VehicleSpec) -> dict[str, Any]:
    return {
        "name": vehicle.name,
        "length_m": _float(vehicle.length_m),
        "diameter_m": _float(vehicle.diameter_m),
        "mass_kg": _float(vehicle.mass_kg),
        "min_speed_mps": _float(vehicle.min_speed_mps),
        "max_speed_mps": _float(vehicle.max_speed_mps),
        "max_accel_mps2": _float(vehicle.max_accel_mps2),
        "max_yaw_rate_rps": _float(vehicle.max_yaw_rate_rps),
        "max_pitch_rate_rps": _float(vehicle.max_pitch_rate_rps),
        "max_rudder_angle_rad": _optional_float(vehicle.max_rudder_angle_rad),
        "max_rudder_rate_rad_s": _optional_float(vehicle.max_rudder_rate_rad_s),
        "max_stern_plane_angle_rad": _optional_float(vehicle.max_stern_plane_angle_rad),
        "max_stern_plane_rate_rad_s": _optional_float(vehicle.max_stern_plane_rate_rad_s),
        "max_propeller_rpm": _optional_float(vehicle.max_propeller_rpm),
    }


def _config_payload(config: RecoveryConfig) -> dict[str, Any]:
    return {
        "horizon_s": _float(config.horizon_s),
        "required_clearance_m": _float(config.required_clearance_m),
        "certificate_validity_s": _float(config.certificate_validity_s),
        "sample_time_s": _float(config.sample_time_s),
        "max_pitch_angle_rad": _float(config.max_pitch_angle_rad),
        "max_translation_step_m": _float(config.max_translation_step_m),
        "max_rotation_step_rad": _float(config.max_rotation_step_rad),
        "error_bounds": {
            "position_m": _float(config.error_bounds.position_m),
            "heading_rad": _float(config.error_bounds.heading_rad),
            "current_mps": _float(config.error_bounds.current_mps),
        },
        "constraint_mode": config.constraint_mode.value,
        "clearance_model": config.clearance_model.value,
        "unbounded_yaw_rate_factor": _float(config.unbounded_yaw_rate_factor),
    }


def _float(value: float) -> str:
    return float(value).hex()


def _optional_float(value: float | None) -> str | None:
    return None if value is None else _float(value)


def _array(value: FloatArray) -> list[str]:
    array = np.asarray(value, dtype=np.float64)
    return [_float(item) for item in array.reshape(-1)]


__all__ = [
    "CHECKER_CACHE_SCHEMA",
    "RecoveryQueryPolicy",
    "RecoveryQueryMode",
    "RecoveryQueryResult",
    "RecoveryQuerySession",
    "RecoveryQueryStats",
    "RecoveryQueryStatus",
    "StatePrefilterResult",
    "prefilter_state",
]
