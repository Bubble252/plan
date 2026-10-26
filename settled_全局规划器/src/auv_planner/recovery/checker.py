from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil
from time import perf_counter
from typing import Any

import numpy as np

from ..contracts import (
    FloatArray,
    RecoveryCertificate,
    State,
    Trajectory,
    VehicleSpec,
    WorldSnapshot,
)
from ..geometry import (
    hull_clearance_details,
    hull_from_vehicle,
    point_clearance_details,
    trajectory_sweep,
)
from .maneuvers import build_maneuvers, coast_trajectory
from .models import (
    DEFAULT_DIAGNOSIS_DELAYS_S,
    DEFAULT_FAULTS,
    ClearanceModel,
    ConstraintMode,
    FaultCheck,
    FaultProfile,
    FaultSpec,
    ManeuverCandidate,
    ManeuverCheck,
    RecoveryAssessment,
    RecoveryConfig,
)


def _validate_point(value: FloatArray | None) -> FloatArray | None:
    if value is None:
        return None
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (3,):
        raise ValueError("safe_anchor_ned_m must have shape (3,)")
    if not np.all(np.isfinite(point)):
        raise ValueError("safe_anchor_ned_m must contain only finite values")
    return point


def _point_sweep(
    trajectory: Trajectory,
    world: WorldSnapshot,
    config: RecoveryConfig,
) -> tuple[float, bool, bool, int]:
    best_clearance = float("inf")
    collision = False
    out_of_bounds = False
    sampled = 0

    def inspect(position: FloatArray) -> None:
        nonlocal best_clearance, collision, out_of_bounds, sampled
        details = point_clearance_details(world.obstacle_map, position)
        best_clearance = min(best_clearance, details.clearance_m)
        collision = collision or details.collision
        out_of_bounds = out_of_bounds or details.out_of_bounds
        sampled += 1

    inspect(trajectory.position_ned_m[0])
    for index in range(trajectory.t_s.shape[0] - 1):
        start = trajectory.position_ned_m[index]
        end = trajectory.position_ned_m[index + 1]
        distance = float(np.linalg.norm(end - start))
        steps = max(1, int(ceil(distance / config.max_translation_step_m)))
        for step in range(1, steps + 1):
            inspect(start + (step / steps) * (end - start))
    return float(best_clearance), collision, out_of_bounds, sampled


def _trajectory_clearance(
    trajectory: Trajectory,
    world: WorldSnapshot,
    vehicle: VehicleSpec,
    config: RecoveryConfig,
) -> tuple[float, bool, bool, int]:
    if config.clearance_model is ClearanceModel.POINT_DIAGNOSTIC:
        return _point_sweep(trajectory, world, config)
    sweep = trajectory_sweep(
        trajectory,
        world.obstacle_map,
        vehicle,
        max_translation_step_m=config.max_translation_step_m,
        max_rotation_step_rad=config.max_rotation_step_rad,
    )
    return sweep.min_clearance_m, sweep.collision, sweep.out_of_bounds, sweep.sampled_poses


def _state_clearance(
    state: State,
    world: WorldSnapshot,
    vehicle: VehicleSpec,
    config: RecoveryConfig,
) -> float:
    if config.clearance_model is ClearanceModel.POINT_DIAGNOSTIC:
        return point_clearance_details(world.obstacle_map, state.position_ned_m).clearance_m
    return hull_clearance_details(world.obstacle_map, state, hull_from_vehicle(vehicle)).clearance_m


def _candidate_check(
    candidate: ManeuverCandidate,
    world: WorldSnapshot,
    vehicle: VehicleSpec,
    config: RecoveryConfig,
    required_clearance_m: float,
) -> ManeuverCheck:
    min_clearance, collision, out_of_bounds, _ = _trajectory_clearance(
        candidate.trajectory,
        world,
        vehicle,
        config,
    )
    reasons: list[str] = []
    if collision:
        reasons.append("collision")
    if out_of_bounds:
        reasons.append("out_of_bounds")
    if min_clearance < required_clearance_m:
        reasons.append("clearance_below_required")
    if config.constraint_mode is ConstraintMode.VEHICLE and candidate.vehicle_limit_exceedances:
        reasons.append("actuator_limit:" + ",".join(candidate.vehicle_limit_exceedances))
    failure_reason = ",".join(dict.fromkeys(reasons)) if reasons else None
    return ManeuverCheck(
        candidate=candidate,
        safe=failure_reason is None,
        min_clearance_m=float(min_clearance),
        required_clearance_m=float(required_clearance_m),
        margin_m=float(min_clearance - required_clearance_m),
        failure_reason=failure_reason,
    )


def _best_check(checks: Sequence[ManeuverCheck]) -> ManeuverCheck | None:
    if not checks:
        return None
    return max(checks, key=lambda check: (check.margin_m, check.candidate.name))


def _fault_check(
    state: State,
    world: WorldSnapshot,
    vehicle: VehicleSpec,
    fault: FaultSpec,
    diagnosis_delay_s: float,
    config: RecoveryConfig,
    *,
    required_clearance_m: float,
    safe_anchor_ned_m: FloatArray | None,
) -> FaultCheck:
    capability = fault.capability(diagnosis_delay_s)
    coast_min_clearance: float | None = None
    coast_safe = True
    coast_failure: str | None = None
    post_delay_state = state

    if diagnosis_delay_s > 0.0:
        coast, _ = coast_trajectory(
            state,
            world,
            vehicle,
            fault,
            config,
            duration_s=min(diagnosis_delay_s, config.horizon_s),
        )
        coast_min_clearance, collision, out_of_bounds, _ = _trajectory_clearance(
            coast,
            world,
            vehicle,
            config,
        )
        coast_reasons: list[str] = []
        if collision:
            coast_reasons.append("collision")
        if out_of_bounds:
            coast_reasons.append("out_of_bounds")
        if coast_min_clearance < required_clearance_m:
            coast_reasons.append("clearance_below_required")
        coast_safe = not coast_reasons
        coast_failure = ",".join(coast_reasons) if coast_reasons else None
        post_delay_state = State(
            t_s=coast.t_s[-1],
            position_ned_m=coast.position_ned_m[-1].copy(),
            quat_xyzw=coast.quat_xyzw[-1].copy(),
            linear_velocity_body_mps=coast.linear_velocity_body_mps[-1].copy(),
            angular_velocity_body_rps=coast.angular_velocity_body_rps[-1].copy(),
            frame_id=coast.frame_id,
        )

    remaining_s = config.horizon_s - diagnosis_delay_s
    if remaining_s <= 1e-9:
        return FaultCheck(
            fault=fault,
            diagnosis_delay_s=diagnosis_delay_s,
            capability=capability,
            safe=False,
            coast_min_clearance_m=coast_min_clearance,
            selected=None,
            checks=(),
            unavailable_maneuvers=(),
            margin_m=None,
            failure_reason="diagnosis_delay_exhausts_horizon",
        )

    candidates = build_maneuvers(
        post_delay_state,
        world,
        vehicle,
        fault,
        config,
        duration_s=remaining_s,
        safe_anchor_ned_m=safe_anchor_ned_m,
    )
    available = tuple(
        candidate
        for candidate in candidates
        if set(candidate.requires_actuators).issubset(capability.available_actuators)
    )
    unavailable = tuple(
        candidate.name
        for candidate in candidates
        if candidate.name not in {available_candidate.name for available_candidate in available}
    )
    checks = tuple(
        _candidate_check(candidate, world, vehicle, config, required_clearance_m)
        for candidate in available
    )
    safe_checks = tuple(check for check in checks if check.safe)
    selected = (
        max(safe_checks, key=lambda check: (check.margin_m, check.candidate.name))
        if safe_checks
        else None
    )
    best_attempt = _best_check(checks)
    combined_margin: float | None = None
    if best_attempt is not None:
        combined_margin = best_attempt.margin_m
    if coast_min_clearance is not None:
        coast_margin = coast_min_clearance - required_clearance_m
        combined_margin = (
            coast_margin if combined_margin is None else min(combined_margin, coast_margin)
        )

    safe = coast_safe and selected is not None
    if safe:
        assert selected is not None
        if coast_min_clearance is not None:
            combined_margin = min(
                selected.margin_m,
                coast_min_clearance - required_clearance_m,
            )
        else:
            combined_margin = selected.margin_m
        failure_reason = None
    elif not coast_safe:
        failure_reason = "diagnosis_coast_unsafe" + (f":{coast_failure}" if coast_failure else "")
    elif not available:
        failure_reason = "no_available_backup_maneuver"
    elif not checks:
        failure_reason = "no_backup_maneuver_generated"
    else:
        failure_reason = "no_safe_backup_maneuver"

    return FaultCheck(
        fault=fault,
        diagnosis_delay_s=float(diagnosis_delay_s),
        capability=capability,
        safe=safe,
        coast_min_clearance_m=coast_min_clearance,
        selected=selected,
        checks=checks,
        unavailable_maneuvers=unavailable,
        margin_m=combined_margin,
        failure_reason=failure_reason,
    )


@dataclass(frozen=True)
class _FaultTask:
    state: State
    world: WorldSnapshot
    vehicle: VehicleSpec
    fault: FaultSpec
    diagnosis_delay_s: float
    config: RecoveryConfig
    required_clearance_m: float
    safe_anchor_ned_m: FloatArray | None


def _fault_check_worker(task: _FaultTask) -> FaultCheck:
    return _fault_check(
        task.state,
        task.world,
        task.vehicle,
        task.fault,
        task.diagnosis_delay_s,
        task.config,
        required_clearance_m=task.required_clearance_m,
        safe_anchor_ned_m=task.safe_anchor_ned_m,
    )


def _run_fault_checks(
    tasks: Sequence[_FaultTask],
    *,
    parallel_workers: int,
) -> tuple[FaultCheck, ...]:
    if parallel_workers <= 1 or len(tasks) <= 1:
        return tuple(_fault_check_worker(task) for task in tasks)

    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=min(parallel_workers, len(tasks))) as executor:
        return tuple(executor.map(_fault_check_worker, tasks))


def _certificate_assumptions(config: RecoveryConfig, vehicle: VehicleSpec) -> tuple[str, ...]:
    inflation = config.error_clearance_inflation_m(vehicle)
    assumptions = [
        (
            f"finite-horizon discrete check: horizon={config.horizon_s:g}s, "
            f"dt={config.sample_time_s:g}s; no continuous-time guarantee"
        ),
        (
            f"clearance threshold={config.required_clearance_m:g}m plus "
            f"unvalidated error inflation={inflation:.3f}m"
        ),
        (
            "error inflation uses position/heading/current bounds "
            f"{config.error_bounds.position_m:g}m, {config.error_bounds.heading_rad:g}rad, "
            f"{config.error_bounds.current_mps:g}m/s"
        ),
        "reduced-order kinematic backup model: body +x water speed plus sampled current; no sway/heave/roll dynamics",
        (
            "fin angle and fin slew are represented by yaw/pitch-rate envelopes; "
            f"pitch commands are limited to +/-{np.rad2deg(config.max_pitch_angle_rad):g}deg"
        ),
        (
            "backup set is finite: brake decelerates only to minimum speed; the safe-corridor "
            "candidate selects a target from a fixed 16x3 direction fan, not a search planner"
        ),
        "diagnosis delay is modeled as a constant-command coast inside the same horizon before a backup starts",
        "thruster loss scales the speed envelope and acceleration; a stuck fin applies a forced residual rate",
        "fault set is limited to thruster efficiency loss, one stuck yaw/pitch fin group, and diagnosis delay",
        "no sensor noise, communication outage, or energy depletion is certified",
        "current is sampled from the supplied world model; M6 experiments use only the M5-validated current envelope",
    ]
    if config.clearance_model is not ClearanceModel.FULL_HULL:
        assumptions.append(
            "diagnostic clearance model is point-only and cannot issue a full-hull certificate"
        )
    if config.constraint_mode is not ConstraintMode.VEHICLE:
        assumptions.append("diagnostic turn envelope relaxes the vehicle yaw-rate limit")
    return tuple(assumptions)


def _fault_summary(check: FaultCheck) -> dict[str, Any]:
    selected = check.selected
    best = _best_check(check.checks)
    return {
        "fault": check.fault.name,
        "delay_s": check.diagnosis_delay_s,
        "safe": check.safe,
        "selected_maneuver": selected.candidate.name if selected is not None else None,
        "selected_margin_m": selected.margin_m if selected is not None else None,
        "best_attempt": best.candidate.name if best is not None else None,
        "best_attempt_clearance_m": best.min_clearance_m if best is not None else None,
        "best_attempt_margin_m": best.margin_m if best is not None else None,
        "coast_min_clearance_m": check.coast_min_clearance_m,
        "failure_reason": check.failure_reason,
        "unavailable_maneuvers": list(check.unavailable_maneuvers),
        "diagnosed_available_actuators": list(check.capability.available_actuators),
    }


class RecoverabilityChecker:
    """Finite-horizon recoverability checker with explicit certificate assumptions."""

    def assess(
        self,
        state: State,
        world: WorldSnapshot,
        vehicle: VehicleSpec,
        *,
        config: RecoveryConfig | None = None,
        profile: FaultProfile | None = None,
        faults: Sequence[FaultSpec] | None = None,
        diagnosis_delays_s: Sequence[float] | None = None,
        safe_anchor_ned_m: FloatArray | None = None,
        early_stop: bool = False,
        parallel_workers: int = 1,
    ) -> RecoveryAssessment:
        if not isinstance(state, State):
            raise TypeError("state must be a State")
        if not isinstance(world, WorldSnapshot):
            raise TypeError("world must be a WorldSnapshot")
        if not isinstance(vehicle, VehicleSpec):
            raise TypeError("vehicle must be a VehicleSpec")
        if not isinstance(early_stop, bool):
            raise TypeError("early_stop must be a bool")
        if not isinstance(parallel_workers, int) or parallel_workers < 1:
            raise ValueError("parallel_workers must be a positive integer")
        if early_stop and parallel_workers > 1:
            raise ValueError("early_stop cannot be combined with parallel_workers > 1")
        active_config = RecoveryConfig() if config is None else config
        if not isinstance(active_config, RecoveryConfig):
            raise TypeError("config must be a RecoveryConfig")
        if profile is not None:
            if not isinstance(profile, FaultProfile):
                raise TypeError("profile must be a FaultProfile")
            if faults is not None or diagnosis_delays_s is not None:
                raise ValueError(
                    "profile cannot be combined with explicit faults or diagnosis delays"
                )
            active_profile = profile
        else:
            active_profile = FaultProfile(
                name="custom",
                faults=tuple(DEFAULT_FAULTS if faults is None else faults),
                diagnosis_delays_s=tuple(
                    DEFAULT_DIAGNOSIS_DELAYS_S if diagnosis_delays_s is None else diagnosis_delays_s
                ),
            )
        if not active_profile.faults:
            raise ValueError("at least one fault specification is required")
        if not active_profile.diagnosis_delays_s:
            raise ValueError("at least one diagnosis delay is required")
        anchor = _validate_point(safe_anchor_ned_m)
        start_time = perf_counter()
        required_clearance = active_config.required_inflated_clearance_m(vehicle)
        current_clearance = _state_clearance(state, world, vehicle, active_config)
        tasks = tuple(
            _FaultTask(
                state=state,
                world=world,
                vehicle=vehicle,
                fault=fault,
                diagnosis_delay_s=float(delay),
                config=active_config,
                required_clearance_m=required_clearance,
                safe_anchor_ned_m=anchor,
            )
            for fault in active_profile.faults
            for delay in active_profile.diagnosis_delays_s
        )
        if early_stop:
            completed: list[FaultCheck] = []
            for task in tasks:
                check = _fault_check_worker(task)
                completed.append(check)
                if not check.safe:
                    break
            checks = tuple(completed)
        else:
            checks = _run_fault_checks(tasks, parallel_workers=parallel_workers)
        early_stopped = len(checks) < len(tasks)
        failures = tuple(check for check in checks if not check.safe)
        diagnostic_mode = (
            active_config.clearance_model is not ClearanceModel.FULL_HULL
            or active_config.constraint_mode is not ConstraintMode.VEHICLE
        )
        valid = not failures and not diagnostic_mode and not early_stopped
        reason: str | None = None
        if failures:
            primary = max(
                failures,
                key=lambda check: (check.diagnosis_delay_s, check.fault.name),
            )
            reason = f"no_recoverable_backup:{primary.label}:{primary.failure_reason}"
        elif diagnostic_mode:
            reason = "diagnostic_mode_does_not_issue_safety_certificate"

        selected_names = tuple(
            dict.fromkeys(
                check.selected.candidate.name for check in checks if check.selected is not None
            )
        )
        margins = tuple(check.margin_m for check in checks if check.margin_m is not None)
        margin = min(margins) if margins else None
        valid_from = float(state.t_s)
        valid_until = float(state.t_s + active_config.certificate_validity_s)
        assumptions = _certificate_assumptions(active_config, vehicle)
        diagnostics: dict[str, Any] = {
            "horizon_s": active_config.horizon_s,
            "sample_time_s": active_config.sample_time_s,
            "required_clearance_m": active_config.required_clearance_m,
            "error_inflation_m": active_config.error_clearance_inflation_m(vehicle),
            "required_inflated_clearance_m": required_clearance,
            "current_clearance_m": current_clearance,
            "constraint_mode": active_config.constraint_mode.value,
            "clearance_model": active_config.clearance_model.value,
            "diagnostic_only": diagnostic_mode,
            "profile_name": active_profile.name,
            "profile_digest": active_profile.digest,
            "profile_version": active_profile.version,
            "profile_check_count": active_profile.check_count,
            "evaluated_check_count": len(checks),
            "early_stopped": early_stopped,
            "evaluation_mode": (
                "online_fast"
                if early_stop
                else ("full_parallel" if parallel_workers > 1 else "full_sequential")
            ),
            "fault_checks": [_fault_summary(check) for check in checks],
        }
        certificate = RecoveryCertificate(
            valid=valid,
            assumptions=assumptions,
            fault_set=tuple(check.label for check in checks),
            backup_maneuvers=selected_names,
            valid_from_s=valid_from,
            valid_until_s=valid_until,
            margin_m=margin,
            reason=reason,
            profile_name=active_profile.name,
            profile_digest=active_profile.digest,
            diagnostics=diagnostics,
        )
        computation_time = perf_counter() - start_time
        return RecoveryAssessment(
            certificate=certificate,
            fault_checks=checks,
            computation_time_s=float(computation_time),
            current_clearance_m=float(current_clearance),
            profile=active_profile,
            early_stopped=early_stopped,
        )


__all__ = ["RecoverabilityChecker"]
