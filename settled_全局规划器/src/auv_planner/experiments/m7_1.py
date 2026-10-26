from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from ..attitude import forward_quaternion
from ..contracts import (
    CapabilitySet,
    FloatArray,
    PlanReason,
    PlanRequest,
    PlanResult,
    PlanStatus,
    ScenarioConfig,
    State,
    TaskSpec,
    Trajectory,
)
from ..evaluation.runner import EvaluationRecord, evaluate_plan, run_scenario
from ..geometry import trajectory_sweep
from ..metrics import compute_path_metrics
from ..planners.astar import AStarPlanner
from ..planners.common import SegmentValidator, build_trajectory
from ..planners.rrt_star import RRTStarPlanner
from ..recovery import (
    FaultProfile,
    RecoveryConfig,
    RecoveryQueryMode,
    RecoveryQueryPolicy,
    RecoveryQueryResult,
    RecoveryQuerySession,
    RecoveryQueryStatus,
    fault_profile_by_name,
    prefilter_state,
)
from .m7_budget import hardware_manifest, m7_probes

M71_AUDIT_INTERVAL_S = 1.0
M71_ONLINE_BUDGET_S = 10.0


@dataclass(frozen=True)
class M71Config:
    audit_interval_s: float = M71_AUDIT_INTERVAL_S
    online_budget_s: float = M71_ONLINE_BUDGET_S
    profile_name: str = "P1_degraded"
    full_workers: int = 8
    seed: int = 0
    astar_resolution_m: float = 0.5
    astar_timeout_s: float = 10.0
    rrt_step_size_m: float = 0.6
    rrt_iterations: int = 2_000
    rrt_timeout_s: float = 3.0
    repair_offsets_m: tuple[float, ...] = (1.0, 1.6, 2.2)
    max_repair_candidates: int = 12
    case_names: tuple[str, ...] | None = None
    planner_names: tuple[str, ...] = ("astar", "rrt_star")

    def __post_init__(self) -> None:
        if not np.isfinite(self.audit_interval_s) or self.audit_interval_s <= 0.0:
            raise ValueError("audit_interval_s must be finite and positive")
        if self.audit_interval_s > 1.0:
            raise ValueError("audit_interval_s must not exceed 1 s")
        if not np.isfinite(self.online_budget_s) or self.online_budget_s <= 0.0:
            raise ValueError("online_budget_s must be finite and positive")
        if not self.profile_name:
            raise ValueError("profile_name must not be empty")
        if not isinstance(self.full_workers, int) or self.full_workers < 1:
            raise ValueError("full_workers must be a positive integer")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for name in (
            "astar_resolution_m",
            "astar_timeout_s",
            "rrt_step_size_m",
            "rrt_timeout_s",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.rrt_iterations, int) or self.rrt_iterations < 1:
            raise ValueError("rrt_iterations must be a positive integer")
        if not self.repair_offsets_m or any(
            not np.isfinite(value) or value <= 0.0 for value in self.repair_offsets_m
        ):
            raise ValueError("repair_offsets_m must contain positive finite values")
        if not isinstance(self.max_repair_candidates, int) or self.max_repair_candidates < 0:
            raise ValueError("max_repair_candidates must be a non-negative integer")
        if self.case_names is not None:
            if not self.case_names or any(not name for name in self.case_names):
                raise ValueError("case_names must be a non-empty sequence of names")
            object.__setattr__(self, "case_names", tuple(self.case_names))
        if not self.planner_names or any(
            name not in {"astar", "rrt_star"} for name in self.planner_names
        ):
            raise ValueError("planner_names must contain astar and/or rrt_star")
        object.__setattr__(self, "planner_names", tuple(self.planner_names))
        object.__setattr__(self, "repair_offsets_m", tuple(float(v) for v in self.repair_offsets_m))

    @property
    def profile(self) -> FaultProfile:
        return fault_profile_by_name(self.profile_name)


@dataclass(frozen=True)
class M71CaseSpec:
    name: str
    scenario: ScenarioConfig
    safe_anchor_ned_m: FloatArray | None
    include_in_aggregate: bool


@dataclass(frozen=True)
class AuditPoint:
    index: int
    t_s: float
    position_ned_m: FloatArray
    quat_xyzw: FloatArray
    clearance_m: float
    prefilter_reason: str | None
    online_status: str
    online_reason: str | None
    online_elapsed_s: float
    online_early_stopped: bool
    online_evaluated_checks: int
    full_status: str
    full_reason: str | None
    full_elapsed_s: float
    full_early_stopped: bool
    full_evaluated_checks: int
    certificate_valid_from_s: float | None
    certificate_valid_until_s: float | None
    failure_source: str | None


@dataclass(frozen=True)
class PathAudit:
    case_name: str
    scenario_name: str
    planner_name: str
    variant: str
    include_in_aggregate: bool
    plan_status: str
    plan_success: bool
    plan_reason: str | None
    planning_time_s: float
    repair_time_s: float
    trajectory: Trajectory | None
    points: tuple[AuditPoint, ...]
    prefilter_time_s: float
    online_time_s: float
    full_time_s: float
    path_length_m: float
    duration_s: float
    min_clearance_m: float
    coverage_ratio: float
    max_uncovered_s: float
    coverage_intervals: tuple[tuple[float, float], ...]
    uncovered_intervals: tuple[tuple[float, float], ...]
    failure_sources: tuple[tuple[str, int], ...]

    @property
    def valid_points(self) -> int:
        return sum(point.full_status == RecoveryQueryStatus.VALID.value for point in self.points)

    @property
    def invalid_points(self) -> int:
        return sum(point.full_status == RecoveryQueryStatus.INVALID.value for point in self.points)

    @property
    def unknown_points(self) -> int:
        return sum(point.full_status == RecoveryQueryStatus.UNKNOWN.value for point in self.points)

    @property
    def end_to_end_s(self) -> float:
        return self.planning_time_s + self.prefilter_time_s + self.online_time_s

    @property
    def full_end_to_end_s(self) -> float:
        return self.end_to_end_s + self.full_time_s


@dataclass(frozen=True)
class RepairAttempt:
    candidate_index: int
    source: str
    via_ned_m: FloatArray | None
    candidate_status: str | None
    geometric_safe: bool
    plan_success: bool
    coverage_ratio: float
    max_uncovered_s: float
    accepted: bool
    reason: str


@dataclass(frozen=True)
class M71CaseResult:
    case_name: str
    scenario_name: str
    planner_name: str
    include_in_aggregate: bool
    baseline: PathAudit
    repair: PathAudit | None
    repair_attempts: tuple[RepairAttempt, ...]
    repair_accepted: bool
    repair_reason: str


@dataclass(frozen=True)
class M71ExperimentResult:
    config: M71Config
    cases: tuple[M71CaseResult, ...]
    elapsed_s: float


def m71_case_specs(config: M71Config | None = None) -> tuple[M71CaseSpec, ...]:
    active = M71Config() if config is None else config
    selected = None if active.case_names is None else set(active.case_names)
    probes = [
        M71CaseSpec(
            name=probe.name,
            scenario=probe.scenario,
            safe_anchor_ned_m=probe.safe_anchor_ned_m,
            include_in_aggregate=probe.include_in_aggregate,
        )
        for probe in m7_probes()
    ]
    if selected is not None:
        known = {probe.name for probe in probes}
        missing = selected - known
        if missing:
            raise ValueError(f"unknown M7.1 cases: {sorted(missing)}")
        probes = [probe for probe in probes if probe.name in selected]
    return tuple(probes)


def _planner(name: str, config: M71Config) -> AStarPlanner | RRTStarPlanner:
    if name == "astar":
        return AStarPlanner(
            resolution_m=config.astar_resolution_m,
            timeout_s=config.astar_timeout_s,
            collision_step_m=0.1,
        )
    if name == "rrt_star":
        return RRTStarPlanner(
            step_size_m=config.rrt_step_size_m,
            max_iterations=config.rrt_iterations,
            timeout_s=config.rrt_timeout_s,
            collision_step_m=0.1,
        )
    raise ValueError(f"unknown planner: {name}")


def _plan_request(
    scenario: ScenarioConfig,
    *,
    request_id: str,
    seed: int,
    start: State | None = None,
    goal_ned_m: FloatArray | None = None,
    goal_radius_m: float | None = None,
    timeout_s: float | None = None,
) -> PlanRequest:
    constraints = dict(scenario.constraints)
    if timeout_s is not None:
        constraints["planning_timeout_s"] = timeout_s
    return PlanRequest(
        request_id=request_id,
        start=scenario.start if start is None else start,
        task=TaskSpec(
            goal_ned_m=scenario.goal_ned_m if goal_ned_m is None else goal_ned_m,
            goal_radius_m=scenario.goal_radius_m if goal_radius_m is None else goal_radius_m,
            constraints=constraints,
        ),
        world=scenario.world,
        vehicle=scenario.vehicle,
        capability=CapabilitySet(
            available_actuators=("propeller", "yaw_fin", "pitch_fin"),
            effectiveness={"propeller": 1.0, "yaw_fin": 1.0, "pitch_fin": 1.0},
        ),
        seed=seed,
        reason=PlanReason.INITIAL,
    )


def sample_trajectory_states(trajectory: Trajectory, interval_s: float) -> tuple[State, ...]:
    if not np.isfinite(interval_s) or interval_s <= 0.0:
        raise ValueError("interval_s must be finite and positive")
    times = trajectory.t_s
    targets = [
        float(value)
        for value in np.arange(float(times[0]), float(times[-1]), interval_s, dtype=np.float64)
    ]
    if not targets or float(targets[-1]) < float(times[-1]) - 1e-12:
        targets.append(float(times[-1]))
    elif abs(float(targets[-1]) - float(times[-1])) <= 1e-12:
        targets[-1] = float(times[-1])
    states: list[State] = []
    for target in targets:
        index = int(np.searchsorted(times, target, side="right") - 1)
        index = int(np.clip(index, 0, times.shape[0] - 1))
        if index >= times.shape[0] - 1 or abs(float(times[index]) - float(target)) <= 1e-12:
            if abs(float(times[index]) - float(target)) <= 1e-12:
                position = trajectory.position_ned_m[index]
                quat = trajectory.quat_xyzw[index]
                velocity = trajectory.linear_velocity_body_mps[index]
                angular = trajectory.angular_velocity_body_rps[index]
            else:
                position = trajectory.position_ned_m[-1]
                quat = trajectory.quat_xyzw[-1]
                velocity = trajectory.linear_velocity_body_mps[-1]
                angular = trajectory.angular_velocity_body_rps[-1]
        else:
            next_index = index + 1
            duration = float(times[next_index] - times[index])
            fraction = 0.0 if duration <= 1e-12 else float((target - times[index]) / duration)
            position = (1.0 - fraction) * trajectory.position_ned_m[index] + fraction * trajectory.position_ned_m[next_index]
            quat = _slerp(
                trajectory.quat_xyzw[index],
                trajectory.quat_xyzw[next_index],
                fraction,
            )
            velocity = (
                (1.0 - fraction) * trajectory.linear_velocity_body_mps[index]
                + fraction * trajectory.linear_velocity_body_mps[next_index]
            )
            angular = (
                (1.0 - fraction) * trajectory.angular_velocity_body_rps[index]
                + fraction * trajectory.angular_velocity_body_rps[next_index]
            )
        states.append(
            State(
                t_s=float(target),
                position_ned_m=np.asarray(position, dtype=np.float64).copy(),
                quat_xyzw=np.asarray(quat, dtype=np.float64).copy(),
                linear_velocity_body_mps=np.asarray(velocity, dtype=np.float64).copy(),
                angular_velocity_body_rps=np.asarray(angular, dtype=np.float64).copy(),
                frame_id=trajectory.frame_id,
            )
        )
    return tuple(states)


def _slerp(quat_a: FloatArray, quat_b: FloatArray, fraction: float) -> FloatArray:
    first = np.asarray(quat_a, dtype=np.float64)
    second = np.asarray(quat_b, dtype=np.float64)
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = first + fraction * (second - first)
        return result / np.linalg.norm(result)
    theta = float(np.arccos(dot))
    sin_theta = float(np.sin(theta))
    if sin_theta <= 1e-12:
        return first.copy()
    first_weight = float(np.sin((1.0 - fraction) * theta) / sin_theta)
    second_weight = float(np.sin(fraction * theta) / sin_theta)
    return first_weight * first + second_weight * second


def _merge_intervals(
    intervals: Iterable[tuple[float, float]],
    *,
    start: float,
    end: float,
) -> tuple[tuple[float, float], ...]:
    clipped = sorted(
        (max(start, float(lo)), min(end, float(hi)))
        for lo, hi in intervals
        if float(hi) > start and float(lo) < end
    )
    if not clipped:
        return ()
    merged: list[tuple[float, float]] = [clipped[0]]
    for lo, hi in clipped[1:]:
        previous_lo, previous_hi = merged[-1]
        if lo <= previous_hi + 1e-12:
            merged[-1] = (previous_lo, max(previous_hi, hi))
        else:
            merged.append((lo, hi))
    return tuple(merged)


def coverage_metrics(
    points: Sequence[AuditPoint],
    *,
    start_s: float,
    end_s: float,
) -> tuple[float, float, tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]]:
    if end_s < start_s:
        raise ValueError("end_s must not precede start_s")
    intervals = _merge_intervals(
        (
            (point.certificate_valid_from_s, point.certificate_valid_until_s)
            for point in points
            if point.full_status == RecoveryQueryStatus.VALID.value
            and point.certificate_valid_from_s is not None
            and point.certificate_valid_until_s is not None
        ),
        start=start_s,
        end=end_s,
    )
    duration = end_s - start_s
    covered = sum(hi - lo for lo, hi in intervals)
    uncovered: list[tuple[float, float]] = []
    cursor = start_s
    for lo, hi in intervals:
        if lo > cursor + 1e-12:
            uncovered.append((cursor, lo))
        cursor = max(cursor, hi)
    if cursor < end_s - 1e-12:
        uncovered.append((cursor, end_s))
    ratio = 0.0 if duration <= 1e-12 else min(1.0, covered / duration)
    max_uncovered = max((hi - lo for lo, hi in uncovered), default=0.0)
    return ratio, max_uncovered, intervals, tuple(uncovered)


def _failure_source(
    *,
    prefilter_reason: str | None,
    full_result: RecoveryQueryResult | None,
) -> str | None:
    if full_result is not None and full_result.status is RecoveryQueryStatus.VALID:
        return None
    prefix = prefilter_reason or ""
    if "collision" in prefix:
        return "initial_obstacle_collision"
    if "out_of_bounds" in prefix:
        return "initial_out_of_bounds"
    if "clearance_below_required" in prefix:
        return "initial_clearance"
    if full_result is None:
        return "certificate_unavailable"
    reason = str(full_result.reason or "")
    if "diagnosis_coast_unsafe" in reason:
        return "diagnosis_coast"
    assessment = full_result.assessment
    if assessment is not None:
        maneuver_reasons = [
            str(check.failure_reason or "")
            for fault_check in assessment.fault_checks
            for check in fault_check.checks
        ]
        fault_reasons = [str(check.failure_reason or "") for check in assessment.fault_checks]
        if any("no_available_backup_maneuver" in text for text in fault_reasons):
            return "backup_unavailable"
        if any("no_backup_maneuver_generated" in text for text in fault_reasons):
            return "backup_unavailable"
        if any("out_of_bounds" in text for text in maneuver_reasons):
            return "backup_out_of_bounds"
        if any("collision" in text for text in maneuver_reasons):
            return "backup_collision"
        if any("clearance_below_required" in text for text in maneuver_reasons):
            return "backup_clearance"
        if any("actuator_limit" in text for text in maneuver_reasons):
            return "actuator_limit"
        if any(not check.checks for check in assessment.fault_checks):
            return "backup_unavailable"
    if "no_available_backup" in reason or "no_backup_maneuver_generated" in reason:
        return "backup_unavailable"
    if "no_safe_backup" in reason:
        return "certificate_model_conservative"
    return "certificate_unclassified"


def audit_trajectory(
    trajectory: Trajectory,
    scenario: ScenarioConfig,
    *,
    case_name: str,
    planner_name: str,
    variant: str,
    include_in_aggregate: bool,
    plan_status: str,
    plan_success: bool,
    plan_reason: str | None,
    planning_time_s: float,
    repair_time_s: float,
    config: M71Config,
    safe_anchor_ned_m: FloatArray | None = None,
) -> PathAudit:
    states = sample_trajectory_states(trajectory, config.audit_interval_s)
    profile = config.profile
    recovery_config = RecoveryConfig()

    prefilters = []
    prefilter_time = 0.0
    for state in states:
        started = perf_counter()
        prefilter = prefilter_state(state, scenario.world, scenario.vehicle, recovery_config)
        prefilter_time += perf_counter() - started
        prefilters.append(prefilter)

    online_session = RecoveryQuerySession(
        policy=RecoveryQueryPolicy(
            budget_s=config.online_budget_s,
            use_cache=True,
            use_prefilter=False,
            mode=RecoveryQueryMode.ONLINE_FAST,
            parallel_workers=1,
        )
    )
    online_results: list[RecoveryQueryResult | None] = []
    online_time = 0.0
    for state, prefilter in zip(states, prefilters, strict=True):
        if prefilter.rejected:
            online_results.append(None)
            continue
        result = online_session.query(
            state,
            scenario.world,
            scenario.vehicle,
            config=recovery_config,
            profile=profile,
            safe_anchor_ned_m=safe_anchor_ned_m,
        )
        online_results.append(result)
        online_time += result.elapsed_s

    full_session = RecoveryQuerySession(
        policy=RecoveryQueryPolicy(
            budget_s=None,
            use_cache=True,
            use_prefilter=False,
            mode=RecoveryQueryMode.FULL,
            parallel_workers=config.full_workers,
        )
    )
    full_results: list[RecoveryQueryResult | None] = []
    full_time = 0.0
    for state, prefilter in zip(states, prefilters, strict=True):
        if prefilter.rejected:
            full_results.append(None)
            continue
        result = full_session.query(
            state,
            scenario.world,
            scenario.vehicle,
            config=recovery_config,
            profile=profile,
            safe_anchor_ned_m=safe_anchor_ned_m,
        )
        full_results.append(result)
        full_time += result.elapsed_s

    points: list[AuditPoint] = []
    online_reason: str | None
    full_reason: str | None
    valid_from: float | None
    valid_until: float | None
    for index, (state, prefilter, online, full) in enumerate(
        zip(states, prefilters, online_results, full_results, strict=True)
    ):
        if prefilter.rejected:
            prefilter_reason: str | None = f"prefilter:{prefilter.reason}"
            online_status = RecoveryQueryStatus.INVALID.value
            full_status = RecoveryQueryStatus.INVALID.value
            online_reason = prefilter_reason
            full_reason = prefilter_reason
            online_elapsed = 0.0
            full_elapsed = 0.0
            online_early = False
            full_early = False
            online_checks = 0
            full_checks = 0
            valid_from = None
            valid_until = None
        else:
            assert online is not None
            online_status = online.status.value
            online_reason = online.reason
            online_elapsed = online.elapsed_s
            online_early = online.early_stopped
            online_checks = online.evaluated_checks
            if full is None:
                full_status = RecoveryQueryStatus.INVALID.value
                full_reason = "missing_full_result"
                full_elapsed = 0.0
                full_early = False
                full_checks = 0
                valid_from = None
                valid_until = None
            else:
                full_status = full.status.value
                full_reason = full.reason
                full_elapsed = full.elapsed_s
                full_early = full.early_stopped
                full_checks = full.evaluated_checks
                certificate = None if full.assessment is None else full.assessment.certificate
                valid_from = None if certificate is None else certificate.valid_from_s
                valid_until = None if certificate is None else certificate.valid_until_s
        points.append(
            AuditPoint(
                index=index,
                t_s=float(state.t_s),
                position_ned_m=state.position_ned_m.copy(),
                quat_xyzw=state.quat_xyzw.copy(),
                clearance_m=float(prefilter.current_clearance_m),
                prefilter_reason=None if not prefilter.rejected else f"prefilter:{prefilter.reason}",
                online_status=online_status,
                online_reason=online_reason,
                online_elapsed_s=float(online_elapsed),
                online_early_stopped=online_early,
                online_evaluated_checks=online_checks,
                full_status=full_status,
                full_reason=full_reason,
                full_elapsed_s=float(full_elapsed),
                full_early_stopped=full_early,
                full_evaluated_checks=full_checks,
                certificate_valid_from_s=valid_from,
                certificate_valid_until_s=valid_until,
                failure_source=_failure_source(
                    prefilter_reason=None if not prefilter.rejected else f"prefilter:{prefilter.reason}",
                    full_result=full,
                ),
            )
        )

    start_s = float(points[0].t_s)
    end_s = float(points[-1].t_s)
    coverage_ratio, max_uncovered, coverage_intervals, uncovered_intervals = coverage_metrics(
        points,
        start_s=start_s,
        end_s=end_s,
    )
    metrics = compute_path_metrics(trajectory, scenario.world.obstacle_map, scenario.vehicle)
    failures = Counter(
        point.failure_source for point in points if point.failure_source is not None
    )
    return PathAudit(
        case_name=case_name,
        scenario_name=scenario.name,
        planner_name=planner_name,
        variant=variant,
        include_in_aggregate=include_in_aggregate,
        plan_status=plan_status,
        plan_success=plan_success,
        plan_reason=plan_reason,
        planning_time_s=float(planning_time_s),
        repair_time_s=float(repair_time_s),
        trajectory=trajectory,
        points=tuple(points),
        prefilter_time_s=float(prefilter_time),
        online_time_s=float(online_time),
        full_time_s=float(full_time),
        path_length_m=float(metrics.path_length_m),
        duration_s=float(metrics.duration_s),
        min_clearance_m=float(metrics.min_clearance_m),
        coverage_ratio=float(coverage_ratio),
        max_uncovered_s=float(max_uncovered),
        coverage_intervals=coverage_intervals,
        uncovered_intervals=uncovered_intervals,
        failure_sources=tuple(sorted(failures.items())),
    )


def _failed_path_audit(
    *,
    case_name: str,
    scenario: ScenarioConfig,
    planner_name: str,
    include_in_aggregate: bool,
    plan: PlanResult,
    config: M71Config,
) -> PathAudit:
    duration = float(scenario.constraints.get("planning_timeout_s", config.astar_timeout_s))
    return PathAudit(
        case_name=case_name,
        scenario_name=scenario.name,
        planner_name=planner_name,
        variant="baseline",
        include_in_aggregate=include_in_aggregate,
        plan_status=plan.status.value,
        plan_success=False,
        plan_reason=plan.reason,
        planning_time_s=float(plan.planning_time_s),
        repair_time_s=0.0,
        trajectory=None,
        points=(),
        prefilter_time_s=0.0,
        online_time_s=0.0,
        full_time_s=0.0,
        path_length_m=0.0,
        duration_s=duration,
        min_clearance_m=0.0,
        coverage_ratio=0.0,
        max_uncovered_s=duration,
        coverage_intervals=(),
        uncovered_intervals=((0.0, duration),) if duration > 0.0 else (),
        failure_sources=(("planner_failed", 1),),
    )


def _prefix_positions(trajectory: Trajectory, end_s: float, end_position: FloatArray) -> list[FloatArray]:
    positions = [
        trajectory.position_ned_m[index].copy()
        for index in range(trajectory.t_s.shape[0])
        if float(trajectory.t_s[index]) < end_s - 1e-12
    ]
    if not positions or float(np.linalg.norm(positions[-1] - end_position)) > 1e-12:
        positions.append(end_position.copy())
    return positions


def _suffix_positions(trajectory: Trajectory, start_s: float, start_position: FloatArray) -> list[FloatArray]:
    positions = [start_position.copy()]
    positions.extend(
        trajectory.position_ned_m[index].copy()
        for index in range(trajectory.t_s.shape[0])
        if float(trajectory.t_s[index]) > start_s + 1e-12
    )
    return positions


def _dedupe_positions(positions: Iterable[FloatArray]) -> FloatArray:
    cleaned: list[FloatArray] = []
    for position in positions:
        candidate = np.asarray(position, dtype=np.float64)
        if not cleaned or float(np.linalg.norm(candidate - cleaned[-1])) > 1e-12:
            cleaned.append(candidate.copy())
    if len(cleaned) < 2:
        cleaned.append(cleaned[0].copy())
    return np.vstack(cleaned)


def _candidate_states(
    scenario: ScenarioConfig,
    baseline: PathAudit,
    spec: M71CaseSpec,
    config: M71Config,
) -> tuple[tuple[State, str], ...]:
    if config.max_repair_candidates <= 0 or not baseline.points:
        return ()
    invalid_index = _first_invalid_index(baseline)
    if invalid_index <= 0 or invalid_index >= len(baseline.points) - 1:
        return ()
    prev_index = max(
        (
            index
            for index in range(invalid_index)
            if baseline.points[index].full_status == RecoveryQueryStatus.VALID.value
        ),
        default=0,
    )
    next_index = min(
        (
            index
            for index in range(invalid_index + 1, len(baseline.points))
            if baseline.points[index].full_status == RecoveryQueryStatus.VALID.value
        ),
        default=len(baseline.points) - 1,
    )
    prev = baseline.points[prev_index]
    invalid = baseline.points[invalid_index]
    next_point = baseline.points[next_index]
    path_direction = next_point.position_ned_m - prev.position_ned_m
    horizontal = np.array([-path_direction[1], path_direction[0], 0.0], dtype=np.float64)
    norm = float(np.linalg.norm(horizontal))
    if norm <= 1e-12:
        horizontal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        horizontal /= norm
    directions = (horizontal, -horizontal, np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 1.0]))
    candidates: list[tuple[State, str]] = []
    bounds = scenario.world.obstacle_map.bounds_ned_m
    seen: set[tuple[float, float, float]] = set()
    if spec.safe_anchor_ned_m is not None:
        point = np.asarray(spec.safe_anchor_ned_m, dtype=np.float64)
        if np.all(point >= bounds[:, 0]) and np.all(point <= bounds[:, 1]):
            key = (float(point[0]), float(point[1]), float(point[2]))
            seen.add(key)
            candidates.append(
                (
                    _candidate_state(invalid, point, next_point.position_ned_m, scenario),
                    "safe_anchor",
                )
            )
            if len(candidates) >= config.max_repair_candidates:
                return tuple(candidates)
    for radius in config.repair_offsets_m:
        for direction in directions:
            point = invalid.position_ned_m + radius * direction
            if np.any(point < bounds[:, 0]) or np.any(point > bounds[:, 1]):
                continue
            key = (float(point[0]), float(point[1]), float(point[2]))
            if key in seen:
                continue
            seen.add(key)
            candidate = _candidate_state(invalid, point, next_point.position_ned_m, scenario)
            candidates.append(
                (
                    candidate,
                    f"offset_{radius:g}m",
                )
            )
            if len(candidates) >= config.max_repair_candidates:
                return tuple(candidates)
    return tuple(candidates)


def _state_at_point(source: AuditPoint, position: FloatArray, scenario: ScenarioConfig) -> State:
    return State(
        t_s=source.t_s,
        position_ned_m=np.asarray(position, dtype=np.float64),
        quat_xyzw=source.quat_xyzw.copy(),
        linear_velocity_body_mps=np.array([scenario.vehicle.min_speed_mps, 0.0, 0.0]),
        angular_velocity_body_rps=np.zeros(3),
    )


def _candidate_state(
    invalid: AuditPoint,
    position: FloatArray,
    next_position: FloatArray,
    scenario: ScenarioConfig,
) -> State:
    direction = np.asarray(next_position, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    if float(np.linalg.norm(direction)) <= 1e-12:
        direction = np.asarray(next_position, dtype=np.float64) - invalid.position_ned_m
    return State(
        t_s=invalid.t_s,
        position_ned_m=np.asarray(position, dtype=np.float64),
        quat_xyzw=forward_quaternion(direction),
        linear_velocity_body_mps=np.array([scenario.vehicle.min_speed_mps, 0.0, 0.0]),
        angular_velocity_body_rps=np.zeros(3),
    )


def _candidate_path_safe(
    previous: State,
    candidate: State,
    following: State,
    scenario: ScenarioConfig,
) -> bool:
    validator = SegmentValidator(scenario.world.obstacle_map, scenario.vehicle, max_step_m=0.1)
    if not validator.state_is_safe(candidate.position_ned_m, candidate.quat_xyzw):
        return False
    return bool(
        validator(previous.position_ned_m, candidate.position_ned_m)
        and validator(candidate.position_ned_m, following.position_ned_m)
    )


def _attempt_repair(
    baseline: PathAudit,
    scenario: ScenarioConfig,
    spec: M71CaseSpec,
    config: M71Config,
) -> tuple[PathAudit | None, tuple[RepairAttempt, ...], float, bool, str]:
    if baseline.trajectory is None:
        return None, (), 0.0, False, "baseline_plan_failed"
    if all(point.full_status == RecoveryQueryStatus.VALID.value for point in baseline.points):
        return None, (), 0.0, False, "no_invalid_points"
    invalid_index = _first_invalid_index(baseline)
    if invalid_index <= 0:
        return None, (), 0.0, False, "invalid_start_state"
    if invalid_index >= len(baseline.points) - 1:
        return None, (), 0.0, False, "invalid_goal_state"
    candidates = _candidate_states(scenario, baseline, spec, config)
    if not candidates:
        return None, (), 0.0, False, "no_repair_candidates"
    profile = config.profile
    recovery_config = RecoveryConfig()
    attempts: list[RepairAttempt] = []
    best: PathAudit | None = None
    best_ratio = baseline.coverage_ratio
    best_uncovered = baseline.max_uncovered_s
    repair_time = 0.0
    request = _plan_request(
        scenario,
        request_id=f"{spec.name}:repair",
        seed=config.seed,
    )
    prev_index = max(
        (
            index
            for index in range(invalid_index)
            if baseline.points[index].full_status == RecoveryQueryStatus.VALID.value
        ),
        default=0,
    )
    next_index = min(
        (
            index
            for index in range(invalid_index + 1, len(baseline.points))
            if baseline.points[index].full_status == RecoveryQueryStatus.VALID.value
        ),
        default=len(baseline.points) - 1,
    )
    prev_point = baseline.points[prev_index]
    next_point = baseline.points[next_index]
    previous = _state_at_point(prev_point, prev_point.position_ned_m, scenario)
    following = _state_at_point(next_point, next_point.position_ned_m, scenario)
    for candidate_index, (candidate, source) in enumerate(candidates):
        started = perf_counter()
        prefilter = prefilter_state(candidate, scenario.world, scenario.vehicle, recovery_config)
        if prefilter.rejected:
            repair_time += perf_counter() - started
            attempts.append(
                RepairAttempt(
                    candidate_index=candidate_index,
                    source=source,
                    via_ned_m=candidate.position_ned_m.copy(),
                    candidate_status=RecoveryQueryStatus.INVALID.value,
                    geometric_safe=False,
                    plan_success=False,
                    coverage_ratio=0.0,
                    max_uncovered_s=baseline.duration_s,
                    accepted=False,
                    reason=f"prefilter:{prefilter.reason}",
                )
            )
            continue
        candidate_session = RecoveryQuerySession(
            policy=RecoveryQueryPolicy(
                budget_s=None,
                use_cache=False,
                use_prefilter=False,
                mode=RecoveryQueryMode.ONLINE_FAST,
            )
        )
        candidate_result = candidate_session.query(
            candidate,
            scenario.world,
            scenario.vehicle,
            config=recovery_config,
            profile=profile,
            safe_anchor_ned_m=spec.safe_anchor_ned_m,
        )
        if candidate_result.status is not RecoveryQueryStatus.VALID:
            repair_time += perf_counter() - started
            attempts.append(
                RepairAttempt(
                    candidate_index=candidate_index,
                    source=source,
                    via_ned_m=candidate.position_ned_m.copy(),
                    candidate_status=candidate_result.status.value,
                    geometric_safe=False,
                    plan_success=False,
                    coverage_ratio=0.0,
                    max_uncovered_s=baseline.duration_s,
                    accepted=False,
                    reason=str(candidate_result.reason or "candidate_certificate_invalid"),
                )
            )
            continue
        if not _candidate_path_safe(previous, candidate, following, scenario):
            repair_time += perf_counter() - started
            attempts.append(
                RepairAttempt(
                    candidate_index=candidate_index,
                    source=source,
                    via_ned_m=candidate.position_ned_m.copy(),
                    candidate_status=candidate_result.status.value,
                    geometric_safe=False,
                    plan_success=False,
                    coverage_ratio=0.0,
                    max_uncovered_s=baseline.duration_s,
                    accepted=False,
                    reason="geometric_path_unsafe",
                )
            )
            continue
        waypoints = _dedupe_positions(
            _prefix_positions(baseline.trajectory, prev_point.t_s, prev_point.position_ned_m)
            + [candidate.position_ned_m]
            + _suffix_positions(baseline.trajectory, next_point.t_s, next_point.position_ned_m)
        )
        repaired_trajectory = build_trajectory(
            waypoints,
            request,
            trajectory_id=f"{spec.name}:{baseline.planner_name}:repair",
        )
        sweep = trajectory_sweep(
            repaired_trajectory,
            scenario.world.obstacle_map,
            scenario.vehicle,
            max_translation_step_m=0.05,
            max_rotation_step_rad=0.05,
        )
        if not sweep.is_safe:
            repair_time += perf_counter() - started
            attempts.append(
                RepairAttempt(
                    candidate_index=candidate_index,
                    source=source,
                    via_ned_m=candidate.position_ned_m.copy(),
                    candidate_status=candidate_result.status.value,
                    geometric_safe=False,
                    plan_success=False,
                    coverage_ratio=0.0,
                    max_uncovered_s=baseline.duration_s,
                    accepted=False,
                    reason=f"trajectory_sweep:{sweep.failure_reason}",
                )
            )
            continue
        candidate_planning_time = perf_counter() - started
        plan = PlanResult(
            status=PlanStatus.OK,
            trajectory=repaired_trajectory,
            cost=float(np.sum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1))),
            certificate=None,
            planning_time_s=baseline.planning_time_s + repair_time + candidate_planning_time,
            plan_id=f"{spec.name}:{baseline.planner_name}:repair",
        )
        evaluation = evaluate_plan(request, plan)
        repaired = audit_trajectory(
            repaired_trajectory,
            scenario,
            case_name=spec.name,
            planner_name=baseline.planner_name,
            variant="repair_attempt",
            include_in_aggregate=spec.include_in_aggregate,
            plan_status=plan.status.value,
            plan_success=evaluation.success,
            plan_reason=plan.reason,
            planning_time_s=plan.planning_time_s,
            repair_time_s=repair_time + candidate_planning_time,
            config=config,
            safe_anchor_ned_m=spec.safe_anchor_ned_m,
        )
        repair_time += candidate_planning_time
        is_better = evaluation.success and (
            repaired.coverage_ratio > best_ratio + 1e-12
            or (
                abs(repaired.coverage_ratio - best_ratio) <= 1e-12
                and repaired.max_uncovered_s < best_uncovered - 1e-12
            )
        )
        if is_better:
            best = repaired
            best_ratio = repaired.coverage_ratio
            best_uncovered = repaired.max_uncovered_s
        attempts.append(
            RepairAttempt(
                candidate_index=candidate_index,
                source=source,
                via_ned_m=candidate.position_ned_m.copy(),
                candidate_status=candidate_result.status.value,
                geometric_safe=True,
                plan_success=evaluation.success,
                coverage_ratio=repaired.coverage_ratio,
                max_uncovered_s=repaired.max_uncovered_s,
                accepted=is_better,
                reason=(
                    "accepted"
                    if is_better
                    else ("plan_failed" if not evaluation.success else "no_improvement")
                ),
            )
        )
        if is_better:
            return repaired, tuple(attempts), repair_time, True, "accepted"
    if best is not None:
        return best, tuple(attempts), repair_time, False, "no_improvement"
    return None, tuple(attempts), repair_time, False, "no_viable_candidate"


def _first_invalid_index(path: PathAudit) -> int:
    for index, point in enumerate(path.points):
        if point.full_status != RecoveryQueryStatus.VALID.value:
            return index
    return len(path.points) - 1


def run_m7_1(output_dir: Path | str, config: M71Config | None = None) -> M71ExperimentResult:
    active = M71Config() if config is None else config
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    cases: list[M71CaseResult] = []
    specs = m71_case_specs(active)
    for spec in specs:
        for planner_name in active.planner_names:
            planner = _planner(planner_name, active)
            record: EvaluationRecord = run_scenario(spec.scenario, planner, seed=active.seed)
            if record.plan.trajectory is None:
                baseline = _failed_path_audit(
                    case_name=spec.name,
                    scenario=spec.scenario,
                    planner_name=planner_name,
                    include_in_aggregate=spec.include_in_aggregate,
                    plan=record.plan,
                    config=active,
                )
                repair = None
                attempts: tuple[RepairAttempt, ...] = ()
                accepted = False
                repair_reason = "baseline_plan_failed"
            else:
                baseline = audit_trajectory(
                    record.plan.trajectory,
                    spec.scenario,
                    case_name=spec.name,
                    planner_name=planner_name,
                    variant="baseline",
                    include_in_aggregate=spec.include_in_aggregate,
                    plan_status=record.plan.status.value,
                    plan_success=record.result.success,
                    plan_reason=record.plan.reason,
                    planning_time_s=record.plan.planning_time_s,
                    repair_time_s=0.0,
                    config=active,
                    safe_anchor_ned_m=spec.safe_anchor_ned_m,
                )
                repair, attempts, _, accepted, repair_reason = _attempt_repair(
                    baseline,
                    spec.scenario,
                    spec,
                    active,
                )
            cases.append(
                M71CaseResult(
                    case_name=spec.name,
                    scenario_name=spec.scenario.name,
                    planner_name=planner_name,
                    include_in_aggregate=spec.include_in_aggregate,
                    baseline=baseline,
                    repair=repair,
                    repair_attempts=attempts,
                    repair_accepted=accepted,
                    repair_reason=repair_reason,
                )
            )
    result = M71ExperimentResult(
        config=active,
        cases=tuple(cases),
        elapsed_s=perf_counter() - started,
    )
    _write_outputs(root, result)
    return result


def _write_outputs(root: Path, result: M71ExperimentResult) -> None:
    _write_paths_csv(root / "paths.csv", result.cases)
    _write_points_csv(root / "points.csv", result.cases)
    _write_trajectories_csv(root / "trajectories.csv", result.cases)
    _write_repair_attempts_csv(root / "repair_attempts.csv", result.cases)
    _write_summary(root / "summary.json", result)
    _write_manifest(root / "manifest.json", result)


def _write_paths_csv(path: Path, cases: Sequence[M71CaseResult]) -> None:
    rows: list[dict[str, Any]] = []
    for case in cases:
        repair_variant = "repair_accepted" if case.repair_accepted else "repair_attempt"
        for audit, variant in ((case.baseline, "baseline"), (case.repair, repair_variant)):
            if audit is None:
                continue
            rows.append(
                {
                    "case": case.case_name,
                    "scenario": case.scenario_name,
                    "planner": case.planner_name,
                    "variant": variant,
                    "include_in_aggregate": case.include_in_aggregate,
                    "plan_status": audit.plan_status,
                    "plan_success": audit.plan_success,
                    "plan_reason": audit.plan_reason or "",
                    "repair_accepted": case.repair_accepted,
                    "repair_reason": case.repair_reason,
                    "sample_count": len(audit.points),
                    "valid_points": audit.valid_points,
                    "invalid_points": audit.invalid_points,
                    "unknown_points": audit.unknown_points,
                    "coverage_ratio": audit.coverage_ratio,
                    "max_uncovered_s": audit.max_uncovered_s,
                    "prefilter_time_s": audit.prefilter_time_s,
                    "online_time_s": audit.online_time_s,
                    "full_time_s": audit.full_time_s,
                    "planning_time_s": audit.planning_time_s,
                    "repair_time_s": audit.repair_time_s,
                    "end_to_end_s": audit.end_to_end_s,
                    "full_end_to_end_s": audit.full_end_to_end_s,
                    "path_length_m": audit.path_length_m,
                    "duration_s": audit.duration_s,
                    "min_clearance_m": audit.min_clearance_m,
                    "coverage_intervals": ";".join(
                        f"{lo:g}-{hi:g}" for lo, hi in audit.coverage_intervals
                    ),
                    "uncovered_intervals": ";".join(
                        f"{lo:g}-{hi:g}" for lo, hi in audit.uncovered_intervals
                    ),
                    "failure_sources": ";".join(f"{name}:{count}" for name, count in audit.failure_sources),
                }
            )
    _write_csv(path, rows)


def _write_points_csv(path: Path, cases: Sequence[M71CaseResult]) -> None:
    rows: list[dict[str, Any]] = []
    for case in cases:
        repair_variant = "repair_accepted" if case.repair_accepted else "repair_attempt"
        for audit, variant in ((case.baseline, "baseline"), (case.repair, repair_variant)):
            if audit is None:
                continue
            for point in audit.points:
                rows.append(
                    {
                        "case": case.case_name,
                        "planner": case.planner_name,
                        "variant": variant,
                        "index": point.index,
                        "t_s": point.t_s,
                        "x_m": point.position_ned_m[0],
                        "y_m": point.position_ned_m[1],
                        "z_m": point.position_ned_m[2],
                        "clearance_m": point.clearance_m,
                        "prefilter_reason": point.prefilter_reason or "",
                        "online_status": point.online_status,
                        "online_reason": point.online_reason or "",
                        "online_elapsed_s": point.online_elapsed_s,
                        "online_early_stopped": point.online_early_stopped,
                        "online_evaluated_checks": point.online_evaluated_checks,
                        "full_status": point.full_status,
                        "full_reason": point.full_reason or "",
                        "full_elapsed_s": point.full_elapsed_s,
                        "full_early_stopped": point.full_early_stopped,
                        "full_evaluated_checks": point.full_evaluated_checks,
                        "certificate_valid_from_s": (
                            "" if point.certificate_valid_from_s is None else point.certificate_valid_from_s
                        ),
                        "certificate_valid_until_s": (
                            "" if point.certificate_valid_until_s is None else point.certificate_valid_until_s
                        ),
                        "failure_source": point.failure_source or "",
                    }
                )
    _write_csv(path, rows)


def _write_trajectories_csv(path: Path, cases: Sequence[M71CaseResult]) -> None:
    rows: list[dict[str, Any]] = []
    for case in cases:
        repair_variant = "repair_accepted" if case.repair_accepted else "repair_attempt"
        for audit, variant in ((case.baseline, "baseline"), (case.repair, repair_variant)):
            if audit is None or audit.trajectory is None:
                continue
            trajectory = audit.trajectory
            for index in range(trajectory.t_s.shape[0]):
                rows.append(
                    {
                        "case": case.case_name,
                        "planner": case.planner_name,
                        "variant": variant,
                        "index": index,
                        "t_s": trajectory.t_s[index],
                        "x_m": trajectory.position_ned_m[index, 0],
                        "y_m": trajectory.position_ned_m[index, 1],
                        "z_m": trajectory.position_ned_m[index, 2],
                        "qx": trajectory.quat_xyzw[index, 0],
                        "qy": trajectory.quat_xyzw[index, 1],
                        "qz": trajectory.quat_xyzw[index, 2],
                        "qw": trajectory.quat_xyzw[index, 3],
                        "vx_body_mps": trajectory.linear_velocity_body_mps[index, 0],
                        "vy_body_mps": trajectory.linear_velocity_body_mps[index, 1],
                        "vz_body_mps": trajectory.linear_velocity_body_mps[index, 2],
                    }
                )
    _write_csv(path, rows)


def _write_repair_attempts_csv(path: Path, cases: Sequence[M71CaseResult]) -> None:
    rows: list[dict[str, Any]] = []
    for case in cases:
        for attempt in case.repair_attempts:
            rows.append(
                {
                    "case": case.case_name,
                    "planner": case.planner_name,
                    "candidate_index": attempt.candidate_index,
                    "source": attempt.source,
                    "via_x_m": "" if attempt.via_ned_m is None else attempt.via_ned_m[0],
                    "via_y_m": "" if attempt.via_ned_m is None else attempt.via_ned_m[1],
                    "via_z_m": "" if attempt.via_ned_m is None else attempt.via_ned_m[2],
                    "candidate_status": attempt.candidate_status or "",
                    "geometric_safe": attempt.geometric_safe,
                    "plan_success": attempt.plan_success,
                    "coverage_ratio": attempt.coverage_ratio,
                    "max_uncovered_s": attempt.max_uncovered_s,
                    "accepted": attempt.accepted,
                    "reason": attempt.reason,
                }
            )
    _write_csv(path, rows)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _timing(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "median": 0.0, "mean": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def _audit_group(audits: Sequence[PathAudit]) -> dict[str, Any]:
    full_end_to_end = _timing([audit.full_end_to_end_s for audit in audits])
    full_median = float(full_end_to_end["median"])
    share = {
        name: 0.0 if full_median <= 0.0 else float(_timing([value(audit) for audit in audits])["median"]) / full_median
        for name, value in (
            ("planning", lambda audit: audit.planning_time_s),
            ("prefilter", lambda audit: audit.prefilter_time_s),
            ("online", lambda audit: audit.online_time_s),
            ("full", lambda audit: audit.full_time_s),
        )
    }
    return {
        "paths": len(audits),
        "plan_success": sum(audit.plan_success for audit in audits),
        "plan_success_rate": 0.0 if not audits else sum(audit.plan_success for audit in audits) / len(audits),
        "sampled_points": sum(len(audit.points) for audit in audits),
        "valid_points": sum(audit.valid_points for audit in audits),
        "invalid_points": sum(audit.invalid_points for audit in audits),
        "unknown_points": sum(audit.unknown_points for audit in audits),
        "coverage_ratio": _timing([audit.coverage_ratio for audit in audits]),
        "max_uncovered_s": _timing([audit.max_uncovered_s for audit in audits]),
        "prefilter_time_s": _timing([audit.prefilter_time_s for audit in audits]),
        "online_time_s": _timing([audit.online_time_s for audit in audits]),
        "full_time_s": _timing([audit.full_time_s for audit in audits]),
        "planning_time_s": _timing([audit.planning_time_s for audit in audits]),
        "end_to_end_s": _timing([audit.end_to_end_s for audit in audits]),
        "full_end_to_end_s": full_end_to_end,
        "time_share_of_full_end_to_end": share,
    }


def _summary_payload(result: M71ExperimentResult) -> dict[str, Any]:
    aggregate_cases = tuple(case for case in result.cases if case.include_in_aggregate)
    baseline = tuple(case.baseline for case in aggregate_cases)
    repair_attempted = tuple(
        case.repair for case in aggregate_cases if case.repair is not None
    )
    repair_accepted = tuple(
        case.repair
        for case in aggregate_cases
        if case.repair is not None and case.repair_accepted
    )
    failure_counts: Counter[str] = Counter()
    for case in result.cases:
        for audit in (case.baseline, case.repair):
            if audit is None:
                continue
            failure_counts.update(dict(audit.failure_sources))
    by_planner: dict[str, Any] = {}
    for planner_name in result.config.planner_names:
        planner_cases = tuple(case for case in aggregate_cases if case.planner_name == planner_name)
        by_planner[planner_name] = {
            "baseline": _audit_group(tuple(case.baseline for case in planner_cases)),
            "repair_attempted": _audit_group(
                tuple(case.repair for case in planner_cases if case.repair is not None)
            ),
            "repair_accepted": _audit_group(
                tuple(
                    case.repair
                    for case in planner_cases
                    if case.repair is not None and case.repair_accepted
                )
            ),
            "repair_accepted_count": sum(case.repair_accepted for case in planner_cases),
        }
    return {
        "schema_version": 1,
        "status": "M7.1 complete; real A*/RRT* paths audited, cost probe excluded",
        "elapsed_s": result.elapsed_s,
        "config": {
            **asdict(result.config),
            "profile_digest": result.config.profile.digest,
        },
        "case_count": len(result.cases),
        "path_count": len(result.cases),
        "aggregate_case_count": len(aggregate_cases),
        "excluded_cases": sorted(
            case.case_name for case in result.cases if not case.include_in_aggregate
        ),
        "baseline": _audit_group(baseline),
        "repair_attempted": _audit_group(repair_attempted),
        "repair_accepted": _audit_group(repair_accepted),
        "repair_accepted_count": sum(case.repair_accepted for case in aggregate_cases),
        "repair_improved_count": sum(
            case.repair is not None
            and case.repair.coverage_ratio > case.baseline.coverage_ratio + 1e-12
            for case in aggregate_cases
        ),
        "by_planner": by_planner,
        "failure_sources": dict(sorted(failure_counts.items())),
        "cases": [
            {
                "case": case.case_name,
                "planner": case.planner_name,
                "include_in_aggregate": case.include_in_aggregate,
                "baseline_coverage_ratio": case.baseline.coverage_ratio,
                "baseline_max_uncovered_s": case.baseline.max_uncovered_s,
                "repair_coverage_ratio": None if case.repair is None else case.repair.coverage_ratio,
                "repair_max_uncovered_s": None if case.repair is None else case.repair.max_uncovered_s,
                "repair_accepted": case.repair_accepted,
                "repair_reason": case.repair_reason,
                "repair_attempts": len(case.repair_attempts),
            }
            for case in result.cases
        ],
    }


def _write_summary(path: Path, result: M71ExperimentResult) -> None:
    path.write_text(
        json.dumps(_summary_payload(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_manifest(path: Path, result: M71ExperimentResult) -> None:
    payload = {
        "schema_version": 1,
        "experiment": "M7.1 real planner trajectory certificate audit",
        "hardware": hardware_manifest(),
        "config": {
            **asdict(result.config),
            "profile": result.config.profile.manifest(),
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "AuditPoint",
    "M71CaseResult",
    "M71CaseSpec",
    "M71Config",
    "M71ExperimentResult",
    "PathAudit",
    "RepairAttempt",
    "audit_trajectory",
    "coverage_metrics",
    "m71_case_specs",
    "run_m7_1",
    "sample_trajectory_states",
]
