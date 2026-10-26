from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from ..contracts import PlanResult, ScenarioConfig
from ..evaluation.runner import EvaluationRecord, run_scenario
from ..planners.astar import AStarPlanner
from ..planners.recoverable import RecoverablePlanner, RecoverablePlannerConfig
from ..planners.rrt_star import RRTStarPlanner
from ..recovery import FaultProfile, RecoveryQueryStatus, fault_profile_by_name
from .m7_1 import M71CaseSpec, M71Config, PathAudit, audit_trajectory, m71_case_specs
from .m7_budget import hardware_manifest


AGGREGATE_VARIANTS = ("astar", "rrt_star", "candidate_on", "candidate_off")


@dataclass(frozen=True)
class M72Config:
    audit_interval_s: float = 1.0
    online_budget_s: float = 10.0
    budget_s: float = 10.0
    profile_name: str = "P1_degraded"
    full_workers: int = 8
    seed: int = 0
    astar_resolution_m: float = 0.5
    rrt_step_size_m: float = 0.6
    rrt_iterations: int = 2_000
    candidate_resolution_m: float = 1.0
    candidate_inner_timeout_s: float = 6.0
    candidate_checkpoint_spacing_m: float = 1.0
    candidate_max_replans: int = 2
    candidate_max_via_candidates: int = 3
    candidate_block_radius_m: float = 1.2
    candidate_via_offset_m: float = 1.0
    candidate_max_queries: int = 64
    case_names: tuple[str, ...] | None = None
    variant_names: tuple[str, ...] = AGGREGATE_VARIANTS

    def __post_init__(self) -> None:
        for name in (
            "audit_interval_s",
            "online_budget_s",
            "budget_s",
            "astar_resolution_m",
            "rrt_step_size_m",
            "candidate_resolution_m",
            "candidate_inner_timeout_s",
            "candidate_checkpoint_spacing_m",
            "candidate_block_radius_m",
            "candidate_via_offset_m",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.audit_interval_s > 1.0:
            raise ValueError("audit_interval_s must not exceed 1 s")
        if not isinstance(self.full_workers, int) or self.full_workers < 1:
            raise ValueError("full_workers must be a positive integer")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.rrt_iterations, int) or self.rrt_iterations < 1:
            raise ValueError("rrt_iterations must be a positive integer")
        if not isinstance(self.candidate_max_replans, int) or self.candidate_max_replans < 0:
            raise ValueError("candidate_max_replans must be a non-negative integer")
        if (
            not isinstance(self.candidate_max_via_candidates, int)
            or self.candidate_max_via_candidates < 0
        ):
            raise ValueError("candidate_max_via_candidates must be a non-negative integer")
        if not isinstance(self.candidate_max_queries, int) or self.candidate_max_queries < 1:
            raise ValueError("candidate_max_queries must be a positive integer")
        if not self.profile_name:
            raise ValueError("profile_name must not be empty")
        if not self.variant_names or any(
            name not in AGGREGATE_VARIANTS for name in self.variant_names
        ):
            raise ValueError("variant_names must contain known M7.2 variants")
        object.__setattr__(self, "variant_names", tuple(self.variant_names))
        if self.case_names is not None:
            if not self.case_names or any(not name for name in self.case_names):
                raise ValueError("case_names must be a non-empty sequence of names")
            object.__setattr__(self, "case_names", tuple(self.case_names))

    @property
    def profile(self) -> FaultProfile:
        return fault_profile_by_name(self.profile_name)


@dataclass(frozen=True)
class M72Run:
    case_name: str
    scenario_name: str
    probe_role: str
    include_in_aggregate: bool
    variant: str
    planner_name: str
    audit: PathAudit
    diagnostics: dict[str, Any]
    nominal_success: bool

    @property
    def start_recovery_infeasible(self) -> bool:
        return self.audit.plan_reason == "infeasible_recovery_at_start"

    @property
    def recovery_complete(self) -> bool:
        return bool(self.diagnostics.get("recovery_complete", False))

    @property
    def internal_coverage_proxy(self) -> float | None:
        value = self.diagnostics.get("recovery_coverage_ratio")
        return None if value is None else float(value)

    @property
    def internal_max_uncovered_s(self) -> float | None:
        value = self.diagnostics.get("recovery_max_uncovered_s")
        return None if value is None else float(value)


@dataclass(frozen=True)
class M72ExperimentResult:
    config: M72Config
    runs: tuple[M72Run, ...]
    elapsed_s: float


def m72_case_specs(config: M72Config | None = None) -> tuple[M71CaseSpec, ...]:
    active = M72Config() if config is None else config
    specs = m71_case_specs(M71Config(case_names=active.case_names))
    selected = None if active.case_names is None else set(active.case_names)
    if selected is None:
        return specs
    known = {spec.name for spec in specs}
    missing = selected - known
    if missing:
        raise ValueError(f"unknown M7.2 cases: {sorted(missing)}")
    return tuple(spec for spec in specs if spec.name in selected)


def _candidate_config(config: M72Config, *, recovery_mode: bool) -> RecoverablePlannerConfig:
    return RecoverablePlannerConfig(
        resolution_m=config.candidate_resolution_m,
        timeout_s=config.budget_s,
        inner_timeout_s=config.candidate_inner_timeout_s,
        checkpoint_spacing_m=config.candidate_checkpoint_spacing_m,
        query_budget_s=config.budget_s,
        max_recovery_queries=config.candidate_max_queries,
        max_replans=config.candidate_max_replans,
        max_via_candidates=config.candidate_max_via_candidates,
        block_radius_m=config.candidate_block_radius_m,
        via_offset_m=config.candidate_via_offset_m,
        recovery_mode=recovery_mode,
        profile_name=config.profile_name,
    )


def _planner(variant: str, config: M72Config) -> tuple[Any, str]:
    if variant == "astar":
        return (
            AStarPlanner(
                resolution_m=config.astar_resolution_m,
                timeout_s=config.budget_s,
                collision_step_m=0.1,
            ),
            "astar",
        )
    if variant == "rrt_star":
        return (
            RRTStarPlanner(
                step_size_m=config.rrt_step_size_m,
                max_iterations=config.rrt_iterations,
                timeout_s=config.budget_s,
                collision_step_m=0.1,
            ),
            "rrt_star",
        )
    if variant == "candidate_on":
        return (
            RecoverablePlanner(_candidate_config(config, recovery_mode=True)),
            "recoverable_planner",
        )
    if variant == "candidate_off":
        return (
            RecoverablePlanner(_candidate_config(config, recovery_mode=False)),
            "recoverable_planner",
        )
    raise ValueError(f"unknown variant: {variant}")


def _failed_path_audit(
    *,
    spec: M71CaseSpec,
    variant: str,
    planner_name: str,
    plan: PlanResult,
    config: M72Config,
) -> PathAudit:
    duration = float(spec.scenario.constraints.get("planning_timeout_s", config.budget_s))
    return PathAudit(
        case_name=spec.name,
        scenario_name=spec.scenario.name,
        planner_name=planner_name,
        variant=variant,
        include_in_aggregate=spec.include_in_aggregate,
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


def _audit_config(config: M72Config) -> M71Config:
    return M71Config(
        audit_interval_s=config.audit_interval_s,
        online_budget_s=config.online_budget_s,
        profile_name=config.profile_name,
        full_workers=config.full_workers,
        seed=config.seed,
        astar_resolution_m=config.astar_resolution_m,
        astar_timeout_s=config.budget_s,
        rrt_step_size_m=config.rrt_step_size_m,
        rrt_iterations=config.rrt_iterations,
        rrt_timeout_s=config.budget_s,
    )


def run_m7_2(output_dir: Path | str, config: M72Config | None = None) -> M72ExperimentResult:
    active = M72Config() if config is None else config
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    audit_config = _audit_config(active)
    runs: list[M72Run] = []
    for spec in m72_case_specs(active):
        for variant in active.variant_names:
            planner, planner_name = _planner(variant, active)
            record: EvaluationRecord = run_scenario(spec.scenario, planner, seed=active.seed)
            if record.plan.trajectory is None:
                audit = _failed_path_audit(
                    spec=spec,
                    variant=variant,
                    planner_name=planner_name,
                    plan=record.plan,
                    config=active,
                )
            else:
                audit = audit_trajectory(
                    record.plan.trajectory,
                    spec.scenario,
                    case_name=spec.name,
                    planner_name=planner_name,
                    variant=variant,
                    include_in_aggregate=spec.include_in_aggregate,
                    plan_status=record.plan.status.value,
                    plan_success=record.result.success,
                    plan_reason=record.plan.reason,
                    planning_time_s=record.plan.planning_time_s,
                    repair_time_s=0.0,
                    config=audit_config,
                    safe_anchor_ned_m=spec.safe_anchor_ned_m,
                )
            runs.append(
                M72Run(
                    case_name=spec.name,
                    scenario_name=spec.scenario.name,
                    probe_role=(
                        "certificate_rejection_unit_test"
                        if not spec.include_in_aggregate
                        else "frozen_m7_case"
                    ),
                    include_in_aggregate=spec.include_in_aggregate,
                    variant=variant,
                    planner_name=planner_name,
                    audit=audit,
                    diagnostics=dict(record.plan.diagnostics),
                    nominal_success=record.result.success,
                )
            )
    result = M72ExperimentResult(
        config=active,
        runs=tuple(runs),
        elapsed_s=perf_counter() - started,
    )
    _write_outputs(root, result)
    return result


def _write_outputs(root: Path, result: M72ExperimentResult) -> None:
    _write_paths_csv(root / "paths.csv", result.runs)
    _write_points_csv(root / "points.csv", result.runs)
    _write_checkpoints_csv(root / "checkpoints.csv", result.runs)
    _write_trajectories_csv(root / "trajectories.csv", result.runs)
    _write_endpoint_recoverability_csv(root / "endpoint_recoverability.csv", result.runs)
    _write_pareto_csv(root / "pareto.csv", result.runs)
    _write_summary(root / "summary.json", result)
    _write_manifest(root / "manifest.json", result)


def _candidate_diagnostic(result: M72Run, key: str, default: Any = "") -> Any:
    value = result.diagnostics.get(key, default)
    return "" if value is None else value


def _write_paths_csv(path: Path, runs: Sequence[M72Run]) -> None:
    rows: list[dict[str, Any]] = []
    for run in runs:
        audit = run.audit
        stats = run.diagnostics.get("recovery_query_stats", {})
        rows.append(
            {
                "case": run.case_name,
                "scenario": run.scenario_name,
                "probe_role": run.probe_role,
                "include_in_aggregate": run.include_in_aggregate,
                "variant": run.variant,
                "planner": run.planner_name,
                "plan_status": audit.plan_status,
                "plan_success": run.nominal_success,
                "plan_reason": audit.plan_reason or "",
                "start_recovery_infeasible": run.start_recovery_infeasible,
                "recovery_complete": run.recovery_complete,
                "sample_count": len(audit.points),
                "valid_points": audit.valid_points,
                "invalid_points": audit.invalid_points,
                "unknown_points": audit.unknown_points,
                "coverage_ratio": audit.coverage_ratio,
                "max_uncovered_s": audit.max_uncovered_s,
                "path_length_m": audit.path_length_m,
                "duration_s": audit.duration_s,
                "min_clearance_m": audit.min_clearance_m,
                "prefilter_time_s": audit.prefilter_time_s,
                "online_time_s": audit.online_time_s,
                "full_time_s": audit.full_time_s,
                "planning_time_s": audit.planning_time_s,
                "end_to_end_s": audit.end_to_end_s,
                "full_end_to_end_s": audit.full_end_to_end_s,
                "coverage_intervals": ";".join(
                    f"{lo:g}-{hi:g}" for lo, hi in audit.coverage_intervals
                ),
                "uncovered_intervals": ";".join(
                    f"{lo:g}-{hi:g}" for lo, hi in audit.uncovered_intervals
                ),
                "failure_sources": ";".join(
                    f"{name}:{count}" for name, count in audit.failure_sources
                ),
                "recovery_queries": _candidate_diagnostic(run, "recovery_queries", 0),
                "cache_hits": stats.get("cache_hits", 0),
                "cache_misses": stats.get("cache_misses", 0),
                "recovery_unknown": _candidate_diagnostic(run, "recovery_unknown", False),
                "budget_exhausted": stats.get("budget_exhausted", 0),
                "replans": _candidate_diagnostic(run, "replans", 0),
                "blocked_regions": _candidate_diagnostic(run, "blocked_regions", 0),
                "selected_attempt": _candidate_diagnostic(run, "selected_attempt", ""),
                "search_time_s": _candidate_diagnostic(run, "search_time_s", 0.0),
                "online_query_time_s": _candidate_diagnostic(run, "online_query_time_s", 0.0),
                "internal_coverage_proxy": _candidate_diagnostic(
                    run, "recovery_coverage_ratio", ""
                ),
                "internal_max_uncovered_s": _candidate_diagnostic(
                    run, "recovery_max_uncovered_s", ""
                ),
                "cost_total": (
                    run.diagnostics.get("cost_breakdown", {}).get("total", "")
                    if isinstance(run.diagnostics.get("cost_breakdown"), dict)
                    else ""
                ),
            }
        )
    _write_csv(path, rows)


def _write_points_csv(path: Path, runs: Sequence[M72Run]) -> None:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for point in run.audit.points:
            rows.append(
                {
                    "case": run.case_name,
                    "variant": run.variant,
                    "index": point.index,
                    "t_s": point.t_s,
                    "x_m": point.position_ned_m[0],
                    "y_m": point.position_ned_m[1],
                    "z_m": point.position_ned_m[2],
                    "clearance_m": point.clearance_m,
                    "prefilter_reason": point.prefilter_reason or "",
                    "online_status": point.online_status,
                    "online_reason": point.online_reason or "",
                    "full_status": point.full_status,
                    "full_reason": point.full_reason or "",
                    "full_elapsed_s": point.full_elapsed_s,
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


def _write_checkpoints_csv(path: Path, runs: Sequence[M72Run]) -> None:
    rows: list[dict[str, Any]] = []
    for run in runs:
        attempts = run.diagnostics.get("attempts", [])
        if not isinstance(attempts, list):
            continue
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            blocker = attempt.get("blocker_added")
            blocker_position = None if not isinstance(blocker, dict) else blocker.get("position_ned_m")
            checkpoints = attempt.get("checkpoints", [])
            if not isinstance(checkpoints, list):
                continue
            for point in checkpoints:
                if not isinstance(point, dict):
                    continue
                position = point.get("position_ned_m", [None, None, None])
                rows.append(
                    {
                        "case": run.case_name,
                        "variant": run.variant,
                        "attempt": attempt.get("iteration", ""),
                        "selected": attempt.get("selected", False),
                        "checkpoint_coverage_ratio": attempt.get(
                            "checkpoint_coverage_ratio", ""
                        ),
                        "checkpoint_max_uncovered_s": attempt.get(
                            "checkpoint_max_uncovered_s", ""
                        ),
                        "distance_m": point.get("distance_m", ""),
                        "t_s": point.get("t_s", ""),
                        "x_m": position[0],
                        "y_m": position[1],
                        "z_m": position[2],
                        "status": point.get("status", ""),
                        "reason": point.get("reason", "") or "",
                        "margin_m": "" if point.get("margin_m") is None else point.get("margin_m"),
                        "query_source": point.get("query_source", ""),
                        "certificate_valid_from_s": point.get("certificate_valid_from_s", ""),
                        "certificate_valid_until_s": point.get("certificate_valid_until_s", ""),
                        "elapsed_s": point.get("elapsed_s", ""),
                        "budget_exhausted": point.get("budget_exhausted", False),
                        "blocker_x_m": "" if blocker_position is None else blocker_position[0],
                        "blocker_y_m": "" if blocker_position is None else blocker_position[1],
                        "blocker_z_m": "" if blocker_position is None else blocker_position[2],
                        "blocker_radius_m": "" if not isinstance(blocker, dict) else blocker.get("radius_m", ""),
                    }
                )
    _write_csv(path, rows)


def _write_trajectories_csv(path: Path, runs: Sequence[M72Run]) -> None:
    rows: list[dict[str, Any]] = []
    for run in runs:
        trajectory = run.audit.trajectory
        if trajectory is None:
            continue
        for index in range(trajectory.t_s.shape[0]):
            rows.append(
                {
                    "case": run.case_name,
                    "variant": run.variant,
                    "index": index,
                    "t_s": trajectory.t_s[index],
                    "x_m": trajectory.position_ned_m[index, 0],
                    "y_m": trajectory.position_ned_m[index, 1],
                    "z_m": trajectory.position_ned_m[index, 2],
                    "qx": trajectory.quat_xyzw[index, 0],
                    "qy": trajectory.quat_xyzw[index, 1],
                    "qz": trajectory.quat_xyzw[index, 2],
                    "qw": trajectory.quat_xyzw[index, 3],
                }
            )
    _write_csv(path, rows)


def _write_endpoint_recoverability_csv(path: Path, runs: Sequence[M72Run]) -> None:
    rows: list[dict[str, Any]] = []
    for run in runs:
        points = run.audit.points
        start = points[0] if points else None
        goal = points[-1] if points else None
        rows.append(
            {
                "case": run.case_name,
                "variant": run.variant,
                "plan_success": run.nominal_success,
                "plan_reason": run.audit.plan_reason or "",
                "start_status": "" if start is None else start.full_status,
                "start_reason": "" if start is None else (start.full_reason or ""),
                "goal_status": "" if goal is None else goal.full_status,
                "goal_reason": "" if goal is None else (goal.full_reason or ""),
            }
        )
    _write_csv(path, rows)


def _write_pareto_csv(path: Path, runs: Sequence[M72Run]) -> None:
    rows = [
        {
            "case": run.case_name,
            "variant": run.variant,
            "plan_success": run.nominal_success,
            "recovery_complete": run.recovery_complete,
            "coverage_ratio": run.audit.coverage_ratio,
            "max_uncovered_s": run.audit.max_uncovered_s,
            "path_length_m": run.audit.path_length_m,
            "planning_time_s": run.audit.planning_time_s,
            "full_end_to_end_s": run.audit.full_end_to_end_s,
        }
        for run in runs
        if run.include_in_aggregate and run.nominal_success
    ]
    _write_csv(path, rows)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _stats(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "median": 0.0, "mean": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.shape[0]),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def _variant_summary(runs: Sequence[M72Run]) -> dict[str, Any]:
    successful = [run for run in runs if run.nominal_success]
    failure_reasons = Counter(
        run.audit.plan_reason or "unknown" for run in runs if not run.nominal_success
    )
    failure_sources = Counter(
        name for run in runs for name, _ in run.audit.failure_sources
    )
    query_stats = [
        run.diagnostics.get("recovery_query_stats", {})
        for run in runs
        if isinstance(run.diagnostics.get("recovery_query_stats"), dict)
    ]
    return {
        "paths": len(runs),
        "plan_success": len(successful),
        "plan_success_rate": 0.0 if not runs else len(successful) / len(runs),
        "recovery_complete": sum(run.recovery_complete for run in runs),
        "start_recovery_infeasible": sum(run.start_recovery_infeasible for run in runs),
        "coverage_ratio_all": _stats([run.audit.coverage_ratio for run in runs]),
        "coverage_ratio_successful": _stats([run.audit.coverage_ratio for run in successful]),
        "max_uncovered_s_all": _stats([run.audit.max_uncovered_s for run in runs]),
        "max_uncovered_s_successful": _stats(
            [run.audit.max_uncovered_s for run in successful]
        ),
        "path_length_m": _stats([run.audit.path_length_m for run in successful]),
        "min_clearance_m": _stats([run.audit.min_clearance_m for run in successful]),
        "planning_time_s": _stats([run.audit.planning_time_s for run in runs]),
        "end_to_end_s": _stats([run.audit.end_to_end_s for run in runs]),
        "full_end_to_end_s": _stats([run.audit.full_end_to_end_s for run in runs]),
        "valid_points": sum(run.audit.valid_points for run in runs),
        "invalid_points": sum(run.audit.invalid_points for run in runs),
        "unknown_points": sum(run.audit.unknown_points for run in runs),
        "recovery_queries": int(sum(int(row.get("queries", 0)) for row in query_stats)),
        "cache_hits": int(sum(int(row.get("cache_hits", 0)) for row in query_stats)),
        "cache_misses": int(sum(int(row.get("cache_misses", 0)) for row in query_stats)),
        "budget_exhausted": int(sum(int(row.get("budget_exhausted", 0)) for row in query_stats)),
        "replans": int(
            sum(int(run.diagnostics.get("replans", 0) or 0) for run in runs)
        ),
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "failure_sources": dict(sorted(failure_sources.items())),
    }


def _path_rms_distance(first: PathAudit, second: PathAudit) -> float | None:
    if first.trajectory is None or second.trajectory is None:
        return None

    def sample(audit: PathAudit) -> np.ndarray:
        assert audit.trajectory is not None
        trajectory = audit.trajectory
        distances = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(trajectory.position_ned_m, axis=0), axis=1)))
        )
        if distances[-1] <= 1e-12:
            return np.repeat(trajectory.position_ned_m[-1][None, :], 200, axis=0)
        targets = np.linspace(0.0, float(distances[-1]), 200)
        return np.vstack(
            [np.interp(targets, distances, trajectory.position_ned_m[:, axis]) for axis in range(3)]
        ).T

    return float(np.sqrt(np.mean(np.sum((sample(first) - sample(second)) ** 2, axis=1))))


def _on_off_summary(runs: Sequence[M72Run]) -> dict[str, Any]:
    by_case: dict[str, dict[str, M72Run]] = {}
    for run in runs:
        if run.include_in_aggregate and run.variant in {"candidate_on", "candidate_off"}:
            by_case.setdefault(run.case_name, {})[run.variant] = run
    comparisons: list[dict[str, Any]] = []
    for case_name in sorted(by_case):
        pair = by_case[case_name]
        if set(pair) != {"candidate_on", "candidate_off"}:
            continue
        on = pair["candidate_on"]
        off = pair["candidate_off"]
        both_success = on.nominal_success and off.nominal_success
        comparisons.append(
            {
                "case": case_name,
                "on_success": on.nominal_success,
                "off_success": off.nominal_success,
                "both_success": both_success,
                "on_failure_reason": None if on.nominal_success else on.audit.plan_reason,
                "off_failure_reason": None if off.nominal_success else off.audit.plan_reason,
                "on_coverage": on.audit.coverage_ratio if on.nominal_success else None,
                "off_coverage": off.audit.coverage_ratio if off.nominal_success else None,
                "coverage_delta": (
                    on.audit.coverage_ratio - off.audit.coverage_ratio
                    if both_success
                    else None
                ),
                "on_gap_s": on.audit.max_uncovered_s if on.nominal_success else None,
                "off_gap_s": off.audit.max_uncovered_s if off.nominal_success else None,
                "gap_delta_s": (
                    on.audit.max_uncovered_s - off.audit.max_uncovered_s
                    if both_success
                    else None
                ),
                "on_path_length_m": on.audit.path_length_m if on.nominal_success else None,
                "off_path_length_m": off.audit.path_length_m if off.nominal_success else None,
                "path_rms_difference_m": (
                    _path_rms_distance(on.audit, off.audit) if both_success else None
                ),
            }
        )
    both_success_cases = [item for item in comparisons if item["both_success"]]
    return {
        "cases": comparisons,
        "both_success": len(both_success_cases),
        "on_failed_off_success": sum(
            not item["on_success"] and item["off_success"] for item in comparisons
        ),
        "off_failed_on_success": sum(
            item["on_success"] and not item["off_success"] for item in comparisons
        ),
        "both_failed": sum(
            not item["on_success"] and not item["off_success"] for item in comparisons
        ),
        "coverage_improved": sum(
            float(item["coverage_delta"]) > 1e-12 for item in both_success_cases
        ),
        "gap_improved": sum(
            float(item["gap_delta_s"]) < -1e-12 for item in both_success_cases
        ),
        "different_paths": sum(
            item["path_rms_difference_m"] is not None
            and float(item["path_rms_difference_m"]) > 0.05
            for item in both_success_cases
        ),
    }


def _summary_payload(result: M72ExperimentResult) -> dict[str, Any]:
    aggregate = [run for run in result.runs if run.include_in_aggregate]
    rejection = [run for run in result.runs if not run.include_in_aggregate]
    return {
        "schema_version": 1,
        "status": "M7.2 recovery-aware candidate planner experiment",
        "elapsed_s": result.elapsed_s,
        "case_count": len({run.case_name for run in result.runs}),
        "aggregate_case_count": len({run.case_name for run in aggregate}),
        "rejection_case_count": len({run.case_name for run in rejection}),
        "aggregate": {
            variant: _variant_summary([run for run in aggregate if run.variant == variant])
            for variant in result.config.variant_names
        },
        "rejection": {
            variant: _variant_summary([run for run in rejection if run.variant == variant])
            for variant in result.config.variant_names
        },
        "candidate_on_off": _on_off_summary(aggregate),
    }


def _write_summary(path: Path, result: M72ExperimentResult) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_summary_payload(result), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_manifest(path: Path, result: M72ExperimentResult) -> None:
    payload = {
        "schema_version": 1,
        "experiment": "M7.2 recovery-aware global candidate planner",
        "config": asdict(result.config),
        "profile": result.config.profile.manifest(),
        "hardware": hardware_manifest(),
        "artifacts": [
            "paths.csv",
            "points.csv",
            "checkpoints.csv",
            "trajectories.csv",
            "endpoint_recoverability.csv",
            "pareto.csv",
            "summary.json",
            "manifest.json",
        ],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


__all__ = [
    "AGGREGATE_VARIANTS",
    "M72Config",
    "M72ExperimentResult",
    "M72Run",
    "m72_case_specs",
    "run_m7_2",
]
