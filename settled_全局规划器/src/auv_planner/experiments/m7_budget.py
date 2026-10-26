from __future__ import annotations

from collections import Counter
from contextlib import redirect_stdout
import csv
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np

from ..contracts import ScenarioConfig, State
from ..recovery import (
    DEFAULT_FAULT_PROFILES,
    DEFAULT_FAULTS,
    FaultProfile,
    RecoverabilityChecker,
    RecoveryConfig,
    RecoveryQueryPolicy,
    RecoveryQueryResult,
    RecoveryQuerySession,
    RecoveryQueryMode,
    RecoveryQueryStatus,
    build_maneuvers,
    prefilter_state,
)
from ..scenarios import m5_multi_obstacle_scene, m5_single_corner_scene
from .m6_recovery import m6_key_nodes


ONLINE_BUDGET_S = 10.0
AUDIT_INTERVAL_S = 1.0
SINGLE_REPEATS = 3


@dataclass(frozen=True)
class M7Probe:
    name: str
    role: str
    scenario: ScenarioConfig
    state: State
    safe_anchor_ned_m: np.ndarray | None = None
    include_in_aggregate: bool = True


def m7_probes() -> tuple[M7Probe, ...]:
    probes = [
        M7Probe(
            name=node.name,
            role=(
                "certificate_rejection_unit_test"
                if node.name == "dead_end_no_recovery"
                else "m6_key_node"
            ),
            scenario=node.scenario,
            state=node.state,
            safe_anchor_ned_m=node.safe_anchor_ned_m,
            include_in_aggregate=node.name != "dead_end_no_recovery",
        )
        for node in m6_key_nodes()
    ]
    for scene in (m5_single_corner_scene(), m5_multi_obstacle_scene()):
        probes.append(
            M7Probe(
                name=f"{scene.name}_start",
                role="m5_scene_start",
                scenario=scene,
                state=scene.start,
            )
        )
    return tuple(probes)


def m7_budget_manifest() -> dict[str, Any]:
    config = RecoveryConfig()
    return {
        "schema_version": 1,
        "status": "M7.0 cost-profiling configuration; no candidate planner",
        "online_budget_s": ONLINE_BUDGET_S,
        "audit_interval_s": AUDIT_INTERVAL_S,
        "single_full_repeats": SINGLE_REPEATS,
        "profiles": [profile.manifest() for profile in DEFAULT_FAULT_PROFILES],
        "model": {
            "horizon_s": config.horizon_s,
            "certificate_validity_s": config.certificate_validity_s,
            "sample_time_s": config.sample_time_s,
            "required_clearance_m": config.required_clearance_m,
            "error_bounds": {
                "position_m": config.error_bounds.position_m,
                "heading_rad": config.error_bounds.heading_rad,
                "current_mps": config.error_bounds.current_mps,
            },
            "clearance_model": config.clearance_model.value,
            "constraint_mode": config.constraint_mode.value,
        },
        "probes": [
            {
                "name": probe.name,
                "role": probe.role,
                "scenario_name": probe.scenario.name,
                "include_in_aggregate": probe.include_in_aggregate,
                "state_position_ned_m": probe.state.position_ned_m.tolist(),
                "state_linear_velocity_body_mps": probe.state.linear_velocity_body_mps.tolist(),
            }
            for probe in m7_probes()
        ],
    }


def m7_audit_states(
    probe: M7Probe,
    config: RecoveryConfig | None = None,
) -> tuple[State, ...]:
    return _audit_states(probe, RecoveryConfig() if config is None else config)


def write_m7_budget_manifest(path: Path | str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(m7_budget_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def run_m7_budget(
    output_dir: Path | str,
    *,
    online_mode: RecoveryQueryMode = RecoveryQueryMode.FULL,
    online_parallel_workers: int = 1,
    full_parallel_workers: int = 1,
) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    probes = m7_probes()
    config = RecoveryConfig()
    hardware = hardware_manifest()
    single_rows = _run_single_full(probes, config)
    node_rows = _run_node_cache(
        probes,
        config,
        mode=online_mode,
        parallel_workers=online_parallel_workers,
    )
    path_rows = _run_path_audits(
        probes,
        config,
        online_mode=online_mode,
        online_parallel_workers=online_parallel_workers,
        full_parallel_workers=full_parallel_workers,
    )
    prefilter_rows = _validate_prefilter(probes, config)
    prefilter_false_negatives = _prefilter_false_negatives(
        single_rows,
        node_rows,
        prefilter_rows,
    )
    summary = _summarize(
        probes,
        single_rows,
        node_rows,
        path_rows,
        prefilter_rows,
        prefilter_false_negatives,
        hardware,
        online_mode=online_mode,
        online_parallel_workers=online_parallel_workers,
        full_parallel_workers=full_parallel_workers,
    )

    _write_csv(root / "single_full.csv", single_rows)
    _write_csv(root / "node_cache.csv", node_rows)
    _write_csv(root / "path_audit.csv", path_rows)
    _write_csv(root / "prefilter_validation.csv", prefilter_rows)
    (root / "profiles.json").write_text(
        json.dumps([profile.manifest() for profile in DEFAULT_FAULT_PROFILES], indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (root / "hardware_manifest.json").write_text(
        json.dumps(hardware, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(root / "report.md", summary)
    return summary


def _run_single_full(probes: tuple[M7Probe, ...], config: RecoveryConfig) -> list[dict[str, Any]]:
    checker = RecoverabilityChecker()
    rows: list[dict[str, Any]] = []
    for probe in probes:
        for profile in DEFAULT_FAULT_PROFILES:
            for repeat in range(SINGLE_REPEATS):
                assessment = checker.assess(
                    probe.state,
                    probe.scenario.world,
                    probe.scenario.vehicle,
                    config=config,
                    profile=profile,
                    safe_anchor_ned_m=probe.safe_anchor_ned_m,
                )
                rows.append(
                    {
                        "probe": probe.name,
                        "scenario": probe.scenario.name,
                        "role": probe.role,
                        "include_in_aggregate": probe.include_in_aggregate,
                        "profile": profile.name,
                        "profile_digest": profile.digest,
                        "profile_check_count": profile.check_count,
                        "repeat": repeat,
                        "status": "valid" if assessment.valid else "invalid",
                        "valid": assessment.valid,
                        "elapsed_s": assessment.computation_time_s,
                        "current_clearance_m": assessment.current_clearance_m,
                        "reason": assessment.certificate.reason,
                    }
                )
    return rows


def _run_node_cache(
    probes: tuple[M7Probe, ...],
    config: RecoveryConfig,
    *,
    mode: RecoveryQueryMode = RecoveryQueryMode.FULL,
    parallel_workers: int = 1,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for probe in probes:
        for profile in DEFAULT_FAULT_PROFILES:
            session = RecoveryQuerySession(
                policy=RecoveryQueryPolicy(
                    budget_s=ONLINE_BUDGET_S,
                    use_cache=True,
                    use_prefilter=True,
                    mode=mode,
                    parallel_workers=parallel_workers,
                )
            )
            for pass_index in range(2):
                result = session.query(
                    probe.state,
                    probe.scenario.world,
                    probe.scenario.vehicle,
                    config=config,
                    profile=profile,
                    safe_anchor_ned_m=probe.safe_anchor_ned_m,
                )
                rows.append(
                    _node_row(probe, profile, pass_index, result, session)
                )
    return rows


def _node_row(
    probe: M7Probe,
    profile: FaultProfile,
    pass_index: int,
    result: RecoveryQueryResult,
    session: RecoveryQuerySession,
) -> dict[str, Any]:
    return {
        "probe": probe.name,
        "scenario": probe.scenario.name,
        "role": probe.role,
        "include_in_aggregate": probe.include_in_aggregate,
        "profile": profile.name,
        "profile_digest": profile.digest,
        "query_mode": result.mode.value,
        "pass": pass_index,
        "status": result.status.value,
        "cache_hit": result.cache_hit,
        "budget_exhausted": result.budget_exhausted,
        "prefilter_reason": result.prefilter_reason,
        "reason": result.reason,
        "elapsed_s": result.elapsed_s,
        "early_stopped": result.early_stopped,
        "evaluated_checks": result.evaluated_checks,
        "remaining_budget_s": session.remaining_s,
        "session_queries": session.stats.queries,
        "session_cache_hits": session.stats.cache_hits,
        "session_full_assessments": session.stats.full_assessments,
        "session_early_stops": session.stats.early_stops,
    }


def _run_path_audits(
    probes: tuple[M7Probe, ...],
    config: RecoveryConfig,
    *,
    online_mode: RecoveryQueryMode = RecoveryQueryMode.FULL,
    online_parallel_workers: int = 1,
    full_parallel_workers: int = 1,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for probe in probes:
        states = _audit_states(probe, config)
        for profile in DEFAULT_FAULT_PROFILES:
            online = _audit_online(
                probe,
                profile,
                states,
                config,
                mode=online_mode,
                parallel_workers=online_parallel_workers,
            )
            rows.append(online)
            rows.extend(
                _audit_full_cached(
                    probe,
                    profile,
                    states,
                    config,
                    parallel_workers=full_parallel_workers,
                )
            )
    return rows


def _audit_online(
    probe: M7Probe,
    profile: FaultProfile,
    states: tuple[State, ...],
    config: RecoveryConfig,
    *,
    mode: RecoveryQueryMode = RecoveryQueryMode.FULL,
    parallel_workers: int = 1,
) -> dict[str, Any]:
    session = RecoveryQuerySession(
        policy=RecoveryQueryPolicy(
            budget_s=ONLINE_BUDGET_S,
            use_cache=True,
            use_prefilter=True,
            mode=mode,
            parallel_workers=parallel_workers,
        )
    )
    results: list[RecoveryQueryResult] = []
    for state in states:
        result = session.query(
            state,
            probe.scenario.world,
            probe.scenario.vehicle,
            config=config,
            profile=profile,
            safe_anchor_ned_m=probe.safe_anchor_ned_m,
        )
        results.append(result)
        if result.budget_exhausted:
            break
    return _path_row(probe, profile, "online_10s", states, results)


def _audit_full_cached(
    probe: M7Probe,
    profile: FaultProfile,
    states: tuple[State, ...],
    config: RecoveryConfig,
    *,
    parallel_workers: int = 1,
) -> list[dict[str, Any]]:
    session = RecoveryQuerySession(
        policy=RecoveryQueryPolicy(
            budget_s=None,
            use_cache=True,
            use_prefilter=True,
            mode=RecoveryQueryMode.FULL,
            parallel_workers=parallel_workers,
        )
    )
    rows: list[dict[str, Any]] = []
    for mode in ("full_cold", "full_warm"):
        results = [
            session.query(
                state,
                probe.scenario.world,
                probe.scenario.vehicle,
                config=config,
                profile=profile,
                safe_anchor_ned_m=probe.safe_anchor_ned_m,
            )
            for state in states
        ]
        rows.append(_path_row(probe, profile, mode, states, results))
    return rows


def _path_row(
    probe: M7Probe,
    profile: FaultProfile,
    mode: str,
    states: tuple[State, ...],
    results: list[RecoveryQueryResult],
) -> dict[str, Any]:
    valid_results = [
        (state, result)
        for state, result in zip(states, results)
        if result.status is RecoveryQueryStatus.VALID
    ]
    planned_duration_s = float(states[-1].t_s - states[0].t_s)
    coverage_s = min(planned_duration_s, _certificate_coverage_s(valid_results))
    cache_hits = sum(result.cache_hit for result in results)
    full_assessments = sum(
        result.assessment is not None and not result.cache_hit for result in results
    )
    reasons = Counter(str(result.reason) for result in results if result.reason is not None)
    return {
        "probe": probe.name,
        "scenario": probe.scenario.name,
        "role": probe.role,
        "include_in_aggregate": probe.include_in_aggregate,
        "profile": profile.name,
        "profile_digest": profile.digest,
        "mode": mode,
        "query_mode": results[0].mode.value if results else "unknown",
        "points_planned": len(states),
        "points_attempted": len(results),
        "valid_points": sum(result.status is RecoveryQueryStatus.VALID for result in results),
        "invalid_points": sum(result.status is RecoveryQueryStatus.INVALID for result in results),
        "unknown_points": sum(result.status is RecoveryQueryStatus.UNKNOWN for result in results),
        "complete": len(results) == len(states) and all(
            result.status is not RecoveryQueryStatus.UNKNOWN for result in results
        ),
        "certificate_coverage_s": coverage_s,
        "coverage_ratio": 0.0 if planned_duration_s <= 0.0 else coverage_s / planned_duration_s,
        "total_s": sum(result.elapsed_s for result in results),
        "cache_hits": cache_hits,
        "full_assessments": full_assessments,
        "early_stopped_points": sum(result.early_stopped for result in results),
        "evaluated_checks": sum(result.evaluated_checks for result in results),
        "budget_exhausted": any(result.budget_exhausted for result in results),
        "failure_reasons": ";".join(f"{reason}:{count}" for reason, count in sorted(reasons.items())),
    }


def _certificate_coverage_s(valid_results: list[tuple[State, RecoveryQueryResult]]) -> float:
    intervals: list[tuple[float, float]] = []
    for state, result in valid_results:
        if result.assessment is None:
            continue
        certificate = result.assessment.certificate
        if certificate.valid_from_s is None or certificate.valid_until_s is None:
            continue
        intervals.append((certificate.valid_from_s, certificate.valid_until_s))
    if not intervals:
        return 0.0
    intervals.sort()
    coverage = 0.0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            coverage += end - start
            start, end = next_start, next_end
    coverage += end - start
    return float(coverage)


def _audit_states(probe: M7Probe, config: RecoveryConfig) -> tuple[State, ...]:
    candidates = build_maneuvers(
        probe.state,
        probe.scenario.world,
        probe.scenario.vehicle,
        DEFAULT_FAULTS[0],
        config,
        duration_s=config.horizon_s,
        safe_anchor_ned_m=probe.safe_anchor_ned_m,
    )
    candidate = next(
        candidate for candidate in candidates if candidate.name == "brake_to_min_speed"
    )
    trajectory = candidate.trajectory
    selected = [0]
    next_time = float(trajectory.t_s[0] + AUDIT_INTERVAL_S)
    for index in range(1, trajectory.t_s.shape[0]):
        if float(trajectory.t_s[index]) + 1e-12 >= next_time:
            selected.append(index)
            next_time = float(trajectory.t_s[index] + AUDIT_INTERVAL_S)
    if selected[-1] != trajectory.t_s.shape[0] - 1:
        selected.append(trajectory.t_s.shape[0] - 1)
    return tuple(
        State(
            t_s=float(trajectory.t_s[index]),
            position_ned_m=trajectory.position_ned_m[index].copy(),
            quat_xyzw=trajectory.quat_xyzw[index].copy(),
            linear_velocity_body_mps=trajectory.linear_velocity_body_mps[index].copy(),
            angular_velocity_body_rps=trajectory.angular_velocity_body_rps[index].copy(),
            frame_id=trajectory.frame_id,
        )
        for index in selected
    )


def _prefilter_false_negatives(
    single_rows: list[dict[str, Any]],
    node_rows: list[dict[str, Any]],
    prefilter_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    direct = {
        (str(row["probe"]), str(row["profile"]), int(row["repeat"])): row
        for row in single_rows
    }
    findings: list[dict[str, Any]] = []
    for row in node_rows:
        if int(row["pass"]) != 0:
            continue
        full = direct[(str(row["probe"]), str(row["profile"]), 0)]
        if bool(full["valid"]) and row["prefilter_reason"] is not None:
            findings.append(
                {
                    "source": "start_node",
                    "probe": row["probe"],
                    "profile": row["profile"],
                    "prefilter_reason": row["prefilter_reason"],
                }
            )
    for row in prefilter_rows:
        if bool(row["false_negative"]):
            findings.append(
                {
                    "source": "audit_path",
                    "probe": row["probe"],
                    "profile": row["profile"],
                    "t_s": row["t_s"],
                    "prefilter_reason": row["prefilter_reason"],
                    "full_reason": row["full_reason"],
                }
            )
    return findings


def _validate_prefilter(
    probes: tuple[M7Probe, ...],
    config: RecoveryConfig,
) -> list[dict[str, Any]]:
    checker = RecoverabilityChecker()
    rows: list[dict[str, Any]] = []
    for probe in probes:
        states = _audit_states(probe, config)
        for profile in DEFAULT_FAULT_PROFILES:
            for state in states:
                prefilter = prefilter_state(
                    state,
                    probe.scenario.world,
                    probe.scenario.vehicle,
                    config,
                )
                if not prefilter.rejected:
                    continue
                assessment = checker.assess(
                    state,
                    probe.scenario.world,
                    probe.scenario.vehicle,
                    config=config,
                    profile=profile,
                    safe_anchor_ned_m=probe.safe_anchor_ned_m,
                )
                rows.append(
                    {
                        "probe": probe.name,
                        "scenario": probe.scenario.name,
                        "role": probe.role,
                        "include_in_aggregate": probe.include_in_aggregate,
                        "profile": profile.name,
                        "profile_digest": profile.digest,
                        "t_s": state.t_s,
                        "prefilter_reason": prefilter.reason,
                        "current_clearance_m": prefilter.current_clearance_m,
                        "full_valid": assessment.valid,
                        "false_negative": assessment.valid,
                        "full_reason": assessment.certificate.reason,
                        "assessment_s": assessment.computation_time_s,
                    }
                )
    return rows


def _summarize(
    probes: tuple[M7Probe, ...],
    single_rows: list[dict[str, Any]],
    node_rows: list[dict[str, Any]],
    path_rows: list[dict[str, Any]],
    prefilter_rows: list[dict[str, Any]],
    prefilter_false_negatives: list[dict[str, Any]],
    hardware: dict[str, Any],
    *,
    online_mode: RecoveryQueryMode,
    online_parallel_workers: int,
    full_parallel_workers: int,
) -> dict[str, Any]:
    aggregate_probes = {probe.name for probe in probes if probe.include_in_aggregate}
    aggregate_single = [row for row in single_rows if row["probe"] in aggregate_probes]
    aggregate_node = [row for row in node_rows if row["probe"] in aggregate_probes]
    aggregate_path = [row for row in path_rows if row["probe"] in aggregate_probes]
    by_profile: dict[str, Any] = {}
    for profile in DEFAULT_FAULT_PROFILES:
        single = [row for row in aggregate_single if row["profile"] == profile.name]
        node = [row for row in aggregate_node if row["profile"] == profile.name]
        path = [row for row in aggregate_path if row["profile"] == profile.name]
        by_profile[profile.name] = {
            "profile_digest": profile.digest,
            "check_count": profile.check_count,
            "single_full": _timing_summary([float(row["elapsed_s"]) for row in single]),
            "single_valid_rate": _bool_rate([bool(row["valid"]) for row in single]),
            "node_cold": _timing_summary(
                [float(row["elapsed_s"]) for row in node if int(row["pass"]) == 0]
            ),
            "node_warm": _timing_summary(
                [float(row["elapsed_s"]) for row in node if int(row["pass"]) == 1]
            ),
            "node_cache_hit_rate": _bool_rate([bool(row["cache_hit"]) for row in node]),
            "path_online_10s": _path_mode_summary(path, "online_10s"),
            "path_full_cold": _path_mode_summary(path, "full_cold"),
            "path_full_warm": _path_mode_summary(path, "full_warm"),
        }
    p1 = by_profile["P1_degraded"]
    p1_online = p1["path_online_10s"]
    p1_time_gate = bool(
        p1_online["all_complete"]
        and p1_online["budget_exhausted_rows"] == 0
        and p1_online["unknown_points"] == 0
    )
    p1_coverage_gate = bool(p1_online["minimum_coverage_ratio"] >= 1.0)
    p1_budget_gate = bool(
        p1_time_gate and p1_coverage_gate
    )
    return {
        "status": "M7.0 complete; cost profiling only, no planning-performance claim",
        "probe_count": len(probes),
        "aggregate_probe_count": len(aggregate_probes),
        "excluded_from_aggregate": sorted(
            probe.name for probe in probes if not probe.include_in_aggregate
        ),
        "profiles": by_profile,
        "query_config": {
            "online_mode": online_mode.value,
            "online_parallel_workers": online_parallel_workers,
            "full_parallel_workers": full_parallel_workers,
        },
        "prefilter_false_negatives": prefilter_false_negatives,
        "prefilter_false_negative_count": len(prefilter_false_negatives),
        "prefilter_validation_points": len(prefilter_rows),
        "prefilter_validation_by_profile": {
            profile.name: sum(row["profile"] == profile.name for row in prefilter_rows)
            for profile in DEFAULT_FAULT_PROFILES
        },
        "p1_online_budget_gate": p1_budget_gate,
        "p1_online_time_gate": p1_time_gate,
        "p1_online_coverage_gate": p1_coverage_gate,
        "hardware": hardware,
    }


def _path_mode_summary(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    selected = [row for row in rows if row["mode"] == mode]
    return {
        "rows": len(selected),
        "all_complete": all(bool(row["complete"]) for row in selected) if selected else False,
        "budget_exhausted_rows": sum(bool(row["budget_exhausted"]) for row in selected),
        "total_s": _timing_summary([float(row["total_s"]) for row in selected]),
        "coverage_ratio": _timing_summary([float(row["coverage_ratio"]) for row in selected]),
        "minimum_coverage_ratio": min(
            (float(row["coverage_ratio"]) for row in selected), default=0.0
        ),
        "cache_hits": sum(int(row["cache_hits"]) for row in selected),
        "full_assessments": sum(int(row["full_assessments"]) for row in selected),
        "early_stopped_points": sum(int(row["early_stopped_points"]) for row in selected),
        "evaluated_checks": sum(int(row["evaluated_checks"]) for row in selected),
        "unknown_points": sum(int(row["unknown_points"]) for row in selected),
        "query_modes": sorted({str(row["query_mode"]) for row in selected}),
    }


def _timing_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean_s": 0.0, "median_s": 0.0, "p95_s": 0.0, "max_s": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean_s": float(np.mean(array)),
        "median_s": float(np.median(array)),
        "p95_s": float(np.quantile(array, 0.95)),
        "max_s": float(np.max(array)),
    }


def _bool_rate(values: list[bool]) -> float:
    return 0.0 if not values else sum(values) / len(values)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def hardware_manifest() -> dict[str, Any]:
    blas = io.StringIO()
    with redirect_stdout(blas):
        np.show_config()
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": affinity,
        "cpu_affinity_count": None if affinity is None else len(affinity),
        "cgroup_cpu_max": _read_text("/sys/fs/cgroup/cpu.max"),
        "cgroup_cpu_quota_us": _read_text("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
        "cgroup_cpu_period_us": _read_text("/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
        "memory_total": _memory_total(),
        "python": sys.version,
        "python_executable": sys.executable,
        "numpy_version": np.__version__,
        "numpy_blas_config": blas.getvalue().strip(),
        "git_commit": git_commit,
    }


def _cpu_model() -> str | None:
    for line in (_read_text("/proc/cpuinfo", default="") or "").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return None


def _memory_total() -> str | None:
    for line in (_read_text("/proc/meminfo", default="") or "").splitlines():
        if line.startswith("MemTotal:"):
            return line.split(":", 1)[1].strip()
    return None


def _read_text(path: str, default: str | None = None) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return default


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    query_config = summary.get("query_config", {})
    lines = [
        "# M7.0 证书查询成本预研",
        "",
        "本报告由 M7.0 实验脚本生成。只测量证书查询成本，不实现候选规划器，",
        "也不比较规划成功率。所有在线预算固定为 10 s；完整离线审计单独计时。",
        "",
        f"- 在线查询模式：`{query_config.get('online_mode', 'unknown')}`",
        f"- 在线并行 worker：`{query_config.get('online_parallel_workers', 'unknown')}`",
        f"- 离线完整审计并行 worker：`{query_config.get('full_parallel_workers', 'unknown')}`",
        "",
        "## 故障档案",
        "",
        "| 档案 | 检查数 | profile_digest |",
        "|---|---:|---|",
    ]
    for profile in DEFAULT_FAULT_PROFILES:
        lines.append(f"| `{profile.name}` | {profile.check_count} | `{profile.digest}` |")
    lines.extend(
        [
            "",
            "## 成本汇总",
            "",
            "| 档案 | 单次证书中位数 | 冷节点查询中位数 | 热缓存中位数 | 在线路径完成 | 在线早停点 | 完整路径冷审计中位数 | 最低覆盖率 |",
            "|---|---:|---:|---:|---|---:|---:|---:|",
        ]
    )
    for name, profile_summary in summary["profiles"].items():
        online = profile_summary["path_online_10s"]
        lines.append(
            f"| `{name}` | {_ms(profile_summary['single_full']['median_s'])} | "
            f"{_ms(profile_summary['node_cold']['median_s'])} | "
            f"{_ms(profile_summary['node_warm']['median_s'])} | "
            f"{online['all_complete']} | "
            f"{online['early_stopped_points']} | "
            f"{profile_summary['path_full_cold']['total_s']['median_s']:.3f} s | "
            f"{online['minimum_coverage_ratio']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 门槛结论",
            "",
            f"- `P1_degraded` 在线 10 s 时间门槛：`{summary['p1_online_time_gate']}`。",
            f"- `P1_degraded` 在线 10 s 覆盖率门槛：`{summary['p1_online_coverage_gate']}`。",
            f"- `P1_degraded` 两者合并门槛：`{summary['p1_online_budget_gate']}`。",
            f"- 预筛假阴性：`{summary['prefilter_false_negative_count']}`。",
            f"- 预筛拒绝点完整复核：`{summary['prefilter_validation_points']}` 个。",
            f"- 聚合时排除的探针：{', '.join(f'`{name}`' for name in summary['excluded_from_aggregate'])}。",
            "",
            "## 解释边界",
            "",
            "- 路径审计使用冻结的 `brake_to_min_speed` 轨迹作为成本探针，不代表候选规划器轨迹。",
            "- 在线预算耗尽记为 `UNKNOWN`，不计为不可行，也不计为安全。",
            "- 证书仍只覆盖有限离散时域；按不超过 1 s 间隔查询只称采样序列证书。",
            "- 误差界、故障严重度和 1 s 证书有效期均沿用 M6 冻结值，没有调整。",
            "",
            f"- 服务器：`{summary['hardware'].get('hostname')}`，Python `{summary['hardware'].get('python_executable')}`。",
            f"- CPU：`{summary['hardware'].get('cpu_model')}`，可用亲和核数 `{summary['hardware'].get('cpu_affinity_count')}`。",
            f"- Git commit：`{summary['hardware'].get('git_commit')}`。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _ms(value: float) -> str:
    return f"{float(value) * 1000.0:.1f} ms"


__all__ = [
    "AUDIT_INTERVAL_S",
    "M7Probe",
    "ONLINE_BUDGET_S",
    "SINGLE_REPEATS",
    "hardware_manifest",
    "m7_budget_manifest",
    "m7_audit_states",
    "m7_probes",
    "run_m7_budget",
    "write_m7_budget_manifest",
]
