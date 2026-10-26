from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from auv_planner.contracts import PlanStatus, Trajectory
from auv_planner.experiments.m7_1 import (
    AuditPoint,
    M71CaseSpec,
    M71Config,
    PathAudit,
    _candidate_states,
    _plan_request,
    coverage_metrics,
    run_m7_1,
    sample_trajectory_states,
)
from auv_planner.scenarios import m5_open_water_scene


def _trajectory() -> Trajectory:
    return Trajectory(
        trajectory_id="sample",
        t_s=np.array([0.0, 0.6, 1.4, 2.4]),
        position_ned_m=np.array(
            [[0.0, 0.0, 0.0], [0.6, 0.0, 0.0], [1.4, 0.0, 0.0], [2.4, 0.0, 0.0]]
        ),
        quat_xyzw=np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (4, 1)),
        linear_velocity_body_mps=np.tile(np.array([1.0, 0.0, 0.0]), (4, 1)),
        angular_velocity_body_rps=np.zeros((4, 3)),
    )


def _point(
    index: int,
    t_s: float,
    status: str,
    *,
    valid_from: float | None = None,
    valid_until: float | None = None,
) -> AuditPoint:
    return AuditPoint(
        index=index,
        t_s=t_s,
        position_ned_m=np.array([float(index), 0.0, 0.0]),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        clearance_m=2.0,
        prefilter_reason=None,
        online_status=status,
        online_reason=None,
        online_elapsed_s=0.0,
        online_early_stopped=False,
        online_evaluated_checks=0,
        full_status=status,
        full_reason=None,
        full_elapsed_s=0.0,
        full_early_stopped=False,
        full_evaluated_checks=0,
        certificate_valid_from_s=valid_from,
        certificate_valid_until_s=valid_until,
        failure_source=None,
    )


def _audit(points: tuple[AuditPoint, ...]) -> PathAudit:
    return PathAudit(
        case_name="unit",
        scenario_name="unit",
        planner_name="astar",
        variant="baseline",
        include_in_aggregate=True,
        plan_status=PlanStatus.OK.value,
        plan_success=True,
        plan_reason=None,
        planning_time_s=0.1,
        repair_time_s=0.0,
        trajectory=None,
        points=points,
        prefilter_time_s=0.0,
        online_time_s=0.0,
        full_time_s=0.0,
        path_length_m=2.0,
        duration_s=2.0,
        min_clearance_m=2.0,
        coverage_ratio=0.0,
        max_uncovered_s=2.0,
        coverage_intervals=(),
        uncovered_intervals=((0.0, 2.0),),
        failure_sources=(),
    )


def test_sample_trajectory_states_includes_both_ends_without_exceeding_interval() -> None:
    states = sample_trajectory_states(_trajectory(), 1.0)

    assert [state.t_s for state in states] == [0.0, 1.0, 2.0, 2.4]
    assert np.allclose(states[1].position_ned_m, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(states[-1].position_ned_m, np.array([2.4, 0.0, 0.0]))
    assert all(
        later.t_s - earlier.t_s <= 1.0 + 1e-12
        for earlier, later in zip(states, states[1:])
    )


def test_coverage_metrics_merges_certificates_and_reports_longest_gap() -> None:
    points = (
        _point(0, 0.0, "valid", valid_from=0.0, valid_until=1.0),
        _point(1, 1.0, "valid", valid_from=1.0, valid_until=2.0),
        _point(2, 2.0, "invalid"),
        _point(3, 3.0, "valid", valid_from=3.0, valid_until=4.0),
        _point(4, 5.0, "invalid"),
    )

    ratio, max_uncovered, covered, uncovered = coverage_metrics(
        points,
        start_s=0.0,
        end_s=5.0,
    )

    assert ratio == 0.6
    assert max_uncovered == 1.0
    assert covered == ((0.0, 2.0), (3.0, 4.0))
    assert uncovered == ((2.0, 3.0), (4.0, 5.0))


def test_plan_request_preserves_scenario_fields_and_timeout_override() -> None:
    scenario = m5_open_water_scene()
    request = _plan_request(
        scenario,
        request_id="m7.1",
        seed=7,
        timeout_s=4.0,
    )

    assert request.start == scenario.start
    assert request.world == scenario.world
    assert request.vehicle == scenario.vehicle
    assert np.array_equal(request.task.goal_ned_m, scenario.goal_ned_m)
    assert request.task.goal_radius_m == scenario.goal_radius_m
    assert request.task.constraints["planning_timeout_s"] == 4.0
    assert request.seed == 7


def test_repair_candidates_are_deterministic_and_use_safe_anchor_first() -> None:
    scenario = m5_open_water_scene()
    points = (
        _point(0, 0.0, "valid"),
        _point(1, 1.0, "invalid"),
        _point(2, 2.0, "valid"),
    )
    baseline = _audit(points)
    spec = M71CaseSpec(
        name="unit",
        scenario=scenario,
        safe_anchor_ned_m=np.array([0.0, 1.0, 0.0]),
        include_in_aggregate=True,
    )
    config = M71Config(repair_offsets_m=(1.0, 1.6), max_repair_candidates=3)

    first = _candidate_states(scenario, baseline, spec, config)
    second = _candidate_states(scenario, baseline, spec, config)

    assert len(first) == 3
    assert [source for _, source in first] == ["safe_anchor", "offset_1m", "offset_1m"]
    assert np.allclose(first[0][0].position_ned_m, np.array([0.0, 1.0, 0.0]))
    assert all(np.array_equal(a[0].position_ned_m, b[0].position_ned_m) for a, b in zip(first, second))


def test_m7_1_writes_real_path_audit_artifacts(tmp_path: Path) -> None:
    config = M71Config(
        profile_name="P0_nominal",
        case_names=("open_water_start",),
        planner_names=("astar",),
        full_workers=1,
        max_repair_candidates=0,
    )

    result = run_m7_1(tmp_path, config)
    case = result.cases[0]

    assert case.baseline.plan_success
    assert case.baseline.trajectory is not None
    assert len(case.baseline.points) >= 2
    assert case.baseline.planning_time_s >= 0.0
    assert case.baseline.end_to_end_s <= case.baseline.full_end_to_end_s
    assert 0.0 <= case.baseline.coverage_ratio <= 1.0
    assert case.repair_reason == "no_repair_candidates"
    for name in (
        "paths.csv",
        "points.csv",
        "trajectories.csv",
        "repair_attempts.csv",
        "summary.json",
        "manifest.json",
    ):
        assert (tmp_path / name).exists()
    with (tmp_path / "paths.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["case"] == "open_water_start"
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["case_count"] == 1
    assert summary["baseline"]["paths"] == 1
