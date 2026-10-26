from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from auv_planner.contracts import State
from auv_planner.experiments.m6_recovery import dead_end_scene
from auv_planner.experiments.m7_1 import _plan_request
from auv_planner.experiments.m7_2 import M72Config, run_m7_2
from auv_planner.metrics import compute_path_metrics
from auv_planner.planners.recoverable import RecoverablePlanner, RecoverablePlannerConfig
from auv_planner.scenarios import m5_open_water_scene


def _planner(
    *,
    recovery_mode: bool,
    profile_name: str = "P1_degraded",
    query_budget_s: float = 10.0,
    max_replans: int = 0,
) -> RecoverablePlanner:
    return RecoverablePlanner(
        RecoverablePlannerConfig(
            resolution_m=1.0,
            timeout_s=10.0,
            inner_timeout_s=6.0,
            checkpoint_spacing_m=2.0,
            query_budget_s=query_budget_s,
            max_replans=max_replans,
            block_radius_m=1.2,
            recovery_mode=recovery_mode,
            profile_name=profile_name,
        )
    )


def test_recoverable_off_is_deterministic_and_collision_safe() -> None:
    scenario = m5_open_water_scene()
    request = _plan_request(scenario, request_id="off", seed=0, timeout_s=10.0)
    planner = _planner(recovery_mode=False)

    first = planner.plan(request)
    second = planner.plan(request)

    assert first.status.value == "ok"
    assert first.trajectory is not None and second.trajectory is not None
    assert np.array_equal(first.trajectory.position_ned_m, second.trajectory.position_ned_m)
    metrics = compute_path_metrics(
        first.trajectory,
        scenario.world.obstacle_map,
        scenario.vehicle,
    )
    assert metrics.is_safe
    assert first.diagnostics["recovery_mode"] == "off"
    assert first.diagnostics["recovery_queries"] == 0


def test_recoverable_on_is_deterministic_on_partial_recovery_path() -> None:
    scenario = m5_open_water_scene()
    request = _plan_request(scenario, request_id="on", seed=0, timeout_s=10.0)
    planner = _planner(recovery_mode=True, max_replans=0)

    first = planner.plan(request)
    second = planner.plan(request)

    assert first.status.value == "ok"
    assert first.trajectory is not None and second.trajectory is not None
    assert np.array_equal(first.trajectory.position_ned_m, second.trajectory.position_ned_m)
    assert first.diagnostics["recovery_complete"] is False
    assert first.diagnostics["recovery_queries"] == second.diagnostics["recovery_queries"]
    assert first.diagnostics["selected_attempt"] == second.diagnostics["selected_attempt"]


def test_recoverable_rejects_invalid_start_with_explicit_reason() -> None:
    scenario = dead_end_scene()
    result = _planner(recovery_mode=True).plan(
        _plan_request(scenario, request_id="dead-end", seed=0, timeout_s=10.0)
    )

    assert result.status.value == "infeasible"
    assert result.reason == "infeasible_recovery_at_start"
    assert result.trajectory is None
    assert result.diagnostics["start_checkpoint"]["status"] == "invalid"


def test_recoverable_query_budget_exhaustion_is_timeout_not_valid() -> None:
    scenario = m5_open_water_scene()
    result = _planner(recovery_mode=True, query_budget_s=1e-9).plan(
        _plan_request(scenario, request_id="tiny-budget", seed=0, timeout_s=10.0)
    )

    assert result.status.value == "timeout"
    assert result.reason == "recovery_budget_exhausted_at_start"
    assert result.diagnostics["start_checkpoint"]["status"] == "unknown"
    assert result.diagnostics["start_checkpoint"]["budget_exhausted"] is True


def test_recoverable_replan_limit_returns_best_nominal_path() -> None:
    scenario = m5_open_water_scene()
    result = _planner(recovery_mode=True, max_replans=0).plan(
        _plan_request(scenario, request_id="limit", seed=0, timeout_s=10.0)
    )

    assert result.status.value == "ok"
    assert result.reason is None
    assert result.diagnostics["recovery_complete"] is False
    assert result.diagnostics["replans"] == 0
    assert len(result.diagnostics["attempts"]) == 1
    assert result.diagnostics["attempts"][0]["selected"] is True


def test_checkpoint_exactly_at_certificate_expiry_is_requeried(
    monkeypatch: Any,
) -> None:
    scenario = m5_open_water_scene()
    request = _plan_request(scenario, request_id="boundary", seed=0, timeout_s=10.0)
    planner = _planner(recovery_mode=True)
    states = (
        (0.0, request.start),
        (1.0, State(
            t_s=1.0,
            position_ned_m=request.start.position_ned_m + np.array([1.0, 0.0, 0.0]),
            quat_xyzw=request.start.quat_xyzw,
            linear_velocity_body_mps=request.start.linear_velocity_body_mps,
            angular_velocity_body_rps=request.start.angular_velocity_body_rps,
        )),
    )
    queries: list[float] = []

    monkeypatch.setattr(
        "auv_planner.planners.recoverable._checkpoint_states",
        lambda trajectory, spacing_m: states,
    )

    def fake_checkpoint(*, distance_m: float, state: State, session: Any, request: Any) -> Any:
        from auv_planner.planners.recoverable import _Checkpoint
        from auv_planner.recovery import RecoveryQueryStatus

        queries.append(state.t_s)
        return _Checkpoint(
            distance_m=distance_m,
            state=state,
            prefilter_reason=None,
            status=RecoveryQueryStatus.VALID,
            reason=None,
            margin_m=0.2,
            certificate_valid_from_s=state.t_s,
            certificate_valid_until_s=state.t_s + 1.0,
            elapsed_s=0.0,
            early_stopped=False,
            evaluated_checks=1,
            budget_exhausted=False,
            query_source="query",
        )

    monkeypatch.setattr(planner, "_checkpoint", fake_checkpoint)
    checkpoints = planner._checkpoints(
        trajectory=object(),  # type: ignore[arg-type]
        session=object(),  # type: ignore[arg-type]
        request=request,
    )

    assert queries == [0.0, 1.0]
    assert all(item.query_source == "query" for item in checkpoints)


def test_m7_2_writes_complete_artifacts(tmp_path: Path) -> None:
    config = M72Config(
        profile_name="P0_nominal",
        case_names=("open_water_start",),
        full_workers=1,
        candidate_max_replans=0,
        candidate_checkpoint_spacing_m=2.0,
    )

    result = run_m7_2(tmp_path, config)

    assert len(result.runs) == 4
    for name in (
        "paths.csv",
        "points.csv",
        "checkpoints.csv",
        "trajectories.csv",
        "endpoint_recoverability.csv",
        "pareto.csv",
        "summary.json",
        "manifest.json",
    ):
        assert (tmp_path / name).exists()
    with (tmp_path / "paths.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["variant"] for row in rows} == {
        "astar",
        "rrt_star",
        "candidate_on",
        "candidate_off",
    }
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["case_count"] == 1
    assert summary["aggregate_case_count"] == 1
    assert summary["aggregate"]["candidate_on"]["paths"] == 1
    assert "candidate_on_off" in summary


def test_m7_2_dead_end_is_rejection_only_for_candidate(tmp_path: Path) -> None:
    config = M72Config(
        case_names=("dead_end_no_recovery",),
        variant_names=("candidate_on",),
        full_workers=1,
    )

    result = run_m7_2(tmp_path, config)

    run = result.runs[0]
    assert run.include_in_aggregate is False
    assert run.start_recovery_infeasible is True
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["aggregate_case_count"] == 0
    assert summary["rejection_case_count"] == 1
    assert summary["rejection"]["candidate_on"]["start_recovery_infeasible"] == 1
