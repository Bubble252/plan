from __future__ import annotations

import csv
import json

from auv_planner.contracts import PlanResult, PlanStatus
from auv_planner.evaluation import (
    run_scenario,
    write_metrics_csv,
    write_run_log,
)
from auv_planner.planners.astar import AStarPlanner
from auv_planner.scenarios import empty_scene

class NoSolutionPlanner:
    def plan(self, request):
        return PlanResult(
            status=PlanStatus.INFEASIBLE,
            trajectory=None,
            cost=0.0,
            certificate=None,
            planning_time_s=0.01,
            plan_id=f"{request.request_id}:none",
            reason="no_solution",
        )


def test_run_scenario_produces_eval_result_and_success() -> None:
    record = run_scenario(
        empty_scene(),
        AStarPlanner(resolution_m=0.5, timeout_s=3.0, collision_step_m=0.1),
        seed=11,
    )

    assert record.plan.status is PlanStatus.OK
    assert record.result.success
    assert record.result.failure_reason is None
    assert record.result.metrics["goal_distance_m"] <= empty_scene().goal_radius_m


def test_no_trajectory_still_produces_failed_eval_result() -> None:
    record = run_scenario(empty_scene(), NoSolutionPlanner(), seed=0)
    result = record.result

    assert not result.success
    assert result.failure_reason == "no_solution"
    assert result.path_length_m == 0.0


def test_metrics_csv_and_run_log_are_written(tmp_path) -> None:
    success_record = run_scenario(
        empty_scene(),
        AStarPlanner(resolution_m=0.5, timeout_s=3.0, collision_step_m=0.1),
        seed=0,
    )
    failure_record = run_scenario(empty_scene(), NoSolutionPlanner(), seed=0)
    csv_path = write_metrics_csv([success_record, failure_record], tmp_path / "metrics.csv")
    log_path = write_run_log(success_record, tmp_path / "logs" / "empty.json")

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with log_path.open(encoding="utf-8") as handle:
        log = json.load(handle)

    assert len(rows) == 2
    assert rows[0]["scenario"] == "m3_empty"
    assert rows[0]["planner"] == "AStarPlanner"
    assert rows[0]["status"] == "ok"
    assert rows[0]["success"] == "True"
    assert rows[1]["planner"] == "NoSolutionPlanner"
    assert rows[1]["success"] == "False"
    assert rows[1]["failure_reason"] == "no_solution"
    assert rows[1]["goal_distance_m"] == ""
    assert log["request_id"].startswith("m3_empty:AStarPlanner")
    assert "diagnostics" in log
