from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import (
    CapabilitySet,
    EvalResult,
    PlanRequest,
    PlanResult,
    PlanStatus,
    ScenarioConfig,
    TaskSpec,
)
from ..interfaces import GlobalPlanner
from ..metrics import compute_path_metrics

CSV_FIELDS: tuple[str, ...] = (
    "scenario",
    "planner",
    "seed",
    "status",
    "success",
    "collision",
    "out_of_bounds",
    "min_clearance_m",
    "path_length_m",
    "duration_s",
    "planning_time_s",
    "goal_distance_m",
    "failure_reason",
    "plan_id",
)


@dataclass(frozen=True)
class EvaluationRecord:
    scenario_name: str
    planner_name: str
    seed: int
    request_id: str
    plan: PlanResult
    result: EvalResult

    def csv_row(self) -> dict[str, Any]:
        trajectory = self.plan.trajectory
        goal_distance = "" if trajectory is None else self.result.metrics.get("goal_distance_m", "")
        return {
            "scenario": self.scenario_name,
            "planner": self.planner_name,
            "seed": self.seed,
            "status": self.plan.status.value,
            "success": self.result.success,
            "collision": self.result.collision,
            "out_of_bounds": self.result.out_of_bounds,
            "min_clearance_m": self.result.min_clearance_m,
            "path_length_m": self.result.path_length_m,
            "duration_s": self.result.duration_s,
            "planning_time_s": self.plan.planning_time_s,
            "goal_distance_m": goal_distance,
            "failure_reason": self.result.failure_reason or "",
            "plan_id": self.plan.plan_id,
        }

    def log_dict(self) -> dict[str, Any]:
        return {
            **self.csv_row(),
            "request_id": self.request_id,
            "diagnostics": dict(self.plan.diagnostics),
            "metrics": dict(self.result.metrics),
        }


def evaluate_plan(request: PlanRequest, plan: PlanResult) -> EvalResult:
    if not isinstance(request, PlanRequest):
        raise TypeError("request must be a PlanRequest")
    if not isinstance(plan, PlanResult):
        raise TypeError("plan must be a PlanResult")
    if plan.trajectory is None:
        reason = plan.reason or plan.status.value
        return EvalResult(
            success=False,
            collision=False,
            out_of_bounds=False,
            min_clearance_m=0.0,
            path_length_m=0.0,
            duration_s=0.0,
            energy_j=0.0,
            tracking_rmse=0.0,
            constraint_violations=0,
            failure_reason=reason,
            metrics={
                "planning_time_s": plan.planning_time_s,
                "seed": float(request.seed),
            },
        )

    path_metrics = compute_path_metrics(
        plan.trajectory,
        request.world.obstacle_map,
        request.vehicle,
    )
    goal_distance = float(
        np.linalg.norm(plan.trajectory.position_ned_m[-1] - request.task.goal_ned_m)
    )
    goal_reached = goal_distance <= request.task.goal_radius_m
    success = plan.status is PlanStatus.OK and goal_reached and path_metrics.is_safe
    reasons: list[str] = []
    if plan.status is not PlanStatus.OK:
        reasons.append(plan.reason or plan.status.value)
    if not path_metrics.is_safe and path_metrics.failure_reason:
        reasons.append(path_metrics.failure_reason)
    if not goal_reached:
        reasons.append("goal_not_reached")
    failure_reason = ",".join(dict.fromkeys(reasons)) if reasons else None
    return EvalResult(
        success=success,
        collision=path_metrics.collision,
        out_of_bounds=path_metrics.out_of_bounds,
        min_clearance_m=path_metrics.min_clearance_m,
        path_length_m=path_metrics.path_length_m,
        duration_s=path_metrics.duration_s,
        energy_j=0.0,
        tracking_rmse=0.0,
        constraint_violations=0,
        failure_reason=failure_reason,
        metrics={
            "planning_time_s": plan.planning_time_s,
            "goal_distance_m": goal_distance,
            "seed": float(request.seed),
            "sampled_poses": float(path_metrics.sampled_poses),
        },
    )


def run_scenario(
    scenario: ScenarioConfig,
    planner: GlobalPlanner,
    seed: int = 0,
) -> EvaluationRecord:
    if not isinstance(scenario, ScenarioConfig):
        raise TypeError("scenario must be a ScenarioConfig")
    if not isinstance(planner, GlobalPlanner):
        raise TypeError("planner must implement GlobalPlanner")
    planner_name = type(planner).__name__
    request_id = f"{scenario.name}:{planner_name}:seed{seed}"
    request = PlanRequest(
        request_id=request_id,
        start=scenario.start,
        task=TaskSpec(
            goal_ned_m=scenario.goal_ned_m,
            goal_radius_m=scenario.goal_radius_m,
            constraints=scenario.constraints,
        ),
        world=scenario.world,
        vehicle=scenario.vehicle,
        capability=CapabilitySet(
            available_actuators=("propeller", "yaw_fin", "pitch_fin"),
            effectiveness={"propeller": 1.0, "yaw_fin": 1.0, "pitch_fin": 1.0},
        ),
        seed=seed,
    )
    plan = planner.plan(request)
    result = evaluate_plan(request, plan)
    return EvaluationRecord(
        scenario_name=scenario.name,
        planner_name=planner_name,
        seed=seed,
        request_id=request_id,
        plan=plan,
        result=result,
    )


def write_metrics_csv(records: Iterable[EvaluationRecord], path: Path | str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.csv_row())
    return output_path


def write_run_log(record: EvaluationRecord, path: Path | str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(record.log_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def run_suite(
    scenarios: Iterable[ScenarioConfig],
    planners: Iterable[GlobalPlanner],
    seeds: Iterable[int],
    *,
    log_dir: Path | str | None = None,
) -> list[EvaluationRecord]:
    records: list[EvaluationRecord] = []
    log_root = None if log_dir is None else Path(log_dir)
    for scenario in scenarios:
        for planner in planners:
            for seed in seeds:
                record = run_scenario(scenario, planner, seed)
                records.append(record)
                if log_root is not None:
                    write_run_log(
                        record,
                        log_root / f"{scenario.name}__{record.planner_name}__seed{seed}.json",
                    )
    return records


__all__ = [
    "CSV_FIELDS",
    "EvaluationRecord",
    "evaluate_plan",
    "run_scenario",
    "run_suite",
    "write_metrics_csv",
    "write_run_log",
]
