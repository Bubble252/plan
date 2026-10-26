from __future__ import annotations

import time

import numpy as np

from ..contracts import PlanRequest, PlanResult, PlanStatus
from .common import SegmentValidator, finalize_path


class RRTStarPlanner:
    """Deterministic-seeded RRT* over full-hull validated position segments."""

    def __init__(
        self,
        step_size_m: float = 0.5,
        max_iterations: int = 3_000,
        timeout_s: float = 10.0,
        goal_bias: float = 0.1,
        rewire_radius_m: float | None = None,
        collision_step_m: float = 0.05,
    ) -> None:
        if not np.isfinite(step_size_m) or step_size_m <= 0.0:
            raise ValueError("step_size_m must be finite and positive")
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if not np.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("timeout_s must be finite and positive")
        if not np.isfinite(goal_bias) or not 0.0 <= goal_bias <= 1.0:
            raise ValueError("goal_bias must be in [0, 1]")
        if rewire_radius_m is None:
            rewire_radius_m = 2.0 * step_size_m
        if not np.isfinite(rewire_radius_m) or rewire_radius_m <= 0.0:
            raise ValueError("rewire_radius_m must be finite and positive")
        if not np.isfinite(collision_step_m) or collision_step_m <= 0.0:
            raise ValueError("collision_step_m must be finite and positive")
        self.step_size_m = float(step_size_m)
        self.max_iterations = int(max_iterations)
        self.timeout_s = float(timeout_s)
        self.goal_bias = float(goal_bias)
        self.rewire_radius_m = float(rewire_radius_m)
        self.collision_step_m = float(collision_step_m)

    def _failure(
        self,
        request: PlanRequest,
        status: PlanStatus,
        reason: str,
        started: float,
        **diagnostics: object,
    ) -> PlanResult:
        return PlanResult(
            status=status,
            trajectory=None,
            cost=0.0,
            certificate=None,
            planning_time_s=time.perf_counter() - started,
            plan_id=f"{request.request_id}:rrt_star",
            reason=reason,
            diagnostics={
                "planner": "rrt_star",
                "seed": request.seed,
                "step_size_m": self.step_size_m,
                **diagnostics,
            },
        )

    def plan(self, request: PlanRequest) -> PlanResult:
        if not isinstance(request, PlanRequest):
            raise TypeError("request must be a PlanRequest")
        started = time.perf_counter()
        timeout_s = min(
            self.timeout_s,
            float(request.task.constraints.get("planning_timeout_s", self.timeout_s)),
        )
        if not np.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("planning_timeout_s must be finite and positive")
        obstacle_map = request.world.obstacle_map
        lower = obstacle_map.bounds_ned_m[:, 0]
        upper = obstacle_map.bounds_ned_m[:, 1]
        start = request.start.position_ned_m.copy()
        goal = request.task.goal_ned_m.copy()
        if not np.all(start >= lower) or not np.all(start <= upper):
            return self._failure(request, PlanStatus.INFEASIBLE, "start_out_of_bounds", started)
        if not np.all(goal >= lower) or not np.all(goal <= upper):
            return self._failure(request, PlanStatus.INFEASIBLE, "goal_out_of_bounds", started)

        validator = SegmentValidator(
            obstacle_map,
            request.vehicle,
            max_step_m=self.collision_step_m,
        )
        if not validator.state_is_safe(start, request.start.quat_xyzw):
            return self._failure(request, PlanStatus.INFEASIBLE, "start_in_collision", started)
        if float(np.linalg.norm(goal - start)) <= request.task.goal_radius_m:
            finalized = finalize_path(
                np.asarray([start]),
                request,
                validator,
                trajectory_id=f"{request.request_id}:rrt_star",
            )
            return PlanResult(
                status=PlanStatus.OK,
                trajectory=finalized.trajectory,
                cost=finalized.path_length_m,
                certificate=None,
                planning_time_s=time.perf_counter() - started,
                plan_id=f"{request.request_id}:rrt_star",
                diagnostics={
                    "planner": "rrt_star",
                    "nodes": 1,
                    "already_in_goal_region": True,
                    "seed": request.seed,
                },
            )
        if validator(start, goal):
            finalized = finalize_path(
                np.vstack((start, goal)),
                request,
                validator,
                trajectory_id=f"{request.request_id}:rrt_star",
            )
            if finalized.sweep.is_safe:
                return PlanResult(
                    status=PlanStatus.OK,
                    trajectory=finalized.trajectory,
                    cost=finalized.path_length_m,
                    certificate=None,
                    planning_time_s=time.perf_counter() - started,
                    plan_id=f"{request.request_id}:rrt_star",
                    diagnostics={
                        "planner": "rrt_star",
                        "nodes": 1,
                        "direct_path": True,
                        "seed": request.seed,
                    },
                )
        rng = np.random.default_rng(request.seed)
        positions = [start.copy()]
        costs = [0.0]
        parents = [-1]
        children: list[list[int]] = [[]]
        goal_candidates: list[tuple[float, int]] = []
        iterations = 0

        while iterations < self.max_iterations:
            if time.perf_counter() - started >= timeout_s:
                break
            iterations += 1
            if rng.random() < self.goal_bias:
                sample = goal.copy()
            else:
                sample = rng.uniform(lower, upper)
            node_array = np.vstack(positions)
            nearest_index = int(np.argmin(np.linalg.norm(node_array - sample, axis=1)))
            nearest = positions[nearest_index]
            delta = sample - nearest
            distance = float(np.linalg.norm(delta))
            if distance <= 1e-12:
                continue
            candidate = sample.copy() if distance <= self.step_size_m else nearest + self.step_size_m * delta / distance
            if np.any(candidate < lower) or np.any(candidate > upper):
                continue
            if not validator(nearest, candidate):
                continue

            near_indices = [
                index
                for index in range(len(positions))
                if float(np.linalg.norm(positions[index] - candidate)) <= self.rewire_radius_m
            ]
            near_indices.sort()
            parent_index = nearest_index
            parent_cost = costs[nearest_index] + float(np.linalg.norm(candidate - nearest))
            for index in near_indices:
                candidate_cost = costs[index] + float(np.linalg.norm(candidate - positions[index]))
                if candidate_cost < parent_cost and validator(positions[index], candidate):
                    parent_index = index
                    parent_cost = candidate_cost

            new_index = len(positions)
            positions.append(candidate)
            costs.append(parent_cost)
            parents.append(parent_index)
            children.append([])
            children[parent_index].append(new_index)

            ancestors = set()
            cursor = parent_index
            while cursor >= 0:
                ancestors.add(cursor)
                cursor = parents[cursor]

            for index in near_indices:
                if index == parent_index or index in ancestors:
                    continue
                rewired_cost = costs[new_index] + float(np.linalg.norm(positions[index] - candidate))
                if rewired_cost + 1e-12 >= costs[index]:
                    continue
                if not validator(candidate, positions[index]):
                    continue
                old_parent = parents[index]
                if old_parent >= 0:
                    children[old_parent].remove(index)
                parents[index] = new_index
                children[new_index].append(index)
                queue = [index]
                while queue:
                    node = queue.pop()
                    for child in children[node]:
                        costs[child] = costs[node] + float(
                            np.linalg.norm(positions[child] - positions[node])
                        )
                        queue.append(child)

            goal_distance = float(np.linalg.norm(goal - candidate))
            if goal_distance <= request.task.goal_radius_m and (
                goal_distance <= 1e-12 or validator(candidate, goal)
            ):
                goal_candidates.append((costs[new_index] + goal_distance, new_index))

        plan_id = f"{request.request_id}:rrt_star"
        if goal_candidates:
            last_failure_reason: str | None = None
            goal_indices = sorted(
                {index for _, index in goal_candidates},
                key=lambda index: (
                    costs[index] + float(np.linalg.norm(goal - positions[index])),
                    index,
                ),
            )
            for goal_index in goal_indices:
                indices = [goal_index]
                cursor = goal_index
                while parents[cursor] >= 0:
                    cursor = parents[cursor]
                    indices.append(cursor)
                indices.reverse()
                waypoints = np.vstack([positions[index] for index in indices] + [goal])
                finalized = finalize_path(
                    waypoints,
                    request,
                    validator,
                    trajectory_id=plan_id,
                )
                if finalized.sweep.is_safe:
                    return PlanResult(
                        status=PlanStatus.OK,
                        trajectory=finalized.trajectory,
                        cost=finalized.path_length_m,
                        certificate=None,
                        planning_time_s=time.perf_counter() - started,
                        plan_id=plan_id,
                        diagnostics={
                            "planner": "rrt_star",
                            "nodes": len(positions),
                            "iterations": iterations,
                            "goal_candidates": len(goal_indices),
                            "tree_waypoints": len(indices) + 1,
                            "trajectory_samples": finalized.trajectory.t_s.shape[0],
                            "seed": request.seed,
                        },
                    )
                last_failure_reason = finalized.sweep.failure_reason
            assert last_failure_reason is not None
            return self._failure(
                request,
                PlanStatus.INFEASIBLE,
                f"postprocess_{last_failure_reason}",
                started,
                nodes=len(positions),
                iterations=iterations,
            )

        timed_out = time.perf_counter() - started >= timeout_s
        return self._failure(
            request,
            PlanStatus.TIMEOUT if timed_out else PlanStatus.INFEASIBLE,
            "timeout" if timed_out else "no_solution",
            started,
            nodes=len(positions),
            iterations=iterations,
        )


__all__ = ["RRTStarPlanner"]
