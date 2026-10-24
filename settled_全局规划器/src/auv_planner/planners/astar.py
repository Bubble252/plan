from __future__ import annotations

import heapq
import itertools
import time

import numpy as np

from ..contracts import PlanRequest, PlanResult, PlanStatus
from .common import SegmentValidator, finalize_path


class AStarPlanner:
    """Three-dimensional 26-neighbour A* over a start-anchored sparse lattice."""

    def __init__(
        self,
        resolution_m: float = 0.25,
        max_expansions: int = 100_000,
        timeout_s: float = 10.0,
        collision_step_m: float | None = None,
    ) -> None:
        if not np.isfinite(resolution_m) or resolution_m <= 0.0:
            raise ValueError("resolution_m must be finite and positive")
        if max_expansions < 1:
            raise ValueError("max_expansions must be positive")
        if not np.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("timeout_s must be finite and positive")
        self.resolution_m = float(resolution_m)
        self.max_expansions = int(max_expansions)
        self.timeout_s = float(timeout_s)
        self.collision_step_m = float(
            min(self.resolution_m / 2.0, 0.05) if collision_step_m is None else collision_step_m
        )
        if not np.isfinite(self.collision_step_m) or self.collision_step_m <= 0.0:
            raise ValueError("collision_step_m must be finite and positive")
        self._neighbour_offsets = tuple(
            offset
            for offset in itertools.product((-1, 0, 1), repeat=3)
            if offset != (0, 0, 0)
        )

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
            plan_id=f"{request.request_id}:astar",
            reason=reason,
            diagnostics={
                "planner": "astar",
                "resolution_m": self.resolution_m,
                "seed": request.seed,
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
        start = request.start.position_ned_m
        goal = request.task.goal_ned_m
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
                trajectory_id=f"{request.request_id}:astar",
            )
            return PlanResult(
                status=PlanStatus.OK,
                trajectory=finalized.trajectory,
                cost=finalized.path_length_m,
                certificate=None,
                planning_time_s=time.perf_counter() - started,
                plan_id=f"{request.request_id}:astar",
                diagnostics={
                    "planner": "astar",
                    "expanded_nodes": 0,
                    "already_in_goal_region": True,
                    "seed": request.seed,
                },
            )
        if validator(start, goal):
            finalized = finalize_path(
                np.vstack((start, goal)),
                request,
                validator,
                trajectory_id=f"{request.request_id}:astar",
            )
            if finalized.sweep.is_safe:
                return PlanResult(
                    status=PlanStatus.OK,
                    trajectory=finalized.trajectory,
                    cost=finalized.path_length_m,
                    certificate=None,
                    planning_time_s=time.perf_counter() - started,
                    plan_id=f"{request.request_id}:astar",
                    diagnostics={
                        "planner": "astar",
                        "expanded_nodes": 0,
                        "direct_path": True,
                        "seed": request.seed,
                    },
                )
        minimum_index = np.ceil((lower - start) / self.resolution_m).astype(int)
        maximum_index = np.floor((upper - start) / self.resolution_m).astype(int)

        def index_to_world(index: tuple[int, int, int]) -> np.ndarray:
            return start + self.resolution_m * np.asarray(index, dtype=np.float64)

        def heuristic(index: tuple[int, int, int]) -> float:
            return float(np.linalg.norm(index_to_world(index) - goal))

        origin = (0, 0, 0)
        frontier: list[tuple[float, int, tuple[int, int, int]]] = [(heuristic(origin), 0, origin)]
        g_score: dict[tuple[int, int, int], float] = {origin: 0.0}
        parent: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        closed: set[tuple[int, int, int]] = set()
        tie_breaker = 0
        expanded_nodes = 0
        postprocess_failure: str | None = None

        while frontier:
            if time.perf_counter() - started >= timeout_s:
                return self._failure(
                    request,
                    PlanStatus.TIMEOUT,
                    "timeout",
                    started,
                    expanded_nodes=expanded_nodes,
                    frontier_size=len(frontier),
                )
            _, _, current = heapq.heappop(frontier)
            if current in closed:
                continue
            closed.add(current)
            expanded_nodes += 1
            if expanded_nodes > self.max_expansions:
                return self._failure(
                    request,
                    PlanStatus.TIMEOUT,
                    "expansion_limit",
                    started,
                    expanded_nodes=expanded_nodes,
                )

            current_position = index_to_world(current)
            if (
                float(np.linalg.norm(current_position - goal)) <= request.task.goal_radius_m
                and validator(current_position, goal)
            ):
                indices = [current]
                cursor = current
                while cursor in parent:
                    cursor = parent[cursor]
                    indices.append(cursor)
                indices.reverse()
                waypoints = np.vstack([index_to_world(index) for index in indices] + [goal])
                finalized = finalize_path(
                    waypoints,
                    request,
                    validator,
                    trajectory_id=f"{request.request_id}:astar",
                )
                if finalized.sweep.is_safe:
                    return PlanResult(
                        status=PlanStatus.OK,
                        trajectory=finalized.trajectory,
                        cost=finalized.path_length_m,
                        certificate=None,
                        planning_time_s=time.perf_counter() - started,
                        plan_id=f"{request.request_id}:astar",
                        diagnostics={
                            "planner": "astar",
                            "expanded_nodes": expanded_nodes,
                            "grid_waypoints": len(indices) + 1,
                            "trajectory_samples": finalized.trajectory.t_s.shape[0],
                            "seed": request.seed,
                        },
                    )
                postprocess_failure = f"postprocess_{finalized.sweep.failure_reason}"

            current_cost = g_score[current]
            for offset in self._neighbour_offsets:
                neighbour = (
                    current[0] + offset[0],
                    current[1] + offset[1],
                    current[2] + offset[2],
                )
                if np.any(np.asarray(neighbour) < minimum_index) or np.any(
                    np.asarray(neighbour) > maximum_index
                ):
                    continue
                if neighbour in closed:
                    continue
                neighbour_position = index_to_world(neighbour)
                if not validator(current_position, neighbour_position):
                    continue
                tentative_cost = current_cost + self.resolution_m * float(np.linalg.norm(offset))
                if tentative_cost >= g_score.get(neighbour, float("inf")):
                    continue
                g_score[neighbour] = tentative_cost
                parent[neighbour] = current
                tie_breaker += 1
                heapq.heappush(
                    frontier,
                    (tentative_cost + heuristic(neighbour), tie_breaker, neighbour),
                )

        reason = postprocess_failure or "no_solution"
        return self._failure(
            request,
            PlanStatus.INFEASIBLE,
            reason,
            started,
            expanded_nodes=expanded_nodes,
            frontier_size=0,
        )


__all__ = ["AStarPlanner"]
