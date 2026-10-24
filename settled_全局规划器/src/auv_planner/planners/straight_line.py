from __future__ import annotations

import time

import numpy as np

from ..contracts import PlanRequest, PlanResult, PlanStatus, Trajectory


class StraightLinePlanner:
    """Geometric smoke-test planner; it does not claim dynamic feasibility."""

    def __init__(self, samples: int = 101) -> None:
        if samples < 2:
            raise ValueError("samples must be at least 2")
        self.samples = samples

    def plan(self, request: PlanRequest) -> PlanResult:
        started = time.perf_counter()
        start = request.start.position_ned_m
        goal = request.task.goal_ned_m
        distance = float(np.linalg.norm(goal - start))
        speed = max(request.vehicle.max_speed_mps, 1e-6)
        duration = max(distance / speed, 1e-6)
        t_s = np.linspace(0.0, duration, self.samples, dtype=np.float64)
        alpha = (t_s / duration).reshape(-1, 1)
        positions = start + alpha * (goal - start)
        quats = np.repeat(request.start.quat_xyzw.reshape(1, 4), self.samples, axis=0)
        velocities = np.zeros((self.samples, 3), dtype=np.float64)
        angular_rates = np.zeros((self.samples, 3), dtype=np.float64)
        trajectory = Trajectory(
            trajectory_id=f"{request.request_id}:straight",
            t_s=t_s,
            position_ned_m=positions,
            quat_xyzw=quats,
            linear_velocity_body_mps=velocities,
            angular_velocity_body_rps=angular_rates,
            clearance_m=np.full(self.samples, np.nan),
            feasibility="unchecked",
        )
        elapsed = time.perf_counter() - started
        return PlanResult(
            status=PlanStatus.OK,
            trajectory=trajectory,
            cost=distance,
            certificate=None,
            planning_time_s=elapsed,
            plan_id=f"{request.request_id}:straight",
            diagnostics={"planner": "straight_line", "samples": self.samples},
        )
