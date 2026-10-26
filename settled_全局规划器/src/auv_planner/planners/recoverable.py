from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Any

import numpy as np

from ..attitude import forward_quaternion, quaternion_angle
from ..contracts import (
    FloatArray,
    Obstacle,
    ObstacleMap,
    PlanRequest,
    PlanResult,
    PlanStatus,
    State,
    TaskSpec,
    Trajectory,
    WorldSnapshot,
)
from ..metrics import compute_path_metrics
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
from .astar import AStarPlanner
from .common import build_trajectory


@dataclass(frozen=True)
class RecoverablePlannerConfig:
    """Lazy recovery-certified A* settings.

    ``checkpoint_spacing_m`` controls where the current path is asked for an
    ``ONLINE_FAST`` certificate. An invalid checkpoint becomes a temporary
    spherical blocked region and the A* search is rerun. The recovery
    checker's horizon, clearance and validity thresholds are not changed.
    """

    resolution_m: float = 0.5
    max_expansions: int = 100_000
    timeout_s: float = 10.0
    inner_timeout_s: float = 6.0
    collision_step_m: float = 0.1
    checkpoint_spacing_m: float = 2.0
    query_budget_s: float = 10.0
    max_recovery_queries: int = 64
    max_replans: int = 2
    max_via_candidates: int = 3
    block_radius_m: float = 1.2
    via_offset_m: float = 1.0
    recovery_mode: bool = True
    profile_name: str = "P1_degraded"
    target_clearance_buffer_m: float = 0.25
    target_recovery_margin_m: float = 0.05
    clearance_weight: float = 0.2
    recovery_margin_weight: float = 0.5
    conservatism_weight: float = 0.02
    turn_weight: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "resolution_m",
            "timeout_s",
            "inner_timeout_s",
            "collision_step_m",
            "checkpoint_spacing_m",
            "query_budget_s",
            "block_radius_m",
            "via_offset_m",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.max_expansions, int) or self.max_expansions < 1:
            raise ValueError("max_expansions must be a positive integer")
        if not isinstance(self.max_recovery_queries, int) or self.max_recovery_queries < 1:
            raise ValueError("max_recovery_queries must be a positive integer")
        if not isinstance(self.max_replans, int) or self.max_replans < 0:
            raise ValueError("max_replans must be a non-negative integer")
        if not isinstance(self.max_via_candidates, int) or self.max_via_candidates < 0:
            raise ValueError("max_via_candidates must be a non-negative integer")
        if not isinstance(self.recovery_mode, bool):
            raise TypeError("recovery_mode must be a bool")
        if not self.profile_name:
            raise ValueError("profile_name must not be empty")
        for name in (
            "target_clearance_buffer_m",
            "target_recovery_margin_m",
            "clearance_weight",
            "recovery_margin_weight",
            "conservatism_weight",
            "turn_weight",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class _Checkpoint:
    distance_m: float
    state: State
    prefilter_reason: str | None
    status: RecoveryQueryStatus | None
    reason: str | None
    margin_m: float | None
    certificate_valid_from_s: float | None
    certificate_valid_until_s: float | None
    elapsed_s: float
    early_stopped: bool
    evaluated_checks: int
    budget_exhausted: bool
    query_source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "distance_m": self.distance_m,
            "t_s": self.state.t_s,
            "position_ned_m": self.state.position_ned_m.tolist(),
            "prefilter_reason": self.prefilter_reason,
            "status": None if self.status is None else self.status.value,
            "reason": self.reason,
            "margin_m": self.margin_m,
            "certificate_valid_from_s": self.certificate_valid_from_s,
            "certificate_valid_until_s": self.certificate_valid_until_s,
            "elapsed_s": self.elapsed_s,
            "early_stopped": self.early_stopped,
            "evaluated_checks": self.evaluated_checks,
            "budget_exhausted": self.budget_exhausted,
            "query_source": self.query_source,
        }


@dataclass(frozen=True)
class _PathCost:
    path_length_m: float
    duration_s: float
    min_clearance_m: float
    min_recovery_margin_m: float | None
    turn_angle_rad: float
    clearance_shortfall_m: float
    recovery_margin_shortfall_m: float
    conservatism_penalty: float
    total: float

    def as_dict(self) -> dict[str, float | None]:
        return {
            "path_length_m": self.path_length_m,
            "duration_s": self.duration_s,
            "min_clearance_m": self.min_clearance_m,
            "min_recovery_margin_m": self.min_recovery_margin_m,
            "turn_angle_rad": self.turn_angle_rad,
            "clearance_shortfall_m": self.clearance_shortfall_m,
            "recovery_margin_shortfall_m": self.recovery_margin_shortfall_m,
            "conservatism_penalty": self.conservatism_penalty,
            "total": self.total,
        }


def _state_from_row(trajectory: Trajectory, index: int) -> State:
    return State(
        t_s=float(trajectory.t_s[index]),
        position_ned_m=trajectory.position_ned_m[index].copy(),
        quat_xyzw=trajectory.quat_xyzw[index].copy(),
        linear_velocity_body_mps=trajectory.linear_velocity_body_mps[index].copy(),
        angular_velocity_body_rps=trajectory.angular_velocity_body_rps[index].copy(),
        frame_id=trajectory.frame_id,
    )


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


def _state_at_distance(
    trajectory: Trajectory,
    distances: FloatArray,
    target_m: float,
) -> State:
    if target_m <= 1e-12:
        return _state_from_row(trajectory, 0)
    if target_m >= float(distances[-1]) - 1e-12:
        return _state_from_row(trajectory, trajectory.t_s.shape[0] - 1)
    index = int(np.searchsorted(distances, target_m, side="right") - 1)
    index = int(np.clip(index, 0, trajectory.t_s.shape[0] - 2))
    segment_length = float(distances[index + 1] - distances[index])
    fraction = (
        0.0
        if segment_length <= 1e-12
        else float((target_m - distances[index]) / segment_length)
    )
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return State(
        t_s=float(
            trajectory.t_s[index]
            + fraction * (trajectory.t_s[index + 1] - trajectory.t_s[index])
        ),
        position_ned_m=(
            (1.0 - fraction) * trajectory.position_ned_m[index]
            + fraction * trajectory.position_ned_m[index + 1]
        ),
        quat_xyzw=_slerp(
            trajectory.quat_xyzw[index],
            trajectory.quat_xyzw[index + 1],
            fraction,
        ),
        linear_velocity_body_mps=(
            (1.0 - fraction) * trajectory.linear_velocity_body_mps[index]
            + fraction * trajectory.linear_velocity_body_mps[index + 1]
        ),
        angular_velocity_body_rps=(
            (1.0 - fraction) * trajectory.angular_velocity_body_rps[index]
            + fraction * trajectory.angular_velocity_body_rps[index + 1]
        ),
        frame_id=trajectory.frame_id,
    )


def _checkpoint_states(
    trajectory: Trajectory,
    spacing_m: float,
) -> tuple[tuple[float, State], ...]:
    positions = trajectory.position_ned_m
    distances = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
    )
    if not np.isfinite(spacing_m) or spacing_m <= 0.0:
        raise ValueError("spacing_m must be finite and positive")
    targets = [0.0]
    value = spacing_m
    while value < distances[-1] - 1e-12:
        targets.append(float(value))
        value += spacing_m
    if abs(float(distances[-1]) - targets[-1]) > 1e-12:
        targets.append(float(distances[-1]))
    checkpoints: list[tuple[float, State]] = []
    seen: set[tuple[float, ...]] = set()
    for target in targets:
        state = _state_at_distance(trajectory, distances, target)
        key = tuple(float(item) for item in state.position_ned_m)
        if key in seen:
            continue
        seen.add(key)
        checkpoints.append((float(target), state))
    return tuple(checkpoints)


def _certificate_coverage(
    checkpoints: tuple[_Checkpoint, ...],
    *,
    start_s: float,
    end_s: float,
) -> tuple[float, float, tuple[tuple[float, float], ...]]:
    intervals = sorted(
        (
            max(start_s, float(item.certificate_valid_from_s)),
            min(end_s, float(item.certificate_valid_until_s)),
        )
        for item in checkpoints
        if item.status is RecoveryQueryStatus.VALID
        and item.certificate_valid_from_s is not None
        and item.certificate_valid_until_s is not None
        and float(item.certificate_valid_until_s) > start_s
        and float(item.certificate_valid_from_s) < end_s
    )
    merged: list[tuple[float, float]] = []
    for low, high in intervals:
        if high <= low + 1e-12:
            continue
        if not merged or low > merged[-1][1] + 1e-12:
            merged.append((low, high))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], high))
    duration = max(0.0, end_s - start_s)
    covered = sum(high - low for low, high in merged)
    cursor = start_s
    uncovered: list[float] = []
    for low, high in merged:
        if low > cursor + 1e-12:
            uncovered.append(low - cursor)
        cursor = max(cursor, high)
    if cursor < end_s - 1e-12:
        uncovered.append(end_s - cursor)
    ratio = 0.0 if duration <= 1e-12 else min(1.0, covered / duration)
    return float(ratio), float(max(uncovered, default=0.0)), tuple(merged)


def _trajectory_length(trajectory: Trajectory) -> float:
    return float(
        np.sum(np.linalg.norm(np.diff(trajectory.position_ned_m, axis=0), axis=1))
    )


def _dedupe_positions(positions: FloatArray) -> FloatArray:
    cleaned: list[FloatArray] = []
    for point in positions:
        if not cleaned or float(np.linalg.norm(point - cleaned[-1])) > 1e-12:
            cleaned.append(np.asarray(point, dtype=np.float64).copy())
    if len(cleaned) < 2:
        cleaned.append(cleaned[0].copy())
    return np.vstack(cleaned)


def _with_blockers(
    world: WorldSnapshot,
    blockers: tuple[tuple[FloatArray, float, str], ...],
) -> WorldSnapshot:
    if not blockers:
        return world
    obstacles = tuple(world.obstacle_map.obstacles) + tuple(
        Obstacle(
            obstacle_id=name,
            kind="sphere",
            center_ned_m=position,
            radius_m=radius_m,
        )
        for position, radius_m, name in blockers
    )
    return WorldSnapshot(
        t_s=world.t_s,
        obstacle_map=ObstacleMap(
            bounds_ned_m=world.obstacle_map.bounds_ned_m.copy(),
            obstacles=obstacles,
        ),
        current_field=world.current_field,
    )


class RecoverablePlanner:
    """Lazy recovery-certified A* with explicit replanning constraints.

    The returned plan is only a search result; the experiment layer must run
    a separate ``FULL`` audit on its trajectory.
    """

    def __init__(
        self,
        config: RecoverablePlannerConfig | None = None,
        *,
        safe_anchor_ned_m: FloatArray | None = None,
        profile: FaultProfile | None = None,
        recovery_config: RecoveryConfig | None = None,
    ) -> None:
        self.config = RecoverablePlannerConfig() if config is None else config
        if not isinstance(self.config, RecoverablePlannerConfig):
            raise TypeError("config must be a RecoverablePlannerConfig")
        self.safe_anchor_ned_m = (
            None
            if safe_anchor_ned_m is None
            else np.asarray(safe_anchor_ned_m, dtype=np.float64).copy()
        )
        if self.safe_anchor_ned_m is not None:
            if self.safe_anchor_ned_m.shape != (3,):
                raise ValueError("safe_anchor_ned_m must have shape (3,)")
            if not np.all(np.isfinite(self.safe_anchor_ned_m)):
                raise ValueError("safe_anchor_ned_m must contain finite values")
        self.profile = (
            fault_profile_by_name(self.config.profile_name)
            if profile is None
            else profile
        )
        if not isinstance(self.profile, FaultProfile):
            raise TypeError("profile must be a FaultProfile")
        self.recovery_config = (
            RecoveryConfig() if recovery_config is None else recovery_config
        )
        if not isinstance(self.recovery_config, RecoveryConfig):
            raise TypeError("recovery_config must be a RecoveryConfig")
        self._inner = AStarPlanner(
            resolution_m=self.config.resolution_m,
            max_expansions=self.config.max_expansions,
            timeout_s=self.config.inner_timeout_s,
            collision_step_m=self.config.collision_step_m,
        )
        self._off_inner = AStarPlanner(
            resolution_m=self.config.resolution_m,
            max_expansions=self.config.max_expansions,
            timeout_s=self.config.timeout_s,
            collision_step_m=self.config.collision_step_m,
        )

    def _diagnostics(self, **values: Any) -> dict[str, Any]:
        return {
            "planner": "recoverable",
            "recovery_mode": "on" if self.config.recovery_mode else "off",
            "profile_name": self.profile.name,
            "profile_digest": self.profile.digest,
            "resolution_m": self.config.resolution_m,
            "checkpoint_spacing_m": self.config.checkpoint_spacing_m,
            **values,
        }

    def _failure(
        self,
        request: PlanRequest,
        status: PlanStatus,
        reason: str,
        started: float,
        **diagnostics: Any,
    ) -> PlanResult:
        return PlanResult(
            status=status,
            trajectory=None,
            cost=0.0,
            certificate=None,
            planning_time_s=time.perf_counter() - started,
            plan_id=f"{request.request_id}:recoverable",
            reason=reason,
            diagnostics=self._diagnostics(**diagnostics),
        )

    def _query(
        self,
        session: RecoveryQuerySession,
        *,
        state: State,
        request: PlanRequest,
    ) -> RecoveryQueryResult:
        return session.query(
            state,
            request.world,
            request.vehicle,
            profile=self.profile,
            config=self.recovery_config,
            safe_anchor_ned_m=self.safe_anchor_ned_m,
        )

    def _checkpoint(
        self,
        *,
        distance_m: float,
        state: State,
        session: RecoveryQuerySession,
        request: PlanRequest,
    ) -> _Checkpoint:
        prefilter = prefilter_state(
            state,
            request.world,
            request.vehicle,
            self.recovery_config,
        )
        if prefilter.rejected:
            return _Checkpoint(
                distance_m=distance_m,
                state=state,
                prefilter_reason=prefilter.reason,
                status=RecoveryQueryStatus.INVALID,
                reason=f"prefilter:{prefilter.reason}",
                margin_m=None,
                certificate_valid_from_s=None,
                certificate_valid_until_s=None,
                elapsed_s=0.0,
                early_stopped=False,
                evaluated_checks=0,
                budget_exhausted=False,
                query_source="prefilter",
            )
        result = self._query(session, state=state, request=request)
        certificate = None if result.assessment is None else result.assessment.certificate
        return _Checkpoint(
            distance_m=distance_m,
            state=state,
            prefilter_reason=None,
            status=result.status,
            reason=result.reason,
            margin_m=None if certificate is None else certificate.margin_m,
            certificate_valid_from_s=(
                None if certificate is None else certificate.valid_from_s
            ),
            certificate_valid_until_s=(
                None if certificate is None else certificate.valid_until_s
            ),
            elapsed_s=float(result.elapsed_s),
            early_stopped=bool(result.early_stopped),
            evaluated_checks=int(result.evaluated_checks),
            budget_exhausted=bool(result.budget_exhausted),
            query_source="cache" if result.cache_hit else "query",
        )

    def _checkpoints(
        self,
        *,
        trajectory: Trajectory,
        session: RecoveryQuerySession,
        request: PlanRequest,
    ) -> tuple[_Checkpoint, ...]:
        checkpoints: list[_Checkpoint] = []
        covered_until_s: float | None = None
        for distance_m, state in _checkpoint_states(
            trajectory, self.config.checkpoint_spacing_m
        ):
            prefilter = prefilter_state(
                state,
                request.world,
                request.vehicle,
                self.recovery_config,
            )
            if not prefilter.rejected and covered_until_s is not None:
                # Do not reuse a certificate at its own expiry instant.  A
                # checkpoint exactly on the boundary would otherwise be
                # counted as covered with a zero-length validity interval,
                # which understates coverage and biases detour selection.
                if state.t_s < covered_until_s - 1e-9:
                    checkpoints.append(
                        _Checkpoint(
                            distance_m=distance_m,
                            state=state,
                            prefilter_reason=None,
                            status=RecoveryQueryStatus.VALID,
                            reason=None,
                            margin_m=None,
                            certificate_valid_from_s=state.t_s,
                            certificate_valid_until_s=covered_until_s,
                            elapsed_s=0.0,
                            early_stopped=False,
                            evaluated_checks=0,
                            budget_exhausted=False,
                            query_source="covered_by_prior_certificate",
                        )
                    )
                    continue
            checkpoint = self._checkpoint(
                distance_m=distance_m,
                state=state,
                session=session,
                request=request,
            )
            checkpoints.append(checkpoint)
            if checkpoint.status is RecoveryQueryStatus.VALID:
                valid_until = checkpoint.certificate_valid_until_s
                if valid_until is not None:
                    covered_until_s = (
                        valid_until
                        if covered_until_s is None
                        else max(covered_until_s, valid_until)
                    )
            elif checkpoint.status is RecoveryQueryStatus.UNKNOWN:
                break
        return tuple(checkpoints)

    def _detour_trajectory(
        self,
        *,
        request: PlanRequest,
        invalid: _Checkpoint,
        later_valid: _Checkpoint,
        started: float,
        timeout_s: float,
    ) -> Trajectory | None:
        if self.config.max_via_candidates <= 0:
            return None
        direction = later_valid.state.position_ned_m - invalid.state.position_ned_m
        horizontal = np.array([-direction[1], direction[0], 0.0], dtype=np.float64)
        norm = float(np.linalg.norm(horizontal))
        if norm <= 1e-12:
            horizontal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            horizontal /= norm
        directions = (
            -horizontal,
            horizontal,
            np.array([0.0, 0.0, -1.0]),
            np.array([0.0, 0.0, 1.0]),
        )
        bounds = request.world.obstacle_map.bounds_ned_m
        for direction_candidate in directions[: self.config.max_via_candidates]:
            via = (
                invalid.state.position_ned_m
                + self.config.via_offset_m * direction_candidate
            )
            if np.any(via < bounds[:, 0]) or np.any(via > bounds[:, 1]):
                continue
            remaining = timeout_s - (time.perf_counter() - started)
            if remaining <= 0.6:
                break
            segment_budget = min(2.0, max(0.2, remaining * 0.45))
            first_constraints = dict(request.task.constraints)
            first_constraints["planning_timeout_s"] = segment_budget
            first_request = replace(
                request,
                request_id=f"{request.request_id}:via{len(via)}",
                task=TaskSpec(
                    goal_ned_m=via.copy(),
                    goal_radius_m=min(request.task.goal_radius_m, 0.25),
                    constraints=first_constraints,
                ),
            )
            first_plan = self._inner.plan(first_request)
            if first_plan.trajectory is None:
                continue
            remaining = timeout_s - (time.perf_counter() - started)
            if remaining <= 0.3:
                break
            segment_budget = min(2.0, max(0.2, remaining * 0.6))
            second_constraints = dict(request.task.constraints)
            second_constraints["planning_timeout_s"] = segment_budget
            via_state = State(
                t_s=0.0,
                position_ned_m=via.copy(),
                quat_xyzw=forward_quaternion(request.task.goal_ned_m - via),
                linear_velocity_body_mps=np.array(
                    [request.vehicle.min_speed_mps, 0.0, 0.0], dtype=np.float64
                ),
                angular_velocity_body_rps=np.zeros(3, dtype=np.float64),
                frame_id=request.start.frame_id,
            )
            second_request = replace(
                request,
                request_id=f"{request.request_id}:via{len(via)}:tail",
                start=via_state,
                task=replace(request.task, constraints=second_constraints),
            )
            second_plan = self._inner.plan(second_request)
            if second_plan.trajectory is None:
                continue
            positions = np.vstack(
                (
                    first_plan.trajectory.position_ned_m,
                    second_plan.trajectory.position_ned_m,
                )
            )
            trajectory = build_trajectory(
                _dedupe_positions(positions),
                replace(
                    request,
                    request_id=f"{request.request_id}:via{len(via)}:combined",
                ),
                f"{request.request_id}:recoverable-via",
            )
            if compute_path_metrics(
                trajectory,
                request.world.obstacle_map,
                request.vehicle,
            ).is_safe:
                return trajectory
        return None

    def _path_cost(
        self,
        *,
        trajectory: Trajectory,
        request: PlanRequest,
        checkpoints: tuple[_Checkpoint, ...],
        blocker_count: int,
    ) -> _PathCost:
        metrics = compute_path_metrics(
            trajectory,
            request.world.obstacle_map,
            request.vehicle,
        )
        turns = [
            quaternion_angle(trajectory.quat_xyzw[index], trajectory.quat_xyzw[index + 1])
            for index in range(trajectory.quat_xyzw.shape[0] - 1)
        ]
        turn_angle = float(np.sum(turns)) if turns else 0.0
        required_clearance = (
            self.recovery_config.required_inflated_clearance_m(request.vehicle)
            + self.config.target_clearance_buffer_m
        )
        clearance_shortfall = max(0.0, required_clearance - metrics.min_clearance_m)
        margins = [item.margin_m for item in checkpoints if item.margin_m is not None]
        minimum_margin = None if not margins else float(min(margins))
        margin_shortfall = (
            0.0
            if minimum_margin is None
            else max(0.0, self.config.target_recovery_margin_m - minimum_margin)
        )
        conservatism_penalty = self.config.conservatism_weight * float(blocker_count)
        total = (
            metrics.path_length_m
            + self.config.turn_weight * turn_angle
            + self.config.clearance_weight * clearance_shortfall
            + self.config.recovery_margin_weight * margin_shortfall
            + conservatism_penalty
        )
        return _PathCost(
            path_length_m=float(metrics.path_length_m),
            duration_s=float(metrics.duration_s),
            min_clearance_m=float(metrics.min_clearance_m),
            min_recovery_margin_m=minimum_margin,
            turn_angle_rad=turn_angle,
            clearance_shortfall_m=float(clearance_shortfall),
            recovery_margin_shortfall_m=float(margin_shortfall),
            conservatism_penalty=float(conservatism_penalty),
            total=float(total),
        )

    def _success(
        self,
        *,
        request: PlanRequest,
        started: float,
        plan: PlanResult,
        checkpoints: tuple[_Checkpoint, ...],
        blockers: tuple[tuple[FloatArray, float, str], ...],
        replans: int,
        search_time_s: float,
        query_session: RecoveryQuerySession,
        attempts: tuple[dict[str, Any], ...],
        selected_attempt: int,
        recovery_complete: bool,
        recovery_coverage_ratio: float,
        recovery_max_uncovered_s: float,
    ) -> PlanResult:
        assert plan.trajectory is not None
        cost = self._path_cost(
            trajectory=plan.trajectory,
            request=request,
            checkpoints=checkpoints,
            blocker_count=len(blockers),
        )
        diagnostics = dict(plan.diagnostics)
        diagnostics.update(
            self._diagnostics(
                checkpoint_count=len(checkpoints),
                checkpoint_queries=len(checkpoints),
                replans=replans,
                search_time_s=search_time_s,
                online_query_time_s=query_session.stats.total_s,
                recovery_queries=query_session.stats.queries,
                recovery_min_margin_m=cost.min_recovery_margin_m,
                recovery_unknown=any(
                    item.status is RecoveryQueryStatus.UNKNOWN for item in checkpoints
                ),
                recovery_complete=recovery_complete,
                recovery_coverage_ratio=recovery_coverage_ratio,
                recovery_max_uncovered_s=recovery_max_uncovered_s,
                selected_attempt=selected_attempt,
                recovery_budget_exhausted=query_session.stats.budget_exhausted > 0,
                recovery_query_stats=query_session.stats.as_dict(),
                checkpoints=[item.as_dict() for item in checkpoints],
                blocked_regions=len(blockers),
                full_audit_required=True,
                cost_breakdown=cost.as_dict(),
                attempts=list(attempts),
            )
        )
        return PlanResult(
            status=PlanStatus.OK,
            trajectory=plan.trajectory,
            cost=cost.total,
            certificate=None,
            planning_time_s=time.perf_counter() - started,
            plan_id=f"{request.request_id}:recoverable",
            diagnostics=diagnostics,
        )

    def plan(self, request: PlanRequest) -> PlanResult:
        if not isinstance(request, PlanRequest):
            raise TypeError("request must be a PlanRequest")
        started = time.perf_counter()
        timeout_s = min(
            self.config.timeout_s,
            float(request.task.constraints.get("planning_timeout_s", self.config.timeout_s)),
        )
        if not np.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("planning_timeout_s must be finite and positive")

        if not self.config.recovery_mode:
            search_started = time.perf_counter()
            plan = self._off_inner.plan(request)
            search_time_s = time.perf_counter() - search_started
            diagnostics = dict(plan.diagnostics)
            cost = None
            if plan.trajectory is not None:
                cost = self._path_cost(
                    trajectory=plan.trajectory,
                    request=request,
                    checkpoints=(),
                    blocker_count=0,
                )
            diagnostics.update(
                self._diagnostics(
                    recovery_queries=0,
                    checkpoint_count=0,
                    replans=0,
                    search_time_s=search_time_s,
                    online_query_time_s=0.0,
                    blocked_regions=0,
                    full_audit_required=False,
                    cost_breakdown=None if cost is None else cost.as_dict(),
                )
            )
            return PlanResult(
                status=plan.status,
                trajectory=plan.trajectory,
                cost=plan.cost if cost is None else cost.total,
                certificate=plan.certificate,
                planning_time_s=time.perf_counter() - started,
                plan_id=f"{request.request_id}:recoverable",
                parent_plan_id=plan.parent_plan_id,
                valid_until_s=plan.valid_until_s,
                reason=plan.reason,
                diagnostics=diagnostics,
            )

        query_session = RecoveryQuerySession(
            policy=RecoveryQueryPolicy(
                budget_s=min(self.config.query_budget_s, timeout_s),
                use_cache=True,
                use_prefilter=False,
                mode=RecoveryQueryMode.ONLINE_FAST,
                parallel_workers=1,
            )
        )
        start_state = State(
            t_s=0.0,
            position_ned_m=request.start.position_ned_m.copy(),
            quat_xyzw=request.start.quat_xyzw.copy(),
            linear_velocity_body_mps=request.start.linear_velocity_body_mps.copy(),
            angular_velocity_body_rps=request.start.angular_velocity_body_rps.copy(),
            frame_id=request.start.frame_id,
        )
        start_checkpoint = self._checkpoint(
            distance_m=0.0,
            state=start_state,
            session=query_session,
            request=request,
        )
        if start_checkpoint.status is RecoveryQueryStatus.INVALID:
            return self._failure(
                request,
                PlanStatus.INFEASIBLE,
                "infeasible_recovery_at_start",
                started,
                search_time_s=0.0,
                start_checkpoint=start_checkpoint.as_dict(),
                recovery_query_stats=query_session.stats.as_dict(),
            )
        if start_checkpoint.status is RecoveryQueryStatus.UNKNOWN:
            return self._failure(
                request,
                PlanStatus.TIMEOUT,
                "recovery_budget_exhausted_at_start",
                started,
                search_time_s=0.0,
                start_checkpoint=start_checkpoint.as_dict(),
                recovery_query_stats=query_session.stats.as_dict(),
            )

        blockers: list[tuple[FloatArray, float, str]] = []
        attempts: list[dict[str, Any]] = []
        search_time_s = 0.0
        best_plan: PlanResult | None = None
        best_checkpoints: tuple[_Checkpoint, ...] = ()
        best_blocker_count = 0
        best_replans = 0
        best_coverage = -1.0
        best_gap = float("inf")
        best_attempt_index = -1
        via_attempted = False
        stopped_reason = "recovery_replan_limit"
        failure_status = PlanStatus.INFEASIBLE
        failure_reason = "no_recovery_safe_solution"

        for iteration in range(self.config.max_replans + 1):
            elapsed = time.perf_counter() - started
            remaining = timeout_s - elapsed
            if remaining <= 0.0:
                stopped_reason = "timeout"
                failure_status = PlanStatus.TIMEOUT
                failure_reason = "timeout"
                break
            if query_session.stats.queries >= self.config.max_recovery_queries:
                stopped_reason = "recovery_query_limit"
                failure_status = PlanStatus.TIMEOUT
                failure_reason = "recovery_query_limit"
                break
            available_for_search = remaining - 0.05
            if available_for_search <= 0.05:
                stopped_reason = "timeout"
                failure_status = PlanStatus.TIMEOUT
                failure_reason = "timeout"
                break
            inner_budget = min(self.config.inner_timeout_s, available_for_search)
            constraints = dict(request.task.constraints)
            constraints["planning_timeout_s"] = inner_budget
            inner_request = replace(
                request,
                request_id=f"{request.request_id}:iter{iteration}",
                task=replace(request.task, constraints=constraints),
                world=_with_blockers(request.world, tuple(blockers)),
            )
            search_started = time.perf_counter()
            plan = self._inner.plan(inner_request)
            search_time_s += time.perf_counter() - search_started
            if plan.status is not PlanStatus.OK or plan.trajectory is None:
                stopped_reason = plan.reason or "inner_planner_failed"
                failure_status = plan.status
                failure_reason = stopped_reason
                break

            checkpoints = self._checkpoints(
                trajectory=plan.trajectory,
                session=query_session,
                request=request,
            )
            start_s = float(plan.trajectory.t_s[0])
            end_s = float(plan.trajectory.t_s[-1])
            coverage, max_uncovered, coverage_intervals = _certificate_coverage(
                checkpoints,
                start_s=start_s,
                end_s=end_s,
            )
            invalid = next(
                (
                    item
                    for item in checkpoints
                    if item.status is not RecoveryQueryStatus.VALID
                ),
                None,
            )
            attempt: dict[str, Any] = {
                "iteration": iteration,
                "path_length_m": float(plan.cost),
                "checkpoint_count": len(checkpoints),
                "queried_checkpoints": sum(
                    item.query_source in {"query", "cache"} for item in checkpoints
                ),
                "covered_checkpoints": sum(
                    item.query_source == "covered_by_prior_certificate"
                    for item in checkpoints
                ),
                "prefilter_rejections": sum(
                    item.query_source == "prefilter" for item in checkpoints
                ),
                "checkpoint_coverage_ratio": coverage,
                "checkpoint_max_uncovered_s": max_uncovered,
                "coverage_intervals": list(coverage_intervals),
                "checkpoints": [item.as_dict() for item in checkpoints],
                "invalid_checkpoint": None if invalid is None else invalid.as_dict(),
                "blocker_added": None,
                "selected": False,
            }
            attempts.append(attempt)

            is_better = coverage > best_coverage + 1e-12 or (
                abs(coverage - best_coverage) <= 1e-12
                and max_uncovered < best_gap - 1e-12
            )
            if is_better:
                best_plan = plan
                best_checkpoints = checkpoints
                best_blocker_count = len(blockers)
                best_replans = iteration
                best_coverage = coverage
                best_gap = max_uncovered
                best_attempt_index = len(attempts) - 1

            if invalid is None:
                attempt["selected"] = True
                return self._success(
                    request=request,
                    started=started,
                    plan=plan,
                    checkpoints=checkpoints,
                    blockers=tuple(blockers),
                    replans=iteration,
                    search_time_s=search_time_s,
                    query_session=query_session,
                    attempts=tuple(attempts),
                    selected_attempt=len(attempts) - 1,
                    recovery_complete=True,
                    recovery_coverage_ratio=coverage,
                    recovery_max_uncovered_s=max_uncovered,
                )
            if invalid.status is RecoveryQueryStatus.UNKNOWN:
                stopped_reason = "recovery_budget_exhausted"
                failure_status = PlanStatus.TIMEOUT
                failure_reason = stopped_reason
                break

            if not via_attempted:
                via_attempted = True
                invalid_index = checkpoints.index(invalid)
                later_valid = next(
                    (
                        item
                        for item in checkpoints[invalid_index + 1 :]
                        if item.status is RecoveryQueryStatus.VALID
                    ),
                    None,
                )
                if later_valid is not None:
                    detour = self._detour_trajectory(
                        request=request,
                        invalid=invalid,
                        later_valid=later_valid,
                        started=started,
                        timeout_s=timeout_s,
                    )
                    if detour is not None:
                        detour_checkpoints = self._checkpoints(
                            trajectory=detour,
                            session=query_session,
                            request=request,
                        )
                        detour_start = float(detour.t_s[0])
                        detour_end = float(detour.t_s[-1])
                        (
                            detour_coverage,
                            detour_gap,
                            detour_intervals,
                        ) = _certificate_coverage(
                            detour_checkpoints,
                            start_s=detour_start,
                            end_s=detour_end,
                        )
                        detour_invalid = next(
                            (
                                item
                                for item in detour_checkpoints
                                if item.status is not RecoveryQueryStatus.VALID
                            ),
                            None,
                        )
                        detour_attempt: dict[str, Any] = {
                            "iteration": iteration,
                            "source": "recovery_detour",
                            "path_length_m": _trajectory_length(detour),
                            "checkpoint_count": len(detour_checkpoints),
                            "queried_checkpoints": sum(
                                item.query_source in {"query", "cache"}
                                for item in detour_checkpoints
                            ),
                            "covered_checkpoints": sum(
                                item.query_source == "covered_by_prior_certificate"
                                for item in detour_checkpoints
                            ),
                            "prefilter_rejections": sum(
                                item.query_source == "prefilter"
                                for item in detour_checkpoints
                            ),
                            "checkpoint_coverage_ratio": detour_coverage,
                            "checkpoint_max_uncovered_s": detour_gap,
                            "coverage_intervals": list(detour_intervals),
                            "checkpoints": [
                                item.as_dict() for item in detour_checkpoints
                            ],
                            "invalid_checkpoint": (
                                None
                                if detour_invalid is None
                                else detour_invalid.as_dict()
                            ),
                            "blocker_added": None,
                            "selected": False,
                        }
                        attempts.append(detour_attempt)
                        detour_is_better = detour_coverage > best_coverage + 1e-12 or (
                            abs(detour_coverage - best_coverage) <= 1e-12
                            and detour_gap < best_gap - 1e-12
                        )
                        if detour_is_better:
                            detour_plan = PlanResult(
                                status=PlanStatus.OK,
                                trajectory=detour,
                                cost=_trajectory_length(detour),
                                certificate=None,
                                planning_time_s=time.perf_counter() - started,
                                plan_id=f"{request.request_id}:recoverable-via",
                                diagnostics={"planner": "recoverable-via"},
                            )
                            best_plan = detour_plan
                            best_checkpoints = detour_checkpoints
                            best_blocker_count = len(blockers)
                            best_replans = iteration
                            best_coverage = detour_coverage
                            best_gap = detour_gap
                            best_attempt_index = len(attempts) - 1
                        if detour_invalid is None:
                            detour_attempt["selected"] = True
                            assert best_plan is not None
                            return self._success(
                                request=request,
                                started=started,
                                plan=best_plan,
                                checkpoints=detour_checkpoints,
                                blockers=tuple(blockers),
                                replans=iteration,
                                search_time_s=search_time_s,
                                query_session=query_session,
                                attempts=tuple(attempts),
                                selected_attempt=len(attempts) - 1,
                                recovery_complete=True,
                                recovery_coverage_ratio=detour_coverage,
                                recovery_max_uncovered_s=detour_gap,
                            )
                        if detour_invalid.status is RecoveryQueryStatus.UNKNOWN:
                            stopped_reason = "recovery_budget_exhausted"
                            failure_status = PlanStatus.TIMEOUT
                            failure_reason = stopped_reason
                            break
            if iteration >= self.config.max_replans:
                stopped_reason = "recovery_replan_limit"
                failure_status = PlanStatus.INFEASIBLE
                failure_reason = stopped_reason
                break

            blockers.append(
                (
                    invalid.state.position_ned_m.copy(),
                    self.config.block_radius_m,
                    f"recovery-block-{iteration}-{len(blockers)}",
                )
            )
            attempt["blocker_added"] = {
                "position_ned_m": invalid.state.position_ned_m.tolist(),
                "radius_m": self.config.block_radius_m,
                "distance_m": invalid.distance_m,
            }

        if best_plan is not None:
            assert best_attempt_index >= 0
            attempts[best_attempt_index]["selected"] = True
            return self._success(
                request=request,
                started=started,
                plan=best_plan,
                checkpoints=best_checkpoints,
                blockers=tuple(blockers[:best_blocker_count]),
                replans=best_replans,
                search_time_s=search_time_s,
                query_session=query_session,
                attempts=tuple(attempts),
                selected_attempt=best_attempt_index,
                recovery_complete=False,
                recovery_coverage_ratio=best_coverage,
                recovery_max_uncovered_s=best_gap,
            )

        return self._failure(
            request,
            failure_status,
            failure_reason,
            started,
            replans=len(attempts),
            search_time_s=search_time_s,
            blocked_regions=len(blockers),
            start_checkpoint=start_checkpoint.as_dict(),
            attempts=attempts,
            stopped_reason=stopped_reason,
            recovery_query_stats=query_session.stats.as_dict(),
        )


__all__ = ["RecoverablePlanner", "RecoverablePlannerConfig"]
