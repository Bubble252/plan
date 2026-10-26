from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from auv_planner.contracts import (
    CapabilitySet,
    Obstacle,
    ObstacleMap,
    PlanRequest,
    PlanStatus,
    ScenarioConfig,
    State,
    TaskSpec,
    UniformCurrent,
    WorldSnapshot,
)
from auv_planner.metrics import compute_path_metrics
from auv_planner.planners.astar import AStarPlanner
from auv_planner.planners.rrt_star import RRTStarPlanner
from auv_planner.scenarios import (
    cylinder_gate_scene,
    fixed_scenarios,
    remus100_vehicle,
    sphere_detour_scene,
)


def make_request(
    obstacle_map: ObstacleMap,
    *,
    start_ned_m: tuple[float, float, float] = (-3.0, 0.0, 0.0),
    goal_ned_m: tuple[float, float, float] = (3.0, 0.0, 0.0),
    goal_radius_m: float = 0.35,
    seed: int = 0,
) -> PlanRequest:
    world = WorldSnapshot(
        t_s=0.0,
        obstacle_map=obstacle_map,
        current_field=UniformCurrent(velocity_ned_mps=np.zeros(3)),
    )
    start = State(
        t_s=0.0,
        position_ned_m=np.asarray(start_ned_m, dtype=np.float64),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        linear_velocity_body_mps=np.array([1.0, 0.0, 0.0]),
        angular_velocity_body_rps=np.zeros(3),
    )
    return PlanRequest(
        request_id="test",
        start=start,
        task=TaskSpec(
            goal_ned_m=np.asarray(goal_ned_m, dtype=np.float64),
            goal_radius_m=goal_radius_m,
        ),
        world=world,
        vehicle=remus100_vehicle(),
        capability=CapabilitySet(available_actuators=("propeller", "yaw_fin", "pitch_fin")),
        seed=seed,
    )


def scenario_request(scenario: ScenarioConfig, *, seed: int = 0) -> PlanRequest:
    return PlanRequest(
        request_id=scenario.name,
        start=scenario.start,
        task=TaskSpec(
            goal_ned_m=scenario.goal_ned_m,
            goal_radius_m=scenario.goal_radius_m,
            constraints=scenario.constraints,
        ),
        world=scenario.world,
        vehicle=scenario.vehicle,
        capability=CapabilitySet(available_actuators=("propeller", "yaw_fin", "pitch_fin")),
        seed=seed,
    )


def make_bounds(half_extent_m: float = 2.5) -> np.ndarray:
    return np.array(
        [
            [-half_extent_m, half_extent_m],
            [-half_extent_m, half_extent_m],
            [-half_extent_m, half_extent_m],
        ],
        dtype=np.float64,
    )


def test_astar_empty_scene_is_deterministic_and_safe() -> None:
    scenario = fixed_scenarios()[0]
    planner = AStarPlanner(resolution_m=0.5, timeout_s=5.0, collision_step_m=0.1)

    first = planner.plan(scenario_request(scenario, seed=3))
    second = planner.plan(scenario_request(scenario, seed=3))

    assert first.status is PlanStatus.OK
    assert first.trajectory is not None
    assert second.trajectory is not None
    assert np.allclose(first.trajectory.position_ned_m, second.trajectory.position_ned_m)
    metrics = compute_path_metrics(
        first.trajectory,
        scenario.world.obstacle_map,
        scenario.vehicle,
    )
    assert metrics.is_safe


def test_astar_solves_all_fixed_scenes() -> None:
    planner = AStarPlanner(resolution_m=0.5, timeout_s=5.0, collision_step_m=0.1)
    for scenario in fixed_scenarios():
        result = planner.plan(scenario_request(scenario))
        assert result.status is PlanStatus.OK, (scenario.name, result.reason)
        assert result.trajectory is not None
        metrics = compute_path_metrics(
            result.trajectory,
            scenario.world.obstacle_map,
            scenario.vehicle,
        )
        assert metrics.is_safe, scenario.name
        assert np.linalg.norm(result.trajectory.position_ned_m[-1] - scenario.goal_ned_m) <= scenario.goal_radius_m


def test_astar_detours_around_sphere() -> None:
    scenario = sphere_detour_scene()
    result = AStarPlanner(resolution_m=0.5, timeout_s=5.0, collision_step_m=0.1).plan(
        scenario_request(scenario)
    )

    assert result.status is PlanStatus.OK
    assert result.cost > 6.05


def test_astar_reports_no_solution_behind_full_wall() -> None:
    wall = Obstacle(
        "wall",
        "box",
        np.zeros(3),
        half_extents_m=np.array([0.3, 3.0, 3.0]),
    )
    obstacle_map = ObstacleMap(
        bounds_ned_m=make_bounds(),
        obstacles=(wall,),
    )
    request = make_request(
        obstacle_map,
        start_ned_m=(-1.5, 0.0, 0.0),
        goal_ned_m=(1.5, 0.0, 0.0),
        goal_radius_m=0.2,
    )

    result = AStarPlanner(resolution_m=0.5, timeout_s=5.0, collision_step_m=0.1).plan(request)

    assert result.status is PlanStatus.INFEASIBLE
    assert result.reason == "no_solution"
    assert result.trajectory is None


@pytest.mark.parametrize(
    "planner",
    [
        AStarPlanner(resolution_m=0.5, timeout_s=2.0),
        RRTStarPlanner(step_size_m=0.5, max_iterations=10, timeout_s=2.0),
    ],
)
def test_start_inside_goal_region_returns_stationary_solution(planner) -> None:
    obstacle_map = ObstacleMap(bounds_ned_m=make_bounds(), obstacles=())
    request = make_request(
        obstacle_map,
        start_ned_m=(0.0, 0.0, 0.0),
        goal_ned_m=(0.1, 0.0, 0.0),
        goal_radius_m=0.5,
    )

    result = planner.plan(request)

    assert result.status is PlanStatus.OK
    assert result.cost == 0.0
    assert result.trajectory is not None
    assert np.allclose(result.trajectory.position_ned_m, np.zeros((2, 3)))


def test_rrt_star_direct_path_is_deterministic_and_safe() -> None:
    scenario = fixed_scenarios()[0]
    planner = RRTStarPlanner(step_size_m=0.5, max_iterations=100, timeout_s=2.0)

    first = planner.plan(scenario_request(scenario, seed=5))
    second = planner.plan(scenario_request(scenario, seed=5))

    assert first.status is PlanStatus.OK
    assert first.trajectory is not None
    assert second.trajectory is not None
    assert np.allclose(first.trajectory.position_ned_m, second.trajectory.position_ned_m)


@pytest.mark.parametrize("scenario", [sphere_detour_scene(), cylinder_gate_scene()])
def test_rrt_star_solves_obstacle_scenes(scenario: ScenarioConfig) -> None:
    planner = RRTStarPlanner(
        step_size_m=0.5,
        max_iterations=400,
        timeout_s=2.0,
        collision_step_m=0.1,
    )
    first = planner.plan(scenario_request(scenario, seed=7))
    second = planner.plan(scenario_request(scenario, seed=7))

    assert first.status is PlanStatus.OK, (scenario.name, first.reason)
    assert second.status is PlanStatus.OK
    assert first.trajectory is not None
    assert second.trajectory is not None
    assert np.allclose(first.trajectory.position_ned_m, second.trajectory.position_ned_m)
    metrics = compute_path_metrics(
        first.trajectory,
        scenario.world.obstacle_map,
        scenario.vehicle,
    )
    assert metrics.is_safe


def test_rrt_star_timeout_has_explicit_reason() -> None:
    scenario = sphere_detour_scene()
    planner = RRTStarPlanner(step_size_m=0.5, max_iterations=10, timeout_s=1e-9)

    result = planner.plan(scenario_request(scenario))

    assert result.status is PlanStatus.TIMEOUT
    assert result.reason == "timeout"


def test_rrt_star_postprocess_failure_has_explicit_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    request = make_request(ObstacleMap(bounds_ned_m=make_bounds(5.0), obstacles=()))
    unsafe_finalize = SimpleNamespace(
        trajectory=None,
        path_length_m=0.0,
        sweep=SimpleNamespace(is_safe=False, failure_reason="unit_unsafe"),
    )
    monkeypatch.setattr(
        "auv_planner.planners.rrt_star.finalize_path",
        lambda *args, **kwargs: unsafe_finalize,
    )
    planner = RRTStarPlanner(
        step_size_m=6.0,
        max_iterations=1,
        timeout_s=1.0,
        goal_bias=1.0,
    )

    result = planner.plan(request)

    assert result.status is PlanStatus.INFEASIBLE
    assert result.reason == "postprocess_unit_unsafe"


def test_rrt_star_without_goal_candidate_reports_no_solution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = make_request(ObstacleMap(bounds_ned_m=make_bounds(5.0), obstacles=()))
    unsafe_finalize = SimpleNamespace(
        trajectory=None,
        path_length_m=0.0,
        sweep=SimpleNamespace(is_safe=False, failure_reason="unit_unsafe"),
    )
    monkeypatch.setattr(
        "auv_planner.planners.rrt_star.finalize_path",
        lambda *args, **kwargs: unsafe_finalize,
    )
    planner = RRTStarPlanner(
        step_size_m=0.5,
        max_iterations=1,
        timeout_s=1.0,
        goal_bias=1.0,
    )

    result = planner.plan(request)

    assert result.status is PlanStatus.INFEASIBLE
    assert result.reason == "no_solution"
