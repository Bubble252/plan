from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .contracts import (
    Obstacle,
    ObstacleMap,
    ScenarioConfig,
    State,
    UniformCurrent,
    VehicleSpec,
    WorldSnapshot,
)


def remus100_vehicle() -> VehicleSpec:
    return VehicleSpec(
        name="REMUS100",
        length_m=1.6,
        diameter_m=0.19,
        mass_kg=31.0,
        min_speed_mps=1.0,
        max_speed_mps=4.0,
        max_accel_mps2=1.0,
        max_yaw_rate_rps=0.5,
        max_pitch_rate_rps=0.5,
    )


def _start_state(position_ned_m: Iterable[float]) -> State:
    return State(
        t_s=0.0,
        position_ned_m=np.asarray(tuple(position_ned_m), dtype=np.float64),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        linear_velocity_body_mps=np.array([1.0, 0.0, 0.0]),
        angular_velocity_body_rps=np.zeros(3),
    )


def _scenario(
    name: str,
    bounds_ned_m: np.ndarray,
    obstacles: tuple[Obstacle, ...],
    start_ned_m: Iterable[float],
    goal_ned_m: Iterable[float],
    goal_radius_m: float = 0.35,
) -> ScenarioConfig:
    world = WorldSnapshot(
        t_s=0.0,
        obstacle_map=ObstacleMap(bounds_ned_m=bounds_ned_m, obstacles=obstacles),
        current_field=UniformCurrent(velocity_ned_mps=np.zeros(3)),
    )
    return ScenarioConfig(
        name=name,
        world=world,
        vehicle=remus100_vehicle(),
        start=_start_state(start_ned_m),
        goal_ned_m=np.asarray(tuple(goal_ned_m), dtype=np.float64),
        goal_radius_m=goal_radius_m,
        constraints={"planning_timeout_s": 10.0},
    )


def _bounds(half_extent_m: float = 4.0) -> np.ndarray:
    return np.array(
        [
            [-half_extent_m, half_extent_m],
            [-half_extent_m, half_extent_m],
            [-half_extent_m, half_extent_m],
        ],
        dtype=np.float64,
    )


def empty_scene() -> ScenarioConfig:
    return _scenario(
        "m3_empty",
        _bounds(),
        (),
        (-3.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
    )


def sphere_detour_scene() -> ScenarioConfig:
    obstacle = Obstacle("sphere-center", "sphere", np.zeros(3), radius_m=0.55)
    return _scenario(
        "m3_sphere_detour",
        _bounds(),
        (obstacle,),
        (-3.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
    )


def cylinder_gate_scene() -> ScenarioConfig:
    open_half = 0.45
    thickness = 0.2
    centre = open_half + thickness / 2.0
    half_thickness = thickness / 2.0
    obstacles = (
        Obstacle(
            "gate-left",
            "box",
            np.array([0.0, -centre, 0.0]),
            half_extents_m=np.array([half_thickness, half_thickness, 1.5]),
        ),
        Obstacle(
            "gate-right",
            "box",
            np.array([0.0, centre, 0.0]),
            half_extents_m=np.array([half_thickness, half_thickness, 1.5]),
        ),
        Obstacle(
            "gate-bottom",
            "box",
            np.array([0.0, 0.0, -centre]),
            half_extents_m=np.array([half_thickness, centre, half_thickness]),
        ),
        Obstacle(
            "gate-top",
            "box",
            np.array([0.0, 0.0, centre]),
            half_extents_m=np.array([half_thickness, centre, half_thickness]),
        ),
        Obstacle(
            "pillar",
            "cylinder",
            np.array([1.8, 0.0, 0.0]),
            radius_m=0.25,
            height_m=3.0,
        ),
    )
    return _scenario(
        "m3_cylinder_gate",
        _bounds(),
        obstacles,
        (-3.0, -0.8, 0.0),
        (3.0, 0.5, 0.0),
        goal_radius_m=0.4,
    )


def corridor_scene() -> ScenarioConfig:
    wall_y = 0.75
    obstacles = (
        Obstacle(
            "corridor-left",
            "box",
            np.array([0.0, -(wall_y + 0.1), 0.0]),
            half_extents_m=np.array([3.0, 0.1, 1.2]),
        ),
        Obstacle(
            "corridor-right",
            "box",
            np.array([0.0, wall_y + 0.1, 0.0]),
            half_extents_m=np.array([3.0, 0.1, 1.2]),
        ),
        Obstacle(
            "corridor-block",
            "sphere",
            np.array([0.0, -0.15, 0.0]),
            radius_m=0.3,
        ),
    )
    return _scenario(
        "m3_corridor",
        _bounds(),
        obstacles,
        (-3.0, -0.3, 0.0),
        (3.0, -0.3, 0.0),
    )


def fixed_scenarios() -> tuple[ScenarioConfig, ...]:
    return (
        empty_scene(),
        sphere_detour_scene(),
        cylinder_gate_scene(),
        corridor_scene(),
    )


__all__ = [
    "corridor_scene",
    "cylinder_gate_scene",
    "empty_scene",
    "fixed_scenarios",
    "remus100_vehicle",
    "sphere_detour_scene",
]
