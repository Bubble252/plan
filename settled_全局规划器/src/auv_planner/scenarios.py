from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .contracts import (
    CurrentField,
    Obstacle,
    ObstacleMap,
    ScenarioConfig,
    State,
    UniformCurrent,
    VehicleSpec,
    WorldSnapshot,
)
from .current import DepthShearCurrent


def remus100_vehicle() -> VehicleSpec:
    max_fin_angle_rad = np.deg2rad(15.0)
    return VehicleSpec(
        name="REMUS100",
        length_m=1.6,
        diameter_m=0.19,
        mass_kg=31.9,
        min_speed_mps=1.0,
        max_speed_mps=2.5,
        max_accel_mps2=1.0,
        max_yaw_rate_rps=0.5,
        max_pitch_rate_rps=0.5,
        max_rudder_angle_rad=max_fin_angle_rad,
        max_rudder_rate_rad_s=max_fin_angle_rad / 0.1,
        max_stern_plane_angle_rad=max_fin_angle_rad,
        max_stern_plane_rate_rad_s=max_fin_angle_rad / 0.1,
        max_propeller_rpm=1525.0,
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


def _m5_scenario(
    name: str,
    *,
    obstacles: tuple[Obstacle, ...],
    start_ned_m: Iterable[float],
    goal_ned_m: Iterable[float],
    current: CurrentField | None = None,
    goal_radius_m: float = 0.5,
) -> ScenarioConfig:
    world = WorldSnapshot(
        t_s=0.0,
        obstacle_map=ObstacleMap(
            bounds_ned_m=np.array([[-8.0, 8.0], [-5.0, 5.0], [-4.0, 4.0]]),
            obstacles=obstacles,
        ),
        current_field=UniformCurrent(velocity_ned_mps=np.zeros(3)) if current is None else current,
    )
    start = _start_state(start_ned_m)
    start = State(
        t_s=start.t_s,
        position_ned_m=start.position_ned_m,
        quat_xyzw=start.quat_xyzw,
        linear_velocity_body_mps=np.array([1.5, 0.0, 0.0]),
        angular_velocity_body_rps=start.angular_velocity_body_rps,
        frame_id=start.frame_id,
    )
    return ScenarioConfig(
        name=name,
        world=world,
        vehicle=remus100_vehicle(),
        start=start,
        goal_ned_m=np.asarray(tuple(goal_ned_m), dtype=np.float64),
        goal_radius_m=goal_radius_m,
        constraints={"planning_timeout_s": 10.0, "controller_sample_time_s": 0.02},
    )


def m5_open_water_scene() -> ScenarioConfig:
    return _m5_scenario(
        "m5_open_water",
        obstacles=(),
        start_ned_m=(-6.0, 0.0, 0.0),
        goal_ned_m=(6.0, 0.5, -1.5),
    )


def m5_sphere_detour_scene() -> ScenarioConfig:
    obstacle = Obstacle("m5-sphere", "sphere", np.array([0.0, 0.0, -0.5]), radius_m=0.9)
    return _m5_scenario(
        "m5_sphere_detour",
        obstacles=(obstacle,),
        start_ned_m=(-6.0, -0.5, 0.5),
        goal_ned_m=(6.0, 0.5, -1.5),
    )


def _m5_gate_obstacles() -> tuple[Obstacle, ...]:
    open_half = 0.5
    thickness_half = 0.1
    centre_z = -1.5
    return (
        Obstacle(
            "m5-gate-left",
            "box",
            np.array([0.0, -(open_half + thickness_half), centre_z]),
            half_extents_m=np.array([thickness_half, thickness_half, 1.25]),
        ),
        Obstacle(
            "m5-gate-right",
            "box",
            np.array([0.0, open_half + thickness_half, centre_z]),
            half_extents_m=np.array([thickness_half, thickness_half, 1.25]),
        ),
        Obstacle(
            "m5-gate-bottom",
            "box",
            np.array([0.0, 0.0, centre_z - open_half - thickness_half]),
            half_extents_m=np.array([thickness_half, open_half, thickness_half]),
        ),
        Obstacle(
            "m5-gate-top",
            "box",
            np.array([0.0, 0.0, centre_z + open_half + thickness_half]),
            half_extents_m=np.array([thickness_half, open_half, thickness_half]),
        ),
        Obstacle(
            "m5-gate-pillar",
            "cylinder",
            np.array([2.8, 0.0, 0.0]),
            radius_m=0.3,
            height_m=3.5,
        ),
    )


def m5_cylinder_gate_scene() -> ScenarioConfig:
    return _m5_scenario(
        "m5_cylinder_gate",
        obstacles=_m5_gate_obstacles(),
        start_ned_m=(-6.0, -1.0, 0.0),
        goal_ned_m=(6.0, 1.0, -1.5),
    )


def m5_s_bend_corridor_scene() -> ScenarioConfig:
    obstacles = (
        Obstacle(
            "m5-sbend-lower-a",
            "box",
            np.array([-3.0, -1.5, -1.0]),
            half_extents_m=np.array([0.12, 1.0, 1.5]),
        ),
        Obstacle(
            "m5-sbend-upper-a",
            "box",
            np.array([-3.0, 1.6, -1.0]),
            half_extents_m=np.array([0.12, 0.9, 1.5]),
        ),
        Obstacle(
            "m5-sbend-lower-b",
            "box",
            np.array([0.0, -1.2, -1.0]),
            half_extents_m=np.array([0.12, 1.3, 1.5]),
        ),
        Obstacle(
            "m5-sbend-upper-b",
            "box",
            np.array([0.0, 1.9, -1.0]),
            half_extents_m=np.array([0.12, 0.6, 1.5]),
        ),
        Obstacle(
            "m5-sbend-lower-c",
            "box",
            np.array([3.0, -0.75, -1.0]),
            half_extents_m=np.array([0.12, 1.75, 1.5]),
        ),
        Obstacle(
            "m5-sbend-upper-c",
            "box",
            np.array([3.0, 2.35, -1.0]),
            half_extents_m=np.array([0.12, 0.15, 1.5]),
        ),
    )
    return _m5_scenario(
        "m5_s_bend_corridor",
        obstacles=obstacles,
        start_ned_m=(-6.0, -0.5, 0.0),
        goal_ned_m=(6.0, 1.8, -1.5),
    )


def m5_single_corner_scene() -> ScenarioConfig:
    obstacle = Obstacle(
        "m5-corner-wall",
        "box",
        np.array([0.0, 1.5, 0.0]),
        half_extents_m=np.array([0.15, 2.5, 2.0]),
    )
    return _m5_scenario(
        "m5_single_corner",
        obstacles=(obstacle,),
        start_ned_m=(-5.0, -1.0, 0.0),
        goal_ned_m=(2.0, 3.5, 1.5),
        goal_radius_m=0.6,
    )


def m5_cross_current_sphere_scene() -> ScenarioConfig:
    obstacle = Obstacle("m5-current-sphere", "sphere", np.array([0.0, 0.0, -0.5]), radius_m=0.9)
    return _m5_scenario(
        "m5_cross_current_sphere",
        obstacles=(obstacle,),
        start_ned_m=(-6.0, -0.5, 0.5),
        goal_ned_m=(6.0, 0.5, -1.5),
        current=UniformCurrent(velocity_ned_mps=np.array([0.0, 0.35, 0.0])),
    )


def m5_depth_shear_gate_scene() -> ScenarioConfig:
    current = DepthShearCurrent(
        reference_velocity_ned_mps=np.array([0.25, 0.0, 0.0]),
        shear_ned_per_m=np.array([0.02, 0.01, 0.0]),
    )
    return _m5_scenario(
        "m5_depth_shear_gate",
        obstacles=_m5_gate_obstacles(),
        start_ned_m=(-6.0, -1.0, 0.0),
        goal_ned_m=(6.0, 1.0, -1.5),
        current=current,
    )


def m5_multi_obstacle_scene() -> ScenarioConfig:
    obstacles = (
        Obstacle("m5-slalom-a", "sphere", np.array([-3.0, 0.0, -0.5]), radius_m=0.6),
        Obstacle(
            "m5-slalom-b",
            "cylinder",
            np.array([0.0, 0.8, 0.5]),
            radius_m=0.45,
            height_m=2.5,
        ),
        Obstacle("m5-slalom-c", "sphere", np.array([3.0, -0.5, 1.0]), radius_m=0.55),
    )
    return _m5_scenario(
        "m5_multi_obstacle",
        obstacles=obstacles,
        start_ned_m=(-6.0, -1.0, 0.0),
        goal_ned_m=(6.0, 1.0, -1.5),
    )


__all__ = [
    "corridor_scene",
    "cylinder_gate_scene",
    "empty_scene",
    "fixed_scenarios",
    "m5_cross_current_sphere_scene",
    "m5_cylinder_gate_scene",
    "m5_depth_shear_gate_scene",
    "m5_multi_obstacle_scene",
    "m5_open_water_scene",
    "m5_s_bend_corridor_scene",
    "m5_single_corner_scene",
    "m5_sphere_detour_scene",
    "remus100_vehicle",
    "sphere_detour_scene",
]
