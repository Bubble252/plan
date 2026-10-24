from __future__ import annotations

import numpy as np
import pytest

from auv_planner.contracts import Obstacle, ObstacleMap, State, Trajectory, VehicleSpec
from auv_planner.geometry import (
    OccupancyGrid,
    hull_clearance,
    hull_clearance_details,
    hull_from_vehicle,
    obstacle_signed_distance,
    point_clearance,
    trajectory_sweep,
)


def make_vehicle(length_m: float = 1.6, diameter_m: float = 0.19) -> VehicleSpec:
    return VehicleSpec(
        name="test-auv",
        length_m=length_m,
        diameter_m=diameter_m,
        mass_kg=31.0,
        min_speed_mps=0.1,
        max_speed_mps=4.0,
        max_accel_mps2=1.0,
        max_yaw_rate_rps=0.5,
        max_pitch_rate_rps=0.5,
    )


def make_state(position: np.ndarray, quat: np.ndarray | None = None) -> State:
    return State(
        t_s=0.0,
        position_ned_m=position,
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]) if quat is None else quat,
        linear_velocity_body_mps=np.zeros(3),
        angular_velocity_body_rps=np.zeros(3),
    )


def make_trajectory(
    points: np.ndarray,
    quats: np.ndarray | None = None,
    t_s: np.ndarray | None = None,
) -> Trajectory:
    positions = np.asarray(points, dtype=np.float64)
    rows = positions.shape[0]
    quaternions = (
        np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (rows, 1))
        if quats is None
        else np.asarray(quats, dtype=np.float64)
    )
    times = np.linspace(0.0, 1.0, rows) if t_s is None else np.asarray(t_s, dtype=np.float64)
    return Trajectory(
        trajectory_id="test",
        t_s=times,
        position_ned_m=positions,
        quat_xyzw=quaternions,
        linear_velocity_body_mps=np.zeros((rows, 3)),
        angular_velocity_body_rps=np.zeros((rows, 3)),
    )


def make_map(obstacles: tuple[Obstacle, ...] = (), half_size_m: float = 5.0) -> ObstacleMap:
    bounds = np.array(
        [
            [-half_size_m, half_size_m],
            [-half_size_m, half_size_m],
            [-half_size_m, half_size_m],
        ],
        dtype=np.float64,
    )
    return ObstacleMap(bounds_ned_m=bounds, obstacles=obstacles)


def make_gate(open_half_size_m: float = 0.3, thickness_m: float = 0.2) -> ObstacleMap:
    side_center = open_half_size_m + thickness_m / 2.0
    half_thickness = thickness_m / 2.0
    obstacles = (
        Obstacle(
            "gate-left",
            "box",
            np.array([0.0, -side_center, 0.0]),
            half_extents_m=np.array([half_thickness, half_thickness, 1.0]),
        ),
        Obstacle(
            "gate-right",
            "box",
            np.array([0.0, side_center, 0.0]),
            half_extents_m=np.array([half_thickness, half_thickness, 1.0]),
        ),
        Obstacle(
            "gate-bottom",
            "box",
            np.array([0.0, 0.0, -side_center]),
            half_extents_m=np.array([half_thickness, side_center, half_thickness]),
        ),
        Obstacle(
            "gate-top",
            "box",
            np.array([0.0, 0.0, side_center]),
            half_extents_m=np.array([half_thickness, side_center, half_thickness]),
        ),
    )
    return make_map(obstacles)


def test_sphere_signed_distance_is_negative_inside() -> None:
    sphere = Obstacle("sphere", "sphere", np.zeros(3), radius_m=1.0)

    assert obstacle_signed_distance(np.array([2.0, 0.0, 0.0]), sphere) == pytest.approx(1.0)
    assert obstacle_signed_distance(np.array([0.5, 0.0, 0.0]), sphere) == pytest.approx(-0.5)


def test_hull_collides_where_point_model_does_not() -> None:
    sphere = Obstacle("sphere", "sphere", np.array([0.85, 0.0, 0.0]), radius_m=0.1)
    obstacle_map = make_map((sphere,))
    vehicle = make_vehicle()
    state = make_state(np.zeros(3))
    hull = hull_from_vehicle(vehicle)

    assert point_clearance(obstacle_map, state.position_ned_m) == pytest.approx(0.75)
    assert hull_clearance(obstacle_map, state, hull) < 0.0
    assert hull_clearance_details(obstacle_map, state, hull).collision


def test_vertical_cylinder_side_top_and_corner_distances() -> None:
    cylinder = Obstacle(
        "pillar",
        "cylinder",
        np.zeros(3),
        radius_m=0.5,
        height_m=2.0,
    )

    assert obstacle_signed_distance(np.array([1.0, 0.0, 0.0]), cylinder) == pytest.approx(0.5)
    assert obstacle_signed_distance(np.array([0.0, 0.0, 1.5]), cylinder) == pytest.approx(0.5)
    assert obstacle_signed_distance(np.array([1.0, 0.0, 1.5]), cylinder) == pytest.approx(
        np.hypot(0.5, 0.5)
    )
    assert obstacle_signed_distance(np.array([0.0, 0.0, 0.9]), cylinder) == pytest.approx(-0.1)


def test_cylinder_blocks_hull_trajectory() -> None:
    cylinder = Obstacle(
        "pillar",
        "cylinder",
        np.zeros(3),
        radius_m=0.3,
        height_m=2.0,
    )
    obstacle_map = make_map((cylinder,), half_size_m=4.0)
    path = make_trajectory(np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))

    result = trajectory_sweep(path, obstacle_map, make_vehicle())

    assert result.collision
    assert result.min_clearance_m < 0.0
    assert result.failure_reason == "collision"


def test_gate_allows_narrow_hull_but_blocks_wide_hull() -> None:
    gate = make_gate()
    path = make_trajectory(np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))

    narrow = trajectory_sweep(path, gate, make_vehicle())
    wide = trajectory_sweep(path, gate, make_vehicle(length_m=1.6, diameter_m=0.7))

    assert narrow.is_safe
    assert narrow.min_clearance_m == pytest.approx(0.205, abs=1e-9)
    assert wide.collision
    assert wide.failure_reason == "collision"


def test_corridor_clearance_decreases_near_wall() -> None:
    wall_offset = 0.6
    obstacles = (
        Obstacle(
            "left-wall",
            "box",
            np.array([0.0, -wall_offset, 0.0]),
            half_extents_m=np.array([2.5, 0.1, 1.0]),
        ),
        Obstacle(
            "right-wall",
            "box",
            np.array([0.0, wall_offset, 0.0]),
            half_extents_m=np.array([2.5, 0.1, 1.0]),
        ),
    )
    obstacle_map = make_map(obstacles, half_size_m=4.0)
    center_path = make_trajectory(np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
    near_wall_path = make_trajectory(np.array([[-2.0, 0.2, 0.0], [2.0, 0.2, 0.0]]))
    vehicle = make_vehicle()

    center = trajectory_sweep(center_path, obstacle_map, vehicle)
    near_wall = trajectory_sweep(near_wall_path, obstacle_map, vehicle)

    assert center.min_clearance_m == pytest.approx(0.405, abs=1e-9)
    assert near_wall.min_clearance_m == pytest.approx(0.205, abs=1e-9)
    assert near_wall.min_clearance_m < center.min_clearance_m


def test_sweep_detects_obstacle_between_safe_endpoints() -> None:
    sphere = Obstacle("sphere", "sphere", np.zeros(3), radius_m=0.4)
    obstacle_map = make_map((sphere,), half_size_m=4.0)
    path = make_trajectory(np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))

    result = trajectory_sweep(path, obstacle_map, make_vehicle())

    assert result.collision
    assert result.min_clearance_m < 0.0
    assert result.sampled_poses > 2
    assert result.failure_reason == "collision"


def test_sweep_detects_rotation_collision() -> None:
    obstacle = Obstacle("sphere", "sphere", np.array([0.0, 0.8, 0.0]), radius_m=0.2)
    obstacle_map = make_map((obstacle,), half_size_m=3.0)
    yaw_90 = np.array([0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)])
    path = make_trajectory(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        quats=np.vstack((np.array([0.0, 0.0, 0.0, 1.0]), yaw_90)),
    )

    result = trajectory_sweep(path, obstacle_map, make_vehicle(length_m=2.0, diameter_m=0.1))

    assert result.collision
    assert result.failure_reason == "collision"


def test_sweep_detects_out_of_bounds() -> None:
    path = make_trajectory(np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))

    result = trajectory_sweep(path, make_map(half_size_m=1.0), make_vehicle())

    assert result.out_of_bounds
    assert result.min_clearance_m < 0.0
    assert result.failure_reason == "out_of_bounds"


def test_occupancy_grid_clearance_and_bounds() -> None:
    occupied = np.zeros((2, 2, 2), dtype=bool)
    occupied[0, 0, 0] = True
    grid = OccupancyGrid(origin_ned_m=np.zeros(3), resolution_m=1.0, occupied=occupied)

    assert grid.contains(np.array([1.5, 0.5, 0.5]))
    assert not grid.contains(np.array([2.0, 0.5, 0.5]))
    assert grid.point_clearance(np.array([0.5, 0.5, 0.5])) == pytest.approx(-0.5)
    assert grid.point_clearance(np.array([1.5, 0.5, 0.5])) == pytest.approx(0.5)
    assert grid.point_clearance(np.array([2.5, 0.5, 0.5])) == pytest.approx(-0.5)

    empty = OccupancyGrid(origin_ned_m=np.zeros(3), resolution_m=1.0, occupied=np.zeros((2, 2, 2), dtype=bool))
    assert empty.point_clearance(np.array([1.0, 1.0, 1.0])) == pytest.approx(1.0)
