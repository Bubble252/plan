from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from auv_planner.geometry import point_clearance
from auv_planner.scene_catalog import (
    m5_scene_catalog,
    scene_catalog_digest,
    scene_catalog_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "configs" / "scenes" / "m5_scene_catalog.json"


def test_m5_scene_catalog_has_eight_unique_fixed_scenes() -> None:
    catalog = m5_scene_catalog()
    names = [spec.scenario.name for spec in catalog]
    assert len(catalog) == 8
    assert len(set(names)) == len(names)
    assert all(spec.evaluation_seeds == tuple(range(10)) for spec in catalog)
    assert scene_catalog_digest() == scene_catalog_digest()


def test_m5_scene_manifest_matches_generated_catalog() -> None:
    committed = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    generated = scene_catalog_manifest()
    assert committed == generated


def test_m5_scenes_have_start_and_goal_inside_bounds_and_clear_of_obstacles() -> None:
    for spec in m5_scene_catalog():
        scenario = spec.scenario
        bounds = scenario.world.obstacle_map.bounds_ned_m
        hull_radius = scenario.vehicle.diameter_m / 2.0
        for point in (scenario.start.position_ned_m, scenario.goal_ned_m):
            assert np.all(point >= bounds[:, 0])
            assert np.all(point <= bounds[:, 1])
            assert point_clearance(scenario.world.obstacle_map, point) > hull_radius


def test_m5_scene_metadata_records_currents_and_obstacles() -> None:
    scenes = {spec.scenario.name: spec for spec in m5_scene_catalog()}
    assert not scenes["m5_open_water"].scenario.world.obstacle_map.obstacles
    assert (
        scenes["m5_cross_current_sphere"].scenario.world.current_field.sample(np.zeros(3), 0.0)[1]
        > 0.0
    )
    assert scenes["m5_depth_shear_gate"].generation_rule.startswith(
        "gate geometry with horizontal current"
    )
    assert len(scenes["m5_multi_obstacle"].scenario.world.obstacle_map.obstacles) == 3
