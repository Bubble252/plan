from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import Obstacle, ScenarioConfig, UniformCurrent
from .current import DepthShearCurrent
from .scenarios import (
    m5_cross_current_sphere_scene,
    m5_cylinder_gate_scene,
    m5_depth_shear_gate_scene,
    m5_multi_obstacle_scene,
    m5_open_water_scene,
    m5_s_bend_corridor_scene,
    m5_single_corner_scene,
    m5_sphere_detour_scene,
)


@dataclass(frozen=True)
class M5SceneSpec:
    scenario: ScenarioConfig
    generation_rule: str
    evaluation_seeds: tuple[int, ...] = tuple(range(10))

    def __post_init__(self) -> None:
        if not self.generation_rule:
            raise ValueError("generation_rule must not be empty")
        object.__setattr__(self, "evaluation_seeds", tuple(self.evaluation_seeds))
        if not self.evaluation_seeds or any(seed < 0 for seed in self.evaluation_seeds):
            raise ValueError("evaluation_seeds must be a non-empty sequence of non-negative ints")


def m5_scene_catalog() -> tuple[M5SceneSpec, ...]:
    return (
        M5SceneSpec(
            m5_open_water_scene(),
            "fixed analytic empty box; straight start/goal with a 1.5 m depth change",
        ),
        M5SceneSpec(
            m5_sphere_detour_scene(),
            "fixed unit sphere at (0,0,-0.5) m with radius 0.9 m; detour in y and z",
        ),
        M5SceneSpec(
            m5_cylinder_gate_scene(),
            "fixed 1.0 m square gate centred at z=-1.5 m plus a vertical cylinder at x=2.8 m",
        ),
        M5SceneSpec(
            m5_s_bend_corridor_scene(),
            "three staggered pairs of vertical wall segments forming an S-shaped channel",
        ),
        M5SceneSpec(
            m5_single_corner_scene(),
            "fixed box wall covering y in [-1,4] m and z in [-2,2] m; path must turn around its lower end",
        ),
        M5SceneSpec(
            m5_cross_current_sphere_scene(),
            "sphere-detour geometry with uniform +y current of 0.35 m/s",
        ),
        M5SceneSpec(
            m5_depth_shear_gate_scene(),
            "gate geometry with horizontal current [0.25,0,0] + z*(0.02,0.01,0) m/s",
        ),
        M5SceneSpec(
            m5_multi_obstacle_scene(),
            "fixed three-obstacle slalom with sphere, vertical cylinder and sphere at staggered depths",
        ),
    )


def scene_catalog_manifest() -> dict[str, Any]:
    scenes = []
    for spec in m5_scene_catalog():
        scenario = spec.scenario
        scenes.append(
            {
                "scenario_name": scenario.name,
                "generation_rule": spec.generation_rule,
                "evaluation_seeds": list(spec.evaluation_seeds),
                "bounds_ned_m": scenario.world.obstacle_map.bounds_ned_m.tolist(),
                "start_ned_m": scenario.start.position_ned_m.tolist(),
                "start_linear_velocity_body_mps": scenario.start.linear_velocity_body_mps.tolist(),
                "goal_ned_m": scenario.goal_ned_m.tolist(),
                "goal_radius_m": scenario.goal_radius_m,
                "obstacles": [_obstacle_manifest(obstacle) for obstacle in scenario.world.obstacle_map.obstacles],
                "current": _current_manifest(scenario),
                "constraints": dict(scenario.constraints),
            }
        )
    return {
        "schema_version": 1,
        "status": "M5-B frozen scene catalog",
        "vehicle": "REMUS100",
        "seed_policy": "all planners use the same evaluation_seeds within each scenario",
        "scenes": scenes,
    }


def write_scene_catalog_manifest(path: Path | str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(scene_catalog_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def _obstacle_manifest(obstacle: Obstacle) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "obstacle_id": obstacle.obstacle_id,
        "kind": obstacle.kind,
        "center_ned_m": obstacle.center_ned_m.tolist(),
    }
    if obstacle.radius_m is not None:
        payload["radius_m"] = obstacle.radius_m
    if obstacle.half_extents_m is not None:
        payload["half_extents_m"] = obstacle.half_extents_m.tolist()
    if obstacle.height_m is not None:
        payload["height_m"] = obstacle.height_m
    return payload


def _current_manifest(scenario: ScenarioConfig) -> dict[str, Any]:
    current = scenario.world.current_field
    if isinstance(current, UniformCurrent):
        return {
            "kind": "uniform",
            "velocity_ned_mps": current.velocity_ned_mps.tolist(),
        }
    if isinstance(current, DepthShearCurrent):
        return {
            "kind": "depth_shear",
            "reference_velocity_ned_mps": current.reference_velocity_ned_mps.tolist(),
            "shear_ned_per_m": current.shear_ned_per_m.tolist(),
            "reference_depth_m": current.reference_depth_m,
        }
    raise TypeError(f"unsupported M5 current type: {type(current).__name__}")


def scene_catalog_digest() -> str:
    payload = json.dumps(scene_catalog_manifest(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "M5SceneSpec",
    "m5_scene_catalog",
    "scene_catalog_digest",
    "scene_catalog_manifest",
    "write_scene_catalog_manifest",
]
