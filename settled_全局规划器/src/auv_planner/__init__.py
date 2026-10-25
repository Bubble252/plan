"""Lazy public exports for the underwater planner package."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

_EXPORTS = {
    "BODY_FRAME": ".contracts",
    "WORLD_FRAME": ".contracts",
    "ActuatorCommand": ".contracts",
    "AlosSolution": ".guidance",
    "CapabilitySet": ".contracts",
    "ControlProposal": ".contracts",
    "CurrentField": ".contracts",
    "ClearanceDetails": ".geometry",
    "DepthShearCurrent": ".current",
    "EvalResult": ".contracts",
    "GlobalPlanner": ".interfaces",
    "Guidance": ".interfaces",
    "HullSpec": ".geometry",
    "LocalObservation": ".contracts",
    "LocalPlanner": ".interfaces",
    "Obstacle": ".contracts",
    "ObstacleMap": ".contracts",
    "OccupancyGrid": ".geometry",
    "PathMetrics": ".metrics",
    "PlanReason": ".contracts",
    "PlanRequest": ".contracts",
    "PlanResult": ".contracts",
    "PlanStatus": ".contracts",
    "RecoveryCertificate": ".contracts",
    "Reference": ".contracts",
    "Remus100ControlDiagnostics": ".control",
    "Remus100ActuatorCommand": ".contracts",
    "Remus100TrackingController": ".control",
    "SafeCommand": ".contracts",
    "SafetyContext": ".contracts",
    "SafetyFilter": ".interfaces",
    "ScenarioConfig": ".contracts",
    "SimulationSnapshot": ".contracts",
    "SimulatorBackend": ".interfaces",
    "State": ".contracts",
    "SweepResult": ".geometry",
    "TaskSpec": ".contracts",
    "Trajectory": ".contracts",
    "ThreeDAlosGuidance": ".guidance",
    "TrackingController": ".interfaces",
    "UniformCurrent": ".contracts",
    "VehicleSpec": ".contracts",
    "WorldSnapshot": ".contracts",
    "compute_path_metrics": ".metrics",
    "hull_clearance": ".geometry",
    "hull_clearance_details": ".geometry",
    "hull_from_vehicle": ".geometry",
    "hull_sphere_centers": ".geometry",
    "obstacle_signed_distance": ".geometry",
    "point_clearance": ".geometry",
    "point_clearance_details": ".geometry",
    "trajectory_sweep": ".geometry",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> object:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:
    from .contracts import (
        BODY_FRAME,
        WORLD_FRAME,
        ActuatorCommand,
        CapabilitySet,
        ControlProposal,
        CurrentField,
        EvalResult,
        LocalObservation,
        Obstacle,
        ObstacleMap,
        PlanReason,
        PlanRequest,
        PlanResult,
        PlanStatus,
        RecoveryCertificate,
        Reference,
        Remus100ActuatorCommand,
        SafeCommand,
        SafetyContext,
        ScenarioConfig,
        SimulationSnapshot,
        State,
        TaskSpec,
        Trajectory,
        UniformCurrent,
        VehicleSpec,
        WorldSnapshot,
    )
    from .control import Remus100ControlDiagnostics, Remus100TrackingController
    from .current import DepthShearCurrent
    from .geometry import (
        ClearanceDetails,
        HullSpec,
        OccupancyGrid,
        SweepResult,
        hull_clearance,
        hull_clearance_details,
        hull_from_vehicle,
        hull_sphere_centers,
        obstacle_signed_distance,
        point_clearance,
        point_clearance_details,
        trajectory_sweep,
    )
    from .guidance import AlosSolution, ThreeDAlosGuidance
    from .interfaces import (
        GlobalPlanner,
        Guidance,
        LocalPlanner,
        SafetyFilter,
        SimulatorBackend,
        TrackingController,
    )
    from .metrics import PathMetrics, compute_path_metrics
