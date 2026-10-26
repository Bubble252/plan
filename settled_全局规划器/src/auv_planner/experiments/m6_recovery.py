from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import Obstacle, ObstacleMap, ScenarioConfig, State, UniformCurrent, WorldSnapshot
from ..current import DepthShearCurrent
from ..recovery import (
    DEFAULT_DIAGNOSIS_DELAYS_S,
    DEFAULT_FAULTS,
    ClearanceModel,
    ConstraintMode,
    FaultCheck,
    ManeuverCheck,
    RecoverabilityChecker,
    RecoveryAssessment,
    RecoveryConfig,
)
from ..scenarios import (
    m5_cylinder_gate_scene,
    m5_depth_shear_gate_scene,
    m5_open_water_scene,
    m5_s_bend_corridor_scene,
    m5_sphere_detour_scene,
)


@dataclass(frozen=True)
class M6KeyNode:
    name: str
    description: str
    scenario: ScenarioConfig
    state: State
    safe_anchor_ned_m: np.ndarray | None = None


@dataclass(frozen=True)
class M6ConstraintVariant:
    name: str
    config: RecoveryConfig


def _state(
    position_ned_m: tuple[float, float, float],
    *,
    speed_mps: float = 1.0,
    body_velocity: tuple[float, float, float] | None = None,
) -> State:
    velocity = (
        np.array(body_velocity, dtype=np.float64)
        if body_velocity is not None
        else np.array([speed_mps, 0.0, 0.0], dtype=np.float64)
    )
    return State(
        t_s=0.0,
        position_ned_m=np.asarray(position_ned_m, dtype=np.float64),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        linear_velocity_body_mps=velocity,
        angular_velocity_body_rps=np.zeros(3, dtype=np.float64),
    )


def dead_end_scene() -> ScenarioConfig:
    """A narrow closed tube with only a long reverse route as a possible escape."""

    obstacles = (
        Obstacle(
            "dead-end-left",
            "box",
            np.array([0.0, -0.48, 0.0]),
            half_extents_m=np.array([2.15, 0.05, 0.5]),
        ),
        Obstacle(
            "dead-end-right",
            "box",
            np.array([0.0, 0.48, 0.0]),
            half_extents_m=np.array([2.15, 0.05, 0.5]),
        ),
        Obstacle(
            "dead-end-floor",
            "box",
            np.array([0.0, 0.0, 0.5]),
            half_extents_m=np.array([2.15, 0.43, 0.05]),
        ),
        Obstacle(
            "dead-end-ceiling",
            "box",
            np.array([0.0, 0.0, -0.5]),
            half_extents_m=np.array([2.15, 0.43, 0.05]),
        ),
        Obstacle(
            "dead-end-back",
            "box",
            np.array([1.4, 0.0, 0.0]),
            half_extents_m=np.array([0.1, 0.43, 0.45]),
        ),
    )
    world = WorldSnapshot(
        t_s=0.0,
        obstacle_map=ObstacleMap(
            bounds_ned_m=np.array([[-6.0, 12.0], [-2.0, 2.0], [-2.0, 2.0]]),
            obstacles=obstacles,
        ),
        current_field=UniformCurrent(velocity_ned_mps=np.zeros(3)),
    )
    return ScenarioConfig(
        name="m6_dead_end",
        world=world,
        vehicle=m5_open_water_scene().vehicle,
        start=_state((0.0, 0.0, 0.0)),
        goal_ned_m=np.array([-4.0, 0.0, 0.0]),
        goal_radius_m=0.4,
    )


def m6_key_nodes() -> tuple[M6KeyNode, ...]:
    open_scene = m5_open_water_scene()
    sphere_scene = m5_sphere_detour_scene()
    gate_scene = m5_cylinder_gate_scene()
    s_bend_scene = m5_s_bend_corridor_scene()
    shear_scene = m5_depth_shear_gate_scene()
    cross_current_scene = replace(
        open_scene,
        name="m6_open_water_cross_current_0.30",
        world=replace(
            open_scene.world,
            current_field=UniformCurrent(velocity_ned_mps=np.array([0.0, 0.3, 0.0])),
        ),
        start=_state((-2.0, 0.0, 0.0)),
    )
    dead_end = dead_end_scene()
    return (
        M6KeyNode(
            "open_water_start",
            "M5 open-water start; checks actuator-space recovery without obstacles",
            open_scene,
            open_scene.start,
        ),
        M6KeyNode(
            "sphere_start",
            "M5 sphere-detour start; detour and clearance margin are available",
            sphere_scene,
            sphere_scene.start,
        ),
        M6KeyNode(
            "gate_start",
            "M5 gate start; recovery must respect the gate and downstream pillar",
            gate_scene,
            gate_scene.start,
        ),
        M6KeyNode(
            "s_bend_start",
            "M5 S-bend start; narrow lateral space stresses minimum-radius turns",
            s_bend_scene,
            s_bend_scene.start,
        ),
        M6KeyNode(
            "shear_gate_start",
            "M5 depth-shear gate start using the validated shear field",
            shear_scene,
            shear_scene.start,
        ),
        M6KeyNode(
            "cross_current_open_center",
            "open-water node with the M5-validated 0.30 m/s uniform cross current",
            cross_current_scene,
            cross_current_scene.start,
        ),
        M6KeyNode(
            "dead_end_no_recovery",
            "artificial narrow closed tube; current pose is clearance-safe but no 10 s backup exists",
            dead_end,
            dead_end.start,
        ),
    )


def m6_constraint_variants() -> tuple[M6ConstraintVariant, ...]:
    return (
        M6ConstraintVariant("full_hull_vehicle_limits", RecoveryConfig()),
        M6ConstraintVariant(
            "point_vehicle_limits",
            RecoveryConfig(clearance_model=ClearanceModel.POINT_DIAGNOSTIC),
        ),
        M6ConstraintVariant(
            "full_hull_unbounded_turn",
            RecoveryConfig(constraint_mode=ConstraintMode.UNBOUNDED_TURN_DIAGNOSTIC),
        ),
        M6ConstraintVariant(
            "point_unbounded_turn",
            RecoveryConfig(
                clearance_model=ClearanceModel.POINT_DIAGNOSTIC,
                constraint_mode=ConstraintMode.UNBOUNDED_TURN_DIAGNOSTIC,
            ),
        ),
    )


def _current_manifest(world: WorldSnapshot) -> dict[str, Any]:
    current = world.current_field
    if isinstance(current, UniformCurrent):
        return {"kind": "uniform", "velocity_ned_mps": current.velocity_ned_mps.tolist()}
    if isinstance(current, DepthShearCurrent):
        return {
            "kind": "depth_shear",
            "reference_velocity_ned_mps": current.reference_velocity_ned_mps.tolist(),
            "shear_ned_per_m": current.shear_ned_per_m.tolist(),
        }
    return {"kind": type(current).__name__}


def m6_config_manifest() -> dict[str, Any]:
    config = RecoveryConfig()
    return {
        "schema_version": 1,
        "status": "M6 frozen recovery configuration",
        "model": {
            "horizon_s": config.horizon_s,
            "required_clearance_m": config.required_clearance_m,
            "certificate_validity_s": config.certificate_validity_s,
            "sample_time_s": config.sample_time_s,
            "max_pitch_angle_deg": float(np.rad2deg(config.max_pitch_angle_rad)),
            "error_bounds": asdict(config.error_bounds),
            "clearance_model": config.clearance_model.value,
            "constraint_mode": config.constraint_mode.value,
            "continuous_time_guarantee": False,
        },
        "faults": [
            {
                "name": fault.name,
                "kind": fault.kind.value,
                "effectiveness": fault.effectiveness,
                "stuck_sign": fault.stuck_sign,
                "residual_rate_fraction": fault.residual_rate_fraction,
            }
            for fault in DEFAULT_FAULTS
        ],
        "diagnosis_delays_s": list(DEFAULT_DIAGNOSIS_DELAYS_S),
        "x0_variants": [
            {
                "name": variant.name,
                "clearance_model": variant.config.clearance_model.value,
                "constraint_mode": variant.config.constraint_mode.value,
            }
            for variant in m6_constraint_variants()
        ],
        "x0_scope": "nominal fault at 0 s; isolates point-vs-hull and turn-envelope effects",
        "certificate_issuance": "only full_hull with vehicle limits may emit valid=true",
        "x1_rule": "first delay in the frozen grid for which any maneuver under the fault is unsafe",
        "key_nodes": [
            {
                "name": node.name,
                "description": node.description,
                "scenario_name": node.scenario.name,
                "state_position_ned_m": node.state.position_ned_m.tolist(),
                "state_linear_velocity_body_mps": node.state.linear_velocity_body_mps.tolist(),
                "current": _current_manifest(node.scenario.world),
            }
            for node in m6_key_nodes()
        ],
        "claims": {
            "full_hull": True,
            "continuous_time": False,
            "sensor_noise": False,
            "communication_faults": False,
            "energy_depletion": False,
            "current_envelope": "M5-validated 0.30 m/s uniform and depth-shear fields only",
        },
    }


def write_m6_config_manifest(path: Path | str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(m6_config_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def _assessment_margin(assessment: RecoveryAssessment) -> float | None:
    return assessment.certificate.margin_m


def _best_or_selected(check: FaultCheck) -> ManeuverCheck | None:
    if check.selected is not None:
        return check.selected
    if not check.checks:
        return None
    return max(check.checks, key=lambda candidate: (candidate.margin_m, candidate.candidate.name))


def run_x0_constraints(nodes: tuple[M6KeyNode, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    checker = RecoverabilityChecker()
    for node in nodes:
        for variant in m6_constraint_variants():
            assessment = checker.assess(
                node.state,
                node.scenario.world,
                node.scenario.vehicle,
                config=variant.config,
                faults=(DEFAULT_FAULTS[0],),
                diagnosis_delays_s=(0.0,),
                safe_anchor_ned_m=node.safe_anchor_ned_m,
            )
            exceedance_count = sum(
                1
                for fault_check in assessment.fault_checks
                for check in fault_check.checks
                if check.candidate.vehicle_limit_exceedances
            )
            rows.append(
                {
                    "node": node.name,
                    "variant": variant.name,
                    "certificate_valid": assessment.valid,
                    "all_fault_checks_safe": all(check.safe for check in assessment.fault_checks),
                    "safe_fault_checks": sum(check.safe for check in assessment.fault_checks),
                    "total_fault_checks": len(assessment.fault_checks),
                    "current_clearance_m": assessment.current_clearance_m,
                    "certificate_margin_m": _assessment_margin(assessment),
                    "vehicle_limit_exceedance_count": exceedance_count,
                    "computation_time_s": assessment.computation_time_s,
                    "reason": assessment.certificate.reason,
                }
            )
    return rows


def run_x1_delays(nodes: tuple[M6KeyNode, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    checker = RecoverabilityChecker()
    for node in nodes:
        for fault in DEFAULT_FAULTS:
            for delay in DEFAULT_DIAGNOSIS_DELAYS_S:
                assessment = checker.assess(
                    node.state,
                    node.scenario.world,
                    node.scenario.vehicle,
                    config=RecoveryConfig(),
                    faults=(fault,),
                    diagnosis_delays_s=(delay,),
                    safe_anchor_ned_m=node.safe_anchor_ned_m,
                )
                fault_check = assessment.fault_checks[0]
                best = _best_or_selected(fault_check)
                rows.append(
                    {
                        "node": node.name,
                        "fault": fault.name,
                        "delay_s": delay,
                        "safe": fault_check.safe,
                        "selected_maneuver": (
                            fault_check.selected.candidate.name
                            if fault_check.selected is not None
                            else None
                        ),
                        "margin_m": fault_check.margin_m,
                        "best_attempt": best.candidate.name if best is not None else None,
                        "best_attempt_clearance_m": best.min_clearance_m
                        if best is not None
                        else None,
                        "best_attempt_failure_reason": best.failure_reason
                        if best is not None
                        else None,
                        "coast_min_clearance_m": fault_check.coast_min_clearance_m,
                        "unavailable_maneuvers": list(fault_check.unavailable_maneuvers),
                        "computation_time_s": assessment.computation_time_s,
                        "certificate_reason": assessment.certificate.reason,
                    }
                )
    return rows


def delay_boundaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["node"]), str(row["fault"])), []).append(row)
    boundaries: list[dict[str, Any]] = []
    for (node, fault), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: float(row["delay_s"]))
        first_invalid = next(
            (float(row["delay_s"]) for row in ordered if not bool(row["safe"])),
            None,
        )
        boundaries.append(
            {
                "node": node,
                "fault": fault,
                "first_invalid_delay_s": first_invalid,
                "delay_results": ";".join(
                    f"{float(row['delay_s']):g}:{'safe' if bool(row['safe']) else 'unsafe'}"
                    for row in ordered
                ),
            }
        )
    return boundaries


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            normalized = {
                key: "|".join(str(item) for item in value) if isinstance(value, list) else value
                for key, value in row.items()
            }
            writer.writerow(normalized)


def _counterexample_details(nodes: tuple[M6KeyNode, ...]) -> dict[str, Any]:
    node = next(node for node in nodes if node.name == "dead_end_no_recovery")
    assessment = RecoverabilityChecker().assess(
        node.state,
        node.scenario.world,
        node.scenario.vehicle,
        faults=(DEFAULT_FAULTS[0],),
        diagnosis_delays_s=(0.0,),
    )
    best = max(
        (check for fault_check in assessment.fault_checks for check in fault_check.checks),
        key=lambda check: (check.margin_m, check.candidate.name),
    )
    config = RecoveryConfig()
    return {
        "node": node.name,
        "current_clearance_m": assessment.current_clearance_m,
        "required_inflated_clearance_m": config.required_inflated_clearance_m(
            node.scenario.vehicle
        ),
        "certificate_valid": assessment.valid,
        "reason": assessment.certificate.reason,
        "best_attempt": best.candidate.name,
        "best_attempt_min_clearance_m": best.min_clearance_m,
        "best_attempt_failure_reason": best.failure_reason,
    }


def _write_report(
    path: Path,
    x0_rows: list[dict[str, Any]],
    x1_rows: list[dict[str, Any]],
    boundaries: list[dict[str, Any]],
    counterexample: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    lines = [
        "# M6 Recovery Certificate Experiment",
        "",
        "This report is generated by `scripts/run_m6_recovery.py`. It is a finite-horizon,",
        "discrete-time diagnostic result, not a continuous-time safety guarantee.",
        "",
        "## Frozen configuration",
        "",
        f"- Horizon: {summary['config']['horizon_s']} s",
        f"- Base clearance: {summary['config']['required_clearance_m']} m",
        f"- Certificate validity: {summary['config']['certificate_validity_s']} s",
        f"- Error inflation: {summary['config']['error_inflation_m']:.3f} m",
        f"- Diagnosis delays: {summary['config']['diagnosis_delays_s']}",
        "",
        "## X0: geometry and actuator-space constraints",
        "",
        "| Variant | Certified nodes | Nominal-safe nodes | Nominal checks safe | Mean margin (m) |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in m6_constraint_variants():
        selected = [row for row in x0_rows if row["variant"] == variant.name]
        margins = [
            float(row["certificate_margin_m"])
            for row in selected
            if row["certificate_margin_m"] is not None
        ]
        if margins:
            certified = sum(bool(row["certificate_valid"]) for row in selected)
            nominal_safe = sum(bool(row["all_fault_checks_safe"]) for row in selected)
            checks_safe = sum(int(row["safe_fault_checks"]) for row in selected)
            checks_total = sum(int(row["total_fault_checks"]) for row in selected)
            lines.append(
                f"| {variant.name} | {certified}/{len(selected)} | {nominal_safe}/{len(selected)} "
                f"| {checks_safe}/{checks_total} | {np.mean(margins):.3f} |"
            )
        else:
            lines.append(f"| {variant.name} | 0/{len(selected)} | 0/{len(selected)} | 0/0 | n/a |")
    lines.extend(
        [
            "",
            "Only `full_hull_vehicle_limits` may issue a certificate. Point-clearance and relaxed-turn",
            "variants are diagnostic and are forced to `valid=false` even when their checks pass.",
            "",
            "## X1: diagnosis-delay scan",
            "",
            "| Delay (s) | Safe fault checks | Fault-check success rate |",
            "|---:|---:|---:|",
        ]
    )
    for delay in DEFAULT_DIAGNOSIS_DELAYS_S:
        selected = [row for row in x1_rows if float(row["delay_s"]) == delay]
        safe = sum(bool(row["safe"]) for row in selected)
        lines.append(f"| {delay:g} | {safe}/{len(selected)} | {safe / len(selected):.1%} |")
    invalid_boundaries = [row for row in boundaries if row["first_invalid_delay_s"] is not None]
    lines.extend(
        [
            "",
            f"First-invalid delay was observed for {len(invalid_boundaries)}/{len(boundaries)} node-fault pairs.",
            f"No-solution rate across the fault-delay grid: {summary['x1_unsafe_rate']:.1%}.",
            f"Mean certificate computation time: {summary['x1_mean_computation_time_s'] * 1000.0:.1f} ms per fault-delay check.",
            "No failure thresholds were tuned to improve this rate.",
            "",
            "## Counterexample: current clearance does not imply recoverability",
            "",
            f"- Node: `{counterexample['node']}`",
            f"- Current full-hull clearance: {counterexample['current_clearance_m']:.3f} m",
            f"- Required inflated clearance: {counterexample['required_inflated_clearance_m']:.3f} m",
            f"- Certificate: `valid={counterexample['certificate_valid']}`",
            f"- Reason: `{counterexample['reason']}`",
            f"- Best attempted maneuver: `{counterexample['best_attempt']}`",
            f"- Best attempted clearance: {counterexample['best_attempt_min_clearance_m']:.3f} m",
            f"- Best attempted failure: `{counterexample['best_attempt_failure_reason']}`",
            "",
            "The tube is wide enough for the current hull pose, but a 180-degree minimum-radius turn",
            "needs more lateral space than the tube provides; minimum speed prevents stopping in place.",
            "The ceiling and floor also block vertical escape before the back wall is reached.",
            "",
            "## Failure-reason counts",
            "",
        ]
    )
    for reason, count in summary["failure_reason_counts"].items():
        lines.append(f"- `{reason}`: {count}")
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Backup trajectories use a reduced-order kinematic model and sampled currents.",
            "- The model does not include sway, heave, roll, sensor noise, communication loss, or energy depletion.",
            "- Stuck-fin faults use the declared worst-case residual rate fraction.",
            "- Passing checks are only valid inside the sampled horizon and frozen error bounds.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_m6_recovery(output_dir: Path | str) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    nodes = m6_key_nodes()
    x0_rows = run_x0_constraints(nodes)
    x1_rows = run_x1_delays(nodes)
    boundaries = delay_boundaries(x1_rows)
    counterexample = _counterexample_details(nodes)

    certificates: list[dict[str, Any]] = []
    checker = RecoverabilityChecker()
    for node in nodes:
        assessment = checker.assess(
            node.state,
            node.scenario.world,
            node.scenario.vehicle,
            config=RecoveryConfig(),
            safe_anchor_ned_m=node.safe_anchor_ned_m,
        )
        certificates.append(
            {
                "node": node.name,
                "computation_time_s": assessment.computation_time_s,
                "certificate": asdict(assessment.certificate),
            }
        )

    failure_reasons = Counter(
        str(row["best_attempt_failure_reason"])
        for row in x1_rows
        if row["best_attempt_failure_reason"] is not None
    )
    config = RecoveryConfig()
    summary: dict[str, Any] = {
        "status": "M6 complete; finite-horizon diagnostic certificate",
        "node_count": len(nodes),
        "x0_rows": len(x0_rows),
        "x1_rows": len(x1_rows),
        "boundary_rows": len(boundaries),
        "config": {
            "horizon_s": config.horizon_s,
            "required_clearance_m": config.required_clearance_m,
            "certificate_validity_s": config.certificate_validity_s,
            "sample_time_s": config.sample_time_s,
            "error_inflation_m": config.error_clearance_inflation_m(nodes[0].scenario.vehicle),
            "diagnosis_delays_s": list(DEFAULT_DIAGNOSIS_DELAYS_S),
            "faults": [fault.name for fault in DEFAULT_FAULTS],
        },
        "x0_certified": sum(bool(row["certificate_valid"]) for row in x0_rows),
        "x0_all_checks_safe": sum(bool(row["all_fault_checks_safe"]) for row in x0_rows),
        "x1_safe": sum(bool(row["safe"]) for row in x1_rows),
        "x1_total": len(x1_rows),
        "x1_unsafe_rate": 1.0 - sum(bool(row["safe"]) for row in x1_rows) / len(x1_rows),
        "x0_mean_computation_time_s": float(
            np.mean([float(row["computation_time_s"]) for row in x0_rows])
        ),
        "x1_mean_computation_time_s": float(
            np.mean([float(row["computation_time_s"]) for row in x1_rows])
        ),
        "first_invalid_boundary_count": sum(
            row["first_invalid_delay_s"] is not None for row in boundaries
        ),
        "failure_reason_counts": dict(sorted(failure_reasons.items())),
        "counterexample": counterexample,
    }

    _write_csv(root / "x0_constraints.csv", x0_rows)
    _write_csv(root / "x1_delay_scan.csv", x1_rows)
    _write_csv(root / "x1_delay_boundaries.csv", boundaries)
    (root / "certificates.json").write_text(
        json.dumps(certificates, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(root / "report.md", x0_rows, x1_rows, boundaries, counterexample, summary)
    return summary


__all__ = [
    "M6ConstraintVariant",
    "M6KeyNode",
    "dead_end_scene",
    "delay_boundaries",
    "m6_config_manifest",
    "m6_constraint_variants",
    "m6_key_nodes",
    "run_m6_recovery",
    "write_m6_config_manifest",
]
