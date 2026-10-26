from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from auv_planner.attitude import euler_from_quaternion
from auv_planner.contracts import (
    Obstacle,
    ObstacleMap,
    State,
    UniformCurrent,
    VehicleSpec,
    WorldSnapshot,
)
from auv_planner.experiments.m6_recovery import (
    dead_end_scene,
    m6_config_manifest,
    m6_key_nodes,
)
from auv_planner.geometry import hull_clearance_details, hull_from_vehicle
from auv_planner.recovery import (
    DEFAULT_FAULTS,
    ClearanceModel,
    ConstraintMode,
    FaultKind,
    ManeuverKind,
    RecoverabilityChecker,
    RecoveryConfig,
    build_maneuvers,
)
from auv_planner.scenarios import m5_open_water_scene, remus100_vehicle

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "configs" / "experiments" / "m6_recovery.json"


def _open_state() -> tuple[State, WorldSnapshot, VehicleSpec]:
    scenario = m5_open_water_scene()
    return scenario.start, scenario.world, scenario.vehicle


def test_maneuver_library_covers_required_kinds_and_respects_vehicle_limits() -> None:
    state, world, vehicle = _open_state()
    config = RecoveryConfig()
    candidates = build_maneuvers(
        state,
        world,
        vehicle,
        DEFAULT_FAULTS[0],
        config,
        duration_s=config.horizon_s,
    )

    assert {candidate.kind for candidate in candidates} == set(ManeuverKind)
    assert {candidate.name for candidate in candidates} == {
        "brake_to_min_speed",
        "ascend",
        "descend",
        "minimum_radius_turn_left",
        "minimum_radius_turn_right",
        "exit_current",
        "enter_safe_corridor",
    }
    for candidate in candidates:
        trajectory = candidate.trajectory
        assert candidate.vehicle_limit_exceedances == ()
        assert trajectory.feasibility == "kinodynamically_feasible"
        assert trajectory.t_s[-1] - trajectory.t_s[0] == pytest.approx(config.horizon_s)
        assert np.all(np.diff(trajectory.t_s) > 0.0)
        assert np.max(np.abs(trajectory.angular_velocity_body_rps[:, 1])) <= vehicle.max_pitch_rate_rps + 1e-9
        assert np.max(np.abs(trajectory.angular_velocity_body_rps[:, 2])) <= vehicle.max_yaw_rate_rps + 1e-9
        pitches = np.vstack([euler_from_quaternion(quat) for quat in trajectory.quat_xyzw])[:, 1]
        assert np.max(np.abs(pitches)) <= config.max_pitch_angle_rad + 1e-9
        speeds = trajectory.linear_velocity_body_mps[:, 0]
        assert np.max(np.abs(np.diff(speeds))) <= vehicle.max_accel_mps2 * config.sample_time_s + 1e-9


def test_open_water_nominal_and_thruster_loss_are_certified() -> None:
    state, world, vehicle = _open_state()
    config = RecoveryConfig()
    assessment = RecoverabilityChecker().assess(
        state,
        world,
        vehicle,
        config=config,
        faults=DEFAULT_FAULTS[:2],
        diagnosis_delays_s=(0.0, 1.0, 3.0, 5.0),
    )

    assert assessment.valid
    assert assessment.certificate.reason is None
    assert assessment.certificate.valid_from_s == pytest.approx(state.t_s)
    assert assessment.certificate.valid_until_s == pytest.approx(
        state.t_s + config.certificate_validity_s
    )
    assert assessment.certificate.backup_maneuvers
    assert any("finite-horizon" in assumption for assumption in assessment.certificate.assumptions)
    assert any("error inflation" in assumption for assumption in assessment.certificate.assumptions)
    assert all(check.safe and check.selected is not None for check in assessment.fault_checks)
    assert assessment.computation_time_s >= 0.0


def test_dead_end_is_clear_now_but_has_no_recovery_fallback() -> None:
    scenario = dead_end_scene()
    config = RecoveryConfig()
    hull = hull_from_vehicle(scenario.vehicle)
    current_details = hull_clearance_details(
        scenario.world.obstacle_map,
        scenario.start,
        hull,
    )
    assessment = RecoverabilityChecker().assess(
        scenario.start,
        scenario.world,
        scenario.vehicle,
        config=config,
        faults=DEFAULT_FAULTS,
        diagnosis_delays_s=(0.0,),
    )

    assert current_details.is_safe
    assert current_details.clearance_m >= config.required_inflated_clearance_m(scenario.vehicle)
    assert not assessment.valid
    assert assessment.certificate.reason is not None
    assert assessment.certificate.reason.startswith("no_recoverable_backup:")
    assert all(not check.safe for check in assessment.fault_checks)
    attempted = [check for fault in assessment.fault_checks for check in fault.checks]
    assert attempted
    assert max(check.margin_m for check in attempted) < 0.0
    assert all(check.failure_reason is not None for check in attempted)


def test_diagnosis_delay_beyond_horizon_is_explicitly_infeasible() -> None:
    state, world, vehicle = _open_state()
    assessment = RecoverabilityChecker().assess(
        state,
        world,
        vehicle,
        config=RecoveryConfig(horizon_s=3.0),
        faults=(DEFAULT_FAULTS[0],),
        diagnosis_delays_s=(5.0,),
    )

    assert not assessment.valid
    assert assessment.fault_checks[0].failure_reason == "diagnosis_delay_exhausts_horizon"


@pytest.mark.parametrize(
    "config",
    [
        RecoveryConfig(clearance_model=ClearanceModel.POINT_DIAGNOSTIC),
        RecoveryConfig(constraint_mode=ConstraintMode.UNBOUNDED_TURN_DIAGNOSTIC),
    ],
)
def test_diagnostic_modes_never_issue_safety_certificates(config: RecoveryConfig) -> None:
    state, world, vehicle = _open_state()
    assessment = RecoverabilityChecker().assess(
        state,
        world,
        vehicle,
        config=config,
        faults=DEFAULT_FAULTS[:2],
        diagnosis_delays_s=(0.0,),
    )

    assert all(check.safe for check in assessment.fault_checks)
    assert not assessment.valid
    assert assessment.certificate.reason == "diagnostic_mode_does_not_issue_safety_certificate"


def test_relaxed_turn_diagnostic_contains_the_vehicle_limited_candidates() -> None:
    state, world, vehicle = _open_state()
    checker = RecoverabilityChecker()
    limited = checker.assess(
        state,
        world,
        vehicle,
        config=RecoveryConfig(),
        faults=(DEFAULT_FAULTS[0],),
        diagnosis_delays_s=(0.0,),
    )
    relaxed = checker.assess(
        state,
        world,
        vehicle,
        config=RecoveryConfig(constraint_mode=ConstraintMode.UNBOUNDED_TURN_DIAGNOSTIC),
        faults=(DEFAULT_FAULTS[0],),
        diagnosis_delays_s=(0.0,),
    )

    assert limited.fault_checks[0].safe
    assert relaxed.fault_checks[0].safe
    assert any(
        name.endswith("_vehicle_rate")
        for name in (check.candidate.name for check in relaxed.fault_checks[0].checks)
    )


def test_stuck_fin_faults_remove_the_maneuvers_that_need_that_axis() -> None:
    state, world, vehicle = _open_state()
    checker = RecoverabilityChecker()
    yaw_fault = next(fault for fault in DEFAULT_FAULTS if fault.kind is FaultKind.YAW_FIN_STUCK)
    pitch_fault = next(fault for fault in DEFAULT_FAULTS if fault.kind is FaultKind.PITCH_FIN_STUCK)

    yaw_assessment = checker.assess(
        state,
        world,
        vehicle,
        faults=(yaw_fault,),
        diagnosis_delays_s=(0.0,),
    )
    pitch_assessment = checker.assess(
        state,
        world,
        vehicle,
        faults=(pitch_fault,),
        diagnosis_delays_s=(0.0,),
    )

    yaw_check = yaw_assessment.fault_checks[0]
    pitch_check = pitch_assessment.fault_checks[0]
    assert "yaw_fin" not in yaw_check.capability.available_actuators
    assert "minimum_radius_turn_left" in yaw_check.unavailable_maneuvers
    assert "minimum_radius_turn_right" in yaw_check.unavailable_maneuvers
    assert "enter_safe_corridor" in yaw_check.unavailable_maneuvers
    assert "pitch_fin" not in pitch_check.capability.available_actuators
    assert "ascend" in pitch_check.unavailable_maneuvers
    assert "descend" in pitch_check.unavailable_maneuvers
    assert "enter_safe_corridor" in pitch_check.unavailable_maneuvers


def test_full_hull_and_point_diagnostics_disagree_near_an_obstacle() -> None:
    obstacle = Obstacle("nearby", "sphere", np.array([0.85, 0.0, 0.0]), radius_m=0.1)
    world = WorldSnapshot(
        t_s=0.0,
        obstacle_map=ObstacleMap(
            bounds_ned_m=np.array([[-4.0, 4.0], [-4.0, 4.0], [-4.0, 4.0]]),
            obstacles=(obstacle,),
        ),
        current_field=UniformCurrent(velocity_ned_mps=np.zeros(3)),
    )
    state = State(
        t_s=0.0,
        position_ned_m=np.zeros(3),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        linear_velocity_body_mps=np.array([1.0, 0.0, 0.0]),
        angular_velocity_body_rps=np.zeros(3),
    )
    vehicle = remus100_vehicle()
    checker = RecoverabilityChecker()
    full = checker.assess(
        state,
        world,
        vehicle,
        config=RecoveryConfig(),
        faults=(DEFAULT_FAULTS[0],),
        diagnosis_delays_s=(0.0,),
    )
    point = checker.assess(
        state,
        world,
        vehicle,
        config=RecoveryConfig(clearance_model=ClearanceModel.POINT_DIAGNOSTIC),
        faults=(DEFAULT_FAULTS[0],),
        diagnosis_delays_s=(0.0,),
    )

    assert full.current_clearance_m < 0.0
    assert point.current_clearance_m > 0.0


def test_certificate_results_are_deterministic_excluding_runtime() -> None:
    state, world, vehicle = _open_state()
    checker = RecoverabilityChecker()
    first = checker.assess(
        state,
        world,
        vehicle,
        faults=DEFAULT_FAULTS[:2],
        diagnosis_delays_s=(0.0, 1.0),
    )
    second = checker.assess(
        state,
        world,
        vehicle,
        faults=DEFAULT_FAULTS[:2],
        diagnosis_delays_s=(0.0, 1.0),
    )

    assert asdict(first.certificate) == asdict(second.certificate)


def test_m6_config_manifest_matches_committed_file() -> None:
    committed = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert committed == m6_config_manifest()
    assert len(m6_key_nodes()) == 7
