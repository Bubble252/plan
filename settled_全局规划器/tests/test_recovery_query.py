from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from auv_planner.contracts import Obstacle, ObstacleMap, State, UniformCurrent, WorldSnapshot
from auv_planner.experiments.m6_recovery import dead_end_scene
from auv_planner.experiments.m7_budget import (
    _audit_full_cached,
    _audit_states,
    m7_budget_manifest,
    m7_probes,
)
from auv_planner.recovery import (
    DEFAULT_FAULT_PROFILES,
    P0_NOMINAL,
    P1_DEGRADED,
    P2_M6_STRESS,
    FaultProfile,
    RecoverabilityChecker,
    RecoveryConfig,
    RecoveryQueryMode,
    RecoveryQueryPolicy,
    RecoveryQuerySession,
    RecoveryQueryStatus,
)
from auv_planner.scenarios import m5_open_water_scene, remus100_vehicle

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "configs" / "experiments" / "m7_budget.json"


def test_fault_profiles_are_versioned_and_have_expected_check_counts() -> None:
    assert [profile.name for profile in DEFAULT_FAULT_PROFILES] == [
        "P0_nominal",
        "P1_degraded",
        "P2_m6_stress",
    ]
    assert P0_NOMINAL.check_count == 1
    assert P1_DEGRADED.check_count == 8
    assert P2_M6_STRESS.check_count == 24
    assert (
        P0_NOMINAL.digest
        == FaultProfile(
            name=P0_NOMINAL.name,
            faults=P0_NOMINAL.faults,
            diagnosis_delays_s=P0_NOMINAL.diagnosis_delays_s,
            description=P0_NOMINAL.description,
            version=P0_NOMINAL.version,
        ).digest
    )
    assert len(P1_DEGRADED.digest) == 64


def test_checker_records_profile_metadata_on_certificate_and_diagnostics() -> None:
    scenario = m5_open_water_scene()
    assessment = RecoverabilityChecker().assess(
        scenario.start,
        scenario.world,
        scenario.vehicle,
        config=RecoveryConfig(),
        profile=P1_DEGRADED,
    )

    assert assessment.profile is P1_DEGRADED
    assert assessment.certificate.profile_name == P1_DEGRADED.name
    assert assessment.certificate.profile_digest == P1_DEGRADED.digest
    assert assessment.certificate.diagnostics["profile_check_count"] == 8


def test_query_cache_reuses_identical_state_and_invalidates_on_world_change() -> None:
    scenario = m5_open_water_scene()
    session = RecoveryQuerySession(
        policy=RecoveryQueryPolicy(budget_s=10.0, use_cache=True, use_prefilter=True)
    )
    first = session.query(
        scenario.start,
        scenario.world,
        scenario.vehicle,
        config=RecoveryConfig(),
        profile=P0_NOMINAL,
    )
    second = session.query(
        scenario.start,
        scenario.world,
        scenario.vehicle,
        config=RecoveryConfig(),
        profile=P0_NOMINAL,
    )
    changed_world = replace(
        scenario.world,
        current_field=UniformCurrent(velocity_ned_mps=np.array([0.0, 0.3, 0.0])),
    )
    third = session.query(
        scenario.start,
        changed_world,
        scenario.vehicle,
        config=RecoveryConfig(),
        profile=P0_NOMINAL,
    )

    assert first.status is RecoveryQueryStatus.VALID
    assert not first.cache_hit
    assert second.cache_hit
    assert not third.cache_hit
    assert third.query_key != first.query_key
    assert session.stats.cache_hits == 1
    assert session.stats.full_assessments == 2


def test_prefilter_has_no_false_negatives_on_smoke_cases() -> None:
    open_scene = m5_open_water_scene()
    dead_end = dead_end_scene()
    near_obstacle = WorldSnapshot(
        t_s=0.0,
        obstacle_map=ObstacleMap(
            bounds_ned_m=np.array([[-4.0, 4.0], [-4.0, 4.0], [-4.0, 4.0]]),
            obstacles=(Obstacle("nearby", "sphere", np.array([0.85, 0.0, 0.0]), radius_m=0.1),),
        ),
        current_field=UniformCurrent(velocity_ned_mps=np.zeros(3)),
    )
    cases = (
        (open_scene.start, open_scene.world, open_scene.vehicle),
        (dead_end.start, dead_end.world, dead_end.vehicle),
        (
            State(
                t_s=0.0,
                position_ned_m=np.zeros(3),
                quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
                linear_velocity_body_mps=np.array([1.0, 0.0, 0.0]),
                angular_velocity_body_rps=np.zeros(3),
            ),
            near_obstacle,
            remus100_vehicle(),
        ),
    )
    checker = RecoverabilityChecker()
    for state, world, vehicle in cases:
        for profile in (P0_NOMINAL, P1_DEGRADED):
            direct = checker.assess(
                state,
                world,
                vehicle,
                config=RecoveryConfig(),
                profile=profile,
            )
            queried = RecoveryQuerySession(
                policy=RecoveryQueryPolicy(budget_s=None, use_cache=True, use_prefilter=True)
            ).query(
                state,
                world,
                vehicle,
                config=RecoveryConfig(),
                profile=profile,
            )
            if queried.prefilter_reason is not None:
                assert not direct.valid
            assert (queried.status is RecoveryQueryStatus.VALID) == direct.valid


def test_budget_exhaustion_is_unknown_not_infeasible() -> None:
    scenario = m5_open_water_scene()
    session = RecoveryQuerySession(policy=RecoveryQueryPolicy(budget_s=1e-12))
    result = session.query(
        scenario.start,
        scenario.world,
        scenario.vehicle,
        config=RecoveryConfig(),
        profile=P0_NOMINAL,
    )

    assert result.status is RecoveryQueryStatus.UNKNOWN
    assert result.budget_exhausted
    assert result.assessment is None


def test_m7_budget_manifest_includes_all_profiles_and_probes() -> None:
    manifest = m7_budget_manifest()
    assert manifest["online_budget_s"] == 10.0
    assert len(manifest["profiles"]) == 3
    assert len(manifest["probes"]) == 9
    assert manifest["model"]["certificate_validity_s"] == pytest.approx(1.0)
    assert any(not probe["include_in_aggregate"] for probe in manifest["probes"])


def test_m7_budget_manifest_matches_committed_file() -> None:
    assert json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) == m7_budget_manifest()


def test_path_audit_clamps_coverage_and_reports_warm_cache_per_pass() -> None:
    probe = m7_probes()[0]
    config = RecoveryConfig()
    states = _audit_states(probe, config)
    assert all(
        later.t_s - earlier.t_s <= 1.0 + 1e-12
        for earlier, later in zip(states, states[1:], strict=False)
    )
    rows = _audit_full_cached(probe, P0_NOMINAL, states, config)
    cold, warm = rows
    assert 0.0 <= float(cold["coverage_ratio"]) <= 1.0
    assert 0.0 <= float(warm["coverage_ratio"]) <= 1.0
    assert int(cold["cache_hits"]) == 0
    assert int(cold["full_assessments"]) > 0
    assert int(warm["cache_hits"]) == len(states)
    assert int(warm["full_assessments"]) == 0


def test_online_fast_stops_after_first_failure_without_issuing_certificate() -> None:
    scenario = dead_end_scene()
    checker = RecoverabilityChecker()
    full = checker.assess(
        scenario.start,
        scenario.world,
        scenario.vehicle,
        config=RecoveryConfig(),
        profile=P1_DEGRADED,
    )
    fast = checker.assess(
        scenario.start,
        scenario.world,
        scenario.vehicle,
        config=RecoveryConfig(),
        profile=P1_DEGRADED,
        early_stop=True,
    )

    assert not full.valid
    assert not fast.valid
    assert fast.early_stopped
    assert len(fast.fault_checks) < len(full.fault_checks)
    assert fast.certificate.reason is not None
    assert fast.certificate.reason.startswith("no_recoverable_backup:")


def test_parallel_full_assessment_matches_sequential_result() -> None:
    probe = next(probe for probe in m7_probes() if probe.name == "open_water_start")
    checker = RecoverabilityChecker()
    sequential = checker.assess(
        probe.state,
        probe.scenario.world,
        probe.scenario.vehicle,
        config=RecoveryConfig(),
        profile=P1_DEGRADED,
        safe_anchor_ned_m=probe.safe_anchor_ned_m,
        parallel_workers=1,
    )
    parallel = checker.assess(
        probe.state,
        probe.scenario.world,
        probe.scenario.vehicle,
        config=RecoveryConfig(),
        profile=P1_DEGRADED,
        safe_anchor_ned_m=probe.safe_anchor_ned_m,
        parallel_workers=2,
    )

    assert parallel.valid == sequential.valid
    assert [check.label for check in parallel.fault_checks] == [
        check.label for check in sequential.fault_checks
    ]
    assert parallel.certificate.reason == sequential.certificate.reason


def test_online_fast_query_mode_is_explicit_and_sequential_only() -> None:
    scenario = dead_end_scene()
    session = RecoveryQuerySession(
        policy=RecoveryQueryPolicy(
            budget_s=10.0,
            mode=RecoveryQueryMode.ONLINE_FAST,
        )
    )
    result = session.query(
        scenario.start,
        scenario.world,
        scenario.vehicle,
        config=RecoveryConfig(),
        profile=P1_DEGRADED,
    )

    assert result.status is RecoveryQueryStatus.INVALID
    assert result.mode is RecoveryQueryMode.ONLINE_FAST
    assert result.early_stopped
    assert 0 < result.evaluated_checks < P1_DEGRADED.check_count
    with pytest.raises(ValueError, match="sequential"):
        RecoveryQueryPolicy(
            mode=RecoveryQueryMode.ONLINE_FAST,
            parallel_workers=2,
        )
