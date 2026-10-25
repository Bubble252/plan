from __future__ import annotations

from typing import Any

from auv_planner.experiments.m5_validation import (
    proposed_model_thresholds,
    run_derivative_comparison,
)


def test_m5_derivative_differences_stay_within_audited_model_bands() -> None:
    rows: dict[str, dict[str, Any]] = {
        str(row["case"]): row for row in run_derivative_comparison()
    }
    calm = ("calm_straight", "calm_rudder", "calm_stern_plane", "combined_attitude", "depth_shear")
    for name in calm:
        assert float(rows[name]["relative_l2"]) < 0.07
    assert float(rows["cross_current"]["relative_l2"]) < 0.35


def test_m5_thresholds_are_proposed_not_paper_success_thresholds() -> None:
    thresholds = proposed_model_thresholds()
    assert thresholds["constraint_violations_max"] == 0.0
    assert "not yet" in str(thresholds["status"])
