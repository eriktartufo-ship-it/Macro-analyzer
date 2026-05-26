"""Test Tier 6.2 — Regime continuo 2D."""
from __future__ import annotations

import pytest

from app.services.regime.regime_2d import (
    Regime2DAssessment,
    assess_regime_2d,
    compute_from_indicators,
    compute_from_probs,
    nearest_quadrant,
)


class TestComputeFromProbs:
    def test_empty_returns_origin(self):
        assert compute_from_probs({}) == (0.0, 0.0)

    def test_pure_reflation_maps_to_pos_pos(self):
        out = compute_from_probs({"reflation": 1.0})
        assert out == (1.0, 1.0)

    def test_pure_stagflation_maps_to_neg_pos(self):
        out = compute_from_probs({"stagflation": 1.0})
        assert out == (-1.0, 1.0)

    def test_pure_deflation_maps_to_neg_neg(self):
        out = compute_from_probs({"deflation": 1.0})
        assert out == (-1.0, -1.0)

    def test_pure_goldilocks_maps_to_pos_neg(self):
        out = compute_from_probs({"goldilocks": 1.0})
        assert out == (1.0, -1.0)

    def test_uniform_blend_centered_origin(self):
        out = compute_from_probs(
            {"reflation": 0.25, "stagflation": 0.25, "deflation": 0.25, "goldilocks": 0.25}
        )
        assert out == (0.0, 0.0)

    def test_refl_gold_split(self):
        """50/50 refl/gold → +growth, 0 inflation."""
        out = compute_from_probs({"reflation": 0.5, "goldilocks": 0.5})
        assert out == (1.0, 0.0)

    def test_unknown_regime_ignored(self):
        out = compute_from_probs({"reflation": 0.5, "phantom": 0.5})
        assert out == (0.5, 0.5)


class TestComputeFromIndicators:
    def test_target_neutral_returns_origin(self):
        out = compute_from_indicators({"gdp_roc": 2.0, "cpi_yoy": 2.0})
        assert out == (0.0, 0.0)

    def test_strong_growth_positive_z(self):
        out = compute_from_indicators({"gdp_roc": 5.0, "cpi_yoy": 2.0})
        assert out[0] == pytest.approx(2.0)  # (5-2)/1.5
        assert out[1] == 0.0

    def test_high_inflation_positive_z(self):
        out = compute_from_indicators({"gdp_roc": 2.0, "cpi_yoy": 6.5})
        assert out[1] == pytest.approx(3.0)  # clamp upper

    def test_clamp_lower_bound(self):
        out = compute_from_indicators({"gdp_roc": -10.0, "cpi_yoy": -3.0})
        assert out[0] == -3.0
        assert out[1] == pytest.approx(-3.0)

    def test_missing_indicators_default_neutral(self):
        out = compute_from_indicators({})
        assert out == (0.0, 0.0)

    def test_non_numeric_safe(self):
        out = compute_from_indicators({"gdp_roc": "n/a", "cpi_yoy": None})
        assert out == (0.0, 0.0)

    def test_nan_safe(self):
        """T7.5 guard: NaN non deve propagare nei coord."""
        import math
        out = compute_from_indicators({"gdp_roc": float("nan"), "cpi_yoy": 2.0})
        assert not math.isnan(out[0])
        assert out == (0.0, 0.0)

    def test_inf_safe(self):
        """T7.5 guard: Inf non deve propagare."""
        import math
        out = compute_from_indicators({"gdp_roc": float("inf"), "cpi_yoy": 2.0})
        assert not math.isinf(out[0])
        assert out == (0.0, 0.0)


class TestNearestQuadrant:
    def test_origin_distances_equal(self):
        _, dists = nearest_quadrant(0.0, 0.0)
        sqrt2 = 2 ** 0.5
        for q in ("reflation", "stagflation", "deflation", "goldilocks"):
            assert dists[q] == pytest.approx(sqrt2, abs=1e-3)

    def test_point_in_reflation_quadrant(self):
        nq, _ = nearest_quadrant(0.7, 0.6)
        assert nq == "reflation"

    def test_point_in_deflation_quadrant(self):
        nq, _ = nearest_quadrant(-0.5, -0.8)
        assert nq == "deflation"


class TestAssessRegime2D:
    def test_probs_only_source(self):
        out = assess_regime_2d(probs={"reflation": 1.0})
        assert isinstance(out, Regime2DAssessment)
        assert out.source == "probs"
        assert out.growth_z == 1.0
        assert out.nearest_quadrant == "reflation"

    def test_indicators_only_source(self):
        out = assess_regime_2d(indicators={"gdp_roc": 4.0, "cpi_yoy": 5.0})
        assert out.source == "indicators"
        # gdp z=(4-2)/1.5=1.33 → norm 0.67; cpi z=2.0 → norm 1.0
        assert out.growth_z > 0
        assert out.inflation_z > 0
        assert out.nearest_quadrant == "reflation"

    def test_blended_source(self):
        out = assess_regime_2d(
            probs={"reflation": 0.6, "stagflation": 0.4},
            indicators={"gdp_roc": 1.5, "cpi_yoy": 4.5},
        )
        assert out.source == "blended"
        assert -1.0 <= out.growth_z <= 1.0
        assert -1.0 <= out.inflation_z <= 1.0

    def test_empty_inputs_returns_origin(self):
        out = assess_regime_2d()
        assert out.source == "empty"
        assert out.growth_z == 0.0
        assert out.inflation_z == 0.0

    def test_blend_zero_equals_probs(self):
        probs = {"reflation": 0.7, "deflation": 0.3}
        indicators = {"gdp_roc": 0.0, "cpi_yoy": 0.0}
        only_probs = assess_regime_2d(probs=probs)
        blended = assess_regime_2d(probs=probs, indicators=indicators, blend=0.0)
        assert blended.growth_z == pytest.approx(only_probs.growth_z, abs=1e-3)
        assert blended.inflation_z == pytest.approx(only_probs.inflation_z, abs=1e-3)

    def test_distances_all_present(self):
        out = assess_regime_2d(probs={"reflation": 1.0})
        assert set(out.distance_to_quadrants.keys()) == {
            "reflation", "stagflation", "deflation", "goldilocks"
        }
        assert out.distance_to_quadrants["reflation"] == 0.0
