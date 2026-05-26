"""Test circuit-breakers."""
from __future__ import annotations

import pytest

from app.services.validation.circuit_breakers import (
    check_consecutive_wrong,
    check_low_confidence,
    check_ood_distribution,
    compute_mahalanobis_distance,
    evaluate_all_circuit_breakers,
    reset_ood_cache,
)


class TestLowConfidence:
    def test_low_confidence_triggers_warning(self):
        out = check_low_confidence(max_prob=0.30)
        assert out.is_triggered is True
        assert out.severity == "warning"

    def test_high_confidence_no_trigger(self):
        out = check_low_confidence(max_prob=0.80)
        assert out.is_triggered is False
        assert out.severity == "info"

    def test_threshold_exact_boundary(self):
        # 0.45 exact: NOT trigger (< is strict)
        out = check_low_confidence(max_prob=0.45)
        assert out.is_triggered is False


class TestOODDetector:
    def setup_method(self):
        reset_ood_cache()

    def test_in_sample_indicators_low_distance(self):
        """Indicators tipici training (es. medi) → low Mahalanobis."""
        normal_ind = {
            "gdp_roc": 2.5, "cpi_yoy": 2.5, "unrate": 5.0, "vix": 18.0,
            "fed_funds": 3.0, "breakeven_10y": 2.2, "baa_10y_spread": 2.0,
        }
        out = check_ood_distribution(normal_ind)
        # Distance non zero ma sotto soglia
        assert out.metric_value is not None
        assert out.metric_value < 10.0

    def test_extreme_indicators_high_distance(self):
        """Indicators estremi (Argentina-style hyperinflation): high Mahalanobis."""
        extreme_ind = {
            "gdp_roc": -15.0, "cpi_yoy": 60.0, "unrate": 22.0, "vix": 90.0,
            "fed_funds": 20.0, "breakeven_10y": 8.0, "baa_10y_spread": 10.0,
        }
        out = check_ood_distribution(extreme_ind)
        # Distance alta → trigger
        assert out.is_triggered is True
        assert out.severity == "critical"

    def test_compute_distance_returns_float(self):
        d = compute_mahalanobis_distance({"gdp_roc": 2.0, "cpi_yoy": 2.0})
        assert isinstance(d, float)
        assert d >= 0

    def test_nan_input_safe(self):
        """NaN nei indicators non rompe Mahalanobis."""
        ind = {"gdp_roc": float("nan"), "cpi_yoy": 2.0}
        d = compute_mahalanobis_distance(ind)
        import math
        assert not math.isnan(d)


class TestConsecutiveWrong:
    def test_two_consecutive_wrong_triggers(self):
        out = check_consecutive_wrong([False, False, True])
        assert out.is_triggered is True
        assert out.severity == "critical"

    def test_one_wrong_no_trigger(self):
        out = check_consecutive_wrong([False, True, True])
        assert out.is_triggered is False

    def test_all_correct_no_trigger(self):
        out = check_consecutive_wrong([True, True, True, True])
        assert out.is_triggered is False

    def test_empty_list_no_trigger(self):
        out = check_consecutive_wrong([])
        assert out.is_triggered is False


class TestEvaluateAllCircuitBreakers:
    def setup_method(self):
        reset_ood_cache()

    def test_normal_mode_when_all_green(self):
        normal_ind = {
            "gdp_roc": 2.5, "cpi_yoy": 2.0, "unrate": 4.5, "vix": 15.0,
            "fed_funds": 3.0, "breakeven_10y": 2.2, "baa_10y_spread": 1.8,
        }
        out = evaluate_all_circuit_breakers(
            max_prob=0.75, indicators=normal_ind,
            recent_predictions_correct=[True, True, True],
        )
        assert out["advisory_mode"] == "normal"
        assert out["any_triggered"] is False

    def test_uncertain_mode_when_low_confidence(self):
        normal_ind = {
            "gdp_roc": 2.5, "cpi_yoy": 2.0, "unrate": 4.5, "vix": 15.0,
            "fed_funds": 3.0, "breakeven_10y": 2.2, "baa_10y_spread": 1.8,
        }
        out = evaluate_all_circuit_breakers(
            max_prob=0.30, indicators=normal_ind,
            recent_predictions_correct=[True, True],
        )
        assert out["advisory_mode"] == "uncertain"
        assert out["any_triggered"] is True
        assert out["any_critical"] is False

    def test_frozen_mode_when_ood_critical(self):
        extreme_ind = {
            "gdp_roc": -15.0, "cpi_yoy": 60.0, "unrate": 22.0, "vix": 90.0,
        }
        out = evaluate_all_circuit_breakers(
            max_prob=0.75, indicators=extreme_ind,
            recent_predictions_correct=[True, True],
        )
        assert out["advisory_mode"] == "frozen"
        assert out["any_critical"] is True

    def test_frozen_when_consecutive_wrong(self):
        normal_ind = {
            "gdp_roc": 2.5, "cpi_yoy": 2.0, "unrate": 4.5, "vix": 15.0,
        }
        out = evaluate_all_circuit_breakers(
            max_prob=0.80, indicators=normal_ind,
            recent_predictions_correct=[False, False, True],
        )
        assert out["advisory_mode"] == "frozen"

    def test_signals_list_complete(self):
        ind = {"gdp_roc": 2.5, "cpi_yoy": 2.0}
        out = evaluate_all_circuit_breakers(
            max_prob=0.5, indicators=ind,
            recent_predictions_correct=[True],
        )
        assert len(out["signals"]) == 3
        names = {s.name for s in out["signals"]}
        assert names == {"low_confidence", "ood_distribution", "consecutive_wrong"}
