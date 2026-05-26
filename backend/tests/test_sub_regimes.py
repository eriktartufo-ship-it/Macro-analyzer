"""Test Tier 6.8 — Hierarchical sub-regime classifier."""
from __future__ import annotations

import pytest

from app.services.regime.sub_regimes import (
    SubRegimeAssessment,
    classify_sub_regime,
    list_sub_regimes_for,
)


class TestListSubRegimes:
    def test_reflation_three(self):
        assert list_sub_regimes_for("reflation") == ["early_cycle", "mid_cycle", "late_cycle"]

    def test_stagflation_three(self):
        assert list_sub_regimes_for("stagflation") == ["oil_shock", "debasement", "wage_spiral"]

    def test_deflation_three(self):
        assert list_sub_regimes_for("deflation") == [
            "financial_crisis", "demand_shock", "disinflation"
        ]

    def test_goldilocks_two(self):
        assert list_sub_regimes_for("goldilocks") == ["complacency", "steady_state"]

    def test_unknown_empty(self):
        assert list_sub_regimes_for("nope") == []


class TestDispatch:
    def test_unknown_regime_returns_none(self):
        assert classify_sub_regime("phantom", {}) is None

    def test_returns_assessment_object(self):
        out = classify_sub_regime("reflation", {"gdp_roc": 2.0})
        assert isinstance(out, SubRegimeAssessment)
        assert out.primary_regime == "reflation"
        assert out.sub_regime in {"early_cycle", "mid_cycle", "late_cycle"}


class TestReflationSubRegimes:
    def test_early_cycle_high_unrate_falling_claims(self):
        """Recovery: unrate ancora alta in calo, fed accomod, yield steep."""
        out = classify_sub_regime("reflation", {
            "gdp_roc": 1.5,
            "unrate": 5.4,
            "fed_funds": 1.5,
            "yield_curve_spread": 1.8,
            "claims_roc": -3.0,
            "breakeven_10y": 2.0,
        })
        assert out.sub_regime == "early_cycle"

    def test_late_cycle_unrate_low_fed_high(self):
        """Late: unrate very low, fed restrictive, yield flat, breakeven >2.5."""
        out = classify_sub_regime("reflation", {
            "gdp_roc": 3.5,
            "unrate": 3.8,
            "fed_funds": 5.0,
            "yield_curve_spread": 0.2,
            "claims_roc": 1.0,
            "breakeven_10y": 2.6,
        })
        assert out.sub_regime == "late_cycle"

    def test_breakdown_keys_complete(self):
        out = classify_sub_regime("reflation", {"gdp_roc": 2.0})
        assert set(out.breakdown.keys()) == {"early_cycle", "mid_cycle", "late_cycle"}


class TestStagflationSubRegimes:
    def test_oil_shock_high_energy(self):
        out = classify_sub_regime("stagflation", {
            "cpi_yoy": 8.5,
            "energy_yoy": 40.0,
            "gold_oil_ratio": 9.0,
            "dedollar_combined": 0.4,
            "payrolls_yoy": 1.0,
            "unrate": 5.5,
        })
        assert out.sub_regime == "oil_shock"

    def test_debasement_dollar_weak_gold_strong(self):
        out = classify_sub_regime("stagflation", {
            "cpi_yoy": 5.5,
            "energy_yoy": 5.0,
            "gold_oil_ratio": 35.0,
            "dedollar_combined": 0.85,
            "payrolls_yoy": 1.5,
            "unrate": 4.5,
        })
        assert out.sub_regime == "debasement"

    def test_wage_spiral_tight_labor(self):
        out = classify_sub_regime("stagflation", {
            "cpi_yoy": 5.0,
            "energy_yoy": 2.0,
            "gold_oil_ratio": 18.0,
            "dedollar_combined": 0.3,
            "payrolls_yoy": 4.0,
            "unrate": 3.4,
        })
        assert out.sub_regime == "wage_spiral"


class TestDeflationSubRegimes:
    def test_financial_crisis_2008_style(self):
        out = classify_sub_regime("deflation", {
            "baa_10y_spread": 5.5,
            "vix": 60.0,
            "nfci": 1.5,
            "gdp_roc": -1.0,
            "unrate_roc": 0.5,
            "claims_roc": 8.0,
            "cpi_yoy": 1.5,
        })
        assert out.sub_regime == "financial_crisis"

    def test_demand_shock_2020_style(self):
        out = classify_sub_regime("deflation", {
            "baa_10y_spread": 2.5,
            "vix": 30.0,
            "nfci": 0.3,
            "gdp_roc": -8.0,
            "unrate_roc": 5.0,
            "claims_roc": 60.0,
            "cpi_yoy": 1.0,
        })
        assert out.sub_regime == "demand_shock"

    def test_disinflation_soft_no_crisis(self):
        out = classify_sub_regime("deflation", {
            "baa_10y_spread": 1.5,
            "vix": 16.0,
            "nfci": -0.1,
            "gdp_roc": 1.0,
            "unrate_roc": 0.05,
            "claims_roc": 0.5,
            "cpi_yoy": 0.5,
        })
        assert out.sub_regime == "disinflation"


class TestGoldilocksSubRegimes:
    def test_complacency_vix_floor(self):
        out = classify_sub_regime("goldilocks", {
            "vix": 9.5,
            "nfci": -0.7,
            "cpi_yoy": 2.0,
            "unrate": 3.5,
            "gdp_roc": 2.5,
        })
        assert out.sub_regime == "complacency"

    def test_steady_state_all_in_range(self):
        out = classify_sub_regime("goldilocks", {
            "vix": 17.0,
            "nfci": -0.2,
            "cpi_yoy": 2.0,
            "unrate": 4.2,
            "gdp_roc": 2.2,
        })
        assert out.sub_regime == "steady_state"

    def test_breakdown_two_categories(self):
        out = classify_sub_regime("goldilocks", {})
        assert set(out.breakdown.keys()) == {"complacency", "steady_state"}


class TestRobustness:
    def test_missing_indicators_no_crash(self):
        # Tutti i sub-classifier devono gestire dict vuoto
        for regime in ("reflation", "stagflation", "deflation", "goldilocks"):
            out = classify_sub_regime(regime, {})
            assert out is not None
            assert out.primary_regime == regime

    def test_non_numeric_indicator_safe(self):
        out = classify_sub_regime("reflation", {"gdp_roc": "n/a"})
        assert out is not None

    def test_nan_indicator_safe(self):
        """T7.5: NaN deve essere trattato come missing → default neutro."""
        import math
        out = classify_sub_regime("stagflation", {
            "gold_oil_ratio": float("nan"),
            "cpi_yoy": 8.0,
            "energy_yoy": 40.0,
        })
        for v in out.breakdown.values():
            assert not math.isnan(v)

    def test_inf_indicator_safe(self):
        """T7.5: Inf deve essere trattato come missing."""
        import math
        out = classify_sub_regime("deflation", {
            "vix": float("inf"),
            "gdp_roc": -5.0,
        })
        for v in out.breakdown.values():
            assert not math.isinf(v)
