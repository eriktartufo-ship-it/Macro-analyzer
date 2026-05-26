"""Test crisis risk indicator (3 tipi di crisi + positioning).

Verifica:
- 3 scenari (deflation crash 2008, stagflation 1973, bubble 2000) producono
  il tipo corretto di crisi + positioning coerente con teoria.
- Trigger evaluation logic.
- Summary IT corretto.
"""
from __future__ import annotations

import pytest

from app.services.regime.crisis_indicator import (
    _classify_crisis_type,
    _compute_risk_level,
    _evaluate_triggers,
    assess_crisis_risk,
)


class TestEvaluateTriggers:
    def test_normal_indicators_no_triggers_fired(self):
        ind = {
            "gdp_roc": 2.0, "cpi_yoy": 2.0, "unrate": 4.0, "unrate_roc": 0.0,
            "vix": 17.0, "baa_spread": 1.8, "yield_curve_10y2y": 0.8,
            "breakeven_10y": 2.0, "nfci": -0.3,
        }
        probs = {"reflation": 0.4, "stagflation": 0.2, "deflation": 0.15, "goldilocks": 0.25}
        triggers = _evaluate_triggers(ind, probs)
        # Pochi triggers fired in scenario benigno (probabilmente max 2-3 borderline)
        fired = [t for t in triggers if t["fired"]]
        assert len(fired) <= 3

    def test_deflation_crash_triggers_fire(self):
        """Scenario tipo 2008: VIX 45, BAA 5.5, yield invertita, GDP -3."""
        ind = {
            "gdp_roc": -3.0, "cpi_yoy": 1.0, "unrate": 7.0, "unrate_roc": 80.0,
            "vix": 45.0, "baa_spread": 5.5, "yield_curve_10y2y": -0.5,
            "breakeven_10y": 1.2, "nfci": 1.5,
        }
        probs = {"reflation": 0.05, "stagflation": 0.10, "deflation": 0.70, "goldilocks": 0.15}
        triggers = _evaluate_triggers(ind, probs)
        fired_names = {t["name"] for t in triggers if t["fired"]}
        # Devono accendersi i triggers deflazione
        assert "deflation_prob_high" in fired_names
        assert "vix_spike" in fired_names
        assert "credit_spread_extreme" in fired_names
        assert "yield_curve_inverted" in fired_names
        assert "cpi_collapse" in fired_names

    def test_stagflation_triggers_fire(self):
        """Scenario tipo 1973: CPI 10%, breakeven 3.5%, GDP -2, unrate rising."""
        ind = {
            "gdp_roc": -2.0, "cpi_yoy": 10.0, "unrate": 7.0, "unrate_roc": 50.0,
            "vix": 25.0, "baa_spread": 2.5, "yield_curve_10y2y": 0.2,
            "breakeven_10y": 3.5, "nfci": 0.4,
        }
        probs = {"reflation": 0.10, "stagflation": 0.65, "deflation": 0.10, "goldilocks": 0.15}
        triggers = _evaluate_triggers(ind, probs)
        fired_names = {t["name"] for t in triggers if t["fired"]}
        assert "stagflation_prob_high" in fired_names
        assert "cpi_hot" in fired_names
        assert "breakeven_high" in fired_names


class TestClassifyCrisisType:
    def test_deflation_crash_2008(self):
        ind = {
            "gdp_roc": -3.0, "cpi_yoy": 1.0, "vix": 45.0, "baa_spread": 5.5,
            "yield_curve_10y2y": -0.5, "breakeven_10y": 1.2, "nfci": 1.5,
            "unrate_roc": 80.0,
        }
        probs = {"reflation": 0.05, "stagflation": 0.10, "deflation": 0.70, "goldilocks": 0.15}
        triggers = _evaluate_triggers(ind, probs)
        crisis_type, conf, analog = _classify_crisis_type(probs, triggers, 0.5)
        assert crisis_type == "deflation_crash"
        assert conf > 0.5
        assert "2008" in analog or "2020" in analog

    def test_stagflation_debasement_1973(self):
        ind = {
            "gdp_roc": -2.0, "cpi_yoy": 10.0, "vix": 22.0, "baa_spread": 2.5,
            "yield_curve_10y2y": 0.2, "breakeven_10y": 3.5, "nfci": 0.3,
            "unrate_roc": 50.0,
        }
        probs = {"reflation": 0.10, "stagflation": 0.65, "deflation": 0.10, "goldilocks": 0.15}
        triggers = _evaluate_triggers(ind, probs)
        # Dedollar alto (debasement signal)
        crisis_type, conf, analog = _classify_crisis_type(probs, triggers, 0.75)
        assert crisis_type == "stagflation_debasement"
        assert "1973" in analog or "2022" in analog

    def test_bubble_goldilocks_2000(self):
        """Scenario complacency pre-burst: VIX 12, regime reflation/goldilocks
        dominante, no stress macro."""
        ind = {
            "gdp_roc": 2.5, "cpi_yoy": 2.0, "vix": 12.0, "baa_spread": 1.5,
            "yield_curve_10y2y": 0.8, "breakeven_10y": 2.0, "nfci": -0.4,
            "unrate_roc": 0.0,
        }
        probs = {"reflation": 0.30, "stagflation": 0.05, "deflation": 0.05, "goldilocks": 0.60}
        triggers = _evaluate_triggers(ind, probs)
        crisis_type, conf, analog = _classify_crisis_type(probs, triggers, 0.4)
        assert crisis_type == "bubble_goldilocks"
        assert "2000" in analog or "2021" in analog

    def test_no_crisis_normal_economy(self):
        ind = {
            "gdp_roc": 2.0, "cpi_yoy": 2.2, "vix": 17.0, "baa_spread": 1.8,
            "yield_curve_10y2y": 0.8, "breakeven_10y": 2.1, "nfci": -0.3,
            "unrate_roc": 0.0,
        }
        probs = {"reflation": 0.35, "stagflation": 0.20, "deflation": 0.15, "goldilocks": 0.30}
        triggers = _evaluate_triggers(ind, probs)
        crisis_type, _, _ = _classify_crisis_type(probs, triggers, 0.4)
        assert crisis_type == "no_crisis"


class TestComputeRiskLevel:
    def test_extreme_level_when_many_critical_fire(self):
        triggers = [
            {"name": "vix_spike", "fired": True},
            {"name": "credit_spread_extreme", "fired": True},
            {"name": "deflation_prob_high", "fired": True},
            {"name": "yield_curve_inverted", "fired": True},
            {"name": "vix_elevated", "fired": True},
            {"name": "credit_spread_wide", "fired": True},
            {"name": "cpi_collapse", "fired": True},
            {"name": "breakeven_low", "fired": True},
            {"name": "nfci_tight", "fired": True},
            {"name": "gdp_weak", "fired": True},
            {"name": "unrate_rising", "fired": True},
        ] + [{"name": f"other_{i}", "fired": False} for i in range(5)]
        level, score = _compute_risk_level(triggers, "deflation_crash")
        assert level == "extreme"
        assert score > 0.5

    def test_low_level_when_no_triggers(self):
        triggers = [{"name": f"trigger_{i}", "fired": False} for i in range(15)]
        level, score = _compute_risk_level(triggers, "no_crisis")
        assert level == "low"
        assert score == 0.0


class TestAssessCrisisRiskEndToEnd:
    """End-to-end tests con default flag OFF (baseline pre-T5/T6).

    Auto-isolate: ogni test garantisce env pulito per testare logica base.
    Test specifici per pillar opt-in vivono in test files dedicati.
    """

    @pytest.fixture(autouse=True)
    def _isolate_flags(self, monkeypatch):
        """T10b: USE_GDP_COLLAPSE_OVERRIDE e USE_ML_REGIME_BLEND default-ON. Force "0"."""
        for k in ("USE_GDP_COLLAPSE_OVERRIDE", "USE_ML_REGIME_BLEND"):
            monkeypatch.setenv(k, "0")
        for k in (
            "USE_DEDOLLAR_PILLAR", "USE_CRISIS_MODULATION", "USE_NEWS_PILLAR",
            "USE_DFM_ASSET_BONUS", "USE_ASSET_FEEDBACK",
            "USE_ADAPTIVE_THRESHOLDS", "USE_CORRELATION_REGIME",
            "USE_CROSS_ASSET_PILLARS",
            "USE_FRESHNESS_WEIGHTING",
        ):
            monkeypatch.delenv(k, raising=False)

    def test_full_2008_scenario(self):
        ind = {
            "gdp_roc": -3.0, "pmi": 35.0, "cpi_yoy": 1.0, "unrate": 8.0,
            "unrate_roc": 80.0, "yield_curve_10y2y": -0.5, "initial_claims_roc": 50.0,
            "lei_roc": -3.0, "fed_funds_rate": 1.0, "vix": 45.0, "baa_spread": 5.5,
            "breakeven_10y": 1.2, "nfci": 1.5,
        }
        result = assess_crisis_risk(ind, dedollar_combined=0.4)
        assert result.crisis_type == "deflation_crash"
        assert result.risk_level in {"elevated", "extreme"}
        # Positioning: bonds long overweight, equities underweight
        assert result.positioning["us_bonds_long"] == "overweight"
        assert result.positioning["us_equities_growth"] == "underweight"
        assert result.positioning["bitcoin"] == "underweight"

    def test_full_1973_scenario(self):
        ind = {
            "gdp_roc": -2.0, "pmi": 45.0, "cpi_yoy": 10.0, "unrate": 7.0,
            "unrate_roc": 50.0, "yield_curve_10y2y": 0.2, "initial_claims_roc": 10.0,
            "lei_roc": -2.0, "fed_funds_rate": 10.0, "core_pce_yoy": 8.0,
            "vix": 22.0, "baa_spread": 2.5, "breakeven_10y": 3.5, "nfci": 0.3,
        }
        result = assess_crisis_risk(ind, dedollar_combined=0.75)
        assert result.crisis_type == "stagflation_debasement"
        # Positioning: gold/commodities overweight, bonds long underweight
        assert result.positioning["gold"] == "overweight"
        assert result.positioning["us_bonds_long"] == "underweight"
        assert result.positioning["broad_commodities"] == "overweight"

    def test_full_bubble_scenario(self):
        ind = {
            "gdp_roc": 2.5, "pmi": 56.0, "cpi_yoy": 2.0, "unrate": 3.8,
            "unrate_roc": 0.0, "yield_curve_10y2y": 0.7, "initial_claims_roc": -3.0,
            "lei_roc": 1.0, "fed_funds_rate": 2.0, "vix": 12.0, "baa_spread": 1.5,
            "breakeven_10y": 2.0, "nfci": -0.4,
        }
        result = assess_crisis_risk(ind, dedollar_combined=0.4)
        assert result.crisis_type == "bubble_goldilocks"
        # Positioning: cash/short bonds overweight, growth underweight
        assert result.positioning["cash_money_market"] == "overweight"
        assert result.positioning["us_equities_growth"] == "underweight"

    def test_summary_in_italian(self):
        """summary deve essere in italiano + contenere il tipo di crisi."""
        ind = {"gdp_roc": -3.0, "vix": 45.0, "baa_spread": 5.5,
               "cpi_yoy": 1.0, "yield_curve_10y2y": -0.5, "nfci": 1.5,
               "breakeven_10y": 1.2, "unrate_roc": 80.0}
        result = assess_crisis_risk(ind, dedollar_combined=0.4)
        # Check parole italiane chiave
        s = result.summary.lower()
        assert "rischio" in s or "crisi" in s
        # Output struttura completa
        assert result.date
        assert isinstance(result.triggers, list)
        assert isinstance(result.positioning, dict)


class TestCorrelationBreakdownTrigger:
    """Tier 6.5 integration — correlation_breakdown trigger in crisis_indicator.

    Verifica:
    - Nessun correlation_assessment passato → trigger NON aggiunto (back-compat).
    - correlation_assessment con is_breakdown=True → trigger fired aggiunto.
    - correlation_assessment con is_breakdown=False → trigger NOT fired.
    """
    def _ind_normal(self):
        return {
            "gdp_roc": 2.0, "cpi_yoy": 2.0, "unrate": 4.0, "unrate_roc": 0.0,
            "vix": 17.0, "baa_spread": 1.8, "yield_curve_10y2y": 0.8,
            "breakeven_10y": 2.0, "nfci": -0.3,
        }

    def test_no_assessment_no_correlation_trigger(self):
        ind = self._ind_normal()
        result = assess_crisis_risk(ind, dedollar_combined=0.4)
        trigger_names = {t["name"] for t in result.triggers}
        assert "correlation_breakdown" not in trigger_names

    def test_breakdown_assessment_adds_fired_trigger(self):
        ind = self._ind_normal()
        # Mock minimale del dataclass CorrelationRegimeAssessment
        class FakeAssessment:
            avg_pairwise_correlation = 0.85
            is_breakdown = True
        result = assess_crisis_risk(
            ind, dedollar_combined=0.4, correlation_assessment=FakeAssessment()
        )
        triggers = {t["name"]: t for t in result.triggers}
        assert "correlation_breakdown" in triggers
        assert triggers["correlation_breakdown"]["fired"] is True
        assert triggers["correlation_breakdown"]["value"] == 0.85

    def test_non_breakdown_assessment_adds_unfired_trigger(self):
        ind = self._ind_normal()
        class FakeAssessment:
            avg_pairwise_correlation = 0.30
            is_breakdown = False
        result = assess_crisis_risk(
            ind, dedollar_combined=0.4, correlation_assessment=FakeAssessment()
        )
        triggers = {t["name"]: t for t in result.triggers}
        assert "correlation_breakdown" in triggers
        assert triggers["correlation_breakdown"]["fired"] is False
