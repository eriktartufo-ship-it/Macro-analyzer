"""Test Tier 5.1 LOOP CLOSURE — apply_crisis_modulation.

Verifica:
- Flag OFF (default): identity (probs invariate).
- Low risk (<0.3): no-op anche con flag ON.
- 2008-style scenario → boost deflation.
- 1973-style scenario → boost stagflation.
- Bubble scenario → small shift verso defl + stag.
- Probs sempre sum=1, non-negative, no key dropping.
- Renormalizzazione robusta (others_sum=0, drift float).
"""
from __future__ import annotations

import pytest

from app.services.regime.crisis_modulation import (
    _shift_toward,
    apply_crisis_modulation,
)


def _approx_sum_one(probs: dict, tol: float = 1e-6) -> bool:
    return abs(sum(probs.values()) - 1.0) < tol


class TestApplyCrisisModulationFlagGating:
    def test_flag_off_is_identity(self, monkeypatch):
        monkeypatch.delenv("USE_CRISIS_MODULATION", raising=False)
        probs = {"reflation": 0.4, "stagflation": 0.2, "deflation": 0.2, "goldilocks": 0.2}
        ind = {"gdp_roc": -3.0, "cpi_yoy": 1.0, "vix": 45.0, "baa_spread": 5.5,
               "yield_curve_10y2y": -0.5, "breakeven_10y": 1.2, "nfci": 1.5,
               "unrate_roc": 80.0}
        out = apply_crisis_modulation(probs, ind, dedollar_combined=0.4)
        assert out == probs

    def test_flag_on_low_risk_is_identity(self, monkeypatch):
        """Anche con flag ON, se risk<0.3 → no-op."""
        monkeypatch.setenv("USE_CRISIS_MODULATION", "1")
        probs = {"reflation": 0.4, "stagflation": 0.2, "deflation": 0.2, "goldilocks": 0.2}
        # Scenario calmo → risk score basso
        ind = {"gdp_roc": 2.0, "cpi_yoy": 2.2, "vix": 17.0, "baa_spread": 1.8,
               "yield_curve_10y2y": 0.8, "breakeven_10y": 2.0, "nfci": -0.3,
               "unrate_roc": 0.0}
        out = apply_crisis_modulation(probs, ind, dedollar_combined=0.4)
        # Differenza trascurabile
        for k in probs:
            assert abs(out[k] - probs[k]) < 1e-9


class TestApplyCrisisModulationScenarios:
    @pytest.fixture(autouse=True)
    def _isolate_other_flags(self, monkeypatch):
        for k in ("USE_ML_REGIME_BLEND", "USE_FRESHNESS_WEIGHTING", "USE_GDP_COLLAPSE_OVERRIDE"):
            monkeypatch.delenv(k, raising=False)

    def test_2008_scenario_boosts_deflation(self, monkeypatch):
        monkeypatch.setenv("USE_CRISIS_MODULATION", "1")
        probs = {"reflation": 0.15, "stagflation": 0.15, "deflation": 0.50, "goldilocks": 0.20}
        ind = {"gdp_roc": -3.0, "cpi_yoy": 1.0, "vix": 45.0, "baa_spread": 5.5,
               "yield_curve_10y2y": -0.5, "breakeven_10y": 1.2, "nfci": 1.5,
               "unrate_roc": 80.0, "unrate": 8.0}
        out = apply_crisis_modulation(probs, ind, dedollar_combined=0.4)
        assert out["deflation"] > probs["deflation"]
        assert _approx_sum_one(out)

    def test_1973_scenario_boosts_stagflation(self, monkeypatch):
        monkeypatch.setenv("USE_CRISIS_MODULATION", "1")
        probs = {"reflation": 0.15, "stagflation": 0.55, "deflation": 0.10, "goldilocks": 0.20}
        ind = {"gdp_roc": -2.0, "cpi_yoy": 10.0, "vix": 22.0, "baa_spread": 2.5,
               "yield_curve_10y2y": 0.2, "breakeven_10y": 3.5, "nfci": 0.3,
               "unrate_roc": 50.0, "unrate": 7.0}
        out = apply_crisis_modulation(probs, ind, dedollar_combined=0.75)
        assert out["stagflation"] > probs["stagflation"]
        assert _approx_sum_one(out)

    def test_bubble_scenario_shifts_toward_defl_stag(self, monkeypatch):
        monkeypatch.setenv("USE_CRISIS_MODULATION", "1")
        # Goldilocks/reflation dominante, VIX compresso, no stress macro.
        # Indicators completi per garantire detection bubble_goldilocks
        # (richiede benign_dom>0.45 + vix_compressed trigger).
        probs = {"reflation": 0.30, "stagflation": 0.05, "deflation": 0.05, "goldilocks": 0.60}
        ind = {
            "gdp_roc": 2.5, "cpi_yoy": 1.8, "vix": 11.0, "baa_spread": 1.4,
            "yield_curve_10y2y": 0.8, "breakeven_10y": 2.0, "nfci": -0.5,
            "unrate_roc": 0.0, "unrate": 3.5, "gdp_yoy_dfm": 2.6,
            "core_pce_yoy": 1.7, "consumer_sentiment": 95.0,
            "payrolls_roc_12m": 2.0, "indpro_roc_12m": 2.5,
        }
        out = apply_crisis_modulation(probs, ind, dedollar_combined=0.3)
        # Defl + stag dovrebbero aumentare entrambi (preludio downside)
        defl_delta = out["deflation"] - probs["deflation"]
        stag_delta = out["stagflation"] - probs["stagflation"]
        assert defl_delta > 0 or stag_delta > 0  # almeno uno boosta
        assert _approx_sum_one(out)


class TestApplyCrisisModulationInvariants:
    def test_probs_sum_to_one_after_modulation(self, monkeypatch):
        monkeypatch.setenv("USE_CRISIS_MODULATION", "1")
        probs = {"reflation": 0.25, "stagflation": 0.25, "deflation": 0.25, "goldilocks": 0.25}
        ind = {"gdp_roc": -3.0, "cpi_yoy": 1.0, "vix": 45.0, "baa_spread": 5.5,
               "yield_curve_10y2y": -0.5, "breakeven_10y": 1.2, "nfci": 1.5,
               "unrate_roc": 80.0}
        out = apply_crisis_modulation(probs, ind, dedollar_combined=0.4)
        assert _approx_sum_one(out)

    def test_no_negative_probs(self, monkeypatch):
        monkeypatch.setenv("USE_CRISIS_MODULATION", "1")
        # Probs estreme: solo deflation 90%, altri 3.33% each
        probs = {"reflation": 0.033, "stagflation": 0.033, "deflation": 0.90, "goldilocks": 0.034}
        ind = {"gdp_roc": -3.0, "cpi_yoy": 1.0, "vix": 45.0, "baa_spread": 5.5,
               "yield_curve_10y2y": -0.5, "breakeven_10y": 1.2, "nfci": 1.5,
               "unrate_roc": 80.0}
        out = apply_crisis_modulation(probs, ind, dedollar_combined=0.4)
        for k, v in out.items():
            assert v >= 0.0, f"{k} negativa: {v}"

    def test_all_keys_preserved(self, monkeypatch):
        monkeypatch.setenv("USE_CRISIS_MODULATION", "1")
        probs = {"reflation": 0.25, "stagflation": 0.25, "deflation": 0.25, "goldilocks": 0.25}
        ind = {"gdp_roc": -2.0, "cpi_yoy": 10.0, "vix": 22.0, "baa_spread": 2.5,
               "yield_curve_10y2y": 0.2, "breakeven_10y": 3.5, "nfci": 0.3,
               "unrate_roc": 50.0}
        out = apply_crisis_modulation(probs, ind, dedollar_combined=0.75)
        assert set(out.keys()) == set(probs.keys())

    def test_alpha_caps_shift(self, monkeypatch):
        """Con alpha=0.05 e risk=1.0, shift singolo non supera ~5pp."""
        monkeypatch.setenv("USE_CRISIS_MODULATION", "1")
        probs = {"reflation": 0.25, "stagflation": 0.25, "deflation": 0.25, "goldilocks": 0.25}
        # Scenario estremo per portare risk vicino a 1.0
        ind = {"gdp_roc": -5.0, "cpi_yoy": 0.5, "vix": 50.0, "baa_spread": 6.0,
               "yield_curve_10y2y": -1.0, "breakeven_10y": 1.0, "nfci": 2.0,
               "unrate_roc": 100.0, "unrate": 10.0}
        out = apply_crisis_modulation(probs, ind, dedollar_combined=0.4, alpha=0.05)
        # Boost massimo a deflation: 0.25 + 0.05 * 1.0 = 0.30, ma probs renormalizzato
        # quindi può essere leggermente > 0.30 dopo renorm. Cap soft <0.40.
        assert out["deflation"] - probs["deflation"] < 0.10


class TestShiftToward:
    def test_basic_shift_redistributes(self):
        probs = {"a": 0.5, "b": 0.3, "c": 0.2}
        _shift_toward(probs, target="a", amount=0.1)
        assert probs["a"] == pytest.approx(0.6, abs=1e-9)
        # b, c sottratti proporzionalmente: 0.3 e 0.2 vanno a 0.4 totale
        # (0.3+0.2-0.1)/0.5 = 0.4/0.5 = 0.8 factor
        assert probs["b"] == pytest.approx(0.3 * 0.8, abs=1e-9)
        assert probs["c"] == pytest.approx(0.2 * 0.8, abs=1e-9)

    def test_unknown_target_noop(self):
        probs = {"a": 0.5, "b": 0.5}
        before = dict(probs)
        _shift_toward(probs, target="zzz", amount=0.1)
        assert probs == before

    def test_amount_capped_at_others_99pct(self):
        """Non possiamo prelevare più del 99% degli altri (floor di sicurezza)."""
        probs = {"a": 0.1, "b": 0.9}
        _shift_toward(probs, target="a", amount=0.95)  # > 99% di 0.9
        # Cap: actual = min(0.95, 0.9*0.99) = 0.891
        assert probs["a"] == pytest.approx(0.1 + 0.891, abs=1e-6)
        assert probs["b"] > 0  # non azzerato

    def test_others_zero_just_adds(self):
        probs = {"a": 0.5, "b": 0.0, "c": 0.0}
        _shift_toward(probs, target="a", amount=0.2)
        # b, c zero: aggiungiamo amount al target ma clip a 1.0
        assert probs["a"] == pytest.approx(0.7, abs=1e-9)
