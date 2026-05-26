"""Test Tier 7.1 — GDP collapse override per recessioni disinflazionistiche.

Coverage:
- Flag OFF: comportamento baseline invariato.
- Flag ON: 4 trigger composti (gdp/unrate_roc/vix/breakeven) fired → defl ↑.
- Flag ON: trigger NON fired → no shift.
- Scenari storici (2008, 2000, 1990) → predicted regime = deflation.
"""
from __future__ import annotations

import pytest

from app.services.regime.classifier import classify_regime


def _baseline_indicators() -> dict[str, float]:
    """Indicators neutri (goldilocks weak)."""
    return {
        "gdp_roc": 2.0,
        "pmi": 51.0,
        "cpi_yoy": 2.0,
        "unrate": 4.5,
        "unrate_roc": 0.0,
        "claims_roc": 0.0,
        "lei_roc": 0.5,
        "yield_curve_spread": 1.0,
        "fed_funds": 2.5,
        "payrolls_yoy": 1.8,
        "indpro_yoy": 2.0,
        "baa_10y_spread": 1.8,
        "housing_starts_yoy": 2.0,
        "nfci": -0.2,
        "vix": 16.0,
        "acm_term_premium": 0.0,
        "core_pce_yoy": 2.0,
        "consumer_sentiment": 75.0,
        "breakeven_10y": 2.0,
    }


def _disinflation_recession() -> dict[str, float]:
    """2008/2000/1990 style: CPI ancora elevato (lagging) + economy collapsing.

    cpi_yoy=4.5 (alto), gdp<1, unrate ↑, vix paura, breakeven crollato.
    Baseline classifier vede "stagflation". Con T7.1 vede "deflation".
    """
    return {
        "gdp_roc": -2.0,
        "pmi": 42.0,
        "cpi_yoy": 4.5,  # ancora alto, lagging
        "unrate": 6.5,
        "unrate_roc": 0.8,  # in salita
        "claims_roc": 15.0,
        "lei_roc": -3.0,
        "yield_curve_spread": -0.3,
        "fed_funds": 2.0,
        "payrolls_yoy": -1.0,
        "indpro_yoy": -4.0,
        "baa_10y_spread": 4.5,
        "housing_starts_yoy": -20.0,
        "nfci": 1.2,
        "vix": 38.0,  # paura
        "core_pce_yoy": 3.5,
        "consumer_sentiment": 55.0,
        "breakeven_10y": 1.5,  # mercato NON crede inflazione (crollato)
    }


class TestFlagGating:
    def test_flag_off_baseline_behavior(self, monkeypatch):
        """Flag OFF → comportamento baseline (no shift).

        T10b: flag promosso default-ON, disable esplicitamente per legacy test.
        """
        monkeypatch.setenv("USE_GDP_COLLAPSE_OVERRIDE", "0")
        out = classify_regime(_disinflation_recession())
        # Senza T7.1 questa configurazione è classificata stagflation/mixed
        # (CPI 4.5 → inflation_high fired, GDP < 1.5 → gdp_weak fired)
        assert "gdp_collapse_triggered" not in out

    def test_flag_on_triggered(self, monkeypatch):
        """Flag ON + scenario disinflation → trigger fired."""
        monkeypatch.setenv("USE_GDP_COLLAPSE_OVERRIDE", "1")
        out = classify_regime(_disinflation_recession())
        assert out.get("gdp_collapse_triggered") is True

    def test_flag_on_not_triggered_in_normal_regime(self, monkeypatch):
        """Flag ON + indicators normali → trigger NON fired."""
        monkeypatch.setenv("USE_GDP_COLLAPSE_OVERRIDE", "1")
        out = classify_regime(_baseline_indicators())
        assert "gdp_collapse_triggered" not in out


class TestRegimeShift:
    def test_disinflation_classified_as_deflation_with_flag(self, monkeypatch):
        """Con flag ON, lo scenario 2008/2000 viene classificato deflation."""
        monkeypatch.setenv("USE_GDP_COLLAPSE_OVERRIDE", "1")
        out = classify_regime(_disinflation_recession())
        # Top regime deve essere deflation
        assert out["regime"] == "deflation"
        assert out["probabilities"]["deflation"] > out["probabilities"]["stagflation"]

    def test_baseline_misclassifies_disinflation_as_stagflation(self, monkeypatch):
        """Senza T7.1 e senza T8.3 ML blend, lo stesso scenario va a stagflation
        (bug documentato che T7.1 e T8.3 risolvono indipendentemente).

        T10b: GDP_COLLAPSE + ML_BLEND promossi default-ON, disable esplicitamente.
        """
        monkeypatch.setenv("USE_GDP_COLLAPSE_OVERRIDE", "0")
        monkeypatch.setenv("USE_ML_REGIME_BLEND", "0")
        out = classify_regime(_disinflation_recession())
        # Verifico il bug noto del puro rule-based: stagflation domina
        assert out["probabilities"]["stagflation"] > out["probabilities"]["deflation"]


class TestTriggerConditions:
    """T9 continualizzato: severity sigmoid, non più AND-rigid.

    severity > 0.5 → triggered. Severity decresce in modo continuo se uno
    dei 4 indicators si avvicina al neutro.
    """

    def test_full_recession_severity_high(self, monkeypatch):
        """Tutti 4 trigger estremi → severity > 0.7."""
        monkeypatch.setenv("USE_GDP_COLLAPSE_OVERRIDE", "1")
        out = classify_regime(_disinflation_recession())
        assert out.get("gdp_collapse_triggered") is True
        assert out["gdp_collapse_severity"] > 0.7

    def test_severity_decreases_with_calmer_vix(self, monkeypatch):
        """VIX scende da 38 → 18 (calmo): severity deve scendere significativamente."""
        monkeypatch.setenv("USE_GDP_COLLAPSE_OVERRIDE", "1")
        severe = classify_regime(_disinflation_recession())
        ind = _disinflation_recession()
        ind["vix"] = 18.0
        mild = classify_regime(ind)
        # Severity più bassa con VIX calmo
        sev_severe = severe.get("gdp_collapse_severity", 0)
        sev_mild = mild.get("gdp_collapse_severity", 0)
        assert sev_mild < sev_severe

    def test_severity_decreases_with_strong_gdp(self, monkeypatch):
        """GDP da -2 → +2: severity deve scendere drammaticamente."""
        monkeypatch.setenv("USE_GDP_COLLAPSE_OVERRIDE", "1")
        ind = _disinflation_recession()
        severe = classify_regime(ind)
        ind["gdp_roc"] = 2.0
        mild = classify_regime(ind)
        sev_severe = severe.get("gdp_collapse_severity", 0)
        sev_mild = mild.get("gdp_collapse_severity", 0)
        assert sev_mild < sev_severe

    def test_no_trigger_in_healthy_regime(self, monkeypatch):
        """Indicators sani (gdp +3, vix 14, unrate falling, breakeven 2.0):
        severity deve essere < 0.5, no trigger."""
        monkeypatch.setenv("USE_GDP_COLLAPSE_OVERRIDE", "1")
        healthy = {
            "gdp_roc": 3.5,
            "pmi": 56.0,
            "cpi_yoy": 2.0,
            "unrate": 4.0,
            "unrate_roc": -0.3,  # disoccupazione scendendo
            "vix": 13.0,  # mercato calmo
            "fed_funds": 2.0,
            "breakeven_10y": 2.0,  # neutral
            "baa_10y_spread": 1.5,
            "core_pce_yoy": 2.0,
            "payrolls_yoy": 2.0,
            "lei_roc": 1.5,
        }
        out = classify_regime(healthy)
        # severity < 0.5 → no triggered flag
        assert "gdp_collapse_triggered" not in out

    def test_severity_continuous_proportional(self, monkeypatch):
        """Test continuità: piccoli cambi → piccoli delta severity."""
        monkeypatch.setenv("USE_GDP_COLLAPSE_OVERRIDE", "1")
        ind1 = _disinflation_recession()
        ind1["gdp_roc"] = -2.0
        out1 = classify_regime(ind1)

        ind2 = _disinflation_recession()
        ind2["gdp_roc"] = -1.8  # cambio piccolo
        out2 = classify_regime(ind2)

        sev1 = out1.get("gdp_collapse_severity", 0)
        sev2 = out2.get("gdp_collapse_severity", 0)
        # Differenza piccola (no salto binary)
        assert abs(sev1 - sev2) < 0.05


class TestInvariants:
    def test_probabilities_still_sum_to_one(self, monkeypatch):
        monkeypatch.setenv("USE_GDP_COLLAPSE_OVERRIDE", "1")
        out = classify_regime(_disinflation_recession())
        assert abs(sum(out["probabilities"].values()) - 1.0) < 1e-6

    def test_no_negative_probs(self, monkeypatch):
        monkeypatch.setenv("USE_GDP_COLLAPSE_OVERRIDE", "1")
        out = classify_regime(_disinflation_recession())
        for p in out["probabilities"].values():
            assert p >= 0
