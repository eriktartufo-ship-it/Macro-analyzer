"""Test T20 — pillar livello-vs-accelerazione (council P2 2026-07-12).

Il pacchetto è FLAG-OFF di default (gate A/B sui 32 episodi: FAIL — vedi
_LEVEL_ACCEL_PILLAR_REBALANCE in classifier.py). I test coprono:
- flag OFF = baseline identica (back-compat)
- flag ON: comportamento dei pillar su scenari storici noti
- invarianti (somma pesi ≈ 1, probs sum=1)
"""
from __future__ import annotations

import pytest


def _reload_classifier():
    import importlib
    import app.services.regime.classifier as clf
    importlib.reload(clf)
    return clf


GOLDILOCKS_2018 = {
    # 2018-style: crescita ok, inflazione bassa, lavoro sano (dati reali FRED)
    "gdp_roc": 2.5, "pmi": 55.0, "cpi_yoy": 2.1, "unrate": 3.9,
    "unrate_roc": -0.2, "yield_curve_10y2y": 0.6, "initial_claims_roc": -2.0,
    "lei_roc": 1.0, "fed_funds_rate": 1.8, "core_pce_yoy": 1.9,
    "payrolls_roc_12m": 1.62, "cpi_yoy_change_6m": 0.68,
    "ulc_yoy": 1.87, "productivity_yoy": 1.90,
}

DEFLATION_2008 = {
    # 2008-12 reale: payrolls -2.56, cpi in caduta -4.96, claims spike
    "gdp_roc": -2.0, "pmi": 38.0, "cpi_yoy": 0.5, "unrate": 7.2,
    "unrate_roc": 1.5, "yield_curve_10y2y": 1.5, "initial_claims_roc": 25.0,
    "lei_roc": -3.0, "fed_funds_rate": 0.5, "core_pce_yoy": 1.5,
    "vix": 60.0, "baa_spread": 5.5, "breakeven_10y": 0.5,
    "payrolls_roc_12m": -2.56, "cpi_yoy_change_6m": -4.96,
    "ulc_yoy": 3.28, "productivity_yoy": 0.21,
}

STAGFLATION_1974 = {
    "gdp_roc": -2.5, "pmi": 42.0, "cpi_yoy": 10.0, "unrate": 8.0,
    "unrate_roc": 0.5, "yield_curve_10y2y": 0.5, "initial_claims_roc": 5.0,
    "lei_roc": -2.0, "fed_funds_rate": 8.0, "core_pce_yoy": 8.0,
    "payrolls_roc_12m": 2.23, "cpi_yoy_change_6m": 1.92,
    "ulc_yoy": 11.22, "productivity_yoy": -2.08,
}


class TestFlagOffBackCompat:
    def test_flag_off_identical_probs(self, monkeypatch):
        monkeypatch.delenv("USE_LEVEL_ACCEL_PILLARS", raising=False)
        monkeypatch.setenv("USE_ML_REGIME_BLEND", "0")
        clf = _reload_classifier()
        base = clf.classify_regime(dict(GOLDILOCKS_2018))
        # Rimuovere i nuovi indicator non cambia nulla con flag OFF
        stripped = {k: v for k, v in GOLDILOCKS_2018.items()
                    if k not in ("ulc_yoy", "productivity_yoy")}
        again = clf.classify_regime(stripped)
        for k in base["probabilities"]:
            assert base["probabilities"][k] == pytest.approx(again["probabilities"][k])


class TestFlagOnBehavior:
    def test_goldilocks_2018_boosted(self, monkeypatch):
        """La diagnosi council: 2017-19 era misclassificato deflation."""
        monkeypatch.setenv("USE_ML_REGIME_BLEND", "0")
        monkeypatch.delenv("USE_LEVEL_ACCEL_PILLARS", raising=False)
        clf = _reload_classifier()
        off = clf.classify_regime(dict(GOLDILOCKS_2018))
        monkeypatch.setenv("USE_LEVEL_ACCEL_PILLARS", "1")
        on = clf.classify_regime(dict(GOLDILOCKS_2018))
        assert on["probabilities"]["goldilocks"] > off["probabilities"]["goldilocks"]
        assert on["probabilities"]["deflation"] < off["probabilities"]["deflation"]

    def test_deflation_2008_preserved(self, monkeypatch):
        """Deflazione macro VERA (payrolls in contrazione) non regredisce."""
        monkeypatch.setenv("USE_ML_REGIME_BLEND", "0")
        monkeypatch.setenv("USE_LEVEL_ACCEL_PILLARS", "1")
        clf = _reload_classifier()
        on = clf.classify_regime(dict(DEFLATION_2008))
        assert on["regime"] == "deflation"

    def test_stagflation_1974_ulc_spiral(self, monkeypatch):
        monkeypatch.setenv("USE_ML_REGIME_BLEND", "0")
        monkeypatch.setenv("USE_LEVEL_ACCEL_PILLARS", "1")
        clf = _reload_classifier()
        on = clf.classify_regime(dict(STAGFLATION_1974))
        assert on["regime"] == "stagflation"

    def test_probs_sum_to_one(self, monkeypatch):
        monkeypatch.setenv("USE_LEVEL_ACCEL_PILLARS", "1")
        clf = _reload_classifier()
        for ind in (GOLDILOCKS_2018, DEFLATION_2008, STAGFLATION_1974):
            out = clf.classify_regime(dict(ind))
            assert sum(out["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)


class TestWeightsSum:
    def test_merged_weights_sum_close_to_one(self, monkeypatch):
        monkeypatch.setenv("USE_LEVEL_ACCEL_PILLARS", "1")
        clf = _reload_classifier()
        merged = clf._merge_level_accel_pillars(clf.REGIME_CONDITIONS)
        for regime, conds in merged.items():
            total = sum(c["weight"] for c in conds.values())
            # I pacchetti sono Σ-preservanti (ADD compensato da REBALANCE)
            assert total == pytest.approx(
                sum(c["weight"] for c in clf.REGIME_CONDITIONS[regime].values()),
                abs=0.011,
            ), f"{regime}: somma pesi {total}"
