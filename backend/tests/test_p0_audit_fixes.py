"""Regression test per i fix P0 dell'audit 2026-07-12.

Coprono:
1. Boundary bug dynamic_n_tranches (confidence esattamente 0.30 → deadlock
   Data-Driven: 0 posizioni per 3 settimane in produzione).
2. Soglie cross-asset in unità raw coerenti con compute_copper_gold_ratio.
3. PMI proxy nel feature extractor ML (NAPM discontinuato → live senza `pmi`).

NOTA 2026-07-15: i test su `data_strategist._decide` sono stati rimossi con il
bot Data-Driven (decisione Erik). Il boundary di `dynamic_n_tranches` resta
coperto qui: il floor 0.30 è usato anche dalle altre strategie.
"""
from __future__ import annotations

import numpy as np

from app.services.ai_portfolio import sizer
from app.services.indicators.cross_asset import (
    compute_copper_gold_ratio,
    get_threshold,
)
from app.services.regime.ml_classifier import extract_features


class TestDynamicNTranchesBoundary:
    """AUDIT 2026-07-12: confidence = max(0.30, fired/10) in data_strategist
    produce ESATTAMENTE 0.30 con <=3 trigger fired. Il vecchio `> 0.30`
    ritornava 0 tranche → "target weight 0" per ogni asset → deadlock."""

    def test_confidence_exactly_030_gives_tranches(self):
        assert sizer.dynamic_n_tranches(0.30) == 5

    def test_below_gate_still_skips(self):
        assert sizer.dynamic_n_tranches(0.29) == 0
        assert sizer.dynamic_n_tranches(0.0) == 0

    def test_upper_bands_unchanged(self):
        assert sizer.dynamic_n_tranches(0.31) == 5
        assert sizer.dynamic_n_tranches(0.51) == 4
        assert sizer.dynamic_n_tranches(0.66) == 3
        assert sizer.dynamic_n_tranches(0.81) == 2


class TestCrossAssetThresholdUnits:
    """AUDIT 2026-07-12: le soglie copper_gold (0.45/0.30) erano ~250x fuori
    scala rispetto al raw ratio $/lb / $/oz → il segnale deflattivo sarebbe
    scattato PER SEMPRE con flag ON. Ora soglie = quantili reali 2000-2026."""

    def test_thresholds_same_scale_as_computed_ratio(self):
        # Episodi storici reali (Yahoo): 2011 reflation peak / 2020 covid
        peak_2011 = compute_copper_gold_ratio(4.45, 1400.0)   # ~0.0032
        covid_2020 = compute_copper_gold_ratio(2.36, 1650.0)  # ~0.0014
        assert peak_2011 > get_threshold("copper_gold_strong")
        assert covid_2020 < get_threshold("copper_gold_weak")

    def test_median_regime_is_neutral(self):
        # p50 storico ~0.0023: non deve scattare né strong né weak
        p50 = 0.0023
        assert p50 < get_threshold("copper_gold_strong")
        assert p50 > get_threshold("copper_gold_weak")


class TestMlPmiProxy:
    """AUDIT 2026-07-12: NAPM discontinuato → live non ha mai `pmi`; l'ML
    blend vedeva sempre l'anchor 50.0 (manifatturiero cieco)."""

    def _base(self) -> dict:
        return {"gdp_roc": 1.0, "cpi_yoy": 3.0, "unrate": 4.5, "vix": 18.0}

    def test_pmi_proxy_from_indpro_when_missing(self):
        # indpro_roc_12m -4% → proxy 50 + 5*(-4) = 30 (clamp floor)
        feats, names = extract_features({**self._base(), "indpro_roc_12m": -4.0})
        pmi_val = feats[names.index("pmi")]
        assert pmi_val < 50.0  # non più anchor fisso
        assert pmi_val == 30.0

    def test_real_pmi_still_wins(self):
        # Episodi storici hanno il PMI vero: non va sovrascritto dal proxy
        feats, names = extract_features({**self._base(), "pmi": 42.0})
        assert feats[names.index("pmi")] == 42.0

    def test_no_data_stays_neutral(self):
        feats, names = extract_features(self._base())
        pmi_val = feats[names.index("pmi")]
        assert pmi_val == 50.0  # indpro assente → proxy neutro

    def test_features_finite(self):
        feats, _ = extract_features({**self._base(), "indpro_roc_12m": 2.0})
        assert np.all(np.isfinite(feats))
