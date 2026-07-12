"""Test Tier 6.3 — Cross-asset ratio signals (copper/gold, gold/oil, HY/IG).

Coverage:
- compute_*_ratio: happy path + edge cases (None, zero, negative).
- 6 condition classifier flag-gated (off → identity, on → shift).
- Soft pillar: non flippa regime con macro forti opposti.
"""
from __future__ import annotations

import pytest

from app.services.indicators.cross_asset import (
    compute_copper_gold_ratio,
    compute_gold_oil_ratio,
    compute_hy_ig_spread_ratio,
    get_threshold,
    list_pillar_conditions,
)


class TestComputeCopperGoldRatio:
    def test_typical_reflation(self):
        # Cu $4.50/lb, Au $2000/oz → ratio ~0.00225 ma in $/oz Cu = ~$4.50/lb
        # Storia: ratio range 0.18-0.55. Per test usiamo input già normalizzati.
        out = compute_copper_gold_ratio(0.45, 1.0)
        assert out == 0.45

    def test_typical_deflation(self):
        out = compute_copper_gold_ratio(0.20, 1.0)
        assert out == 0.20

    def test_none_returns_none(self):
        assert compute_copper_gold_ratio(None, 100.0) is None
        assert compute_copper_gold_ratio(4.0, None) is None
        assert compute_copper_gold_ratio(None, None) is None

    def test_zero_returns_none(self):
        assert compute_copper_gold_ratio(0, 100.0) is None
        assert compute_copper_gold_ratio(4.0, 0) is None

    def test_negative_returns_none(self):
        assert compute_copper_gold_ratio(-1.0, 100.0) is None
        assert compute_copper_gold_ratio(4.0, -100.0) is None

    def test_invalid_type_returns_none(self):
        assert compute_copper_gold_ratio("abc", 100.0) is None


class TestComputeGoldOilRatio:
    def test_typical_normal(self):
        # Gold $2000, oil $75 → 26.67 (normal)
        out = compute_gold_oil_ratio(2000.0, 75.0)
        assert abs(out - 26.667) < 0.01

    def test_deflation_extreme(self):
        # 2020-Q1 covid: Gold $1500, oil $20 → 75 (extreme deflation)
        out = compute_gold_oil_ratio(1500.0, 20.0)
        assert out == 75.0

    def test_stagflation_oil_shock(self):
        # 1973: Gold $100, oil $11 → 9 (stagflation oil-driven)
        out = compute_gold_oil_ratio(100.0, 11.0)
        assert abs(out - 9.09) < 0.01

    def test_none_returns_none(self):
        assert compute_gold_oil_ratio(None, 75.0) is None
        assert compute_gold_oil_ratio(2000.0, None) is None

    def test_zero_oil_returns_none(self):
        """No division by zero."""
        assert compute_gold_oil_ratio(2000.0, 0) is None


class TestComputeHyIgSpreadRatio:
    def test_normal_credit(self):
        out = compute_hy_ig_spread_ratio(4.5, 1.5)
        assert out == 3.0

    def test_credit_stress(self):
        # 2008: HY 18%, IG 3% → ratio 6 (stress severo)
        out = compute_hy_ig_spread_ratio(18.0, 3.0)
        assert out == 6.0

    def test_credit_euphoria(self):
        # 2021: HY 3.5%, IG 1.5% → ratio 2.33 (risk-on euforico)
        out = compute_hy_ig_spread_ratio(3.5, 1.5)
        assert abs(out - 2.333) < 0.01

    def test_none_returns_none(self):
        assert compute_hy_ig_spread_ratio(None, 1.5) is None
        assert compute_hy_ig_spread_ratio(5.0, None) is None

    def test_zero_ig_returns_none(self):
        assert compute_hy_ig_spread_ratio(5.0, 0) is None


class TestThresholds:
    def test_all_thresholds_defined(self):
        for name in (
            "copper_gold_strong", "copper_gold_weak",
            "gold_oil_high", "gold_oil_low",
            "hy_ig_stress", "hy_ig_tight",
        ):
            assert get_threshold(name) > 0

    def test_threshold_ordering(self):
        """copper_gold_strong > copper_gold_weak (strong = high ratio)."""
        assert get_threshold("copper_gold_strong") > get_threshold("copper_gold_weak")
        assert get_threshold("gold_oil_high") > get_threshold("gold_oil_low")
        assert get_threshold("hy_ig_stress") > get_threshold("hy_ig_tight")


class TestListPillarConditions:
    def test_lists_6_conditions(self):
        out = list_pillar_conditions()
        assert len(out) == 6
        assert "copper_gold_strong" in out
        assert "hy_ig_stress" in out


# --- Integration tests with classifier ---


class TestClassifierWithCrossAssetPillars:
    """Verifica 6 condition flag-gated."""

    def _base_neutral(self) -> dict:
        return {
            "gdp_roc": 1.5, "pmi": 50.0, "cpi_yoy": 2.5, "unrate": 4.5,
            "unrate_roc": 0.0, "yield_curve_10y2y": 0.5, "initial_claims_roc": 0.0,
            "lei_roc": 0.0, "fed_funds_rate": 3.0,
        }

    def test_flag_off_preserves_baseline(self, monkeypatch):
        """T10b: disable GDP_COLLAPSE/ML_BLEND default-ON per baseline puro."""
        monkeypatch.delenv("USE_CROSS_ASSET_PILLARS", raising=False)
        monkeypatch.setenv("USE_GDP_COLLAPSE_OVERRIDE", "0")
        monkeypatch.setenv("USE_ML_REGIME_BLEND", "0")
        from app.services.regime.classifier import classify_regime

        ind = self._base_neutral()
        # Ratios estremi: flag off → no effect
        r_no = classify_regime(ind)
        r_extreme = classify_regime({
            **ind,
            "copper_gold_ratio": 0.0032,  # very strong cyclical (2011 peak, raw units)
            "gold_oil_ratio": 60.0,       # gold dominates (deflation)
            "hy_ig_spread_ratio": 8.0,    # credit stress
        })
        for k in r_no["probabilities"]:
            assert r_no["probabilities"][k] == r_extreme["probabilities"][k]

    def test_flag_on_copper_gold_strong_boosts_reflation(self, monkeypatch):
        monkeypatch.setenv("USE_CROSS_ASSET_PILLARS", "1")
        from app.services.regime.classifier import classify_regime

        ind = self._base_neutral()
        # AUDIT 2026-07-12: valori raw reali ($/lb / $/oz) — p50 e 2011 peak
        baseline = classify_regime({**ind, "copper_gold_ratio": 0.0023})  # neutral p50
        cyclical = classify_regime({**ind, "copper_gold_ratio": 0.0032})  # strong (2011)

        assert cyclical["probabilities"]["reflation"] > baseline["probabilities"]["reflation"]

    def test_flag_on_copper_gold_weak_boosts_deflation(self, monkeypatch):
        monkeypatch.setenv("USE_CROSS_ASSET_PILLARS", "1")
        from app.services.regime.classifier import classify_regime

        ind = self._base_neutral()
        baseline = classify_regime({**ind, "copper_gold_ratio": 0.0023})
        crash = classify_regime({**ind, "copper_gold_ratio": 0.0014})  # 2020 covid (raw)

        assert crash["probabilities"]["deflation"] > baseline["probabilities"]["deflation"]

    def test_flag_on_gold_oil_high_boosts_deflation(self, monkeypatch):
        monkeypatch.setenv("USE_CROSS_ASSET_PILLARS", "1")
        from app.services.regime.classifier import classify_regime

        ind = self._base_neutral()
        baseline = classify_regime({**ind, "gold_oil_ratio": 17.0})
        defl = classify_regime({**ind, "gold_oil_ratio": 60.0})  # 2020 covid

        assert defl["probabilities"]["deflation"] > baseline["probabilities"]["deflation"]

    def test_flag_on_gold_oil_low_boosts_stagflation(self, monkeypatch):
        monkeypatch.setenv("USE_CROSS_ASSET_PILLARS", "1")
        from app.services.regime.classifier import classify_regime

        ind = self._base_neutral()
        baseline = classify_regime({**ind, "gold_oil_ratio": 17.0})
        stag = classify_regime({**ind, "gold_oil_ratio": 9.0})  # 1973 oil shock

        assert stag["probabilities"]["stagflation"] > baseline["probabilities"]["stagflation"]

    def test_flag_on_hy_ig_stress_boosts_deflation(self, monkeypatch):
        monkeypatch.setenv("USE_CROSS_ASSET_PILLARS", "1")
        from app.services.regime.classifier import classify_regime

        ind = self._base_neutral()
        baseline = classify_regime({**ind, "hy_ig_spread_ratio": 3.5})
        stress = classify_regime({**ind, "hy_ig_spread_ratio": 7.0})  # 2008 GFC

        assert stress["probabilities"]["deflation"] > baseline["probabilities"]["deflation"]

    def test_flag_on_hy_ig_tight_boosts_reflation(self, monkeypatch):
        monkeypatch.setenv("USE_CROSS_ASSET_PILLARS", "1")
        from app.services.regime.classifier import classify_regime

        ind = self._base_neutral()
        baseline = classify_regime({**ind, "hy_ig_spread_ratio": 3.5})
        euph = classify_regime({**ind, "hy_ig_spread_ratio": 2.3})  # 2007 pre-GFC

        assert euph["probabilities"]["reflation"] > baseline["probabilities"]["reflation"]

    def test_pillar_does_not_flip_regime_with_strong_macro(self, monkeypatch):
        """Soft pillar (0.02-0.03) non flippa con macro forte opposto."""
        monkeypatch.setenv("USE_CROSS_ASSET_PILLARS", "1")
        from app.services.regime.classifier import classify_regime

        # Macro forte deflation: GDP -2, CPI 1, VIX 35
        deflation_macro = {
            "gdp_roc": -2.0, "pmi": 42.0, "cpi_yoy": 1.0, "unrate": 6.5,
            "unrate_roc": 1.0, "yield_curve_10y2y": -0.3, "initial_claims_roc": 15.0,
            "lei_roc": -2.0, "fed_funds_rate": 1.0, "vix": 35.0, "baa_spread": 4.0,
            # Ratios anomali pro-reflation (raw units)
            "copper_gold_ratio": 0.0032, "hy_ig_spread_ratio": 2.3,
        }
        result = classify_regime(deflation_macro)
        assert result["regime"] == "deflation"

    def test_flag_on_neutral_ratios_minimal_impact(self, monkeypatch):
        """Ratios neutrali (default) → diff < 1pp vs no ratios."""
        monkeypatch.setenv("USE_CROSS_ASSET_PILLARS", "1")
        from app.services.regime.classifier import classify_regime

        ind = self._base_neutral()
        baseline = classify_regime(ind)  # no ratios → defaults neutri
        neutral = classify_regime({
            **ind,
            "copper_gold_ratio": 0.0023,
            "gold_oil_ratio": 17.0,
            "hy_ig_spread_ratio": 3.5,
        })
        for k in baseline["probabilities"]:
            assert abs(baseline["probabilities"][k] - neutral["probabilities"][k]) < 0.01
