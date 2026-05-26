"""Test Tier 6.9 — Freshness weighting per condition classifier.

Coverage:
- freshness_for: mapped vs unmapped (default 0.7).
- effective_weight: base × freshness, range.
- Classifier flag OFF: comportamento identico baseline (no `freshness` keys).
- Classifier flag ON: `freshness` key presente in conditions_detail.
- Flag ON: GDP quarterly pesa meno, VIX daily pesa pieno.
"""
from __future__ import annotations

import pytest

from app.services.regime.classifier import REGIME_CONDITIONS, classify_regime
from app.services.regime.freshness import (
    CONDITION_FRESHNESS,
    DEFAULT_FRESHNESS,
    effective_weight,
    freshness_for,
)


class TestFreshnessFor:
    def test_gdp_is_quarterly_low_score(self):
        assert freshness_for("gdp_strong") == 0.4
        assert freshness_for("gdp_weak") == 0.4

    def test_vix_is_daily_full_score(self):
        assert freshness_for("vix_low") == 1.0
        assert freshness_for("vix_spike") == 1.0

    def test_news_is_near_daily(self):
        assert freshness_for("news_panic") == 0.95

    def test_unmapped_uses_default(self):
        assert freshness_for("nonexistent_condition") == DEFAULT_FRESHNESS

    def test_all_mapped_in_range(self):
        for name, score in CONDITION_FRESHNESS.items():
            assert 0.0 <= score <= 1.0, f"{name} freshness out of range: {score}"


class TestEffectiveWeight:
    def test_full_freshness_returns_base(self):
        assert effective_weight("vix_low", 0.10) == pytest.approx(0.10)

    def test_quarterly_halves_base(self):
        assert effective_weight("gdp_strong", 0.10) == pytest.approx(0.04)

    def test_unmapped_uses_default(self):
        assert effective_weight("xxx", 0.10) == pytest.approx(0.07)


class TestClassifierFlagGating:
    """Quando USE_FRESHNESS_WEIGHTING è OFF, comportamento identico baseline."""

    def _base_indicators(self) -> dict:
        return {
            "gdp_roc": 2.5,
            "pmi": 54.0,
            "cpi_yoy": 2.2,
            "unrate": 4.0,
            "claims_roc": -1.0,
            "lei_roc": 0.5,
            "yield_curve_spread": 0.8,
            "fed_funds": 2.5,
            "payrolls_yoy": 2.0,
            "indpro_yoy": 2.5,
            "baa_10y_spread": 1.8,
            "housing_starts_yoy": 4.0,
            "nfci": -0.5,
            "vix": 14.0,
            "acm_term_premium": 0.0,
            "core_pce_yoy": 2.0,
            "consumer_sentiment": 80.0,
            "breakeven_10y": 2.0,
        }

    def test_flag_off_no_freshness_keys(self, monkeypatch):
        monkeypatch.delenv("USE_FRESHNESS_WEIGHTING", raising=False)
        out = classify_regime(self._base_indicators())
        sample = next(iter(out["conditions_detail"]["reflation"].values()))
        assert "freshness" not in sample
        assert "effective_weight" not in sample

    def test_flag_on_freshness_keys_present(self, monkeypatch):
        monkeypatch.setenv("USE_FRESHNESS_WEIGHTING", "1")
        out = classify_regime(self._base_indicators())
        cond_detail = out["conditions_detail"]["reflation"]
        gdp_detail = cond_detail.get("gdp_strong")
        vix_detail = cond_detail.get("vix_low")
        assert gdp_detail is not None
        assert vix_detail is not None
        assert gdp_detail["freshness"] == 0.4
        assert vix_detail["freshness"] == 1.0
        assert gdp_detail["effective_weight"] == pytest.approx(
            gdp_detail["weight"] * 0.4
        )

    def test_flag_on_probabilities_still_sum_one(self, monkeypatch):
        monkeypatch.setenv("USE_FRESHNESS_WEIGHTING", "1")
        out = classify_regime(self._base_indicators())
        assert abs(sum(out["probabilities"].values()) - 1.0) < 1e-6

    def test_flag_on_gdp_contributes_less(self, monkeypatch):
        """GDP quarterly weighted_score deve essere ~40% del valore senza freshness."""
        indicators = self._base_indicators()

        monkeypatch.delenv("USE_FRESHNESS_WEIGHTING", raising=False)
        baseline = classify_regime(indicators)
        base_gdp = baseline["conditions_detail"]["reflation"]["gdp_strong"]["weighted_score"]

        monkeypatch.setenv("USE_FRESHNESS_WEIGHTING", "1")
        with_fresh = classify_regime(indicators)
        fresh_gdp = with_fresh["conditions_detail"]["reflation"]["gdp_strong"]["weighted_score"]

        # GDP freshness 0.4 → weighted_score ridotto 60% rispetto baseline
        if base_gdp > 0.0001:
            assert fresh_gdp < base_gdp
            assert fresh_gdp == pytest.approx(base_gdp * 0.4, rel=0.05)
