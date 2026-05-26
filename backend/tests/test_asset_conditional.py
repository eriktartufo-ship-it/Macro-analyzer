"""Test conditional asset scoring P(asset | regime, stress)."""
from __future__ import annotations

import pytest

from app.services.scoring.asset_conditional import (
    CONDITIONAL_ASSET_TOP_N,
    assess_stress_level,
    calculate_conditional_scores,
    get_conditional_top_n,
)


class TestStressLevel:
    def test_calm_market_low_stress(self):
        ind = {"vix": 13.0, "baa_10y_spread": 1.5, "nfci": -0.4, "unrate_roc": 0.0}
        out = assess_stress_level(ind)
        assert out.level == "low"
        assert out.score < 0.30

    def test_medium_stress(self):
        # Medium stress realistico: VIX 25 + BAA 2.8 + NFCI 0.3 + unrate ↑
        ind = {"vix": 25.0, "baa_10y_spread": 2.8, "nfci": 0.3, "unrate_roc": 0.5}
        out = assess_stress_level(ind)
        assert out.level == "medium"

    def test_high_stress(self):
        ind = {"vix": 45.0, "baa_10y_spread": 5.0, "nfci": 1.5, "unrate_roc": 2.0}
        out = assess_stress_level(ind)
        assert out.level == "high"
        assert out.score > 0.55

    def test_nan_safe(self):
        ind = {"vix": float("nan"), "baa_10y_spread": 2.0}
        out = assess_stress_level(ind)
        assert 0.0 <= out.score <= 1.0


class TestConditionalTopN:
    def test_all_buckets_have_assets(self):
        for (regime, stress), assets in CONDITIONAL_ASSET_TOP_N.items():
            assert len(assets) >= 5, f"{regime}/{stress} has only {len(assets)} assets"

    def test_deflation_high_includes_bonds_cash(self):
        out = get_conditional_top_n("deflation", "high", top_n=5)
        defensive = {"us_bonds_long", "us_bonds_short", "cash_money_market"}
        assert len(set(out) & defensive) >= 2

    def test_goldilocks_low_includes_equities(self):
        out = get_conditional_top_n("goldilocks", "low", top_n=5)
        equities = {"us_equities_growth", "us_equities_value", "international_dm_equities"}
        assert len(set(out) & equities) >= 2

    def test_stagflation_includes_commodities(self):
        out = get_conditional_top_n("stagflation", "medium", top_n=5)
        real_assets = {"energy", "broad_commodities", "gold", "tips_inflation_bonds"}
        assert len(set(out) & real_assets) >= 3

    def test_top_n_respects_size(self):
        out = get_conditional_top_n("reflation", "low", top_n=3)
        assert len(out) == 3

    def test_unknown_regime_fallback(self):
        # Regime sconosciuto → empty fallback
        out = get_conditional_top_n("nonexistent", "low", top_n=5)
        assert out == []


class TestCalculateConditionalScores:
    def test_returns_scores_dict_and_stress(self):
        probs = {"reflation": 0.6, "goldilocks": 0.3, "stagflation": 0.05, "deflation": 0.05}
        ind = {"vix": 15.0, "baa_10y_spread": 1.6}
        assets = ["us_equities_growth", "us_bonds_long", "gold", "cash_money_market"]
        scores, stress = calculate_conditional_scores(probs, ind, assets)
        assert isinstance(scores, dict)
        assert all(a in scores for a in assets)
        assert stress.level in {"low", "medium", "high"}

    def test_reflation_low_stress_favors_equities(self):
        probs = {"reflation": 0.9, "goldilocks": 0.05, "stagflation": 0.03, "deflation": 0.02}
        ind = {"vix": 13.0, "baa_10y_spread": 1.5}  # low stress
        assets = ["us_equities_growth", "us_bonds_long", "gold", "cash_money_market"]
        scores, stress = calculate_conditional_scores(probs, ind, assets)
        assert stress.level == "low"
        assert scores["us_equities_growth"] > scores["us_bonds_long"]
        assert scores["us_equities_growth"] > scores["cash_money_market"]

    def test_deflation_high_stress_favors_bonds_cash(self):
        probs = {"deflation": 0.85, "stagflation": 0.05, "reflation": 0.05, "goldilocks": 0.05}
        ind = {"vix": 50.0, "baa_10y_spread": 5.5, "nfci": 1.5}  # high stress
        assets = ["us_equities_growth", "us_bonds_long", "cash_money_market", "gold"]
        scores, stress = calculate_conditional_scores(probs, ind, assets)
        assert stress.level == "high"
        assert scores["us_bonds_long"] > scores["us_equities_growth"]
        assert scores["cash_money_market"] > scores["us_equities_growth"]

    def test_additive_bonus_with_base_scores(self):
        """T9-AUDIT-FIX-2 (rev2): semantica additive bonus.
        Base scores invariati per asset NON in top-3 conditional;
        asset in top-3 ricevono bonus +12/+8/+5 × prob_regime.
        """
        probs = {"reflation": 1.0, "stagflation": 0, "deflation": 0, "goldilocks": 0}
        ind = {"vix": 13.0, "baa_10y_spread": 1.5}  # reflation low stress
        assets = ["us_equities_growth", "bitcoin"]
        base = {"us_equities_growth": 50.0, "bitcoin": 80.0}
        scores, _ = calculate_conditional_scores(probs, ind, assets, base_scores=base)
        # us_equities_growth è top-1 in reflation/low → bonus +12 → 62
        assert scores["us_equities_growth"] > 50.0
        # bitcoin è top-6 in reflation/low (non top-3) → no bonus → resta 80
        assert scores["bitcoin"] == 80.0
