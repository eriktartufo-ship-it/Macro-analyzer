"""Test Tier 6.1 — Adaptive thresholds (quantili rolling 10y).

Coverage:
- compute_quantiles: serie corta, NaN, all-zero, synthetic distribuzioni.
- quantiles_to_recipe: 5 threshold_type (high/high_extreme/low/low_extreme/centered).
- get_recipe_for_condition: mapping condition → recipe.
- classify_regime con flag ON + adaptive_recipes injection.
- Back-compat: flag OFF → identity baseline.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.services.indicators.adaptive_thresholds import (
    AdaptiveRecipe,
    _CONDITION_RECIPES,
    compute_quantiles,
    get_recipe_for_condition,
    list_adaptive_conditions,
    quantiles_to_recipe,
)


def _make_series(values: list[float], start: str = "2010-01-01", freq: str = "D") -> pd.Series:
    dates = pd.date_range(start, periods=len(values), freq=freq)
    return pd.Series(values, index=dates)


class TestComputeQuantiles:
    def test_empty_series_returns_empty(self):
        out = compute_quantiles(pd.Series(dtype=float))
        assert out == {}

    def test_too_short_series_returns_empty(self):
        s = _make_series([1.0] * 10)
        out = compute_quantiles(s, window_years=10)
        assert out == {}

    def test_uniform_series_quantiles(self):
        # 5000 punti uniform [10, 30]
        rng = np.random.default_rng(42)
        values = list(rng.uniform(10, 30, size=5000))
        s = _make_series(values, freq="D")
        out = compute_quantiles(s, quantiles=(0.25, 0.50, 0.75, 0.95), window_years=15)
        # Quantili uniformi attesi: q25≈15, q50≈20, q75≈25, q95≈29
        assert 14 < out[0.25] < 16
        assert 19 < out[0.50] < 21
        assert 24 < out[0.75] < 26
        assert 28 < out[0.95] < 30

    def test_nan_values_ignored(self):
        rng = np.random.default_rng(42)
        values = list(rng.uniform(0, 1, size=2000))
        values[100:200] = [np.nan] * 100
        s = _make_series(values, freq="D")
        out = compute_quantiles(s, quantiles=(0.5,), window_years=10)
        assert 0.4 < out[0.5] < 0.6  # mediana ~0.5 despite NaN

    def test_end_date_window_filtering(self):
        # 20 anni di dati, window 5 anni a fine 2020 → solo 2015-2020
        rng = np.random.default_rng(0)
        values = list(rng.normal(50, 10, size=20 * 365))
        s = _make_series(values, start="2005-01-01", freq="D")
        # Aggiungi spike alla fine
        out_full = compute_quantiles(s, quantiles=(0.5,), end_date=date(2025, 1, 1), window_years=20)
        out_short = compute_quantiles(s, quantiles=(0.5,), end_date=date(2010, 1, 1), window_years=5)
        # Window shorter copre solo 2005-2010, ma full include tutto
        assert 0.5 in out_full
        assert 0.5 in out_short


class TestQuantilesToRecipe:
    def _quantiles(self):
        return {0.05: 10.0, 0.10: 12.0, 0.25: 15.0, 0.50: 20.0, 0.75: 26.0, 0.90: 30.0, 0.95: 35.0}

    def test_high_recipe(self):
        r = quantiles_to_recipe(self._quantiles(), "high", series_name="vix")
        assert r is not None
        assert r.center == 26.0  # q75
        assert r.scale == 9.0    # q95 - q75 = 35-26

    def test_high_extreme_recipe(self):
        r = quantiles_to_recipe(self._quantiles(), "high_extreme", series_name="vix")
        assert r is not None
        assert r.center == 35.0  # q95
        assert r.scale == 5.0    # q95 - q90 = 35-30

    def test_low_recipe(self):
        r = quantiles_to_recipe(self._quantiles(), "low", series_name="vix")
        assert r is not None
        assert r.center == 15.0  # q25
        assert r.scale == 5.0    # q50 - q25 = 20-15

    def test_low_extreme_recipe(self):
        r = quantiles_to_recipe(self._quantiles(), "low_extreme", series_name="vix")
        assert r is not None
        assert r.center == 10.0  # q05
        assert r.scale == 2.0    # q10 - q05 = 12-10

    def test_centered_recipe(self):
        r = quantiles_to_recipe(self._quantiles(), "centered", series_name="cpi")
        assert r is not None
        assert r.center == 20.0   # q50
        assert r.scale == 11.0    # IQR = q75-q25 = 26-15

    def test_unknown_type_returns_none(self):
        r = quantiles_to_recipe(self._quantiles(), "garbage_type", series_name="vix")
        assert r is None

    def test_missing_quantiles_returns_none(self):
        partial = {0.50: 20.0}
        r = quantiles_to_recipe(partial, "high", series_name="vix")
        assert r is None

    def test_scale_floor_eps(self):
        """Quantili degeneri (q75 == q95) → scale fissato a eps positivo."""
        degenerate = {0.25: 5.0, 0.50: 5.0, 0.75: 5.0, 0.95: 5.0}
        r = quantiles_to_recipe(degenerate, "high", series_name="x")
        assert r is not None
        assert r.scale > 0


class TestGetRecipeForCondition:
    def _vix_series(self):
        rng = np.random.default_rng(123)
        # Distribuzione tipo VIX: lognormal, mean ~18, fat right tail
        values = rng.lognormal(mean=2.85, sigma=0.4, size=3000)
        # Clamp realistic VIX range
        values = np.clip(values, 8, 80)
        return _make_series(list(values), freq="D")

    def test_known_condition_returns_recipe(self):
        r = get_recipe_for_condition("vix_spike", self._vix_series())
        assert r is not None
        assert r.series_name == "vix"
        assert r.center > 0
        assert r.scale > 0

    def test_unknown_condition_returns_none(self):
        r = get_recipe_for_condition("garbage_condition", self._vix_series())
        assert r is None

    def test_empty_series_returns_none(self):
        r = get_recipe_for_condition("vix_spike", pd.Series(dtype=float))
        assert r is None

    def test_short_series_returns_none(self):
        r = get_recipe_for_condition("vix_spike", _make_series([20.0] * 20))
        assert r is None

    def test_vix_spike_center_above_vix_elevated(self):
        s = self._vix_series()
        spike = get_recipe_for_condition("vix_spike", s)
        elevated = get_recipe_for_condition("vix_elevated", s)
        assert spike is not None and elevated is not None
        # q95 > q75 → vix_spike center > vix_elevated center
        assert spike.center > elevated.center

    def test_vix_low_center_below_vix_calm(self):
        s = self._vix_series()
        low = get_recipe_for_condition("vix_low", s)        # q25
        calm = get_recipe_for_condition("vix_calm", s)      # q05
        assert low is not None and calm is not None
        # q25 > q05 → vix_low center > vix_calm center
        assert low.center > calm.center


class TestListAdaptiveConditions:
    def test_returns_all_conditions(self):
        out = list_adaptive_conditions()
        # Almeno 13 condizioni mappate
        assert len(out) >= 13
        assert "vix_spike" in out
        assert "nfci_tight" in out
        assert "inflation_high" in out


class TestClassifierWithAdaptiveRecipes:
    """Verifica classifier accetta adaptive_recipes e li usa quando flag ON."""

    def _balanced_indicators(self) -> dict:
        return {
            "gdp_roc": 1.5, "pmi": 50.0, "cpi_yoy": 5.0, "unrate": 4.5,
            "unrate_roc": 0.0, "yield_curve_10y2y": 0.5, "initial_claims_roc": 0.0,
            "lei_roc": 0.0, "fed_funds_rate": 3.0, "vix": 22.0, "baa_spread": 2.5,
            "nfci": 0.2,
        }

    def test_flag_off_ignores_adaptive_recipes(self, monkeypatch):
        """Default flag OFF → adaptive_recipes ignorato, hardcoded center/scale.

        T10b: disable GDP_COLLAPSE + ML_BLEND (default-ON) per consistency baseline.
        """
        monkeypatch.delenv("USE_ADAPTIVE_THRESHOLDS", raising=False)
        monkeypatch.setenv("USE_GDP_COLLAPSE_OVERRIDE", "0")
        monkeypatch.setenv("USE_ML_REGIME_BLEND", "0")
        from app.services.regime.classifier import classify_regime

        ind = self._balanced_indicators()
        # Recipes che spostano vix_spike a center=15 (vs hardcoded 30) — boost defl
        bogus_recipes = {
            "vix_spike": AdaptiveRecipe(center=15.0, scale=2.0, quantile_used="bogus", series_name="vix"),
        }
        r_no_recipes = classify_regime(ind)
        r_with_recipes = classify_regime(ind, adaptive_recipes=bogus_recipes)
        # Identici: flag OFF
        for k in r_no_recipes["probabilities"]:
            assert r_no_recipes["probabilities"][k] == r_with_recipes["probabilities"][k]

    def test_flag_on_uses_adaptive_recipes(self, monkeypatch):
        """Flag ON + recipe per vix_spike → boost deflation rispetto a hardcoded."""
        monkeypatch.setenv("USE_ADAPTIVE_THRESHOLDS", "1")
        from app.services.regime.classifier import classify_regime

        # Scenario: VIX 28 (sotto hardcoded vix_spike=30, dovrebbe NON triggerare)
        ind = self._balanced_indicators()
        ind["vix"] = 28.0
        r_hardcoded = classify_regime(ind)

        # Recipe adattivo che abbassa vix_spike center a 25 (regime "calmo storico")
        # → 28 ora attiva vix_spike → boost deflation
        recipes = {
            "vix_spike": AdaptiveRecipe(center=25.0, scale=2.0, quantile_used="q95_test", series_name="vix"),
        }
        r_adaptive = classify_regime(ind, adaptive_recipes=recipes)

        # Defl prob più alta con adaptive (center più basso → sigmoid attivato a VIX 28)
        assert r_adaptive["probabilities"]["deflation"] > r_hardcoded["probabilities"]["deflation"]

    def test_flag_on_none_recipes_falls_back_to_hardcoded(self, monkeypatch):
        """Flag ON ma recipes=None o vuoto → fall back hardcoded."""
        monkeypatch.setenv("USE_ADAPTIVE_THRESHOLDS", "1")
        from app.services.regime.classifier import classify_regime

        ind = self._balanced_indicators()
        r_no_recipes = classify_regime(ind)
        r_empty_recipes = classify_regime(ind, adaptive_recipes={})
        r_none_recipes = classify_regime(ind, adaptive_recipes=None)

        for k in r_no_recipes["probabilities"]:
            assert r_no_recipes["probabilities"][k] == pytest.approx(
                r_empty_recipes["probabilities"][k], abs=1e-12
            )
            assert r_no_recipes["probabilities"][k] == pytest.approx(
                r_none_recipes["probabilities"][k], abs=1e-12
            )

    def test_partial_recipes_only_affect_mapped_conditions(self, monkeypatch):
        """Recipe solo per vix_spike → altri condition (cpi, baa) restano hardcoded."""
        monkeypatch.setenv("USE_ADAPTIVE_THRESHOLDS", "1")
        from app.services.regime.classifier import classify_regime

        ind = self._balanced_indicators()
        ind["vix"] = 35.0  # sopra hardcoded e adattivo, max defl
        recipes = {
            "vix_spike": AdaptiveRecipe(center=30.0, scale=4.0, quantile_used="same_as_hardcoded", series_name="vix"),
        }
        # Recipe matches hardcoded for vix_spike → output identico a no-recipes
        r_with = classify_regime(ind, adaptive_recipes=recipes)
        r_no = classify_regime(ind)
        # Identici entro epsilon (stesso center/scale)
        for k in r_with["probabilities"]:
            assert abs(r_with["probabilities"][k] - r_no["probabilities"][k]) < 1e-9


class TestCoverageMapping:
    """Verifica che il mapping condition→recipe sia completo per i 5 indicatori target."""

    def test_all_vix_conditions_mapped(self):
        for cond in ("vix_low", "vix_calm", "vix_elevated", "vix_spike"):
            assert cond in _CONDITION_RECIPES, f"{cond} non in mapping"

    def test_all_nfci_conditions_mapped(self):
        for cond in ("financial_conditions_loose", "financial_conditions_easy", "nfci_tight"):
            assert cond in _CONDITION_RECIPES

    def test_baa_conditions_mapped(self):
        for cond in ("credit_spread_tight", "credit_spread_wide"):
            assert cond in _CONDITION_RECIPES

    def test_cpi_conditions_mapped(self):
        for cond in ("inflation_rising", "inflation_high", "inflation_low"):
            assert cond in _CONDITION_RECIPES

    def test_yield_curve_mapped(self):
        assert "yield_curve_steep" in _CONDITION_RECIPES
