"""Test DFM nowcast GDP (Tier 2.5 roadmap Bridgewater).

Dynamic Factor Model su indicatori coincidenti FRED (industrial_production,
nonfarm_payrolls, retail_sales, consumer_sentiment, housing_starts) per
estrarre un fattore latente "growth nowcast" + mappare a GDP YoY via OLS.

Output: nowcast mensile vs gdp_roc trimestrale lagged 1-2 mesi.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.indicators.dfm_nowcast import (
    _standardize_panel,
    _yoy_transform,
    fit_factor_model,
)


def _make_panel(rng_seed: int = 42, n_months: int = 120) -> pd.DataFrame:
    """Genera 5 serie con un factor latente comune + noise individuale.
    Il DFM dovrebbe estrarre il factor con correlation ~1 al ground truth."""
    rng = np.random.default_rng(rng_seed)
    dates = pd.date_range("2010-01-31", periods=n_months, freq="ME")
    # Factor latente: trend + ciclo
    t = np.arange(n_months)
    factor = np.sin(t / 12.0) + 0.05 * t
    # 5 serie = factor × loading + noise
    loadings = [1.0, 0.8, 1.2, 0.9, 0.7]
    panel = pd.DataFrame({
        f"series_{i}": loadings[i] * factor + rng.normal(0, 0.5, n_months)
        for i in range(5)
    }, index=dates)
    return panel, factor


class TestYoYTransform:
    def test_yoy_basic(self):
        """YoY = (val_t - val_{t-12}) / val_{t-12} × 100."""
        s = pd.Series(
            [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 110],
            index=pd.date_range("2024-01-31", periods=13, freq="ME"),
        )
        out = _yoy_transform(s)
        # Last point: (110 - 100) / 100 * 100 = 10%
        assert out.iloc[-1] == pytest.approx(10.0, abs=0.01)

    def test_yoy_drops_first_year(self):
        """Primi 12 valori sono NaN (no lag disponibile)."""
        s = pd.Series(
            range(1, 25),  # avoid zero (which → NaN in YoY)
            index=pd.date_range("2024-01-31", periods=24, freq="ME"),
        )
        out = _yoy_transform(s)
        assert out.iloc[:12].isna().all()
        assert out.iloc[12:].notna().all()

    def test_yoy_handles_zero_division(self):
        """Valore precedente zero → NaN, non inf."""
        s = pd.Series(
            [0] * 12 + [10],
            index=pd.date_range("2024-01-31", periods=13, freq="ME"),
        )
        out = _yoy_transform(s)
        assert pd.isna(out.iloc[-1])


class TestStandardizePanel:
    def test_zscore_mean_zero(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.normal(5, 2, (100, 3)), columns=["a", "b", "c"])
        out = _standardize_panel(df)
        for col in out.columns:
            assert abs(out[col].mean()) < 1e-9
            assert abs(out[col].std(ddof=0) - 1.0) < 1e-9

    def test_handles_constant_column(self):
        """Colonna costante → output 0 (no division by zero)."""
        df = pd.DataFrame({"a": [1.0] * 50, "b": np.arange(50, dtype=float)})
        out = _standardize_panel(df)
        assert (out["a"] == 0.0).all()
        # b standardizzata normale
        assert abs(out["b"].mean()) < 1e-9


class TestFitFactorModel:
    """Smoke test: il factor estratto deve correlare col ground truth."""

    def test_factor_recovers_synthetic_signal(self):
        panel, true_factor = _make_panel(rng_seed=42, n_months=120)
        std_panel = _standardize_panel(panel)
        result = fit_factor_model(std_panel, n_factors=1, max_iter=50)

        assert result is not None
        assert result.factor.shape == (120,)
        # Correlazione factor estratto vs ground truth (può essere negativa
        # per via dell'ambiguità di sign del modello → uso |corr|)
        std_factor_truth = (true_factor - true_factor.mean()) / true_factor.std()
        corr = float(np.corrcoef(result.factor, std_factor_truth)[0, 1])
        assert abs(corr) > 0.8, f"Factor recovery weak: corr={corr:.3f}"

    def test_loadings_have_consistent_sign(self):
        """Tutti i loadings dovrebbero avere lo stesso sign (caso 1-factor con
        serie tutte positivamente correlate al factor latente)."""
        panel, _ = _make_panel(rng_seed=42, n_months=120)
        std_panel = _standardize_panel(panel)
        result = fit_factor_model(std_panel, n_factors=1, max_iter=50)
        assert result is not None
        signs = np.sign(result.loadings)
        # Ammetto un eventuale outlier: almeno 4 su 5 stessa direzione
        same_sign_count = max((signs > 0).sum(), (signs < 0).sum())
        assert same_sign_count >= 4

    def test_returns_none_on_insufficient_data(self):
        """Panel con < 12 osservazioni → None (no fit affidabile)."""
        small = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = fit_factor_model(small, n_factors=1)
        assert result is None
