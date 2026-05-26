"""Test Tier 6.5 — Correlation regime detection.

Coverage:
- compute_avg_pairwise_correlation: empty, single asset, 2-4 assets.
- compute_correlation_regime: 4 regimi (crisis/elevated/moderate/diversified)
  con synthetic returns generati per produrre correlation target.
- Edge cases: NaN, window troppo corta, asset insufficienti.
- returns_from_prices helper.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.regime.correlation_regime import (
    _classify,
    compute_avg_pairwise_correlation,
    compute_correlation_regime,
    list_default_assets,
    returns_from_prices,
)


def _synthetic_returns(
    correlations: dict[tuple[str, str], float],
    assets: list[str],
    n_days: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Genera returns DataFrame con correlazioni target via Cholesky.

    Args:
        correlations: dict {(asset_a, asset_b): target_corr}.
            Solo coppie specificate, default 0 per altre.
        assets: lista asset (ordered).
        n_days: numero giorni.

    Returns:
        pd.DataFrame returns indexed by date, columns by asset.
    """
    n = len(assets)
    # Build target correlation matrix
    corr_mat = np.eye(n)
    for (a, b), c in correlations.items():
        ia, ib = assets.index(a), assets.index(b)
        corr_mat[ia, ib] = c
        corr_mat[ib, ia] = c

    # Cholesky decomposition (must be PSD)
    try:
        L = np.linalg.cholesky(corr_mat)
    except np.linalg.LinAlgError:
        # Add small jitter to diagonal
        L = np.linalg.cholesky(corr_mat + np.eye(n) * 0.01)

    rng = np.random.default_rng(seed)
    iid = rng.normal(size=(n_days, n))
    correlated = iid @ L.T

    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    return pd.DataFrame(correlated, index=dates, columns=assets)


class TestComputeAvgPairwiseCorrelation:
    def test_empty_matrix_returns_zero(self):
        assert compute_avg_pairwise_correlation(pd.DataFrame()) == 0.0

    def test_single_asset_returns_zero(self):
        m = pd.DataFrame([[1.0]], columns=["a"], index=["a"])
        assert compute_avg_pairwise_correlation(m) == 0.0

    def test_uniform_correlation(self):
        """Matrix con tutte le correlazioni = 0.5."""
        n = 4
        arr = np.ones((n, n)) * 0.5
        np.fill_diagonal(arr, 1.0)
        m = pd.DataFrame(arr, columns=list("abcd"), index=list("abcd"))
        # 6 coppie unique con valore 0.5 → avg = 0.5
        assert abs(compute_avg_pairwise_correlation(m) - 0.5) < 1e-6

    def test_mixed_correlations(self):
        arr = np.array([
            [1.0, 0.8, 0.2, 0.6],
            [0.8, 1.0, 0.4, 0.5],
            [0.2, 0.4, 1.0, 0.3],
            [0.6, 0.5, 0.3, 1.0],
        ])
        m = pd.DataFrame(arr, columns=list("abcd"), index=list("abcd"))
        # 6 unique upper-tri: 0.8+0.2+0.6+0.4+0.5+0.3 = 2.8 / 6 = 0.4667
        assert abs(compute_avg_pairwise_correlation(m) - 0.4666666) < 1e-3


class TestClassify:
    def test_crisis_threshold(self):
        assert _classify(0.75) == "correlation_crisis"
        assert _classify(0.71) == "correlation_crisis"

    def test_elevated_threshold(self):
        assert _classify(0.55) == "correlation_elevated"
        assert _classify(0.65) == "correlation_elevated"

    def test_moderate_threshold(self):
        assert _classify(0.30) == "correlation_moderate"
        assert _classify(0.45) == "correlation_moderate"

    def test_diversified_threshold(self):
        assert _classify(0.15) == "correlation_diversified"
        assert _classify(0.0) == "correlation_diversified"
        assert _classify(-0.1) == "correlation_diversified"


class TestComputeCorrelationRegime:
    def _assets(self) -> list[str]:
        return ["us_equities_growth", "us_bonds_long", "gold", "broad_commodities"]

    def test_empty_returns_moderate(self):
        out = compute_correlation_regime(pd.DataFrame(), window_days=60)
        assert out.regime == "correlation_moderate"
        assert out.n_assets == 0
        assert out.is_breakdown is False

    def test_single_asset_insufficient(self):
        df = pd.DataFrame({"a": [0.01] * 50}, index=pd.date_range("2020-01-01", periods=50))
        out = compute_correlation_regime(df)
        assert "insufficienti" in out.description.lower() or out.n_assets < 2

    def test_short_window_insufficient(self):
        assets = self._assets()
        df = _synthetic_returns({}, assets, n_days=10)
        out = compute_correlation_regime(df, min_obs=20)
        assert "corta" in out.description.lower()

    def test_diversified_scenario(self):
        """Asset poco correlati (corr ~ 0) → correlation_diversified."""
        assets = self._assets()
        # No correlations specified = all ~0 by Cholesky construction
        df = _synthetic_returns({}, assets, n_days=200, seed=1)
        out = compute_correlation_regime(df, window_days=100)
        assert out.regime == "correlation_diversified"
        assert out.avg_pairwise_correlation <= 0.2
        assert out.is_breakdown is False

    def test_elevated_scenario(self):
        """Asset moderatamente correlati (0.6) → correlation_elevated."""
        assets = self._assets()
        corrs = {
            (assets[0], assets[1]): 0.6,
            (assets[0], assets[2]): 0.6,
            (assets[0], assets[3]): 0.6,
            (assets[1], assets[2]): 0.6,
            (assets[1], assets[3]): 0.6,
            (assets[2], assets[3]): 0.6,
        }
        df = _synthetic_returns(corrs, assets, n_days=300, seed=2)
        out = compute_correlation_regime(df, window_days=200)
        assert out.regime == "correlation_elevated"
        assert 0.5 < out.avg_pairwise_correlation <= 0.7

    def test_crisis_scenario(self):
        """Asset altamente correlati (0.85) → correlation_crisis."""
        assets = self._assets()
        corrs = {
            (assets[0], assets[1]): 0.85,
            (assets[0], assets[2]): 0.85,
            (assets[0], assets[3]): 0.85,
            (assets[1], assets[2]): 0.85,
            (assets[1], assets[3]): 0.85,
            (assets[2], assets[3]): 0.85,
        }
        df = _synthetic_returns(corrs, assets, n_days=300, seed=3)
        out = compute_correlation_regime(df, window_days=200)
        assert out.regime == "correlation_crisis"
        assert out.avg_pairwise_correlation > 0.7
        assert out.is_breakdown is True

    def test_moderate_scenario(self):
        """Asset con correlazione ~0.3 → correlation_moderate."""
        assets = self._assets()
        corrs = {
            (assets[0], assets[1]): 0.3,
            (assets[0], assets[2]): 0.3,
            (assets[0], assets[3]): 0.3,
            (assets[1], assets[2]): 0.3,
            (assets[1], assets[3]): 0.3,
            (assets[2], assets[3]): 0.3,
        }
        df = _synthetic_returns(corrs, assets, n_days=300, seed=4)
        out = compute_correlation_regime(df, window_days=200)
        assert out.regime == "correlation_moderate"
        assert 0.2 < out.avg_pairwise_correlation <= 0.5

    def test_assets_subset_filtering(self):
        """`assets` param restringe a subset di colonne."""
        all_assets = ["a", "b", "c", "d"]
        df = _synthetic_returns({}, all_assets, n_days=100)
        # Restringi a 2 asset (corr ~0 random)
        out = compute_correlation_regime(df, assets=["a", "b"], window_days=80)
        assert out.n_assets == 2

    def test_top_pairs_returned(self):
        """matrix_excerpt contiene top-3 correlazioni."""
        assets = self._assets()
        corrs = {
            (assets[0], assets[1]): 0.85,
            (assets[2], assets[3]): 0.30,
        }
        df = _synthetic_returns(corrs, assets, n_days=300, seed=5)
        out = compute_correlation_regime(df, window_days=200)
        assert len(out.matrix_excerpt) > 0
        # Highest absolute corr first
        top1 = out.matrix_excerpt[0]
        assert abs(top1[2]) >= 0.5  # high pair captured

    def test_description_includes_regime_label(self):
        assets = self._assets()
        df = _synthetic_returns({}, assets, n_days=100)
        out = compute_correlation_regime(df, window_days=80)
        assert "correlazion" in out.description.lower()


class TestReturnsFromPrices:
    def test_empty_returns_empty(self):
        out = returns_from_prices(pd.DataFrame())
        assert out.empty

    def test_log_returns_computation(self):
        dates = pd.date_range("2020-01-01", periods=5)
        prices = pd.DataFrame({"a": [100, 110, 99, 120, 132]}, index=dates)
        ret = returns_from_prices(prices)
        # Primo row dropped, 4 returns
        assert len(ret) == 4
        # log(110/100) = log(1.1) ≈ 0.0953
        assert abs(ret["a"].iloc[0] - np.log(1.10)) < 1e-6


class TestListDefaultAssets:
    def test_includes_main_classes(self):
        out = list_default_assets()
        assert "us_equities_growth" in out
        assert "us_bonds_long" in out
        assert "gold" in out
        assert len(out) >= 4
