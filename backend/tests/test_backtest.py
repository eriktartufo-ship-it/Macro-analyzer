"""Test backtest framework: metrics, portfolio simulator, strategies."""
import numpy as np
import pandas as pd
import pytest

from app.services.backtest.metrics import (
    alpha_vs_benchmark,
    compute_drawdown_series,
    compute_nav,
    compute_stats,
)


class TestMetrics:
    def test_nav_starts_at_initial(self):
        rets = pd.Series([0.0, 0.0, 0.0],
                         index=pd.date_range("2020-01-31", periods=3, freq="ME"))
        nav = compute_nav(rets, initial=1.0)
        assert nav.iloc[0] == 1.0

    def test_nav_compounds(self):
        rets = pd.Series([0.10, 0.10],
                         index=pd.date_range("2020-01-31", periods=2, freq="ME"))
        nav = compute_nav(rets)
        assert abs(nav.iloc[-1] - 1.21) < 1e-9

    def test_drawdown_zero_for_monotonic(self):
        nav = pd.Series([1.0, 1.05, 1.10, 1.15],
                        index=pd.date_range("2020-01-31", periods=4, freq="ME"))
        dd = compute_drawdown_series(nav)
        assert (dd <= 0).all()
        assert dd.iloc[-1] == 0.0

    def test_drawdown_captures_decline(self):
        nav = pd.Series([1.0, 1.20, 0.80, 0.90],
                        index=pd.date_range("2020-01-31", periods=4, freq="ME"))
        dd = compute_drawdown_series(nav)
        # Peak 1.20, trough 0.80 -> dd = -33.3%
        assert abs(dd.min() - (-1 / 3)) < 1e-9

    def test_stats_known_returns(self):
        # 12 mesi tutti +1% mensile -> CAGR ~12.68%
        rets = pd.Series([0.01] * 24,
                         index=pd.date_range("2020-01-31", periods=24, freq="ME"))
        stats = compute_stats(rets)
        assert stats.n_months == 24
        assert abs(stats.annualized_return - 0.1268) < 0.005
        assert stats.win_rate == 1.0
        assert stats.max_drawdown == 0.0

    def test_alpha_correlated_returns(self):
        rng = np.random.default_rng(42)
        b = pd.Series(rng.normal(0.005, 0.04, 60),
                      index=pd.date_range("2020-01-31", periods=60, freq="ME"))
        # Portfolio = beta=0.5 * benchmark + alpha=0.002 monthly
        p = 0.5 * b + 0.002
        a = alpha_vs_benchmark(p, b)
        assert 0.45 < a["beta"] < 0.55
        # Annualized alpha ~= 0.002*12 = 2.4%
        assert 0.015 < a["alpha"] < 0.035


class TestPortfolioSimulator:
    def test_zero_weights_zero_returns(self):
        from app.services.backtest.portfolio import run_backtest
        idx = pd.date_range("2020-01-31", periods=12, freq="ME")
        weights = pd.DataFrame(0.0, index=idx, columns=["A", "B"])
        rets = pd.DataFrame(0.05, index=idx, columns=["A", "B"])
        run = run_backtest(weights, rets, cost_bps=10.0)
        assert (run.monthly_returns.abs() < 1e-9).all()

    def test_full_allocation_matches_asset_return(self):
        from app.services.backtest.portfolio import run_backtest
        idx = pd.date_range("2020-01-31", periods=12, freq="ME")
        weights = pd.DataFrame({"A": [1.0] * 12}, index=idx)
        rets = pd.DataFrame({"A": [0.02] * 12}, index=idx)
        run = run_backtest(weights, rets, cost_bps=0.0)
        assert (abs(run.monthly_returns - 0.02) < 1e-9).all()

    def test_cost_reduces_returns(self):
        from app.services.backtest.portfolio import run_backtest
        idx = pd.date_range("2020-01-31", periods=12, freq="ME")
        # Alterna 100% A vs 100% B ogni mese -> turnover massimo
        weights = pd.DataFrame({
            "A": [1.0 if i % 2 == 0 else 0.0 for i in range(12)],
            "B": [0.0 if i % 2 == 0 else 1.0 for i in range(12)],
        }, index=idx)
        rets = pd.DataFrame({"A": [0.02] * 12, "B": [0.02] * 12}, index=idx)
        run0 = run_backtest(weights, rets, cost_bps=0.0)
        run100 = run_backtest(weights, rets, cost_bps=100.0)
        # Con cost > 0, return netto inferiore
        assert run100.monthly_returns.sum() < run0.monthly_returns.sum()


class TestLeadTimeRecessionParser:
    def test_extracts_recession_spans(self):
        from app.services.backtest.lead_time import _list_nber_recessions

        idx = pd.date_range("2010-01-01", periods=12, freq="MS")
        # 0,0,1,1,1,0,0,1,0,0,0,0
        s = pd.Series([0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0], index=idx)
        spans = _list_nber_recessions(s)
        assert len(spans) == 2
        # Prima recession: marzo->maggio (end = giugno - 1 day = 31 maggio)
        assert spans[0][0].month == 3
        assert spans[1][0].month == 8
