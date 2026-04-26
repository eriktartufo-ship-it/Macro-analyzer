"""Test integrazione Yahoo Finance + real returns calculator.

Questi test sono `live` (richiedono rete) ma sono leggeri e cachano su disco.
Vengono saltati se yfinance non disponibile.
"""
import pytest

pytest.importorskip("yfinance")


class TestYahooFetcher:
    def test_fetch_spy_has_history(self):
        from app.services.prices.yahoo_fetcher import YahooFetcher
        y = YahooFetcher()
        s = y.fetch("SPY")
        assert len(s) > 1000  # SPY ha 30+ anni di storia
        assert s.iloc[-1] > 50  # prezzo SPY mai sceso sotto $50 dal 1995

    def test_fetch_unknown_asset_raises(self):
        from app.services.prices.yahoo_fetcher import YahooFetcher
        y = YahooFetcher()
        with pytest.raises(ValueError):
            y.fetch_asset("nonexistent_asset_class")

    def test_fetch_asset_routes_to_ticker(self):
        from app.services.prices.yahoo_fetcher import YahooFetcher
        y = YahooFetcher()
        s = y.fetch_asset("us_equities_growth")  # QQQ + Nasdaq Comp backfill
        # Nasdaq Composite parte 1971
        assert s.index.min().year <= 1985
        assert s.iloc[-1] > 100


class TestRealReturns:
    def test_real_return_series_for_spy(self):
        from app.services.prices.returns import real_return_series
        rr = real_return_series("us_equities_growth", horizon_months=12)
        # Deve avere centinaia di osservazioni mensili
        assert len(rr) > 200
        # Equity reale long-run: tipicamente +5-8% media (anche con dot-com / GFC)
        assert -0.3 < rr.mean() < 0.5
        # Volatilita storicamente ~15-25%
        assert 0.05 < rr.std() < 0.40

    def test_metrics_by_regime_runs(self):
        """Smoke test: pipeline regime_probs_dataframe -> metrics_by_regime non crasha."""
        import pandas as pd
        from app.services.prices.returns import metrics_by_regime

        # Costruisci probs sintetiche (12 mesi tutti reflation)
        idx = pd.date_range("2010-01-31", periods=200, freq="ME")
        probs = pd.DataFrame({
            "reflation": [0.80] * 200,
            "stagflation": [0.10] * 200,
            "deflation": [0.05] * 200,
            "goldilocks": [0.05] * 200,
        }, index=idx)
        metrics = metrics_by_regime("gold", probs, horizon_months=12, threshold=0.50)
        # Almeno reflation deve avere n > 0 (sintesi sempre 0.80 reflation)
        refl = next(m for m in metrics if m.regime == "reflation")
        assert refl.n_observations > 0
