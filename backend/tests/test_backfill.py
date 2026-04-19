"""Test per il backfill storico delle classificazioni regime."""

from datetime import date

import pandas as pd
import pytest

from app.services.regime.backfill import _build_indicators_as_of


def _monthly(values: list[float], end: str = "2026-04-01") -> pd.Series:
    idx = pd.date_range(end=end, periods=len(values), freq="MS")
    return pd.Series(values, index=idx)


def _quarterly(values: list[float], end: str = "2026-04-01") -> pd.Series:
    idx = pd.date_range(end=end, periods=len(values), freq="QS")
    return pd.Series(values, index=idx)


def _daily(values: list[float], end: str = "2026-04-15") -> pd.Series:
    idx = pd.date_range(end=end, periods=len(values), freq="D")
    return pd.Series(values, index=idx)


class TestBuildIndicatorsAsOf:
    def test_truncates_series_to_as_of(self):
        """Valori posteriori a `as_of` non devono influenzare lo snapshot."""
        cpi_past = [250.0 + i for i in range(24)]
        cpi_spike = cpi_past + [300.0, 310.0]  # lo spike è DOPO la data di interesse
        series = {
            "cpi": _monthly(cpi_spike, end="2026-04-01"),
        }

        # `as_of` = 2026-01-01 esclude gli ultimi 3 mesi (lo spike)
        indicators = _build_indicators_as_of(series, date(2026, 1, 1))

        assert "cpi_yoy" in indicators
        # Il YoY calcolato su dati pre-spike deve essere moderato
        assert indicators["cpi_yoy"] < 8.0

    def test_returns_empty_when_no_data(self):
        """Se nessuna serie è presente, output vuoto."""
        indicators = _build_indicators_as_of({}, date(2026, 4, 1))
        assert indicators == {}

    def test_includes_new_indicators(self):
        """I nuovi indicatori (core_pce, payrolls, indpro, baa, sentiment) sono estratti."""
        series = {
            # 13 mesi per avere roc_12m
            "core_pce": _monthly([120.0 + i * 0.2 for i in range(13)]),
            "nonfarm_payrolls": _monthly([150000 + i * 1000 for i in range(13)]),
            "industrial_production": _monthly([100.0 + i * 0.1 for i in range(13)]),
            "baa_spread": _daily([2.3] * 10),
            "consumer_sentiment": _monthly([80.0] * 3),
        }

        indicators = _build_indicators_as_of(series, date(2026, 4, 15))

        assert "core_pce_yoy" in indicators
        assert "payrolls_roc_12m" in indicators
        assert "indpro_roc_12m" in indicators
        assert indicators["baa_spread"] == 2.3
        assert indicators["consumer_sentiment"] == 80.0

    def test_gdp_roc_from_quarterly(self):
        """GDP ROC calcolato su 1 trimestre (indice quarterly)."""
        series = {
            "real_gdp": _quarterly([100.0, 101.0, 102.0, 103.0]),
        }
        indicators = _build_indicators_as_of(series, date(2026, 4, 1))
        assert "gdp_roc" in indicators
        assert indicators["gdp_roc"] == pytest.approx(100 * (103.0 / 102.0 - 1), abs=0.01)
