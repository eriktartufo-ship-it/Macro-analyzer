"""Test della daily returns matrix (B2) — no rete: fake YahooFetcher iniettato."""
from __future__ import annotations

import pandas as pd
import pytest

from app.services.backtest.daily_returns_matrix import (
    build_daily_returns_matrix,
    trading_days,
)


class _FakeYahoo:
    """Fetcher stub: ritorna serie prezzo daily predefinite per asset."""

    def __init__(self, prices: dict[str, pd.Series]):
        self._prices = prices

    def fetch_asset(self, asset: str) -> pd.Series:
        return self._prices.get(asset, pd.Series(dtype=float))


def _price(dates: list[str], vals: list[float]) -> pd.Series:
    return pd.Series(vals, index=pd.to_datetime(dates))


def test_pct_change_correct_and_first_day_dropped():
    px = _price(["2020-01-02", "2020-01-03", "2020-01-06"], [100.0, 110.0, 99.0])
    fake = _FakeYahoo({"us_equities_growth": px})
    df = build_daily_returns_matrix(["us_equities_growth"], use_cache=False, yahoo=fake)
    # primo giorno droppato (no return), poi +10% e -10%
    assert len(df) == 2
    assert df["us_equities_growth"].iloc[0] == pytest.approx(0.10)
    assert df["us_equities_growth"].iloc[1] == pytest.approx(-0.10)


def test_missing_history_is_nan_not_zero():
    a = _price(["2020-01-02", "2020-01-03", "2020-01-06"], [100.0, 101.0, 102.0])
    # b parte più tardi → i primi giorni comuni devono essere NaN, non 0
    b = _price(["2020-01-06", "2020-01-07"], [50.0, 55.0])
    fake = _FakeYahoo({"us_equities_growth": a, "gold": b})
    df = build_daily_returns_matrix(["us_equities_growth", "gold"], use_cache=False, yahoo=fake)
    assert pd.isna(df.loc[pd.Timestamp("2020-01-03"), "gold"])
    assert df.loc[pd.Timestamp("2020-01-07"), "gold"] == pytest.approx(0.10)


def test_duplicate_index_deduped():
    px = _price(["2020-01-02", "2020-01-03", "2020-01-03", "2020-01-06"],
                [100.0, 110.0, 110.0, 121.0])
    fake = _FakeYahoo({"us_equities_growth": px})
    df = build_daily_returns_matrix(["us_equities_growth"], use_cache=False, yahoo=fake)
    # 3 date uniche → 2 rendimenti; nessun indice duplicato
    assert not df.index.has_duplicates
    assert len(df) == 2


def test_unknown_asset_skipped():
    fake = _FakeYahoo({})
    df = build_daily_returns_matrix(["not_an_asset"], use_cache=False, yahoo=fake)
    assert df.empty


def test_trading_days_window_filter():
    px = _price(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"],
                [100.0, 101.0, 102.0, 103.0])
    fake = _FakeYahoo({"us_equities_growth": px})
    df = build_daily_returns_matrix(["us_equities_growth"], use_cache=False, yahoo=fake)
    td = trading_days(df, "2020-01-03", "2020-01-06")
    # index dei return = [01-03, 01-06, 01-07]; finestra [03,06] → [03, 06]
    assert list(td) == [pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-06")]
