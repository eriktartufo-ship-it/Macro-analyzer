"""Cache YahooFetcher: la chiave deve includere il RANGE, non solo la fine.

BUG REALE 2026-07-15 (trovato al primo scan live di risk_managed_sp):
`cache_key = f"{ticker}|{end_date}"` → chi chiedeva PIÙ storia di un chiamante
precedente (stesso ticker, stessa TTL 1h) riceveva la serie CORTA di quello,
senza alcun errore.

Caso reale: llm_strategist fetcha SPY su 200 giorni di calendario (~135 di borsa)
→ cache; poi risk_managed_sp chiede 3 anni per la MA200 → cache HIT → 135 righe →
MA200 impossibile → il bot saltava ogni giorno in prod.

La regola era già in memoria (`feedback_cache_key_include_daterange`): era stata
applicata solo a metà (end sì, start no).
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from app.services.prices.yahoo_fetcher import YahooFetcher


@pytest.fixture
def fetcher(monkeypatch, tmp_path):
    f = YahooFetcher()
    # niente disco: isoliamo la mem-cache
    monkeypatch.setattr(f, "_load_disk", lambda *a, **k: None)
    monkeypatch.setattr(f, "_save_disk", lambda *a, **k: None)
    return f


def _fake_yf(monkeypatch, calls):
    """Simula yfinance: ritorna tante righe quanti i giorni richiesti."""
    import types

    def fake_download(ticker, start, end, **kw):
        calls.append((ticker, start, end))
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        idx = pd.bdate_range(s, e)
        return pd.DataFrame({"Close": [100.0 + i for i in range(len(idx))]}, index=idx)

    mod = types.SimpleNamespace(download=fake_download)
    monkeypatch.setitem(__import__("sys").modules, "yfinance", mod)


def test_richiesta_piu_lunga_non_riceve_la_serie_corta(fetcher, monkeypatch):
    """IL BUG: prima questo test falliva — la 2a chiamata riceveva 135 righe."""
    calls = []
    _fake_yf(monkeypatch, calls)
    end = date(2026, 7, 15)

    # 1) chiamante "llm_strategist": 200 giorni di calendario
    short = fetcher.fetch("SPY", start_date=end - timedelta(days=200), end_date=end)
    # 2) chiamante "risk_managed_sp": 3 anni (serve la MA200)
    long = fetcher.fetch("SPY", start_date=end - timedelta(days=3 * 365), end_date=end)

    assert len(long) > len(short), (
        "chi chiede 3 anni DEVE ricevere 3 anni, non la cache corta del chiamante "
        f"precedente (short={len(short)}, long={len(long)})"
    )
    assert len(long) > 200, "servono >200 giorni di borsa per calcolare la MA200"
    assert len(calls) == 2, "range diversi = fetch diversi, non un cache-hit sbagliato"


def test_stesso_range_usa_la_cache(fetcher, monkeypatch):
    """La cache deve comunque funzionare: stesso range = 1 sola fetch."""
    calls = []
    _fake_yf(monkeypatch, calls)
    end = date(2026, 7, 15)
    start = end - timedelta(days=365)
    a = fetcher.fetch("SPY", start_date=start, end_date=end)
    b = fetcher.fetch("SPY", start_date=start, end_date=end)
    assert len(calls) == 1, "stesso range → una sola chiamata a Yahoo"
    assert len(a) == len(b)


def test_end_date_diverso_non_riusa_la_cache(fetcher, monkeypatch):
    """Regressione del fix 2026-06-19 (prezzi stale tra giorni): resta valida."""
    calls = []
    _fake_yf(monkeypatch, calls)
    start = date(2025, 1, 1)
    fetcher.fetch("SPY", start_date=start, end_date=date(2026, 7, 14))
    fetcher.fetch("SPY", start_date=start, end_date=date(2026, 7, 15))
    assert len(calls) == 2, "end_date diverso → fetch nuova (no serie stale)"


def test_ticker_diverso_non_collide(fetcher, monkeypatch):
    calls = []
    _fake_yf(monkeypatch, calls)
    end = date(2026, 7, 15)
    start = end - timedelta(days=365)
    fetcher.fetch("SPY", start_date=start, end_date=end)
    fetcher.fetch("GLD", start_date=start, end_date=end)
    assert len(calls) == 2
