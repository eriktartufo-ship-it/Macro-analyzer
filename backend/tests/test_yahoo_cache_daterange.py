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


# ---------------------------------------------------------------------------
# La cache su DISCO: deve solo CRESCERE, mai regredire.
# BUG 2026-07-15 (il vero colpevole in prod): `_save_disk` faceva `to_parquet`
# secco → un consumer con finestra corta (llm_strategist, 200gg) SOVRASCRIVEVA la
# storia lunga sul disco, e `_load_disk` validava solo la freschezza, mai la
# copertura → chi chiedeva 3 anni riceveva 135 righe.
# ---------------------------------------------------------------------------

class TestDiskCacheNonRegredisce:
    def _f(self, tmp_path, monkeypatch):
        import app.services.prices.yahoo_fetcher as YF
        monkeypatch.setattr(YF, "_cache_path", lambda t: tmp_path / f"{t}.parquet")
        monkeypatch.setattr(YF, "_is_fresh", lambda p: p.exists())
        return YF.YahooFetcher()

    def test_save_unisce_invece_di_sovrascrivere(self, tmp_path, monkeypatch):
        f = self._f(tmp_path, monkeypatch)
        lunga = pd.Series(
            [100.0] * 500, index=pd.bdate_range("2024-01-01", periods=500), name="close")
        f._save_disk("SPY", lunga)
        # arriva un consumer con finestra CORTA (come llm_strategist)
        corta = pd.Series(
            [200.0] * 10, index=pd.bdate_range("2025-11-25", periods=10), name="close")
        f._save_disk("SPY", corta)

        on_disk = f._load_disk("SPY")
        assert len(on_disk) >= 500, (
            "una finestra corta NON deve distruggere la storia lunga sul disco "
            f"(righe rimaste: {len(on_disk)})")

    def test_i_dati_nuovi_vincono_a_parita_di_data(self, tmp_path, monkeypatch):
        f = self._f(tmp_path, monkeypatch)
        idx = pd.bdate_range("2025-01-01", periods=5)
        f._save_disk("SPY", pd.Series([100.0] * 5, index=idx, name="close"))
        f._save_disk("SPY", pd.Series([111.0] * 5, index=idx, name="close"))
        s = f._load_disk("SPY")
        assert s.iloc[-1] == 111.0, "il prezzo adjusted rivisto deve vincere"

    def test_load_rifiuta_il_disco_che_non_copre_lo_start(self, tmp_path, monkeypatch):
        f = self._f(tmp_path, monkeypatch)
        # disco con solo 6 mesi
        f._save_disk("SPY", pd.Series(
            [100.0] * 130, index=pd.bdate_range("2026-01-01", periods=130), name="close"))
        # chiedo 3 anni indietro → il disco NON copre → deve dire None (→ re-fetch)
        got = f._load_disk("SPY", want_end=date(2026, 7, 1), want_start=date(2023, 7, 1))
        assert got is None

    def test_load_accetta_il_disco_che_copre(self, tmp_path, monkeypatch):
        f = self._f(tmp_path, monkeypatch)
        f._save_disk("SPY", pd.Series(
            [100.0] * 700, index=pd.bdate_range("2023-01-02", periods=700), name="close"))
        got = f._load_disk("SPY", want_end=date(2025, 9, 1), want_start=date(2023, 6, 1))
        assert got is not None and len(got) == 700

    def test_asset_giovane_non_va_in_refetch_loop(self, tmp_path, monkeypatch):
        """ETF nato ieri: il disco NON può coprire il 1970, ma è già il massimo
        disponibile → tolleranza, non re-fetch infinito."""
        f = self._f(tmp_path, monkeypatch)
        idx = pd.bdate_range("2026-01-01", periods=100)
        f._save_disk("NEW", pd.Series([10.0] * 100, index=idx, name="close"))
        got = f._load_disk("NEW", want_end=date(2026, 5, 20),
                           want_start=date(2026, 1, 1))
        assert got is not None, "start uguale al primo dato disponibile → accetta"
