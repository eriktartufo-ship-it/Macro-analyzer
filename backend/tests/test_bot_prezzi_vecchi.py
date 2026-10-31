"""Prezzi dei bot: la cache su disco non deve far contare zero (o un prezzo di ieri) ai bot.

BUG REALE (misurato in PROD il 2026-10-31): dal 23/10 i NAV dei bot Modello e LLM restavano fermi
al centesimo in 6 giri su 8, mentre la chiusura contata avanzava e l'S&P si muoveva. Il refresh
delle 00:00 UTC scarica i prezzi quando Yahoo spesso non ha ancora la chiusura del giorno prima;
alle 05:30 il bot riusa quel file (ha meno di 6 ore e arriva entro 3 giorni dalla data chiesta)
⇒ prezzo della chiusura nuova == prezzo della chiusura già contata ⇒ rendimento esattamente 0.
L'unico giro che si è mosso (27/10, +0,4435%) era il rendimento vero 23→26/10 (+0,443%).
"""
from __future__ import annotations

import sys
import types
from datetime import date, timedelta

import pandas as pd
import pytest

from app.services.ai_portfolio import market_calendar as mc
from app.services.ai_portfolio import pnl_tracker
from app.services.prices import yahoo_fetcher as yfm
from app.services.prices.yahoo_fetcher import YahooFetcher

GIA_CONTATA = date(2026, 10, 28)  # mercoledì
NUOVA = date(2026, 10, 29)        # giovedì


def _serie(fino_a: date, giorni: int = 30) -> pd.Series:
    idx = pd.bdate_range(fino_a - timedelta(days=giorni), fino_a)
    return pd.Series([100.0 + i for i in range(len(idx))], index=idx, name="close")


@pytest.fixture
def cache_scritta_a_mezzanotte(monkeypatch, tmp_path):
    """Disco con una serie che finisce alla chiusura GIÀ contata, file appena scritto (fresco per
    l'età). Yahoo, se interrogato, ha anche la chiusura nuova. Ritorna la lista delle chiamate."""
    monkeypatch.setattr(yfm, "_CACHE_ROOT", tmp_path)
    calls: list = []

    def fake_download(ticker, start, end, **kw):
        calls.append(ticker)
        # yfinance: `end` esclusivo
        idx = pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end) - pd.Timedelta(days=1))
        return pd.DataFrame({"Close": [100.0 + i for i in range(len(idx))]}, index=idx)

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=fake_download))
    f = YahooFetcher()
    for t in ("XLE", "QUAL"):
        f._save_disk(t, _serie(GIA_CONTATA, giorni=400))
    monkeypatch.setattr(pnl_tracker, "_fetcher_singleton", f)
    return calls


def test_rendimento_fra_chiusure_non_e_zero_con_la_cache_di_mezzanotte(cache_scritta_a_mezzanotte):
    """IL BUG: sul codice del 31/10 questo dava 0.0 (prezzo del 29 = prezzo del 28)."""
    r = pnl_tracker.get_return_between_for_ticker("XLE", GIA_CONTATA, NUOVA)
    assert r is not None
    assert r != 0.0, "la chiusura nuova non è nella cache: va riscaricata, non contata come zero"
    assert cache_scritta_a_mezzanotte == ["XLE"], "una sola chiamata a Yahoo, per il ticker scoperto"


def test_nessuna_chiamata_se_la_cache_ha_gia_la_chiusura(cache_scritta_a_mezzanotte):
    """Il caso sano tace: se il disco arriva già alla chiusura nuova non si riscarica."""
    r = pnl_tracker.get_return_between_for_ticker("XLE", date(2026, 10, 27), GIA_CONTATA)
    assert r is not None and r != 0.0
    assert cache_scritta_a_mezzanotte == []


def test_nessuna_chiusura_nuova_vale_zero_senza_chiamate(cache_scritta_a_mezzanotte):
    """Stessa chiusura (weekend): 0 per definizione, e Yahoo non serve."""
    assert pnl_tracker.get_return_between_for_ticker("XLE", GIA_CONTATA, GIA_CONTATA) == 0.0
    assert cache_scritta_a_mezzanotte == []


def test_prezzo_attuale_dei_bot_e_quello_dell_ultima_seduta(cache_scritta_a_mezzanotte):
    """Il prezzo che decide stop, take profit e prezzo d'ingresso: alle 05:30 del venerdì 30/10
    serve la chiusura di giovedì 29, non quella di mercoledì rimasta in cache."""
    p = pnl_tracker.get_latest_price("factor_quality", today=date(2026, 10, 30))
    stale = float(_serie(GIA_CONTATA, giorni=400).iloc[-1])
    assert p is not None and p != stale
    assert cache_scritta_a_mezzanotte == ["QUAL"]


def test_fetch_senza_need_through_resta_com_era(cache_scritta_a_mezzanotte):
    """Chi non chiede una chiusura precisa (il resto dell'app) usa la cache come prima."""
    s = pnl_tracker._fetcher().fetch("XLE", start_date=GIA_CONTATA - timedelta(days=20), end_date=NUOVA)
    assert s.index.max().date() == GIA_CONTATA
    assert cache_scritta_a_mezzanotte == []


@pytest.mark.parametrize("oggi, attesa", [
    (date(2026, 10, 30), date(2026, 10, 29)),  # venerdì → giovedì
    (date(2026, 10, 31), date(2026, 10, 30)),  # sabato → venerdì
    (date(2026, 11, 1), date(2026, 10, 30)),   # domenica → venerdì
    (date(2026, 11, 2), date(2026, 10, 30)),   # lunedì → venerdì
])
def test_ultimo_giorno_feriale_prima_di(oggi, attesa):
    assert mc.last_weekday_before(oggi) == attesa
