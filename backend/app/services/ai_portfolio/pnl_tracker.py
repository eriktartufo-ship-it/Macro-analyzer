"""Real PnL tracker — calcola PnL effettivo da Yahoo daily prices.

Fase 2 sostituzione: prima `estimated_pnl_pct` veniva calcolato come
drift score × sensitivity. Ora usa il prezzo Yahoo corrente vs prezzo
catturato all'entry della prima tranche.

Volume Weighted Avg Entry Price (VWAP-like): se più tranche, media pesata
per size_pct di ciascuna entry.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from app.services.ai_portfolio.momentum import asset_to_ticker
from app.services.prices.yahoo_fetcher import YahooFetcher

logger = logging.getLogger(__name__)


_fetcher_singleton: YahooFetcher | None = None


def _fetcher() -> YahooFetcher:
    global _fetcher_singleton
    if _fetcher_singleton is None:
        _fetcher_singleton = YahooFetcher()
    return _fetcher_singleton


def get_latest_price(asset_class: str) -> Optional[float]:
    """Ritorna l'ultimo close Yahoo per l'asset, o None se fallisce."""
    ticker = asset_to_ticker(asset_class)
    if not ticker:
        return None
    try:
        from datetime import date, timedelta

        end = date.today()
        start = end - timedelta(days=10)
        s = _fetcher().fetch(ticker, start_date=start, end_date=end)
        if s is None or s.empty:
            return None
        return float(s.iloc[-1])
    except Exception as e:
        logger.warning(f"pnl_tracker: fetch failed for {asset_class}/{ticker}: {e}")
        return None


def get_daily_return_for_ticker(ticker: str) -> Optional[float]:
    """Return giornaliero reale Yahoo per un ticker arbitrario: (last/prev-1).

    Usato per benchmark esterni all'universe (SPY per S&P500, TLT per bonds).
    """
    try:
        from datetime import date, timedelta

        end = date.today()
        start = end - timedelta(days=15)  # buffer weekend+holiday
        s = _fetcher().fetch(ticker, start_date=start, end_date=end)
        if s is None or s.empty or len(s) < 2:
            return None
        last = float(s.iloc[-1])
        prev = float(s.iloc[-2])
        if prev == 0:
            return None
        return (last - prev) / prev
    except Exception as e:
        logger.warning(f"pnl_tracker: daily_return failed for {ticker}: {e}")
        return None


def get_daily_return(asset_class: str) -> Optional[float]:
    """Return giornaliero reale Yahoo per un asset interno via mapping ticker."""
    ticker = asset_to_ticker(asset_class)
    if not ticker:
        return None
    return get_daily_return_for_ticker(ticker)


def get_return_between_for_ticker(ticker: str, from_close, to_close) -> Optional[float]:
    """Rendimento fra la chiusura vista al giro precedente e quella vista oggi (market_calendar).

    Sostituisce `last/prev` nello snapshot: con `last/prev` il venerdì veniva contato a ogni
    giro senza una chiusura nuova (sabato, domenica, lunedì notte). Qui, senza chiusura nuova,
    il rendimento vale 0.
    """
    from datetime import timedelta

    from app.services.ai_portfolio.market_calendar import return_between

    try:
        s = _fetcher().fetch(ticker, start_date=from_close - timedelta(days=15), end_date=to_close)
        if s is None or s.empty:
            return None
        return return_between(s, from_close, to_close)
    except Exception as e:
        logger.warning(f"pnl_tracker: return_between failed for {ticker}: {e}")
        return None


def get_return_between(asset_class: str, from_close, to_close) -> Optional[float]:
    ticker = asset_to_ticker(asset_class)
    if not ticker:
        return None
    return get_return_between_for_ticker(ticker, from_close, to_close)


def spy_closes(until_run_date, days: int = 25):
    """Date di chiusura di SPY disponibili ORA (fetch fresco), per decidere cosa è nuovo."""
    from datetime import timedelta

    from app.services.ai_portfolio.market_calendar import close_dates

    try:
        s = _fetcher().fetch("SPY", start_date=until_run_date - timedelta(days=days), end_date=until_run_date,
                             force_refresh=True)
        return close_dates(s) if s is not None and not s.empty else []
    except Exception as e:
        logger.warning(f"pnl_tracker: chiusure SPY non disponibili: {e}")
        return []


def compute_pnl_pct(entry_price: float | None, current_price: float | None) -> float:
    """PnL % = (current - entry) / entry. 0 se uno dei due è None o entry=0."""
    if not entry_price or not current_price or entry_price == 0:
        return 0.0
    return (current_price - entry_price) / entry_price


def update_avg_entry_price(
    current_avg: float | None,
    tranches_filled_before: int,
    new_entry_price: float,
) -> float:
    """Update VWAP avg entry price quando si apre una tranche aggiuntiva.

    Pesi: ogni tranche conta uguale (semplificazione — same size).
    """
    if current_avg is None or tranches_filled_before == 0:
        return new_entry_price
    n_old = tranches_filled_before
    return (current_avg * n_old + new_entry_price) / (n_old + 1)
