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
