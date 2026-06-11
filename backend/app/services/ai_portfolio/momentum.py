"""Momentum indicators (MA50 + RSI14) per AI Portfolio entry/exit triggers.

Source: Yahoo Finance prices via asset_universe ticker mapping.
Cache: in-memory per request, no persistent (chiamato 1x/giorno).

Output enum: BULLISH | NEUTRAL | BEARISH
- BULLISH: price > MA50 AND RSI > 50
- BEARISH: price < MA50 AND RSI < 50
- NEUTRAL: mixed signals
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from app.services.prices.asset_universe import ASSET_TICKERS

logger = logging.getLogger(__name__)


def compute_rsi(prices: pd.Series, period: int = 14) -> Optional[float]:
    """RSI Wilder (smoothed). Returns last value or None se < period+1 prices."""
    if len(prices) < period + 1:
        return None
    deltas = prices.diff().dropna()
    if len(deltas) < period:
        return None
    gains = deltas.clip(lower=0)
    losses = -deltas.clip(upper=0)
    avg_gain = gains.rolling(window=period).mean().iloc[-1]
    avg_loss = losses.rolling(window=period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi) if not pd.isna(rsi) else None


def compute_ma_cross(prices: pd.Series, window: int = 50) -> Optional[bool]:
    """True se latest price > MA(window), False altrimenti, None se prices < window."""
    if len(prices) < window:
        return None
    ma = prices.rolling(window=window).mean().iloc[-1]
    last = float(prices.iloc[-1])
    if pd.isna(ma):
        return None
    return last > float(ma)


def compute_vol_annualized(prices: pd.Series, window: int = 60) -> Optional[float]:
    """Volatility annualizzata (std daily returns × sqrt(252))."""
    if len(prices) < window:
        return None
    rets = prices.pct_change().dropna().tail(window)
    if len(rets) < 5:
        return None
    return float(rets.std() * np.sqrt(252))


def momentum_signal(prices: pd.Series) -> str:
    """Combined signal: BULLISH | NEUTRAL | BEARISH."""
    above_ma50 = compute_ma_cross(prices, window=50)
    rsi = compute_rsi(prices, period=14)

    if above_ma50 is None or rsi is None:
        return "NEUTRAL"
    if above_ma50 and rsi > 50:
        return "BULLISH"
    if not above_ma50 and rsi < 50:
        return "BEARISH"
    return "NEUTRAL"


def get_asset_signal_and_vol(
    asset_class: str,
    price_history: dict[str, pd.Series],
) -> tuple[str, Optional[float]]:
    """Helper: (signal, vol_60d_annualized) per un asset.

    Args:
        asset_class: chiave interna (es. "sector_energy")
        price_history: dict {asset_class: pd.Series of daily prices, indexed by date}
    """
    series = price_history.get(asset_class)
    if series is None or series.empty:
        return ("NEUTRAL", None)
    sig = momentum_signal(series)
    vol = compute_vol_annualized(series, window=60)
    return (sig, vol)


def asset_to_ticker(asset_class: str) -> Optional[str]:
    """Mapping asset interno → Yahoo ticker."""
    cfg = ASSET_TICKERS.get(asset_class, {})
    return cfg.get("ticker")
