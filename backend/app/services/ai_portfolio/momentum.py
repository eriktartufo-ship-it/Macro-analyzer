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


# ---------------------------------------------------------------------------
# Lettura di mercato per il decisore LLM (v6, 2026-07-15)
# Fino alla v5 il bot Gemini decideva CIECO sui prezzi: il price_history veniva
# fetchato DOPO la chiamata e usato solo per loggare la decisione. Questi helper
# trasformano i prezzi Yahoo (che già scarichiamo) in un blocco leggibile.
# ---------------------------------------------------------------------------

def trailing_return(prices: pd.Series, days: int) -> Optional[float]:
    """Rendimento cumulato degli ultimi `days` giorni di borsa. None se dati insuff."""
    s = prices.dropna()
    if len(s) < days + 1:
        return None
    return float(s.iloc[-1] / s.iloc[-(days + 1)] - 1.0)


def dist_from_ma_pct(prices: pd.Series, window: int = 50) -> Optional[float]:
    """Distanza % del prezzo dalla sua MA(window). Positiva = sopra la media.

    Più informativo del booleano `compute_ma_cross`: dice QUANTO sopra/sotto,
    non solo da che parte.
    """
    s = prices.dropna()
    if len(s) < window:
        return None
    ma = s.rolling(window=window).mean().iloc[-1]
    if pd.isna(ma) or float(ma) == 0.0:
        return None
    return float(s.iloc[-1] / float(ma) - 1.0)


def relative_strength(
    prices: pd.Series, benchmark: pd.Series, days: int = 63,
) -> Optional[float]:
    """Forza relativa vs benchmark = ret(asset, days) − ret(benchmark, days).

    È la misura OPERATIVA della "rotazione": i flussi veri dei fondi non sono
    osservabili gratis, ma quando il capitale ruota da un'asset class a un'altra
    lo si vede nella performance RELATIVA. >0 = sta sovraperformando il benchmark.
    """
    a = trailing_return(prices, days)
    b = trailing_return(benchmark, days)
    if a is None or b is None:
        return None
    return a - b


def market_snapshot(
    asset_class: str,
    price_history: dict[str, pd.Series],
    benchmark: Optional[pd.Series] = None,
) -> dict:
    """Fotografia di mercato di un asset dai prezzi Yahoo.

    Ritorna dict con: signal, rsi, dist_ma50, vol_60d, ret_1m/3m/6m e (se dato
    un benchmark) rel_strength_3m. Valori None = dati insufficienti (il consumer
    li omette invece di inventare default).
    """
    s = price_history.get(asset_class)
    if s is None or s.empty:
        return {}
    out = {
        "signal": momentum_signal(s),
        "rsi": compute_rsi(s, period=14),
        "dist_ma50": dist_from_ma_pct(s, window=50),
        "vol_60d": compute_vol_annualized(s, window=60),
        "ret_1m": trailing_return(s, 21),
        "ret_3m": trailing_return(s, 63),
        "ret_6m": trailing_return(s, 126),
    }
    if benchmark is not None and not benchmark.empty:
        out["rel_strength_3m"] = relative_strength(s, benchmark, days=63)
    return out
