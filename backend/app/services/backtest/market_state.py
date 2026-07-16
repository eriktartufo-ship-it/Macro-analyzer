"""STATO DI STRESS DI MERCATO — affidabilita' del segnale di flusso (S43 #4).

Il diagnostico `scripts/flow_regime_filter.py` ha misurato: il momentum settoriale e' robusto in
CALMA (S&P sopra la media) e CROLLA in STRESS (S&P sotto la media) -> "momentum in stress =
trappola". Questo modulo espone lo stato corrente cosi' le viste flussi possono dire QUANDO
fidarsi. Trend filter classico: S&P vs SMA200 giorni (~10 mesi). NO FUTURE LEAK (SMA trailing).
"""
from __future__ import annotations

import pandas as pd

SP = "us_equities_growth"


def market_stress_state(daily_matrix: pd.DataFrame, sma_days: int = 200) -> dict:
    """Stato corrente: stress se il livello S&P e' sotto la sua SMA a `sma_days` giorni.

    Ritorna dict con: reliable (bool, True in calma), stress (bool), dist_pct (S&P vs SMA, %),
    sma_days. Se dati insufficienti -> reliable None (sconosciuto).
    """
    if SP not in daily_matrix.columns:
        return {"reliable": None, "stress": None, "dist_pct": None, "sma_days": sma_days}
    s = daily_matrix[SP].dropna()
    if len(s) < sma_days + 1:
        return {"reliable": None, "stress": None, "dist_pct": None, "sma_days": sma_days}
    level = (1.0 + s).cumprod()
    sma = level.rolling(sma_days).mean()
    lv_now = float(level.iloc[-1])
    sma_now = float(sma.iloc[-1])
    dist = (lv_now / sma_now - 1.0) * 100.0
    stress = lv_now < sma_now
    return {
        "reliable": (not stress),
        "stress": bool(stress),
        "dist_pct": round(dist, 2),
        "sma_days": sma_days,
    }
