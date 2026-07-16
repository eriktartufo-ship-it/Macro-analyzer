"""Test stato di stress di mercato (affidabilita' segnale flusso). Spec: S43 #4."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.backtest.market_state import market_stress_state


def _sp(drift, n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"us_equities_growth": rng.normal(drift, 0.003, n)}, index=idx)


def test_calm_when_uptrend():
    # trend positivo -> livello sopra SMA -> calmo/affidabile
    st = market_stress_state(_sp(0.0015), sma_days=100)
    assert st["stress"] is False
    assert st["reliable"] is True
    assert st["dist_pct"] > 0


def test_stress_when_downtrend():
    # trend negativo recente -> livello sotto SMA -> stress
    df = _sp(0.001, n=400)
    df.iloc[-120:] = df.iloc[-120:] - 0.004  # crollo nelle ultime ~24 settimane
    st = market_stress_state(df, sma_days=100)
    assert st["stress"] is True
    assert st["reliable"] is False
    assert st["dist_pct"] < 0


def test_insufficient_data():
    st = market_stress_state(_sp(0.001, n=50), sma_days=200)
    assert st["reliable"] is None


def test_no_leak_uses_only_trailing():
    df = _sp(0.001, n=400)
    base = market_stress_state(df, sma_days=100)
    tampered = df.copy()
    tampered.iloc[:100] = 0.05  # passato remoto assurdo (comune -> non cambia il rapporto ultimo/SMA? )
    # NB: la SMA a 100 giorni sull'ULTIMO punto usa gli ultimi 100 giorni; manomettere i primi 100
    # (di 400) non li tocca -> stato invariato.
    re = market_stress_state(tampered, sma_days=100)
    assert re["stress"] == base["stress"]
