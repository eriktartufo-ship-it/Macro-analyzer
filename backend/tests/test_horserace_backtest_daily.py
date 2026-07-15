"""Test dell'harness daily (B2) — funzioni pure + e2e offline con grid fake.

No rete/DB: daily_matrix sintetica, regime grid fake. Esercita la logica REALE
_decide/sizer/scoring a cadenza daily.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from scripts.horserace_backtest_daily import (
    _momentum_daily,
    _vol_daily,
    _bench_return_bh,
    run_horserace_trade_daily,
)


def _matrix(dates, cols: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(cols, index=pd.to_datetime(dates))


# ---------- _momentum_daily ----------

def test_momentum_bullish_on_positive_trend():
    dates = pd.bdate_range("2020-01-01", periods=70)
    # +0.2%/giorno costante → cumulato positivo
    m = _matrix(dates, {"gold": [0.002] * 70})
    sig = _momentum_daily(m, "gold", pd.Timestamp("2020-04-20"), threshold=0.0, lookback=63)
    assert sig == "BULLISH"


def test_momentum_bearish_on_negative_trend():
    dates = pd.bdate_range("2020-01-01", periods=70)
    m = _matrix(dates, {"gold": [-0.002] * 70})
    sig = _momentum_daily(m, "gold", pd.Timestamp("2020-04-20"), threshold=0.0, lookback=63)
    assert sig == "BEARISH"


def test_momentum_neutral_insufficient_history():
    dates = pd.bdate_range("2020-01-01", periods=5)
    m = _matrix(dates, {"gold": [0.01] * 5})
    assert _momentum_daily(m, "gold", pd.Timestamp("2020-01-10"), 0.0) == "NEUTRAL"


def test_momentum_no_future_leak():
    # ritorni negativi PRIMA del target, positivi DOPO: il segnale non deve vedere il dopo
    dates = pd.bdate_range("2020-01-01", periods=130)
    vals = [-0.003] * 65 + [0.01] * 65
    m = _matrix(dates, {"gold": vals})
    target = dates[65]  # confine
    sig = _momentum_daily(m, "gold", target, threshold=0.0, lookback=63)
    assert sig == "BEARISH"  # usa solo il passato negativo


# ---------- _vol_daily ----------

def test_vol_annualized_positive():
    dates = pd.bdate_range("2020-01-01", periods=70)
    rng = np.random.default_rng(0)
    m = _matrix(dates, {"gold": rng.normal(0, 0.01, 70).tolist()})
    v = _vol_daily(m, "gold", pd.Timestamp("2020-04-20"), lookback=63)
    assert 0.05 < v < 0.40  # ~0.01*sqrt(252) ≈ 0.16


def test_vol_default_when_insufficient():
    dates = pd.bdate_range("2020-01-01", periods=5)
    m = _matrix(dates, {"gold": [0.01] * 5})
    assert _vol_daily(m, "gold", pd.Timestamp("2020-01-10")) == 0.20


# ---------- _bench_return_bh ----------

def test_bench_6040_compounded():
    dates = pd.bdate_range("2020-01-01", periods=3)
    m = _matrix(dates, {
        "us_equities_growth": [0.10, 0.0, 0.0],
        "us_bonds_long": [0.0, 0.05, 0.0],
    })
    # giorno1: 0.6*0.10=0.06; giorno2: 0.4*0.05=0.02; giorno3: 0
    # (1.06)(1.02)-1 = 0.0812
    r = _bench_return_bh(m, dates, {"us_equities_growth": 0.60, "us_bonds_long": 0.40})
    assert r == pytest_approx(0.0812)


# ---------- e2e con grid fake ----------

class _FakeGrid:
    def __init__(self, probs, conf):
        self._probs, self._conf = probs, conf

    def get(self, day):
        return {"regime": max(self._probs, key=self._probs.get),
                "confidence": self._conf, "probs": self._probs,
                "grid_date": pd.Timestamp(day)}


def test_e2e_daily_runs_and_deploys():
    from app.services.scoring.engine import ASSET_REGIME_DATA
    assets = list(ASSET_REGIME_DATA.keys())
    dates = pd.bdate_range("2020-01-01", periods=120)
    rng = np.random.default_rng(1)
    # trend leggermente positivo per tutti → momentum BULLISH dopo il warmup
    cols = {a: (rng.normal(0.0004, 0.008, 120)).tolist() for a in assets}
    m = _matrix(dates, cols)
    grid = _FakeGrid({"reflation": 0.15, "stagflation": 0.1,
                      "deflation": 0.1, "goldilocks": 0.65}, conf=0.7)
    t = run_horserace_trade_daily(date(2020, 1, 2), m, grid, horizon_days=90, cadence="daily")
    assert t is not None
    assert t.realized_return is not None
    assert np.isfinite(t.nav_path[-1])
    assert t.avg_deployed >= 0.0
    assert -1.0 <= t.max_drawdown <= 0.0


def test_e2e_low_confidence_blocks_entries():
    from app.services.scoring.engine import ASSET_REGIME_DATA
    assets = list(ASSET_REGIME_DATA.keys())
    dates = pd.bdate_range("2020-01-01", periods=120)
    m = _matrix(dates, {a: [0.001] * 120 for a in assets})
    # confidence < 0.30 → uncertainty gate blocca tutti gli ingressi
    grid = _FakeGrid({"reflation": 0.25, "stagflation": 0.25,
                      "deflation": 0.25, "goldilocks": 0.25}, conf=0.20)
    t = run_horserace_trade_daily(date(2020, 1, 2), m, grid, horizon_days=90, cadence="daily")
    assert t is not None
    assert t.avg_deployed == 0.0  # nessun deployment


def pytest_approx(x, tol=1e-4):
    import pytest
    return pytest.approx(x, abs=tol)
