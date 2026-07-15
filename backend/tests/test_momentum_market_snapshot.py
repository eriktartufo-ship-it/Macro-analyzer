"""Helper di lettura di mercato per il decisore LLM v6 (2026-07-15).

Fino alla v5 Gemini decideva CIECO sui prezzi. Questi helper trasformano i
prezzi Yahoo in un blocco leggibile. No rete: serie sintetiche.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.ai_portfolio.momentum import (
    dist_from_ma_pct,
    market_snapshot,
    relative_strength,
    trailing_return,
)


def _series(vals):
    idx = pd.bdate_range("2025-01-01", periods=len(vals))
    return pd.Series(vals, index=idx)


# ---------- trailing_return ----------

def test_trailing_return_semplice():
    s = _series([100.0] * 20 + [110.0])  # 21 giorni fa era 100, oggi 110
    assert trailing_return(s, 20) == pytest.approx(0.10)


def test_trailing_return_none_se_storia_corta():
    assert trailing_return(_series([100.0, 101.0]), 63) is None


# ---------- dist_from_ma_pct ----------

def test_dist_ma_positiva_sopra_media():
    # 50 giorni a 100, poi sale: il prezzo sta sopra la sua MA50
    s = _series([100.0] * 50 + [120.0])
    d = dist_from_ma_pct(s, window=50)
    assert d is not None and d > 0


def test_dist_ma_negativa_sotto_media():
    s = _series([100.0] * 50 + [80.0])
    d = dist_from_ma_pct(s, window=50)
    assert d is not None and d < 0


def test_dist_ma_none_se_storia_corta():
    assert dist_from_ma_pct(_series([100.0] * 10), window=50) is None


# ---------- relative_strength ----------

def test_rel_strength_positiva_se_batte_il_benchmark():
    # asset +20% in 63g, benchmark +5% -> forza relativa ~ +15pp
    a = _series([100.0] * 63 + [120.0])
    b = _series([100.0] * 63 + [105.0])
    rs = relative_strength(a, b, days=63)
    assert rs == pytest.approx(0.15, abs=1e-6)


def test_rel_strength_negativa_se_sottoperforma():
    a = _series([100.0] * 63 + [95.0])
    b = _series([100.0] * 63 + [110.0])
    assert relative_strength(a, b, days=63) < 0


def test_rel_strength_none_se_benchmark_corto():
    a = _series([100.0] * 70)
    b = _series([100.0] * 5)
    assert relative_strength(a, b, days=63) is None


# ---------- market_snapshot ----------

def test_snapshot_contiene_le_chiavi_attese():
    rng = np.random.default_rng(0)
    prices = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 200))
    hist = {"gold": _series(prices.tolist())}
    bench = _series((100 * np.cumprod(1 + rng.normal(0.0002, 0.01, 200))).tolist())
    snap = market_snapshot("gold", hist, benchmark=bench)
    for k in ("signal", "rsi", "dist_ma50", "vol_60d", "ret_1m", "ret_3m", "ret_6m",
              "rel_strength_3m"):
        assert k in snap


def test_snapshot_vuoto_se_asset_assente():
    assert market_snapshot("non_esiste", {}) == {}


def test_snapshot_senza_benchmark_omette_rel_strength():
    hist = {"gold": _series([100.0] * 200)}
    snap = market_snapshot("gold", hist)
    assert "rel_strength_3m" not in snap
    assert "signal" in snap


def test_snapshot_usa_solo_dati_disponibili_no_invenzioni():
    # storia corta: le metriche lunghe devono essere None, NON default inventati
    hist = {"gold": _series([100.0, 101.0, 102.0])}
    snap = market_snapshot("gold", hist)
    assert snap["ret_6m"] is None
    assert snap["dist_ma50"] is None
