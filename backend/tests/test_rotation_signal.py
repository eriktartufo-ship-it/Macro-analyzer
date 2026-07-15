"""Segnale di rotazione dual-momentum (2026-07-15, richiesta Erik).

Blinda le due proprietà che rendono il segnale ciò che Erik ha chiesto:
1. momentum RELATIVO → sceglie dove "stanno andando i soldi";
2. momentum ASSOLUTO → nei bear va in CASH (smorza le discese).
Più il no-future-leak. No rete: matrici sintetiche.
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.services.ai_portfolio.rotation import (
    momentum_score,
    rank_assets,
    select_rotation_portfolio,
    trailing_return_at,
)

CASH = "cash_money_market"


def _matrix(cols: dict[str, float], n: int = 200):
    """Matrice di rendimenti daily costanti per asset (semplice e deterministica)."""
    idx = pd.bdate_range("2025-01-01", periods=n)
    return pd.DataFrame({k: [v] * n for k, v in cols.items()}, index=idx)


# ---------- trailing_return_at ----------

def test_trailing_return_no_future_leak():
    idx = pd.bdate_range("2025-01-01", periods=100)
    vals = [0.01] * 50 + [-0.05] * 50  # crolla DOPO il giorno 50
    m = pd.DataFrame({"gold": vals}, index=idx)
    day = idx[50]
    r = trailing_return_at(m, "gold", day, lookback=50)
    assert r > 0, "deve vedere solo il passato positivo, non il crollo futuro"


def test_trailing_return_none_se_asset_assente():
    m = _matrix({"gold": 0.001})
    assert trailing_return_at(m, "non_esiste", m.index[100], 63) is None


# ---------- momentum relativo (dove vanno i soldi) ----------

def test_rank_mette_il_piu_forte_per_primo():
    m = _matrix({"gold": 0.003, "sector_technology": 0.001, "energy": -0.002})
    ranked = rank_assets(m, m.index[190], ["gold", "sector_technology", "energy"])
    assert [a for a, _ in ranked] == ["gold", "sector_technology", "energy"]


def test_momentum_score_media_piu_orizzonti():
    m = _matrix({"gold": 0.002})
    s = momentum_score(m, "gold", m.index[190], lookbacks=(63, 126))
    assert s is not None and s > 0


# ---------- il cuore: absolute momentum smorza le discese ----------

def test_bear_generalizzato_va_tutto_in_cash():
    """Se NESSUN asset batte il cash, il portafoglio deve essere 100% liquido.
    È il meccanismo che smorza le discese richiesto da Erik."""
    m = _matrix({
        "gold": -0.002, "sector_technology": -0.003, "energy": -0.004,
        CASH: 0.0002,
    })
    w = select_rotation_portfolio(
        m, m.index[190], ["gold", "sector_technology", "energy", CASH], top_k=3)
    assert w == {CASH: 1.0}


def test_bull_compra_i_top_k_e_non_il_cash():
    m = _matrix({
        "gold": 0.003, "sector_technology": 0.002, "energy": 0.001,
        "silver": -0.001, CASH: 0.0002,
    })
    w = select_rotation_portfolio(
        m, m.index[190], ["gold", "sector_technology", "energy", "silver", CASH],
        top_k=3)
    assert set(w) == {"gold", "sector_technology", "energy"}
    assert all(abs(v - 1/3) < 1e-9 for v in w.values())


def test_bear_parziale_riempie_il_resto_di_cash():
    """Solo 1 asset batte il cash su 3 slot → 1/3 investito, 2/3 in cash."""
    m = _matrix({
        "gold": 0.003, "sector_technology": -0.002, "energy": -0.003,
        CASH: 0.0002,
    })
    w = select_rotation_portfolio(
        m, m.index[190], ["gold", "sector_technology", "energy", CASH], top_k=3)
    assert w["gold"] == pytest.approx(1/3)
    assert w[CASH] == pytest.approx(2/3)
    assert sum(w.values()) == pytest.approx(1.0)


def test_senza_absolute_momentum_compra_anche_nel_bear():
    """Controprova: disattivando il filtro, il portafoglio resta investito nel bear
    (è proprio ciò che il filtro serve a evitare)."""
    m = _matrix({
        "gold": -0.002, "sector_technology": -0.003, "energy": -0.004,
        CASH: 0.0002,
    })
    w = select_rotation_portfolio(
        m, m.index[190], ["gold", "sector_technology", "energy", CASH], top_k=3,
        use_absolute_momentum=False)
    assert CASH not in w or w.get(CASH, 0) < 0.5
    assert "gold" in w


def test_pesi_sommano_sempre_a_uno():
    m = _matrix({"gold": 0.003, "energy": 0.001, CASH: 0.0002})
    for k in (1, 2, 3, 5):
        w = select_rotation_portfolio(m, m.index[190], ["gold", "energy", CASH], top_k=k)
        assert sum(w.values()) == pytest.approx(1.0)


def test_universo_vuoto_finisce_in_cash():
    m = _matrix({CASH: 0.0002})
    assert select_rotation_portfolio(m, m.index[190], [CASH], top_k=3) == {CASH: 1.0}
