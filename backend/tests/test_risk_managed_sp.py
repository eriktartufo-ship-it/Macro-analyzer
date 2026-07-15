"""Bot "S&P a rischio gestito" — logica pura (2026-07-15, richiesta Erik).

Blinda le proprietà che lo rendono ciò che è stato misurato:
sopra MA200 = beta azionario · sotto = de-risk · default prudente se non sa.
No rete/DB.
"""
from __future__ import annotations

import pandas as pd

from app.services.ai_portfolio.risk_managed_sp import (
    CASH_ASSET,
    SP_ASSET,
    decide_target,
    is_above_trend,
    pick_refuge,
)


def _s(vals):
    return pd.Series(vals, index=pd.bdate_range("2024-01-01", periods=len(vals)))


# ---------- is_above_trend ----------

def test_sopra_trend_se_prezzo_sale():
    # 200 giorni a 100, poi sale a 130 -> sopra la MA200
    assert is_above_trend(_s([100.0] * 200 + [130.0]), window=200) is True


def test_sotto_trend_se_prezzo_crolla():
    assert is_above_trend(_s([100.0] * 200 + [70.0]), window=200) is False


def test_none_se_storia_insufficiente():
    assert is_above_trend(_s([100.0] * 50), window=200) is None


# ---------- decide_target ----------

def test_bull_resta_sul_beta_azionario():
    hist = {SP_ASSET: _s([100.0] * 200 + [130.0])}
    target, reason = decide_target(hist)
    assert target == SP_ASSET
    assert "sopra MA200" in reason


def test_bear_esce_dal_sp():
    hist = {SP_ASSET: _s([100.0] * 200 + [70.0]), CASH_ASSET: _s([100.0] * 201)}
    target, _ = decide_target(hist)
    assert target != SP_ASSET


def test_default_prudente_senza_prezzi():
    """Se Yahoo è giù NON deve restare a caso sull'azionario."""
    target, reason = decide_target({})
    assert target == CASH_ASSET
    assert "non disponibili" in reason


def test_default_prudente_con_storia_corta():
    hist = {SP_ASSET: _s([100.0] * 30)}
    target, reason = decide_target(hist)
    assert target == CASH_ASSET
    assert "prudenziale" in reason


# ---------- pick_refuge ----------

def test_rifugio_sceglie_il_migliore_per_momentum():
    """L'oro sale, i bond scendono -> il capitale va nell'oro."""
    hist = {
        "us_bonds_long": _s([100.0] * 64 + [90.0]),
        "gold": _s([100.0] * 64 + [120.0]),
        CASH_ASSET: _s([100.0] * 65),
    }
    assert pick_refuge(hist, lookback=63, use_best=True) == "gold"


def test_rifugio_cash_se_disattivato():
    hist = {
        "gold": _s([100.0] * 64 + [120.0]),
        CASH_ASSET: _s([100.0] * 65),
    }
    assert pick_refuge(hist, lookback=63, use_best=False) == CASH_ASSET


def test_rifugio_cash_se_nessun_candidato_ha_storia():
    assert pick_refuge({}, lookback=63, use_best=True) == CASH_ASSET


def test_bear_con_rifugio_va_sul_migliore_non_su_cash():
    hist = {
        SP_ASSET: _s([100.0] * 200 + [70.0]),
        "gold": _s([100.0] * 200 + [130.0]),
        "us_bonds_long": _s([100.0] * 200 + [95.0]),
        CASH_ASSET: _s([100.0] * 201),
    }
    target, reason = decide_target(hist, use_best_refuge=True)
    assert target == "gold"
    assert "de-risk" in reason
