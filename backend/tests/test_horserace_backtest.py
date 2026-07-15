"""Test funzioni pure di stato dell'harness horserace (no rete/DB).

Coprono: diluizione PnL su tranche aggiuntiva, return mensile, drift+rinormalizzazione
dei pesi, e la state-machine _apply_sim vs le azioni dello strategist reale.
"""
from __future__ import annotations

import pytest

from scripts.horserace_backtest import (
    HorseraceTrade,
    _apply_floor,
    _apply_sim,
    dilute_pnl_on_add,
    drift_weights,
    hedge_month_contribution,
    portfolio_month_return,
    vol_target_scale,
)
from datetime import date


def _trade():
    return HorseraceTrade(start_date=date(2022, 1, 1), end_date=date(2022, 12, 31))


# ---------- dilute_pnl_on_add ----------

def test_dilute_pnl_halves_on_equal_add():
    # pnl +20% su cost 0.10, aggiungo altra 0.10 a prezzo corrente → pnl si dimezza
    assert dilute_pnl_on_add(0.20, 0.10, 0.10) == pytest.approx(0.10)


def test_dilute_pnl_zero_cost_guard():
    assert dilute_pnl_on_add(0.5, 0.0, 0.0) == 0.0


# ---------- portfolio_month_return ----------

def test_portfolio_month_return_positions_plus_cash():
    positions = {"gold": {"weight": 0.3}, "energy": {"weight": 0.2}}
    r = portfolio_month_return(positions, cash_weight=0.5,
                               asset_returns={"gold": 0.10, "energy": -0.05},
                               cash_return=0.004)
    # 0.3*0.10 + 0.2*(-0.05) + 0.5*0.004 = 0.03 -0.01 +0.002 = 0.022
    assert r == pytest.approx(0.022)


def test_portfolio_month_return_missing_asset_return_ignored():
    positions = {"gold": {"weight": 0.4}}
    r = portfolio_month_return(positions, 0.6, {"gold": None}, 0.001)
    assert r == pytest.approx(0.6 * 0.001)


# ---------- drift_weights ----------

def test_drift_weights_renormalizes_to_one():
    positions = {"gold": {"weight": 0.5, "pnl_pct": 0.0}}
    cash = 0.5
    pr = portfolio_month_return(positions, cash, {"gold": 0.20}, 0.0)  # 0.10
    new_cash = drift_weights(positions, cash, {"gold": 0.20}, 0.0, pr)
    total = positions["gold"]["weight"] + new_cash
    assert total == pytest.approx(1.0)
    # gold pesa di più dopo un +20% mese
    assert positions["gold"]["weight"] > 0.5
    assert positions["gold"]["pnl_pct"] == pytest.approx(0.20)


# ---------- _apply_sim state machine ----------

def test_apply_open_creates_position():
    positions, tr = {}, _trade()
    _apply_sim(positions, "gold", "OPEN_TRANCHE", size=0.05, score=60,
               n_tranches=5, regime="reflation", trade=tr)
    assert positions["gold"]["weight"] == pytest.approx(0.05)
    assert positions["gold"]["tranches_filled"] == 1
    assert tr.n_opens == 1


def test_apply_open_second_tranche_adds_and_dilutes():
    positions, tr = {}, _trade()
    _apply_sim(positions, "gold", "OPEN_TRANCHE", 0.05, 60, 5, "reflation", tr)
    positions["gold"]["pnl_pct"] = 0.20  # simuliamo un guadagno prima della 2a tranche
    _apply_sim(positions, "gold", "OPEN_TRANCHE", 0.05, 62, 5, "reflation", tr)
    assert positions["gold"]["weight"] == pytest.approx(0.10)
    assert positions["gold"]["tranches_filled"] == 2
    assert positions["gold"]["pnl_pct"] == pytest.approx(0.10)  # diluito


def test_apply_full_close_removes_position():
    positions, tr = {}, _trade()
    _apply_sim(positions, "gold", "OPEN_TRANCHE", 0.05, 60, 5, "reflation", tr)
    _apply_sim(positions, "gold", "FULL_CLOSE", 0.0, 40, 5, "reflation", tr)
    assert "gold" not in positions
    assert tr.n_closes == 1


def test_apply_close_tranche_partial_then_removes_when_empty():
    positions, tr = {}, _trade()
    _apply_sim(positions, "gold", "OPEN_TRANCHE", 0.05, 60, 1, "reflation", tr)
    # 1 sola tranche filled → close_tranche svuota e rimuove
    _apply_sim(positions, "gold", "CLOSE_TRANCHE", 0.0, 60, 1, "reflation", tr)
    assert "gold" not in positions
    assert tr.n_closes == 1


# ---------- _apply_floor (leva deployment minimo) ----------

def test_apply_floor_deploys_deficit_to_top_regime():
    positions = {}
    scores = {"gold": 70, "energy": 65, "silver": 60, "junk": 10}
    regime_top = {"gold", "energy", "silver"}  # junk fuori dal regime → escluso
    # deployment attuale 0 → floor 0.30 su 3 candidati = 0.10 ciascuno
    _apply_floor(positions, scores, regime_top, floor_pct=0.30, cash_weight=1.0, regime="reflation")
    deployed = sum(p["weight"] for p in positions.values())
    assert deployed == pytest.approx(0.30, abs=1e-6)
    assert "junk" not in positions
    assert positions["gold"]["weight"] == pytest.approx(0.10)


def test_apply_floor_respects_cap_and_noop_above_floor():
    # già sopra il floor → nessuna modifica
    positions = {"gold": {"weight": 0.5, "pnl_pct": 0.0}}
    _apply_floor(positions, {"gold": 70}, {"gold"}, floor_pct=0.30, cash_weight=0.5, regime="reflation")
    assert positions["gold"]["weight"] == pytest.approx(0.5)


# ---------- _apply_sim ritorna il notional scambiato (leva costs) ----------

def test_apply_sim_returns_traded_notional():
    positions, tr = {}, _trade()
    # open → traded = size
    assert _apply_sim(positions, "gold", "OPEN_TRANCHE", 0.05, 60, 5, "reflation", tr) == pytest.approx(0.05)
    # add tranche → traded = size aggiunta
    assert _apply_sim(positions, "gold", "OPEN_TRANCHE", 0.05, 62, 5, "reflation", tr) == pytest.approx(0.05)
    # full close → traded = peso residuo (0.10)
    assert _apply_sim(positions, "gold", "FULL_CLOSE", 0.0, 40, 5, "reflation", tr) == pytest.approx(0.10)
    # azione su posizione inesistente → 0
    assert _apply_sim(positions, "ghost", "CLOSE_TRANCHE", 0.0, 50, 1, "reflation", tr) == 0.0


# ---------- vol_target_scale (leva voltarget #2) ----------

def test_vol_target_scale_caps_at_one_when_below_budget():
    # portafoglio a bassa vol → nessun leverage (cap 1.0), non scala su
    weights = {"bonds": 0.20}
    vols = {"bonds": 0.05}
    assert vol_target_scale(weights, vols, target_vol=0.10) == pytest.approx(1.0)


def test_vol_target_scale_reduces_when_above_budget():
    # single asset vol 40%, peso pieno → scale = 0.10/0.40 = 0.25
    assert vol_target_scale({"eq": 1.0}, {"eq": 0.40}, target_vol=0.10) == pytest.approx(0.25)


def test_vol_target_scale_empty_portfolio_is_noop():
    assert vol_target_scale({}, {}, target_vol=0.10) == 1.0


def test_vol_target_scale_correlation_raises_port_vol():
    # due asset correlati (ρ=0.3) hanno vol > della somma in quadratura pura
    w = {"a": 0.5, "b": 0.5}
    v = {"a": 0.20, "b": 0.20}
    # contribs 0.1,0.1 → var = 0.01+0.01 + 2*0.3*0.1*0.1 = 0.02+0.006 = 0.026 → pv≈0.1612
    scale = vol_target_scale(w, v, target_vol=0.10)
    assert scale == pytest.approx(0.10 / (0.026 ** 0.5), rel=1e-6)


# ---------- hedge_month_contribution (leva hedge #4, PROXY) ----------

def test_hedge_noop_when_zero_pct():
    assert hedge_month_contribution(0.0, equity_ret=-0.20, cash_ret=0.003) == 0.0


def test_hedge_negative_carry_in_calm_month():
    # mese calmo (equity +2%) → solo carry negativo al netto del cash
    c = hedge_month_contribution(0.05, equity_ret=0.02, cash_ret=0.003)
    # 0.05 * (-0.004 - 0.003) = 0.05 * -0.007 = -0.00035
    assert c == pytest.approx(0.05 * (-0.004 - 0.003))


def test_hedge_convex_payoff_in_crash():
    # crash -15%: payoff = k*( -(-0.15) - 0.08 ) = 3*(0.15-0.08)=0.21 → sleeve +21%
    c = hedge_month_contribution(0.05, equity_ret=-0.15, cash_ret=0.0, k=3.0, thresh=-0.08)
    assert c == pytest.approx(0.05 * (3.0 * (0.15 - 0.08)))
    assert c > 0  # in crash la sleeve è nettamente positiva
