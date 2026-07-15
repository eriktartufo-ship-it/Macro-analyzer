"""Test del modello tail-hedge reale (Black-Scholes put + contributo rolled-put)."""
from __future__ import annotations

import math

import pytest

from scripts.hedge_model import bs_put, make_put_contribution


# ---------- bs_put ----------

def test_bs_put_atm_sanity():
    # ATM 1M put, σ=18%: approx S*σ*sqrt(T)*0.4 ≈ 0.0208
    p = bs_put(S=1.0, K=1.0, T=1 / 12, r=0.0, sigma=0.18)
    assert p == pytest.approx(0.0208, abs=0.004)


def test_bs_put_otm_cheaper_than_atm():
    atm = bs_put(1.0, 1.00, 1 / 12, 0.03, 0.18)
    otm = bs_put(1.0, 0.90, 1 / 12, 0.03, 0.18)
    assert 0 < otm < atm


def test_bs_put_higher_vol_more_expensive():
    lo = bs_put(1.0, 0.90, 1 / 12, 0.03, 0.15)
    hi = bs_put(1.0, 0.90, 1 / 12, 0.03, 0.30)
    assert hi > lo


def test_bs_put_degenerate_returns_intrinsic():
    assert bs_put(1.0, 1.10, 0.0, 0.03, 0.18) == pytest.approx(0.10)


# ---------- make_put_contribution ----------

def test_put_contrib_off_when_hedge_pct_zero():
    f = make_put_contribution(prem=0.005, otm=0.10, sigma=0.18)
    assert f(0.0, equity_ret=-0.20, cash_ret=0.0) == 0.0


def test_put_contrib_negative_carry_in_calm_month():
    # equity +2% → put scade OTM, payoff 0 → contributo = -prem
    f = make_put_contribution(prem=0.005, otm=0.10, sigma=0.18)
    assert f(0.05, equity_ret=0.02, cash_ret=0.0) == pytest.approx(-0.005)


def test_put_contrib_convex_payoff_in_crash():
    # crash -25%: K=0.90, S=0.75 → payoff/unità 0.15; notional = prem/premio_put
    prem, otm, sigma = 0.005, 0.10, 0.18
    f = make_put_contribution(prem, otm, sigma)
    p = f.premium
    expected = (prem / p) * (0.90 - 0.75) - prem
    got = f(0.05, equity_ret=-0.25, cash_ret=0.0)
    assert got == pytest.approx(expected)
    assert got > 0  # in un crash serio la sleeve è nettamente positiva


def test_put_contrib_leverage_from_real_price():
    # il moltiplicatore è prem/premio_BS, NON un k arbitrario
    f = make_put_contribution(prem=0.005, otm=0.10, sigma=0.18)
    assert f.premium == pytest.approx(bs_put(1.0, 0.90, 1 / 12, 0.03, 0.18))
    assert f.premium > 0
