"""Test cash idle remunerato all'overnight (fix bias horserace 2026-07-13).

Il capitale NON schierato dalle strategie AI prima rendeva 0%. I benchmark
(SPY, 60/40) sono full-invested → il residuo a 0% penalizzava artificiosamente
le AI del risk-free. Ora il residuo rende l'overnight (BIL ≈ €STR/XEON in EUR).

Logica pura, no rete: BIL daily return mockato.
"""
from __future__ import annotations

import pytest

from app.services import config_flags
from app.services.ai_portfolio import performance


# ---------- idle_cash_contribution (logica pura) ----------

def test_idle_cash_contribution_partial_deploy():
    # 60% schierato → 40% idle a overnight 0.01%/gg = 0.00004
    assert performance.idle_cash_contribution(0.6, 0.0001) == pytest.approx(0.4 * 0.0001)


def test_idle_cash_contribution_fully_deployed_is_zero():
    assert performance.idle_cash_contribution(1.0, 0.0001) == 0.0


def test_idle_cash_contribution_leverage_no_negative_cash():
    # deployed > 1 (leva): niente cash idle, mai contributo negativo
    assert performance.idle_cash_contribution(1.36, 0.0001) == 0.0


def test_idle_cash_contribution_zero_deploy_full_overnight():
    # 0% schierato → 100% a overnight
    assert performance.idle_cash_contribution(0.0, 0.0002) == pytest.approx(0.0002)


# ---------- _cash_overnight_daily_return (BIL, mockato) ----------

def test_cash_overnight_uses_bil(monkeypatch):
    called = {}

    def fake_dr(ticker):
        called["ticker"] = ticker
        return 0.00012

    monkeypatch.setattr(performance.pnl_tracker, "get_daily_return_for_ticker", fake_dr)
    assert performance._cash_overnight_daily_return() == pytest.approx(0.00012)
    assert called["ticker"] == performance.CASH_OVERNIGHT_TICKER == "BIL"


def test_cash_overnight_fallback_zero_on_yahoo_fail(monkeypatch):
    monkeypatch.setattr(
        performance.pnl_tracker, "get_daily_return_for_ticker", lambda t: None
    )
    assert performance._cash_overnight_daily_return() == 0.0


# ---------- flag ----------

def test_flag_default_on(monkeypatch):
    # Nessun override runtime/env → default True (fix attivo di default)
    config_flags.set_runtime_flag("USE_AI_PORTFOLIO_CASH_OVERNIGHT", None, persist=False)
    monkeypatch.delenv("USE_AI_PORTFOLIO_CASH_OVERNIGHT", raising=False)
    assert config_flags.use_ai_portfolio_cash_overnight() is True


def test_flag_can_be_disabled(monkeypatch):
    config_flags.set_runtime_flag("USE_AI_PORTFOLIO_CASH_OVERNIGHT", False, persist=False)
    try:
        assert config_flags.use_ai_portfolio_cash_overnight() is False
    finally:
        config_flags.set_runtime_flag("USE_AI_PORTFOLIO_CASH_OVERNIGHT", None, persist=False)


def test_flag_in_registry():
    state = config_flags.get_all_flags_state()
    assert "USE_AI_PORTFOLIO_CASH_OVERNIGHT" in state
    assert state["USE_AI_PORTFOLIO_CASH_OVERNIGHT"]["value"] is True
