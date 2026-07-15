"""Test deploy-boost combo (USE_AI_PORTFOLIO_DEPLOY_BOOST, PROMOSSO default ON 2026-07-13).

Copre le leve pure (fastramp cap tranche, deadband soglia incremental) con flag
on/off + il default del flag. La logica floor è testata come mirror in
test_horserace_backtest.py (_apply_floor) — qui verifichiamo che rispetti il gate.
"""
from __future__ import annotations

from app.services import config_flags
from app.services.ai_portfolio import deploy_boost

FLAG = "USE_AI_PORTFOLIO_DEPLOY_BOOST"


def _set(v):
    config_flags.set_runtime_flag(FLAG, v, persist=False)


def teardown_function():
    config_flags.set_runtime_flag(FLAG, None, persist=False)


# ---------- flag default ----------

def test_flag_default_on(monkeypatch):
    # PROMOSSO 2026-07-13 dopo walk-forward crisi (2008/2020/all_crisis).
    _set(None)
    monkeypatch.delenv(FLAG, raising=False)
    assert config_flags.use_ai_portfolio_deploy_boost() is True
    assert deploy_boost.enabled() is True


def test_flag_env_can_disable(monkeypatch):
    # reversibile: env=0 spegne il combo (rollback senza deploy)
    _set(None)
    monkeypatch.setenv(FLAG, "0")
    assert config_flags.use_ai_portfolio_deploy_boost() is False


def test_flag_in_registry():
    state = config_flags.get_all_flags_state()
    assert FLAG in state
    assert state[FLAG]["value"] is True


# ---------- fastramp: cap_n_tranches ----------

def test_cap_n_tranches_off_is_passthrough():
    _set(False)
    assert deploy_boost.cap_n_tranches(5) == 5
    assert deploy_boost.cap_n_tranches(2) == 2


def test_cap_n_tranches_on_caps_at_2():
    _set(True)
    assert deploy_boost.cap_n_tranches(5) == 2
    assert deploy_boost.cap_n_tranches(3) == 2
    assert deploy_boost.cap_n_tranches(2) == 2
    assert deploy_boost.cap_n_tranches(1) == 1  # sotto il cap resta
    assert deploy_boost.cap_n_tranches(0) == 0  # 0 (uncertainty gate) resta 0


# ---------- deadband: incremental_threshold ----------

def test_incremental_threshold_off_uses_default():
    _set(False)
    assert deploy_boost.incremental_threshold(60.0, 55.0) == 60.0


def test_incremental_threshold_on_uses_entry():
    _set(True)
    assert deploy_boost.incremental_threshold(60.0, 55.0) == 55.0


# ---------- floor gate: no-op quando OFF ----------

def test_floor_noop_when_disabled():
    _set(False)
    # db=None: se tentasse una query esploderebbe → prova che esce subito sul gate
    n = deploy_boost.apply_deployment_floor(
        db=None, strategy_type="model_driven", scores={"gold": 70},
        eligible={"gold"}, target_date=None, regime="reflation",
    )
    assert n == 0
