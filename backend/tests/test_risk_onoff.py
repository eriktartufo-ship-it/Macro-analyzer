"""Test cross-asset risk-on/off. Spec: scripts/RRG_SPEC.md."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.backtest.risk_onoff import compute_risk_onoff


def _matrix(equity_drift, bond_drift, gold_drift, cash_drift=0.00005,
            n_days=300, crypto_days=20, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    data = {
        "us_equities_growth": rng.normal(equity_drift, 0.004, n_days),
        "us_bonds_long": rng.normal(bond_drift, 0.003, n_days),
        "gold": rng.normal(gold_drift, 0.004, n_days),
        "cash_money_market": rng.normal(cash_drift, 0.0002, n_days),
    }
    df = pd.DataFrame(data, index=idx)
    # cripto: solo ultimi `crypto_days` giorni (storia corta come nel reale)
    crypto = np.full(n_days, np.nan)
    crypto[-crypto_days:] = rng.normal(0.001, 0.02, crypto_days)
    df["bitcoin"] = crypto
    return df


def test_risk_on_when_equity_leads():
    dm = _matrix(equity_drift=0.0012, bond_drift=-0.0002, gold_drift=0.0)
    r = compute_risk_onoff(dm, lookback_weeks=8)
    assert r.gauge > 0
    assert r.regime == "risk_on"


def test_risk_off_when_refuges_lead():
    dm = _matrix(equity_drift=-0.0010, bond_drift=0.0008, gold_drift=0.0009)
    r = compute_risk_onoff(dm, lookback_weeks=8)
    assert r.gauge < 0
    assert r.regime == "risk_off"


def test_best_refuge_is_highest_momentum_safe():
    # oro forte, bond debole, cash ~0 -> rifugio = oro
    dm = _matrix(equity_drift=-0.0005, bond_drift=-0.0003, gold_drift=0.0015)
    r = compute_risk_onoff(dm, lookback_weeks=8)
    assert r.best_refuge == "gold"


def test_crypto_insufficient_data_graceful():
    # cripto ha solo 20 giorni (~4 settimane) -> None con lookback 8 settimane, no crash
    dm = _matrix(equity_drift=0.0008, bond_drift=0.0, gold_drift=0.0, crypto_days=20)
    r = compute_risk_onoff(dm, lookback_weeks=8)
    crypto = next(c for c in r.classes if c.asset == "bitcoin")
    assert crypto.momentum_pct is None
    assert crypto.weeks_available < 8
    assert isinstance(r.gauge, float)


def test_no_leak_bounded_lookback():
    dm = _matrix(equity_drift=0.0009, bond_drift=0.0, gold_drift=0.0)
    base = compute_risk_onoff(dm, lookback_weeks=8)
    tampered = dm.copy()
    # manometto SOLO le colonne contigue (non bitcoin, che ha un buco -> altrimenti creerei
    # dati cripto artificiali). Per una serie contigua il momentum e' un rapporto che cancella
    # il passato comune -> invariante = no leak.
    for c in ["us_equities_growth", "us_bonds_long", "gold", "cash_money_market"]:
        tampered.iloc[:150, tampered.columns.get_loc(c)] = 0.09
    re = compute_risk_onoff(tampered, lookback_weeks=8)
    assert re.as_of == base.as_of
    assert abs(re.gauge - base.gauge) < 1e-9


def test_shape():
    dm = _matrix(0.0008, 0.0, 0.0)
    d = compute_risk_onoff(dm, lookback_weeks=8).to_dict()
    assert d["as_of"] and d["regime"] in ("risk_on", "risk_off", "neutral")
    assert len(d["classes"]) == 5
    assert set(d["classes"][0]) == {"asset", "label", "role", "momentum_pct", "weeks_available"}
