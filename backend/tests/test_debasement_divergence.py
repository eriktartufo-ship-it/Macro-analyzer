"""Test Debasement divergence signal (DREAM-style gold vs real-rate fair-value gap).

Spec: R1 DREAM. Il segnale-firma di Vito Lops (DREAM Index): oro e tassi reali sono
inversamente correlati; su una finestra rolling 60m si stima il fair-value implicito
dell'oro dai tassi reali e si misura di quanto l'oro attuale sta SOPRA quel fair value.

Riproduzione live (spike 2026-07, oro spot GC=F $4013 vs DFII10 2.35%): gap +39.9%,
fair value ~$2907 — coerente col +39.5% / ~$2900 dichiarato nel video. Qui i test
coprono la MATEMATICA in modo deterministico (no rete).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from pytest import approx as pytest_approx

from app.services.backtest.debasement_divergence import (
    WINDOW_MONTHS,
    compute_divergence,
)


def _monthly(values: list[float], start: str = "2015-01-31") -> pd.Series:
    idx = pd.date_range(start=start, periods=len(values), freq="ME")
    return pd.Series(values, index=idx)


def test_exact_gap_reproduced_from_known_relationship():
    """Se i primi 60 mesi seguono ln(gold)=a+b*rr esattamente, la finestra recupera
    (a,b) e il gap dell'ultimo punto = exp(delta)-1 con delta perturbazione nota."""
    a, b = 8.0, -0.15  # ln(gold) ~ a + b*real_yield (b<0: reali su -> oro giu)
    rr = list(np.linspace(-1.0, 3.0, WINDOW_MONTHS))  # 60 tassi reali distinti
    ln_gold = [a + b * r for r in rr]

    # 61esimo punto: perturbazione nota sopra il fair value
    rr_last = 2.35
    delta = 0.20
    rr.append(rr_last)
    ln_gold.append(a + b * rr_last + delta)

    gold = _monthly([math.exp(x) for x in ln_gold])
    real_yield = _monthly(rr)

    res = compute_divergence(gold, real_yield, window=WINDOW_MONTHS)
    assert res is not None
    assert res["gap"] == pytest_approx(math.exp(delta) - 1)          # ~+22.1%
    assert res["fair_value"] == pytest_approx(math.exp(a + b * rr_last))
    assert res["real_yield"] == pytest_approx(rr_last)
    assert res["n"] == WINDOW_MONTHS + 1


def test_no_future_leak_window_excludes_current_point():
    """Il fair value dell'ultimo punto NON deve dipendere dal valore dell'oro
    dell'ultimo punto (finestra trailing esclusiva)."""
    a, b = 8.0, -0.15
    rr = list(np.linspace(-1.0, 3.0, WINDOW_MONTHS)) + [2.0]
    base = [a + b * r for r in rr]

    g1 = _monthly([math.exp(x) for x in base])
    # raddoppio SOLO l'ultimo oro: il fair value deve restare identico, il gap deve salire
    base2 = base.copy()
    base2[-1] += math.log(2)
    g2 = _monthly([math.exp(x) for x in base2])
    ry = _monthly(rr)

    r1 = compute_divergence(g1, ry, window=WINDOW_MONTHS)
    r2 = compute_divergence(g2, ry, window=WINDOW_MONTHS)
    assert r1["fair_value"] == pytest_approx(r2["fair_value"])       # invariato
    assert r2["gap"] > r1["gap"]                                     # gap raddoppiato


def test_gold_above_fair_gives_positive_gap():
    a, b = 8.0, -0.15
    rr = list(np.linspace(0.0, 2.0, WINDOW_MONTHS)) + [1.0]
    ln_gold = [a + b * r for r in rr]
    ln_gold[-1] += 0.10  # sopra il fair value
    res = compute_divergence(
        _monthly([math.exp(x) for x in ln_gold]), _monthly(rr), window=WINDOW_MONTHS
    )
    assert res["gap"] > 0
    assert res["gap_pct"] == pytest_approx(res["gap"] * 100)


def test_graceful_on_short_series():
    """Meno di window+1 punti -> None (graceful, mai crash)."""
    short = _monthly([1800.0, 1850.0, 1900.0])
    assert compute_divergence(short, _monthly([1.0, 1.1, 1.2]), window=WINDOW_MONTHS) is None


def test_history_series_present_and_dated():
    a, b = 8.0, -0.15
    rr = list(np.linspace(0.0, 2.0, WINDOW_MONTHS + 12))
    ln_gold = [a + b * r for r in rr]
    res = compute_divergence(
        _monthly([math.exp(x) for x in ln_gold]), _monthly(rr), window=WINDOW_MONTHS
    )
    # 12 punti di storia dopo il warmup di 60
    assert len(res["history"]) == 12
    assert set(res["history"][0]) >= {"date", "gap"}
    assert res["asof"] == res["history"][-1]["date"]
