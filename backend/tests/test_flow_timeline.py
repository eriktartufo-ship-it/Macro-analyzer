"""Test flow timeline (serie storica settimanale dei flussi). Spec: S45."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.backtest.flow_timeline import compute_flow_timeline

# 6 asset reali di FLOW_ASSETS con drift diversi
ASSETS = ["cash_money_market", "us_bonds_long", "gold", "us_equities_value",
          "us_equities_growth", "em_equities"]
DRIFTS = [0.00005, -0.0002, 0.0001, 0.0004, 0.0009, -0.0003]


def _matrix(n_days=260, seed=2):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame({a: rng.normal(d, 0.005, n_days) for a, d in zip(ASSETS, DRIFTS)}, index=idx)


def _tl(dm, **kw):
    return compute_flow_timeline(dm, n_weeks=kw.get("n_weeks", 8), rrg_window=kw.get("rrg_window", 12),
                                 smooth=kw.get("smooth", 3), flow_lookback=kw.get("flow_lookback", 4),
                                 sma_days=kw.get("sma_days", 100))


def test_shape_weeks_match_frames():
    tl = _tl(_matrix())
    assert len(tl["weeks"]) == len(tl["frames"])
    assert len(tl["weeks"]) <= 8
    assert len(tl["assets"]) >= 4
    f = tl["frames"][-1]
    assert set(f) >= {"week", "rrg", "flows", "edges", "rotated_pp"}


def test_flows_conserved_each_week():
    tl = _tl(_matrix())
    for f in tl["frames"]:
        tot = sum(x["extra"] for x in f["flows"])
        assert abs(tot) < 0.5  # Sigma extra-perf ~ 0 (arrotondamenti a 2 decimali)


def test_edges_only_loser_to_winner():
    tl = _tl(_matrix())
    for f in tl["frames"]:
        flow = {x["a"]: x["extra"] for x in f["flows"]}
        for e in f["edges"]:
            assert flow[e["s"]] < 0  # source cede
            assert flow[e["t"]] > 0  # target attrae


def test_no_leak_bounded():
    dm = _matrix()
    base = _tl(dm)
    tampered = dm.copy()
    tampered.iloc[:80] = 0.06  # passato remoto assurdo
    re = _tl(tampered)
    assert re["frames"][-1]["week"] == base["frames"][-1]["week"]
    # l'ultima settimana non deve cambiare (rolling trailing bounded)
    b = {x["a"]: x["extra"] for x in base["frames"][-1]["flows"]}
    r = {x["a"]: x["extra"] for x in re["frames"][-1]["flows"]}
    for a in b:
        assert abs(b[a] - r.get(a, 0)) < 0.01


def test_assets_have_meta():
    tl = _tl(_matrix())
    for a in tl["assets"]:
        assert set(a) == {"asset", "label", "area", "risk"}
        assert 0.0 <= a["risk"] <= 1.0
