"""Test RRG (Relative Rotation Graph). Spec: scripts/RRG_SPEC.md."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.backtest.rrg import compute_rrg, quadrant


def test_quadrant_four_cases_and_center():
    assert quadrant(102, 101) == "leading"      # forte + accelera
    assert quadrant(102, 99) == "weakening"     # forte + rallenta
    assert quadrant(98, 99) == "lagging"        # debole + peggiora
    assert quadrant(98, 101) == "improving"     # debole + accelera
    assert quadrant(100, 100) == "leading"      # centro incluso (>=)


def _synth_matrix(n_days=400, seed=0):
    """Matrice daily sintetica: un winner (drift+), un loser (drift-), 4 neutri."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n_days)
    drifts = {"winner": 0.0010, "loser": -0.0008,
              "n1": 0.0, "n2": 0.0, "n3": 0.0, "n4": 0.0}
    data = {a: rng.normal(d, 0.006, n_days) for a, d in drifts.items()}
    return pd.DataFrame(data, index=idx)


UNIV = ["winner", "loser", "n1", "n2", "n3", "n4"]


def test_winner_is_on_strong_side_loser_on_weak_side():
    dm = _synth_matrix()
    res = compute_rrg(dm, universe=UNIV, window_weeks=12, tail=6)
    by = {p.asset: p for p in res.points}
    # il winner sovraperforma il basket -> ratio >= 100 (leading o weakening)
    assert by["winner"].ratio >= 100.0
    assert by["winner"].quadrant in ("leading", "weakening")
    # il loser sottoperforma -> ratio < 100 (lagging o improving)
    assert by["loser"].ratio < 100.0
    assert by["loser"].quadrant in ("lagging", "improving")


def test_no_future_leak():
    """Il punto alla data T deve essere IDENTICO se calcolato con dati fino a T o fino a T+k.
    Prova che ratio/momentum usano solo il passato (rolling trailing)."""
    dm_full = _synth_matrix(n_days=400)
    cut = 320
    dm_short = dm_full.iloc[:cut]
    res_short = compute_rrg(dm_short, universe=UNIV, window_weeks=12, tail=3)
    # as_of di res_short = una settimana T ben dentro dm_full
    t = res_short.as_of
    res_full = compute_rrg(dm_full, universe=UNIV, window_weeks=12, tail=60)
    # trova nel tail di res_full il punto con data == T per il winner
    w_full = next(p for p in res_full.points if p.asset == "winner")
    w_short = next(p for p in res_short.points if p.asset == "winner")
    pt_full = next((x for x in w_full.tail if x["date"] == t), None)
    assert pt_full is not None, "la data T deve comparire nel tail lungo di res_full"
    # confronto like-with-like: il punto a T nel tail di short (ultimo) vs quello di full,
    # entrambi arrotondati uguale -> devono coincidere ESATTAMENTE (nessun uso del futuro).
    pt_short = w_short.tail[-1]
    assert pt_short["date"] == t
    assert pt_full["ratio"] == pt_short["ratio"]
    assert pt_full["momentum"] == pt_short["momentum"]


def test_output_shape_and_tail():
    dm = _synth_matrix()
    res = compute_rrg(dm, universe=UNIV, window_weeks=12, tail=5)
    d = res.to_dict()
    assert d["benchmark"] == "equal_weight_universe"
    assert d["as_of"]
    assert len(d["points"]) >= 4
    p0 = d["points"][0]
    assert set(p0) >= {"asset", "label", "quadrant", "ratio", "momentum", "tail"}
    assert len(p0["tail"]) <= 5 and len(p0["tail"]) >= 1
    assert set(p0["tail"][0]) == {"date", "ratio", "momentum"}


def test_missing_asset_excluded_gracefully():
    dm = _synth_matrix()
    # asset richiesto ma assente dalla matrice -> semplicemente escluso, no crash
    res = compute_rrg(dm, universe=UNIV + ["does_not_exist"], window_weeks=12, tail=4)
    assert all(p.asset != "does_not_exist" for p in res.points)
