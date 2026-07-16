"""Test rete di rotazione (flow network). Spec: scripts/RRG_SPEC.md."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.backtest.flow_network import compute_flow_network


def _synth(n_days=300, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    drifts = {"winner": 0.0015, "win2": 0.0008, "loser": -0.0012, "los2": -0.0006,
              "flat1": 0.0, "flat2": 0.0}
    return pd.DataFrame({a: rng.normal(d, 0.005, n_days) for a, d in drifts.items()}, index=idx)


UNIV = ["winner", "win2", "loser", "los2", "flat1", "flat2"]


def test_conservation_extra_sums_zero_and_edges_sum_to_rotated():
    dm = _synth()
    net = compute_flow_network(dm, universe=UNIV, lookback_weeks=8, max_edges=99, min_share=0.0)
    # extra-performance somma a zero (i soldi sono sempre quelli)
    assert abs(sum(n.extra_pp for n in net.nodes)) < 1e-6
    # le frecce sommano ESATTAMENTE al capitale ruotato M
    assert abs(sum(e.weight_pp for e in net.edges) - net.rotated_pp) < 1e-6
    # somma guadagni = somma perdite
    gains = sum(n.extra_pp for n in net.nodes if n.extra_pp > 0)
    losses = -sum(n.extra_pp for n in net.nodes if n.extra_pp < 0)
    assert abs(gains - losses) < 1e-6
    assert abs(gains - net.rotated_pp) < 1e-6


def test_direction_winners_only_receive_losers_only_send():
    dm = _synth()
    net = compute_flow_network(dm, universe=UNIV, lookback_weeks=8, max_edges=99, min_share=0.0)
    roles = {n.asset: n.role for n in net.nodes}
    for e in net.edges:
        assert roles[e.source] == "loser", f"{e.source} manda ma non e' loser"
        assert roles[e.target] == "winner", f"{e.target} riceve ma non e' winner"


def test_bounded_lookback_no_leak():
    """La rete a una data usa SOLO le ultime `lookback` settimane: modificare pesantemente il
    PASSATO remoto (ben prima del lookback) non cambia il risultato all'as_of. Prova che la
    finestra e' delimitata e trailing (nessuna dipendenza da dati fuori dal lookback)."""
    dm = _synth(n_days=300)
    net_base = compute_flow_network(dm, universe=UNIV, lookback_weeks=8)
    # manometto la prima meta' della storia (mesi PRIMA dell'as_of - 8 settimane)
    dm_tamper = dm.copy()
    dm_tamper.iloc[:150] = dm_tamper.iloc[:150] * 0 + 0.05  # valori assurdi nel passato remoto
    net_tamper = compute_flow_network(dm_tamper, universe=UNIV, lookback_weeks=8)
    assert net_tamper.as_of == net_base.as_of
    assert abs(net_tamper.rotated_pp - net_base.rotated_pp) < 1e-9
    e_base = {(e.source, e.target): e.weight_pp for e in net_base.edges}
    e_tamper = {(e.source, e.target): e.weight_pp for e in net_tamper.edges}
    assert e_base.keys() == e_tamper.keys()
    for k in e_base:
        assert abs(e_base[k] - e_tamper[k]) < 1e-9


def test_winner_is_winner_loser_is_loser():
    dm = _synth()
    net = compute_flow_network(dm, universe=UNIV, lookback_weeks=8)
    roles = {n.asset: n.role for n in net.nodes}
    assert roles["winner"] == "winner"
    assert roles["loser"] == "loser"


def test_shape_and_edge_filter():
    dm = _synth()
    net = compute_flow_network(dm, universe=UNIV, lookback_weeks=8, max_edges=3, min_share=0.0)
    d = net.to_dict()
    assert d["as_of"] and d["lookback_weeks"] == 8
    assert d["edges_shown"] <= 3
    assert d["edges_total"] >= d["edges_shown"]
    if d["edges"]:
        e = d["edges"][0]
        assert set(e) == {"source", "target", "weight_pp", "share"}
