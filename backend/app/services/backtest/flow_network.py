"""RETE DI ROTAZIONE — i flussi (relativi) tra asset come grafo diretto con frecce pesate.

Idea di Erik (2026-07-16): "i soldi sono sempre quelli, si trasferiscono da una parte
all'altra; disegnamoli come una rete di frecce, con la portata del flusso". VINCOLO DI VERITA'
(gia' sollevato in S41b): i flussi netti in $ tra asset NON sono misurabili dai prezzi (ogni
compratore ha un venditore). Cio' che e' reale e misurabile = la ROTAZIONE RELATIVA.

MODELLO (conservativo, rispetta "i soldi sono sempre quelli"):
  su un lookback recente L, extra-performance di ogni asset e_i = r_i - media(universo).
  Per costruzione SOMMA(e_i) = 0 -> chi guadagna forza relativa (e_i>0) la prende da chi la
  cede (e_i<0). Trasporto proporzionale: flow(perdente a -> vincitore b) = |e_a| * e_b / M,
  con M = somma dei guadagni = somma delle perdite. Le frecce sommano ESATTAMENTE a M.
  Unita' = punti percentuali di rotazione relativa (NON dollari). NO FUTURE LEAK (solo <= t).

Spec: scripts/RRG_SPEC.md (sezione rete).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.services.backtest.rrg import DEFAULT_UNIVERSE, LABELS, _weekly_levels


@dataclass
class FlowNode:
    asset: str
    label: str
    role: str        # "winner" | "loser" | "flat"
    extra_pp: float  # extra-performance in punti percentuali (segno = ha attratto/ceduto)


@dataclass
class FlowEdge:
    source: str      # perdente (cede forza relativa)
    target: str      # vincitore (attrae)
    weight_pp: float  # portata in punti percentuali di rotazione
    share: float     # quota sul totale ruotato (%)


@dataclass
class FlowNetwork:
    as_of: str
    lookback_weeks: int
    rotated_pp: float          # M = capitale relativo totale ruotato (punti percentuali)
    nodes: list[FlowNode]
    edges: list[FlowEdge]
    edges_shown: int
    edges_total: int

    def to_dict(self) -> dict:
        return {
            "as_of": self.as_of,
            "lookback_weeks": self.lookback_weeks,
            "rotated_pp": round(self.rotated_pp, 2),
            "edges_shown": self.edges_shown,
            "edges_total": self.edges_total,
            "nodes": [
                {"asset": n.asset, "label": n.label, "role": n.role,
                 "extra_pp": round(n.extra_pp, 2)}
                for n in self.nodes
            ],
            "edges": [
                {"source": e.source, "target": e.target,
                 "weight_pp": round(e.weight_pp, 3), "share": round(e.share, 1)}
                for e in self.edges
            ],
        }


def compute_flow_network(
    daily_matrix: pd.DataFrame,
    universe: list[str] | None = None,
    lookback_weeks: int = 8,
    max_edges: int = 14,
    min_share: float = 1.0,
) -> FlowNetwork:
    """Grafo diretto della rotazione relativa (perdenti -> vincitori) su un lookback."""
    universe = universe or DEFAULT_UNIVERSE
    cols = [a for a in universe if a in daily_matrix.columns]
    L = _weekly_levels(daily_matrix, cols)
    if L.empty or len(L) < lookback_weeks + 1:
        return FlowNetwork("", lookback_weeks, 0.0, [], [], 0, 0)

    # rendimento cumulato sull'ultimo lookback (settimane <= as_of -> no leak)
    tail = L.iloc[-(lookback_weeks + 1):]
    as_of = tail.index[-1]
    ret = tail.iloc[-1] / tail.iloc[0] - 1.0            # cum ret per asset sul lookback
    ret = ret.dropna()
    present = [a for a in cols if a in ret.index]
    if len(present) < 4:
        return FlowNetwork(as_of.date().isoformat(), lookback_weeks, 0.0, [], [], 0, 0)

    r = ret[present]
    extra = (r - r.mean()) * 100.0                     # extra-performance pp, SOMMA = 0

    nodes = [
        FlowNode(a, LABELS.get(a, a),
                 "winner" if extra[a] > 1e-9 else "loser" if extra[a] < -1e-9 else "flat",
                 float(extra[a]))
        for a in present
    ]

    winners = {a: extra[a] for a in present if extra[a] > 1e-9}
    losers = {a: -extra[a] for a in present if extra[a] < -1e-9}   # magnitudo positiva
    M = sum(winners.values())                          # = sum(losers.values())
    edges: list[FlowEdge] = []
    if M > 1e-9:
        for a, la in losers.items():                   # perdente
            for b, wb in winners.items():              # vincitore
                w = la * wb / M                         # trasporto proporzionale (pp)
                if w > 1e-9:
                    edges.append(FlowEdge(a, b, w, w / M * 100.0))

    edges.sort(key=lambda e: e.weight_pp, reverse=True)
    total = len(edges)
    shown = [e for e in edges if e.share >= min_share][:max_edges]
    return FlowNetwork(
        as_of.date().isoformat(), lookback_weeks, float(M),
        nodes, shown, len(shown), total,
    )
