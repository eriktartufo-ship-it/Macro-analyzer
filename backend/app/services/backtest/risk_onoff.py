"""CROSS-ASSET RISK-ON/OFF — il "rifugio" di Erik (S43 #3).

Posizionamento a livello di MACRO-CLASSI (non i 17 settori): dove sta il capitale marginale,
in modalita' rischio (azioni/cripto guidano) o rifugio (bond/oro/cash guidano), e QUALE rifugio
sta funzionando ora. Momentum = rendimento cumulato reale sul lookback (fotografia, non predizione).
NO FUTURE LEAK (solo <= as_of). Gestione graceful degli asset con storia corta (cripto ~recente).

Spec: scripts/RRG_SPEC.md (sezione cross-asset).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.services.backtest.rrg import _weekly_levels

# macro-classi: (asset_key, label, role). role = "risk" (rischio) | "safe" (rifugio)
CLASSES = [
    ("us_equities_growth", "Azioni", "risk"),
    ("bitcoin", "Cripto", "risk"),
    ("gold", "Oro", "safe"),
    ("us_bonds_long", "Bond", "safe"),
    ("cash_money_market", "Cash", "safe"),
]


@dataclass
class AssetMomentum:
    asset: str
    label: str
    role: str
    momentum_pct: float | None   # rendimento cumulato sul lookback (None = dati insufficienti)
    weeks_available: int


@dataclass
class RiskOnOff:
    as_of: str
    lookback_weeks: int
    gauge: float                 # >0 risk-on, <0 risk-off (spread azioni-rifugi, pp)
    regime: str                  # "risk_on" | "neutral" | "risk_off"
    best_refuge: str | None      # miglior rifugio per momentum (asset key)
    best_refuge_label: str | None
    classes: list[AssetMomentum]

    def to_dict(self) -> dict:
        return {
            "as_of": self.as_of, "lookback_weeks": self.lookback_weeks,
            "gauge": round(self.gauge, 2), "regime": self.regime,
            "best_refuge": self.best_refuge, "best_refuge_label": self.best_refuge_label,
            "classes": [
                {"asset": c.asset, "label": c.label, "role": c.role,
                 "momentum_pct": None if c.momentum_pct is None else round(c.momentum_pct, 2),
                 "weeks_available": c.weeks_available}
                for c in self.classes
            ],
        }


def compute_risk_onoff(
    daily_matrix: pd.DataFrame,
    lookback_weeks: int = 13,
    neutral_band: float = 1.0,
) -> RiskOnOff:
    """Momentum per macro-classe + gauge risk-on/off + rifugio migliore."""
    keys = [k for k, _, _ in CLASSES]
    cols = [k for k in keys if k in daily_matrix.columns]
    L = _weekly_levels(daily_matrix, cols)
    if L.empty:
        return RiskOnOff("", lookback_weeks, 0.0, "neutral", None, None, [])

    as_of = L.index[-1]
    out: list[AssetMomentum] = []
    for key, label, role in CLASSES:
        if key not in L.columns:
            out.append(AssetMomentum(key, label, role, None, 0))
            continue
        s = L[key].dropna()
        avail = len(s)
        if avail >= lookback_weeks + 1:
            window = s.iloc[-(lookback_weeks + 1):]
            mom = float(window.iloc[-1] / window.iloc[0] - 1.0) * 100.0
            out.append(AssetMomentum(key, label, role, mom, avail))
        else:
            # dati insufficienti per il lookback pieno -> None (mostrato ma fuori dal gauge)
            out.append(AssetMomentum(key, label, role, None, avail))

    risk = [c.momentum_pct for c in out if c.role == "risk" and c.momentum_pct is not None]
    safe = [c.momentum_pct for c in out if c.role == "safe" and c.momentum_pct is not None]
    gauge = (float(np.mean(risk)) - float(np.mean(safe))) if risk and safe else 0.0
    if gauge > neutral_band:
        regime = "risk_on"
    elif gauge < -neutral_band:
        regime = "risk_off"
    else:
        regime = "neutral"

    refuges = [(c.asset, c.label, c.momentum_pct) for c in out
               if c.role == "safe" and c.momentum_pct is not None]
    best = max(refuges, key=lambda x: x[2]) if refuges else None

    return RiskOnOff(
        as_of.date().isoformat(), lookback_weeks, gauge, regime,
        best[0] if best else None, best[1] if best else None, out,
    )
