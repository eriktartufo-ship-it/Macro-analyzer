"""RRG — Relative Rotation Graph (flussi settoriali/asset-class).

Il segnale flussi è reale ma non batte l'S&P long-only (`scripts/FLOW_TRADER_CRITERIA.md`) →
il suo valore è INFORMATIVO: mostrare DOVE vanno i soldi. Standard Julius de Kempenaer, centro 100.
Spec: `scripts/RRG_SPEC.md`.

Benchmark = basket equal-weight dell'universo (scelta cross-asset: equity/bond/oro/cripto insieme
→ un indice equity non è benchmark valido per i bond). NO FUTURE LEAK (rolling trailing).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# universo flussi = settori GICS + asset class (rifugi inclusi) + semiconduttori (S43 #3).
# I semiconduttori sono aggiunti SOLO alle viste flussi (non a ASSET_REGIME_DATA) -> lo scoring
# del regime resta intatto. Vedi EXTRA_FLOW_ASSETS.
DEFAULT_UNIVERSE = [
    "us_equities_growth", "us_equities_value", "international_dm_equities", "em_equities",
    "us_bonds_long", "tips_inflation_bonds", "gold", "silver", "broad_commodities",
    "energy", "real_estate_reits",
    "sector_energy", "sector_financials", "sector_utilities", "sector_technology",
    "sector_healthcare", "sector_staples", "sector_semiconductors",
]

# asset presenti in ASSET_TICKERS ma NON in ASSET_REGIME_DATA: gli endpoint flows li aggiungono
# alla matrice rendimenti senza perturbare lo scoring del regime.
EXTRA_FLOW_ASSETS = ["sector_semiconductors"]

LABELS = {
    "us_equities_growth": "US Growth", "us_equities_value": "US Value",
    "international_dm_equities": "Intl DM", "em_equities": "Emerging",
    "us_bonds_long": "Long Bonds", "tips_inflation_bonds": "TIPS",
    "gold": "Gold", "silver": "Silver", "broad_commodities": "Commodities",
    "energy": "Energy", "real_estate_reits": "REITs",
    "sector_energy": "Sect. Energy", "sector_financials": "Financials",
    "sector_utilities": "Utilities", "sector_technology": "Technology",
    "sector_healthcare": "Healthcare", "sector_staples": "Staples",
    "sector_semiconductors": "Semiconductors",
}


def quadrant(ratio: float, momentum: float) -> str:
    """4 quadranti dal centro (100,100). Il centro è INCLUSO in leading/lagging (>=)."""
    strong = ratio >= 100.0
    accel = momentum >= 100.0
    if strong and accel:
        return "leading"
    if strong and not accel:
        return "weakening"
    if not strong and not accel:
        return "lagging"
    return "improving"


@dataclass
class RrgPoint:
    asset: str
    label: str
    quadrant: str
    ratio: float
    momentum: float
    tail: list[dict]  # [{date, ratio, momentum}], vecchio -> nuovo


@dataclass
class RrgResult:
    as_of: str
    benchmark: str
    window_weeks: int
    tail: int
    points: list[RrgPoint]

    def to_dict(self) -> dict:
        return {
            "as_of": self.as_of, "benchmark": self.benchmark,
            "window_weeks": self.window_weeks, "tail": self.tail,
            "points": [
                {"asset": p.asset, "label": p.label, "quadrant": p.quadrant,
                 "ratio": round(p.ratio, 2), "momentum": round(p.momentum, 2),
                 "tail": p.tail}
                for p in self.points
            ],
        }


def _weekly_levels(daily_matrix: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Livelli (NAV) a fine settimana per asset, ognuno con base sulla sua prima data."""
    out = {}
    for a in cols:
        s = daily_matrix[a].dropna()
        if s.empty:
            continue
        lv = (1.0 + s).cumprod()
        out[a] = lv.resample("W-FRI").last()
    return pd.DataFrame(out)


def compute_rrg(
    daily_matrix: pd.DataFrame,
    universe: list[str] | None = None,
    window_weeks: int = 52,
    tail: int = 8,
    smooth: int = 5,
) -> RrgResult:
    """Calcola RS-Ratio / RS-Momentum (centrati a 100, LISCIATI) + tail per ogni asset.

    Come JdK/StockCharts, ratio e momentum sono lisciati (SMA `smooth` settimane, trailing =
    no leak): senza smoothing lo z-score settimanale zigzaga e le code diventano illeggibili.
    """
    universe = universe or DEFAULT_UNIVERSE
    cols = [a for a in universe if a in daily_matrix.columns]
    L = _weekly_levels(daily_matrix, cols)
    if L.empty:
        return RrgResult("", "equal_weight_universe", window_weeks, tail, [])

    # benchmark = basket equal-weight (media dei rendimenti settimanali dei presenti)
    wk_ret = L.pct_change()
    bench_ret = wk_ret.mean(axis=1, skipna=True)
    bench_lv = (1.0 + bench_ret.fillna(0.0)).cumprod()

    rs = L.div(bench_lv, axis=0)                       # RS per asset (NaN dove asset assente)
    mu = rs.rolling(window_weeks).mean()              # trailing -> no leak
    sd = rs.rolling(window_weeks).std()
    rs_ratio = (100.0 + (rs - mu) / sd).rolling(smooth).mean()   # JdK RS-Ratio (smoothed)

    roc = rs_ratio.diff(1)                            # accelerazione della forza relativa
    rmu = roc.rolling(window_weeks).mean()
    rsd = roc.rolling(window_weeks).std()
    rs_mom = (100.0 + (roc - rmu) / rsd).rolling(smooth).mean()  # RS-Momentum (smoothed)

    # as_of = ultima settimana con almeno un asset valido su entrambe le serie
    valid_mask = rs_ratio.notna() & rs_mom.notna()
    rows_ok = valid_mask.any(axis=1)
    if not rows_ok.any():
        return RrgResult("", "equal_weight_universe", window_weeks, tail, [])
    as_of_idx = rows_ok[rows_ok].index[-1]
    as_of_pos = L.index.get_loc(as_of_idx)

    points: list[RrgPoint] = []
    for a in cols:
        r_now = rs_ratio.at[as_of_idx, a] if a in rs_ratio.columns else np.nan
        m_now = rs_mom.at[as_of_idx, a] if a in rs_mom.columns else np.nan
        if pd.isna(r_now) or pd.isna(m_now):
            continue
        # tail: ultimi `tail` punti fino all'as_of (vecchio -> nuovo)
        lo = max(0, as_of_pos - tail + 1)
        tail_pts = []
        for pos in range(lo, as_of_pos + 1):
            idx = L.index[pos]
            rv, mv = rs_ratio.at[idx, a], rs_mom.at[idx, a]
            if pd.isna(rv) or pd.isna(mv):
                continue
            tail_pts.append({"date": idx.date().isoformat(),
                             "ratio": round(float(rv), 2), "momentum": round(float(mv), 2)})
        points.append(RrgPoint(
            asset=a, label=LABELS.get(a, a), quadrant=quadrant(float(r_now), float(m_now)),
            ratio=float(r_now), momentum=float(m_now), tail=tail_pts,
        ))

    # ordine: per quadrante (leading, weakening, lagging, improving) poi per forza
    order = {"leading": 0, "weakening": 1, "lagging": 2, "improving": 3}
    points.sort(key=lambda p: (order[p.quadrant], -p.ratio))
    return RrgResult(as_of_idx.date().isoformat(), "equal_weight_universe",
                     window_weeks, tail, points)
