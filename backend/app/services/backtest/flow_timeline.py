"""FLOW TIMELINE (S45) — serie storica settimanale dei flussi per la pagina Flussi scrubabile.

Un unico calcolo restituisce, per le ultime N settimane, gli snapshot di:
  - RRG (rs_ratio x rs_momentum centrati a 100, lisciati JdK) per i 25 asset,
  - flussi/rete (extra-perf conservativa Sigma=0 -> edges perdenti->vincitori),
  - spettro safe->risky (flusso per asset + candidati asimmetrici),
  - affidabilita' (S&P vs SMA200) a quella settimana.
Cosi' il frontend fa UNA fetch e lo slider globale scorre le settimane. NO FUTURE LEAK (ogni
settimana usa solo dati <= quella settimana: rolling trailing). Universo/metadati in flow_meta.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.backtest.flow_meta import FLOW_ASSETS, META
from app.services.backtest.rrg import quadrant


def _weekly_levels(daily_matrix: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = {}
    for a in cols:
        s = daily_matrix[a].dropna()
        if s.empty:
            continue
        out[a] = (1.0 + s).cumprod().resample("W-FRI").last()
    return pd.DataFrame(out)


def _edges_for(extra: pd.Series, max_edges: int = 12) -> tuple[list[dict], float]:
    """Da un vettore extra-perf (pp, Sigma~0) costruisce gli edges perdenti->vincitori."""
    winners = {a: v for a, v in extra.items() if v > 1e-9}
    losers = {a: -v for a, v in extra.items() if v < -1e-9}
    M = sum(winners.values())
    edges = []
    if M > 1e-9:
        for a, la in losers.items():
            for b, wb in winners.items():
                w = la * wb / M
                if w > 1e-9:
                    edges.append({"s": a, "t": b, "w": round(w, 3), "sh": round(w / M * 100, 1)})
    edges.sort(key=lambda e: e["w"], reverse=True)
    return edges[:max_edges], float(M)


def compute_flow_timeline(
    daily_matrix: pd.DataFrame,
    n_weeks: int = 52,
    rrg_window: int = 52,
    smooth: int = 5,
    flow_lookback: int = 8,
    sma_days: int = 200,
) -> dict:
    cols = [a for a in FLOW_ASSETS if a in daily_matrix.columns]
    L = _weekly_levels(daily_matrix, cols)
    if L.empty or len(L) < rrg_window + smooth + 2:
        return {"weeks": [], "assets": [], "frames": []}

    # --- RRG rolling (come rrg.py) ---
    wk_ret = L.pct_change()
    bench_ret = wk_ret.mean(axis=1, skipna=True)
    bench_lv = (1.0 + bench_ret.fillna(0.0)).cumprod()
    rs = L.div(bench_lv, axis=0)
    rs_ratio = (100.0 + (rs - rs.rolling(rrg_window).mean()) / rs.rolling(rrg_window).std()).rolling(smooth).mean()
    roc = rs_ratio.diff(1)
    rs_mom = (100.0 + (roc - roc.rolling(rrg_window).mean()) / roc.rolling(rrg_window).std()).rolling(smooth).mean()

    # --- flussi rolling: extra-perf su flow_lookback (cum ret finestra - media universo) ---
    cum_lb = L / L.shift(flow_lookback) - 1.0
    extra = cum_lb.sub(cum_lb.mean(axis=1), axis=0) * 100.0   # pp, Sigma~0 per riga

    # --- affidabilita' (S&P vs SMA200) per settimana ---
    SP = "us_equities_growth"
    stress_by_week = {}
    if SP in daily_matrix.columns:
        sp = daily_matrix[SP].dropna()
        splv = (1.0 + sp).cumprod()
        spsma = splv.rolling(sma_days).mean()
        rel = (splv / spsma - 1.0) * 100.0
        rel_wk = rel.resample("W-FRI").last()
        for wk, v in rel_wk.items():
            if not pd.isna(v):
                stress_by_week[wk] = float(v)

    # settimane valide = quelle con rs_ratio/rs_mom non-NaN per abbastanza asset
    valid_rows = (rs_ratio.notna() & rs_mom.notna()).any(axis=1)
    valid_idx = [i for i in L.index if valid_rows.get(i, False)]
    weeks_idx = valid_idx[-n_weeks:]

    frames = []
    for wk in weeks_idx:
        pos = L.index.get_loc(wk)
        # RRG points
        rrg_pts = []
        for a in cols:
            r, m = rs_ratio.at[wk, a], rs_mom.at[wk, a]
            if pd.isna(r) or pd.isna(m):
                continue
            rrg_pts.append({"a": a, "ratio": round(float(r), 2), "mom": round(float(m), 2),
                            "q": quadrant(float(r), float(m))})
        # flussi della settimana
        ex = extra.loc[wk].dropna()
        flows = [{"a": a, "extra": round(float(ex[a]), 2)} for a in ex.index]
        # candidati asimmetrici: flusso in risalita da territorio negativo (finito l'outflow),
        # rischio non estremo -> ingresso asimmetrico. Guardo extra ora vs 4 settimane fa.
        cand = set()
        if pos - 4 >= 0:
            ex_prev = extra.iloc[pos - 4]
            for a in ex.index:
                p = ex_prev.get(a)
                if p is not None and not pd.isna(p):
                    # candidato: era in outflow SIGNIFICATIVO (< -1.5pp), ora RISALE, rischio non estremo
                    if float(ex[a]) > float(p) + 0.3 and float(p) < -1.5 and META.get(a, {}).get("risk", 1) <= 0.72:
                        cand.add(a)
        for f in flows:
            f["cand"] = f["a"] in cand
        # rete
        edges, rotated = _edges_for(ex)
        frames.append({
            "week": wk.date().isoformat(),
            "rrg": rrg_pts, "flows": flows, "edges": edges,
            "rotated_pp": round(rotated, 2),
            "sp_vs_sma_pct": round(stress_by_week.get(wk), 2) if wk in stress_by_week else None,
        })

    assets = [{"asset": a, "label": META[a]["label"], "area": META[a]["area"],
               "risk": META[a]["risk"]} for a in cols if a in META]
    return {
        "weeks": [f["week"] for f in frames],
        "assets": assets,
        "flow_lookback": flow_lookback, "rrg_window": rrg_window,
        "frames": frames,
    }
