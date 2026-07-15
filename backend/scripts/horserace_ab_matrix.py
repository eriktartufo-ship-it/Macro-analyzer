"""Matrice A/B horserace — testa N leve candidate sullo STESSO set di sim.

Costruisce real_matrix + wf_cache UNA volta, poi per ogni braccio (combinazione
di leve) applica gli override a runtime (salva/ripristina le costanti), gira
broad + crisi-2022 con seed fissi, e stampa una tabella comparativa.

Obiettivo (domanda Erik): quali leve recuperano l'edge del motore nei mercati
normali SENZA rovinare la protezione nelle crisi? Leggi il Δ vs baseline.

Uso: python scripts/horserace_ab_matrix.py [--broad-sims 40] [--crisis-sims 30]
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import date

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts/ per horserace_backtest
from horserace_backtest import run_horserace_trade  # noqa: E402 (importa anche il path per 'app')

# Ogni braccio: label, set-leve, entry_override, incremental_override, top_n, ramp_cap
ARMS = [
    ("baseline",        set(),                              None, None, None, None),
    ("deadband",        {"deadband"},                       None, 55.0, None, None),
    ("fastramp",        {"fastramp"},                       None, None, None, 2),
    ("momneutral",      {"momneutral"},                     None, None, None, None),
    ("entry50",         set(),                              50.0, None, None, None),
    ("top18",           set(),                              None, None, 18,   None),
    ("floor30",         {"floor"},                          None, None, None, None),
    ("combo(db+fr+fl)", {"deadband", "fastramp", "floor"}, None, 55.0, None, 2),
]


def _crisis_pool_2022():
    pool = []
    y, m = 2021, 12
    while (y, m) <= (2022, 12):
        pool.append(date(y, m, 1))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return pool


def _run_arm(label, levers, entry_ov, incr_ov, top_n, ramp_cap,
             real_matrix, wf_cache, broad_sims, crisis_sims, floor_pct):
    from app.services.ai_portfolio import strategist as S, sizer as SZ

    # Salva stato, applica override
    orig_entry, orig_incr, orig_ntr = (
        S.ENTRY_SCORE_THRESHOLD, S.INCREMENTAL_SCORE_THRESHOLD, SZ.dynamic_n_tranches,
    )
    if entry_ov is not None:
        S.ENTRY_SCORE_THRESHOLD = entry_ov
    if incr_ov is not None:
        S.INCREMENTAL_SCORE_THRESHOLD = incr_ov
    if ramp_cap is not None:
        SZ.dynamic_n_tranches = lambda c, _o=orig_ntr, _k=ramp_cap: (min(_k, _o(c)) or 1) if _o(c) > 0 else 0

    def _batch(pool, n, seed):
        rng = random.Random(seed)
        out = []
        for _ in range(n):
            if pool:
                d = rng.choice(pool)
            else:
                d = date(rng.randint(1995, 2023), rng.randint(1, 12), 1)
            t = run_horserace_trade(d, real_matrix, wf_cache, 12, 0.0,
                                    levers=levers, top_n_regime=top_n, floor_pct=floor_pct)
            if t and t.realized_return is not None:
                out.append(t)
        return out

    broad = _batch(None, broad_sims, 7)
    crisis = _batch(_crisis_pool_2022(), crisis_sims, 42)

    # Ripristina stato
    S.ENTRY_SCORE_THRESHOLD, S.INCREMENTAL_SCORE_THRESHOLD, SZ.dynamic_n_tranches = (
        orig_entry, orig_incr, orig_ntr,
    )

    def _agg(ts):
        if not ts:
            return None
        m = lambda a: float(np.mean([getattr(t, a) for t in ts if getattr(t, a) is not None]))
        sev = sum(1 for t in ts if t.severity == "severe")
        return {
            "deployed": m("avg_deployed"), "a60": m("alpha_vs_60_40"),
            "aspx": m("alpha_vs_sp500"), "dd": m("max_drawdown"),
            "sev": sev / len(ts), "n": len(ts),
        }

    return {"label": label, "broad": _agg(broad), "crisis": _agg(crisis)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--broad-sims", type=int, default=40)
    p.add_argument("--crisis-sims", type=int, default=30)
    p.add_argument("--floor-pct", type=float, default=0.30)
    args = p.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.environ.setdefault("USE_MOMENTUM_PILLARS", "1")

    from app.services.scoring.engine import ASSET_REGIME_DATA
    from app.services.backtest.real_returns_matrix import build_real_returns_matrix
    from app.services.backtest.walk_forward import WalkForwardCache, preload_walk_forward_series

    print("Building real returns matrix + FRED cache (una volta)...")
    real_matrix = build_real_returns_matrix(list(ASSET_REGIME_DATA.keys()))
    series = preload_walk_forward_series(start_date=date(1993, 1, 1), end_date=date(2025, 12, 31))
    wf_cache = WalkForwardCache(series)
    print(f"  Matrix {real_matrix.shape}. Running {len(ARMS)} arms x2 windows...\n")

    results = []
    for arm in ARMS:
        r = _run_arm(*arm, real_matrix, wf_cache, args.broad_sims, args.crisis_sims, args.floor_pct)
        results.append(r)
        print(f"  [done] {arm[0]}")

    def _fmt(a):
        if a is None:
            return "   n/a"
        return f"dep{a['deployed']*100:3.0f}% a60{a['a60']*100:+5.1f} aSPX{a['aspx']*100:+5.1f} DD{a['dd']*100:5.1f} sev{a['sev']*100:3.0f}%"

    print("\n" + "=" * 100)
    print("A/B MATRIX — BROAD (1995-2023) | CRISI 2022   [dep=deployed, a60/aSPX=alpha pp, DD=maxdrawdown, sev=%severe]")
    print("=" * 100)
    base = results[0]
    for r in results:
        d60 = ""
        if r["broad"] and base["broad"]:
            delta = (r["broad"]["a60"] - base["broad"]["a60"]) * 100
            d60 = f"  d(a60broad) {delta:+.1f}pp"
        print(f"  {r['label']:16s} | BROAD {_fmt(r['broad'])} | CRISI {_fmt(r['crisis'])}{d60}")


if __name__ == "__main__":
    main()
