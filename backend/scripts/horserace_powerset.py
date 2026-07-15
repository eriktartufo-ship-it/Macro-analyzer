"""Powerset A/B — testa ogni COMBINAZIONE di leve candidate sullo stesso set di sim.

Generalizza `horserace_ab_matrix.py`: invece di una lista hardcoded di bracci,
un REGISTRY di leve ortogonali + un runner che genera i bracci come:
  - singles: baseline + ogni leva da sola (screening degli effetti marginali)
  - all:     baseline + tutte-ON
  - powerset: baseline + tutte le 2^N combinazioni (interazioni complete)
  - lista esplicita: --arms "voltarget+tightstop,deployboost"

Fedeltà (dichiarata nel registry): REAL = rigioca meccanica reale · PROXY = approssima.
Il #5 (ensemble/dibattito AI) NON è qui: richiede LLM live → binario A/B paper-trading.

Uso:
    python scripts/horserace_powerset.py --mode singles [--broad-sims 40] [--crisis-sims 30]
    python scripts/horserace_powerset.py --mode powerset --levers deployboost,voltarget,tightstop
    python scripts/horserace_powerset.py --mode all
"""
from __future__ import annotations

import argparse
import itertools
import os
import random
import sys
from datetime import date

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from horserace_backtest import run_horserace_trade  # noqa: E402


# --- Registry leve ---------------------------------------------------------
# Ogni leva: kind + payload. `patch(S, SZ)` muta i moduli e ritorna una restore-thunk.
def _patch_deployboost(S, SZ):
    """deadband(INCR->ENTRY) + fastramp(<=2 tranche). floor via kwargs."""
    o_incr, o_ntr = S.INCREMENTAL_SCORE_THRESHOLD, SZ.dynamic_n_tranches
    S.INCREMENTAL_SCORE_THRESHOLD = S.ENTRY_SCORE_THRESHOLD
    SZ.dynamic_n_tranches = lambda c, _o=o_ntr: (min(2, _o(c)) or 1) if _o(c) > 0 else 0

    def _restore():
        S.INCREMENTAL_SCORE_THRESHOLD, SZ.dynamic_n_tranches = o_incr, o_ntr
    return _restore


def _patch_tightstop(S, SZ):
    """Uscita più svelta: stop -10%->-7%, exit-score 45->48 (taglia i loser prima)."""
    o_stop, o_exit = S.STOP_LOSS_PCT, S.EXIT_SCORE_THRESHOLD
    S.STOP_LOSS_PCT, S.EXIT_SCORE_THRESHOLD = -0.07, 48.0

    def _restore():
        S.STOP_LOSS_PCT, S.EXIT_SCORE_THRESHOLD = o_stop, o_exit
    return _restore


# name -> {fidelity, patch, levers, kwargs, note}
LEVERS: dict[str, dict] = {
    "deployboost": {"fid": "REAL",  "patch": _patch_deployboost,
                    "levers": {"floor"}, "kwargs": {"floor_pct": 0.30},
                    "note": "#0 metodo d'ingresso validato (deadband+fastramp+floor)"},
    "voltarget":   {"fid": "REAL",  "kwargs": {"vol_target": 0.10},
                    "note": "#2 scala i target alla vol-budget 10% (cap 1.0, no leva)"},
    "tightstop":   {"fid": "REAL",  "patch": _patch_tightstop,
                    "note": "#1 uscita: stop -7% + exit-score 48"},
    "momneutral":  {"fid": "REAL",  "levers": {"momneutral"},
                    "note": "#1 (re)ingresso: basta momentum non-BEARISH"},
    "derisk":      {"fid": "PROXY", "kwargs": {"derisk": True},
                    "note": "#3 dimezza deployment nei mesi di transizione/incertezza"},
    "hedge":       {"fid": "PROXY", "kwargs": {"hedge_pct": 0.05},
                    "note": "#4 overlay convesso tail-hedge 5% (proxy, non put reale)"},
    "costs":       {"fid": "REAL",  "kwargs": {"cost_bps": 10.0},
                    "note": "#6 costo di transazione 10bps sul turnover (igiene)"},
    "cadence":     {"fid": "REAL",  "kwargs": {"cadence_months": 3},
                    "note": "#6 ribilancio trimestrale invece che mensile"},
}


def _apply_arm(names):
    """Applica le leve `names`: ritorna (restore_thunk, run_kwargs)."""
    from app.services.ai_portfolio import strategist as S, sizer as SZ
    restores, levers, kwargs = [], set(), {}
    for n in names:
        spec = LEVERS[n]
        if spec.get("patch"):
            restores.append(spec["patch"](S, SZ))
        levers |= spec.get("levers", set())
        kwargs.update(spec.get("kwargs", {}))
    if levers:
        kwargs["levers"] = levers

    def _restore():
        for r in reversed(restores):
            r()
    return _restore, kwargs


_CRISIS_WINDOWS = {
    "2008": [((2007, 10), (2008, 12))],
    "2020": [((2019, 12), (2020, 8))],
    "2022": [((2021, 12), (2022, 12))],
    "all_crisis": [((2007, 10), (2008, 12)), ((2019, 12), (2020, 8)), ((2021, 12), (2022, 12))],
}


def _crisis_pool(window="2022"):
    pool = []
    for (y0, m0), (y1, m1) in _CRISIS_WINDOWS[window]:
        y, m = y0, m0
        while (y, m) <= (y1, m1):
            pool.append(date(y, m, 1))
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return pool


def _batch(pool, n, seed, real_matrix, wf_cache, kwargs):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        d = rng.choice(pool) if pool else date(rng.randint(1995, 2023), rng.randint(1, 12), 1)
        t = run_horserace_trade(d, real_matrix, wf_cache, 12, 0.0, **kwargs)
        if t and t.realized_return is not None:
            out.append(t)
    return out


def _agg(ts):
    if not ts:
        return None
    m = lambda a: float(np.mean([getattr(t, a) for t in ts if getattr(t, a) is not None]))
    sev = sum(1 for t in ts if t.severity == "severe")
    return {"deployed": m("avg_deployed"), "a60": m("alpha_vs_60_40"),
            "aspx": m("alpha_vs_sp500"), "dd": m("max_drawdown"),
            "sev": sev / len(ts), "n": len(ts)}


def _run_arm(names, real_matrix, wf_cache, broad_sims, crisis_sims, crisis_window="2022"):
    restore, kwargs = _apply_arm(names)
    try:
        broad = _batch(None, broad_sims, 7, real_matrix, wf_cache, kwargs)
        crisis = _batch(_crisis_pool(crisis_window), crisis_sims, 42, real_matrix, wf_cache, kwargs)
    finally:
        restore()
    label = "baseline" if not names else "+".join(names)
    return {"label": label, "broad": _agg(broad), "crisis": _agg(crisis)}


def _build_arms(mode, levers):
    keys = levers or list(LEVERS.keys())
    if mode == "singles":
        return [()] + [(k,) for k in keys]
    if mode == "all":
        return [(), tuple(keys)]
    if mode == "powerset":
        combos = []
        for r in range(len(keys) + 1):
            combos += [c for c in itertools.combinations(keys, r)]
        return combos
    raise ValueError(mode)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["singles", "all", "powerset"], default="singles")
    p.add_argument("--levers", default="", help="Sottoinsieme leve (csv). Vuoto = tutte.")
    p.add_argument("--broad-sims", type=int, default=40)
    p.add_argument("--crisis-sims", type=int, default=30)
    p.add_argument("--crisis", choices=["2022", "2020", "2008", "all_crisis"], default="2022")
    args = p.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.environ.setdefault("USE_MOMENTUM_PILLARS", "1")
    # Igiene A/B: deploy_boost è default-ON nel live ma qui è la leva `deployboost`.
    # Forza OFF il flag così il baseline è pulito e i Δ sono marginali (vedi harness).
    os.environ.setdefault("USE_AI_PORTFOLIO_DEPLOY_BOOST", "0")

    levers = [x.strip() for x in args.levers.split(",") if x.strip()]
    for k in levers:
        if k not in LEVERS:
            raise SystemExit(f"leva sconosciuta: {k}. Note: {list(LEVERS)}")
    arms = _build_arms(args.mode, levers)

    from app.services.scoring.engine import ASSET_REGIME_DATA
    from app.services.backtest.real_returns_matrix import build_real_returns_matrix
    from app.services.backtest.walk_forward import WalkForwardCache, preload_walk_forward_series

    registry_str = ", ".join(f"{k}[{v['fid']}]" for k, v in LEVERS.items())
    print(f"Registry leve: {registry_str}")
    print("Building real returns matrix + FRED cache (una volta)...")
    real_matrix = build_real_returns_matrix(list(ASSET_REGIME_DATA.keys()))
    series = preload_walk_forward_series(start_date=date(1993, 1, 1), end_date=date(2025, 12, 31))
    wf_cache = WalkForwardCache(series)
    print(f"  Matrix {real_matrix.shape}. {len(arms)} bracci x2 finestre "
          f"({args.broad_sims}+{args.crisis_sims} sim)...\n")

    results = []
    for names in arms:
        results.append(_run_arm(names, real_matrix, wf_cache, args.broad_sims,
                                args.crisis_sims, args.crisis))
        print(f"  [done] {results[-1]['label']}", flush=True)

    def _fmt(a):
        if a is None:
            return "   n/a"
        return (f"dep{a['deployed']*100:3.0f}% a60{a['a60']*100:+5.1f} aSPX{a['aspx']*100:+5.1f} "
                f"DD{a['dd']*100:5.1f} sev{a['sev']*100:3.0f}%")

    base = results[0]
    print("\n" + "=" * 110)
    print(f"POWERSET A/B — BROAD (1995-2023) | CRISI {args.crisis}  "
          "[dep=deployed, a60/aSPX=alpha pp, DD=maxDD, sev=%severe]")
    print("=" * 110)
    for r in results:
        d60 = ""
        if r["broad"] and base["broad"]:
            db = (r["broad"]["a60"] - base["broad"]["a60"]) * 100
            dc = ((r["crisis"]["a60"] - base["crisis"]["a60"]) * 100
                  if r["crisis"] and base["crisis"] else 0.0)
            d60 = f"  Δbull {db:+.1f} Δcrisi {dc:+.1f}pp"
        print(f"  {r['label'][:34]:34s} | B {_fmt(r['broad'])} | C {_fmt(r['crisis'])}{d60}")


if __name__ == "__main__":
    main()
