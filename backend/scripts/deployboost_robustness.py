"""Robustezza multi-seed di deploy_boost — risponde all'audit raydalio (NO-GO).

Ray: i numeri del gate (2008 +5.40→+6.56, broad −2.95→−1.24) sono seed-dipendenti e
mescolano seed diversi per finestra senza dichiararlo. Qui: per OGNI finestra, Δ
(combo − baseline) su N seed → media ± std. Se il Δ è robustamente >0 deploy_boost
regge; se è fragile/≈0 ray ha ragione e si torna opt-in.

Include ANCHE il max-DD (per verificare il "quasi raddoppia") e una colonna
cost-adjusted (10bps sul turnover).
"""
from __future__ import annotations

import os
import sys
from datetime import date

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from horserace_backtest import run_horserace_trade  # noqa: E402
from horserace_powerset import _apply_arm, _crisis_pool  # noqa: E402

SEEDS = [1, 5, 11, 42, 99]
WINDOWS = ["broad", "2008", "2020", "all_crisis"]


def _batch_alpha_dd(pool, n, seed, real_matrix, wf_cache, kwargs):
    import random
    rng = random.Random(seed)
    a, dd = [], []
    for _ in range(n):
        d = rng.choice(pool) if pool else date(rng.randint(1995, 2023), rng.randint(1, 12), 1)
        t = run_horserace_trade(d, real_matrix, wf_cache, 12, 0.0, **kwargs)
        if t and t.alpha_vs_60_40 is not None:
            a.append(t.alpha_vs_60_40)
            dd.append(t.max_drawdown)
    return (float(np.mean(a)) if a else float("nan"),
            float(np.mean(dd)) if dd else float("nan"))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.environ.setdefault("USE_MOMENTUM_PILLARS", "1")
    os.environ.setdefault("USE_AI_PORTFOLIO_DEPLOY_BOOST", "0")

    from app.services.scoring.engine import ASSET_REGIME_DATA
    from app.services.backtest.real_returns_matrix import build_real_returns_matrix
    from app.services.backtest.walk_forward import WalkForwardCache, preload_walk_forward_series

    print("Building matrix + FRED...")
    real_matrix = build_real_returns_matrix(list(ASSET_REGIME_DATA.keys()))
    series = preload_walk_forward_series(start_date=date(1993, 1, 1), end_date=date(2025, 12, 31))
    wf_cache = WalkForwardCache(series)

    n_broad, n_crisis = 40, 40
    print(f"\nΔ = alpha_combo − alpha_baseline (pp). Seeds {SEEDS}. "
          f"broad n={n_broad}, crisi n={n_crisis}\n")
    print(f"{'window':11s} | {'Δ per seed (pp)':40s} | {'Δ media±std':14s} | {'DD base→combo':16s} | cost-adj Δ")
    print("-" * 110)

    for win in WINDOWS:
        pool = None if win == "broad" else _crisis_pool(win)
        n = n_broad if win == "broad" else n_crisis
        deltas, dd_base_l, dd_combo_l, deltas_cost = [], [], [], []
        for seed in SEEDS:
            # baseline
            r_base, _ = _apply_arm(())
            ab, ddb = _batch_alpha_dd(pool, n, seed, real_matrix, wf_cache, {})
            r_base()
            # combo deployboost
            r_c, kw = _apply_arm(("deployboost",))
            ac, ddc = _batch_alpha_dd(pool, n, seed, real_matrix, wf_cache, kw)
            r_c()
            # combo + costi 10bps
            r_c2, kw2 = _apply_arm(("deployboost",))
            kw2 = dict(kw2, cost_bps=10.0)
            ac_cost, _ = _batch_alpha_dd(pool, n, seed, real_matrix, wf_cache, kw2)
            r_c2()

            deltas.append((ac - ab) * 100)
            deltas_cost.append((ac_cost - ab) * 100)
            dd_base_l.append(ddb * 100)
            dd_combo_l.append(ddc * 100)

        dser = " ".join(f"{d:+5.1f}" for d in deltas)
        mean, std = np.mean(deltas), np.std(deltas)
        ddb_m, ddc_m = np.mean(dd_base_l), np.mean(dd_combo_l)
        cost_m = np.mean(deltas_cost)
        print(f"{win:11s} | {dser:40s} | {mean:+5.1f}±{std:4.1f}pp | "
              f"{ddb_m:5.1f}→{ddc_m:5.1f}% | {cost_m:+5.1f}pp")


if __name__ == "__main__":
    main()
