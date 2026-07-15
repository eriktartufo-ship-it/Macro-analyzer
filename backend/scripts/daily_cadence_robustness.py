"""B2 (S41) — effetto della CADENZA di decisione, isolato sul motore daily.

Stessa base di rendimenti (daily nominale) e stesso regime (griglia settimanale
ffill): l'unica cosa che cambia è ogni-quanti-giorni si DECIDE (monthly=21 /
weekly=5 / daily=1). Risponde a S39b ("il mensile aggiunge ±7pp di rumore-fortuna"):
se la cadenza fine RIDUCE la std multi-seed dell'alpha → conferma; e mostra se
cattura meglio le V (return/DD nelle crisi).

Per ogni (finestra, cadenza): batch multi-seed → media ± std di alpha vs 60/40,
return, DD, deployed. La griglia settimanale è cachata su disco (build una volta).
"""
from __future__ import annotations

import os
import sys
from datetime import date

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from horserace_backtest_daily import run_horserace_trade_daily, CRISIS_WINDOWS_D  # noqa: E402

SEEDS = [1, 5, 11, 42, 99]
CADENCES = ["monthly", "weekly", "daily"]
WINDOWS = ["broad", "2008", "2020", "2022"]


def _batch(pool, n, seed, dm, grid, cadence, horizon_days, yr):
    import random
    rng = random.Random(seed)
    a, asp, ret, dd, dep = [], [], [], [], []
    for _ in range(n):
        if pool is not None:
            d = rng.choice(pool).date()
        else:
            d = date(rng.randint(yr[0], yr[1]), rng.randint(1, 12), 1)
        t = run_horserace_trade_daily(d, dm, grid, horizon_days, cadence)
        if t and t.alpha_vs_60_40 is not None:
            a.append(t.alpha_vs_60_40); asp.append(t.alpha_vs_sp500)
            ret.append(t.realized_return)
            dd.append(t.max_drawdown); dep.append(t.avg_deployed)
    f = lambda x: (float(np.mean(x)) if x else float("nan"),
                   float(np.std(x)) if x else float("nan"))
    return f(a), f(asp), f(ret), f(dd), f(dep)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.environ.setdefault("USE_MOMENTUM_PILLARS", "1")
    os.environ.setdefault("USE_AI_PORTFOLIO_DEPLOY_BOOST", "0")

    from app.services.scoring.engine import ASSET_REGIME_DATA
    from app.services.backtest.daily_returns_matrix import build_daily_returns_matrix
    from app.services.backtest.daily_regime_grid import build_weekly_grid
    from app.services.backtest.walk_forward import preload_walk_forward_series

    print("Loading daily matrix + weekly regime grid (cache)...")
    dm = build_daily_returns_matrix(list(ASSET_REGIME_DATA.keys()))
    series = preload_walk_forward_series(start_date=date(1993, 1, 1), end_date=date(2025, 12, 31))
    grid = build_weekly_grid(series, date(1995, 1, 1), date(2025, 12, 31))
    print(f"  daily matrix {dm.shape}, grid {len(grid)} punti\n")

    n = 40
    horizon = 252  # ~1 anno di borsa
    print(f"Seeds {SEEDS}, n={n}/cella, horizon={horizon}g. alpha/DD/dep = media±std multi-seed.\n")
    for win in WINDOWS:
        pool = None if win == "broad" else [d for d in dm.index
                                            if CRISIS_WINDOWS_D[win][0] <= d.date() <= CRISIS_WINDOWS_D[win][1]]
        yr = (1998, 2023)
        print("=" * 100)
        print(f"WINDOW: {win}")
        print("=" * 100)
        print(f"{'cadence':8s} | {'alpha vs60/40':15s} | {'alpha vsSP500':15s} | {'return':9s} | {'maxDD':8s} | dep")
        print("-" * 100)
        for cad in CADENCES:
            # multi-seed: media delle medie per-seed + std TRA i seed (= rumore-fortuna)
            per_seed_alpha, per_seed_sp = [], []
            arr_a, arr_sp, arr_r, arr_dd, arr_dep = [], [], [], [], []
            for s in SEEDS:
                (a_m, _), (sp_m, _), (r_m, _), (dd_m, _), (dep_m, _) = _batch(pool, n, s, dm, grid, cad, horizon, yr)
                per_seed_alpha.append(a_m * 100); per_seed_sp.append(sp_m * 100)
                arr_a.append(a_m); arr_sp.append(sp_m); arr_r.append(r_m)
                arr_dd.append(dd_m); arr_dep.append(dep_m)
            a_mean, a_std = np.mean(arr_a) * 100, np.std(per_seed_alpha)
            sp_mean, sp_std = np.mean(arr_sp) * 100, np.std(per_seed_sp)
            r_mean = np.mean(arr_r) * 100
            dd_mean = np.mean(arr_dd) * 100
            dep_mean = np.mean(arr_dep) * 100
            print(f"{cad:8s} | {a_mean:+6.1f} ±{a_std:4.1f}pp | {sp_mean:+6.1f} ±{sp_std:4.1f}pp | "
                  f"{r_mean:+6.1f}% | {dd_mean:6.1f}% | {dep_mean:3.0f}%")
        print()


if __name__ == "__main__":
    main()
