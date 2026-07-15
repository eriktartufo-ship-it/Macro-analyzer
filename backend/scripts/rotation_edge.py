"""Ha edge la ROTAZIONE vs S&P500? (richiesta Erik 2026-07-15)

Erik: "creare un trader che si concentra nel battere s&p500 smorzando le discese,
stando più aggressivo capendo dove stanno andando i soldi".

PRIMA di costruire il bot, si misura (regola: backtest prima di creare/toccare una
strategia). Se il Δ vs S&P non c'è, il bot non si costruisce.

Strategia testata = **dual momentum** (formalizzazione della richiesta):
  - relativo: compra i top-K per momentum (dove va il capitale);
  - assoluto: se un asset non batte il cash, quel peso va in cash (smorza le discese).

Dati REALI: matrice daily 24 asset (cache di B2). Benchmark: S&P (buy&hold) e 60/40.
Rendimenti applicati DAILY, decisione ogni `rebal` giorni. Δ multi-seed su start
casuali. Include i COSTI (bps sul turnover) perché la rotazione TRADA: ignorarli
sarebbe adulare la strategia.

NO FUTURE LEAK: il ranking usa solo giorni < day.
"""
from __future__ import annotations

import os
import random
import sys
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.ai_portfolio.rotation import select_rotation_portfolio  # noqa: E402

CASH = "cash_money_market"
SP = "us_equities_growth"
BOND = "us_bonds_long"
SEEDS = [1, 5, 11, 42, 99]

CRISIS = {
    "2008": (date(2007, 10, 1), date(2008, 12, 31)),
    "2020": (date(2019, 12, 1), date(2020, 8, 31)),
    "2022": (date(2021, 12, 1), date(2022, 12, 31)),
}


def _bh(dm, days, weights):
    acc = 1.0
    for d in days:
        row = dm.loc[d]
        r, w = 0.0, 0.0
        for a, wt in weights.items():
            v = row.get(a)
            if v is not None and not pd.isna(v):
                r += wt * float(v); w += wt
        if w > 0:
            acc *= (1.0 + r / w)
    return acc - 1.0


def run_rotation(dm, start, horizon, assets, top_k, rebal, lookbacks, cost_bps=10.0,
                 use_abs=True):
    idx = dm.index
    win = idx[(idx >= pd.Timestamp(start))][:horizon]
    if len(win) < 30:
        return None
    w = {CASH: 1.0}
    nav = peak = 1.0
    maxdd = 0.0
    for i, day in enumerate(win):
        if i % rebal == 0:
            new_w = select_rotation_portfolio(
                dm, day, assets, top_k=top_k, lookbacks=lookbacks,
                cash_asset=CASH, use_absolute_momentum=use_abs)
            # costo sul turnover (somma delle variazioni di peso / 2)
            turnover = sum(abs(new_w.get(a, 0) - w.get(a, 0))
                           for a in set(new_w) | set(w)) / 2.0
            nav *= (1.0 - (cost_bps / 1e4) * turnover)
            w = new_w
        row = dm.loc[day]
        r = 0.0
        for a, wt in w.items():
            v = row.get(a)
            if v is not None and not pd.isna(v):
                r += wt * float(v)
        nav *= (1.0 + r)
        peak = max(peak, nav)
        maxdd = min(maxdd, nav / peak - 1.0)
    ret = nav - 1.0
    return {
        "ret": ret, "dd": maxdd,
        "a_sp": ret - _bh(dm, win, {SP: 1.0}),
        "a_6040": ret - _bh(dm, win, {SP: 0.6, BOND: 0.4}),
        "sp_dd": None,
    }


def _sp_dd(dm, start, horizon):
    idx = dm.index
    win = idx[(idx >= pd.Timestamp(start))][:horizon]
    nav = peak = 1.0
    dd = 0.0
    for d in win:
        v = dm.loc[d].get(SP)
        nav *= (1.0 + (float(v) if v is not None and not pd.isna(v) else 0.0))
        peak = max(peak, nav)
        dd = min(dd, nav / peak - 1.0)
    return dd


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    from app.services.scoring.engine import ASSET_REGIME_DATA
    from app.services.backtest.daily_returns_matrix import build_daily_returns_matrix

    print("Loading daily matrix (cache B2)...")
    dm = build_daily_returns_matrix(list(ASSET_REGIME_DATA.keys()))
    assets = [a for a in dm.columns]
    print(f"  {dm.shape}, {dm.index.min().date()} -> {dm.index.max().date()}\n")

    horizon = 252
    print(f"ROTAZIONE dual-momentum vs S&P — seeds {SEEDS}, n=40/cella, horizon={horizon}g,")
    print("costi 10bps sul turnover. a_SP = alpha vs S&P (+ = batte l'S&P).\n")

    configs = [
        ("top3 rebal-21 lb(63,126)", dict(top_k=3, rebal=21, lookbacks=(63, 126))),
        ("top5 rebal-21 lb(63,126)", dict(top_k=5, rebal=21, lookbacks=(63, 126))),
        ("top3 rebal-5  lb(63,126)", dict(top_k=3, rebal=5, lookbacks=(63, 126))),
        ("top3 rebal-21 lb(126,252)", dict(top_k=3, rebal=21, lookbacks=(126, 252))),
        ("top3 NO-abs-mom (controprova)", dict(top_k=3, rebal=21, lookbacks=(63, 126), use_abs=False)),
    ]

    for win_name in ["broad", "2008", "2020", "2022"]:
        print("=" * 104)
        print(f"WINDOW: {win_name}")
        print("=" * 104)
        print(f"{'config':32s} | {'a_SP media±std':16s} | {'a_60/40':9s} | {'ret':8s} | {'DD rot':8s} | DD S&P")
        print("-" * 104)
        for label, cfg in configs:
            per_seed_asp, arr = [], {"asp": [], "a60": [], "ret": [], "dd": [], "spdd": []}
            for s in SEEDS:
                rng = random.Random(s)
                vals = {"asp": [], "a60": [], "ret": [], "dd": [], "spdd": []}
                for _ in range(40):
                    if win_name == "broad":
                        d = date(rng.randint(1998, 2023), rng.randint(1, 12), 1)
                    else:
                        d0, d1 = CRISIS[win_name]
                        pool = [x for x in dm.index if d0 <= x.date() <= d1]
                        d = rng.choice(pool).date()
                    r = run_rotation(dm, d, horizon, assets, lookbacks=cfg["lookbacks"],
                                     top_k=cfg["top_k"], rebal=cfg["rebal"],
                                     use_abs=cfg.get("use_abs", True))
                    if r:
                        vals["asp"].append(r["a_sp"]); vals["a60"].append(r["a_6040"])
                        vals["ret"].append(r["ret"]); vals["dd"].append(r["dd"])
                        vals["spdd"].append(_sp_dd(dm, d, horizon))
                if not vals["asp"]:
                    continue
                per_seed_asp.append(float(np.mean(vals["asp"])) * 100)
                for k in arr:
                    arr[k].append(float(np.mean(vals[k])))
            if not per_seed_asp:
                continue
            print(f"{label:32s} | {np.mean(arr['asp'])*100:+6.1f} ±{np.std(per_seed_asp):4.1f}pp | "
                  f"{np.mean(arr['a60'])*100:+6.1f}pp | {np.mean(arr['ret'])*100:+6.1f}% | "
                  f"{np.mean(arr['dd'])*100:6.1f}% | {np.mean(arr['spdd'])*100:6.1f}%")
        print()


if __name__ == "__main__":
    main()
