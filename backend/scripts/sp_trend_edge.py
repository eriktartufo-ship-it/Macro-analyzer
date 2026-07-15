"""Batte l'S&P un "S&P + filtro di trend"? (richiesta Erik 2026-07-15)

Il test precedente (rotation_edge.py) ha misurato la cosa SBAGLIATA per l'obiettivo:
diluire su 24 asset rinuncia per costruzione al beta dell'S&P → non può batterlo in
un campione bull-heavy (infatti: −7.4pp broad). Ha però dimezzato il drawdown.

Qui il design coerente con la richiesta ("battere s&p500 smorzando le discese"):
**resta sull'S&P, esci solo quando il trend gira**. È l'absolute momentum di
Antonacci nella sua forma più semplice, applicato al solo S&P.

Bracci:
  - SPY buy&hold                       (il benchmark da battere)
  - SPY + trend MA(w)                  (sopra MA = 100% SPY, sotto = cash)
  - SPY + trend MA(w) + rifugio        (sotto MA va nel MIGLIORE fra bond/gold/cash
                                        invece che solo cash = "dove vanno i soldi")
  - SPY leva-1.0 sopra trend, cash sotto, con conferma a N giorni (anti-whipsaw)

Costi 10bps sul turnover. Δ multi-seed su start casuali. NO FUTURE LEAK: la media
mobile usa solo giorni < day.
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

SP = "us_equities_growth"
CASH = "cash_money_market"
BOND = "us_bonds_long"
GOLD = "gold"
SEEDS = [1, 5, 11, 42, 99]

CRISIS = {
    "2008": (date(2007, 10, 1), date(2008, 12, 31)),
    "2020": (date(2019, 12, 1), date(2020, 8, 31)),
    "2022": (date(2021, 12, 1), date(2022, 12, 31)),
}


def _price_index(dm: pd.DataFrame, asset: str) -> pd.Series:
    """Ricostruisce un indice di prezzo dai rendimenti daily (per la MA)."""
    r = dm[asset].fillna(0.0)
    return (1.0 + r).cumprod()


def _above_trend(px: pd.Series, day, window: int) -> bool | None:
    """Il prezzo è sopra la sua MA(window)? Usa SOLO giorni < day (no leak)."""
    hist = px.loc[px.index < day]
    if len(hist) < window:
        return None
    ma = hist.tail(window).mean()
    return float(hist.iloc[-1]) > float(ma)


def _best_refuge(dm, day, lookback=63):
    """Rifugio = fra bond/gold/cash quello col miglior trailing return (dove va il capitale)."""
    best, best_r = CASH, -9e9
    for a in (BOND, GOLD, CASH):
        if a not in dm.columns:
            continue
        hist = dm.loc[dm.index < day, a].dropna().tail(lookback)
        if len(hist) < lookback // 2:
            continue
        r = float((1.0 + hist).prod() - 1.0)
        if r > best_r:
            best, best_r = a, r
    return best


def simulate(dm, start, horizon, mode, window=200, confirm=1, cost_bps=10.0):
    idx = dm.index
    win = idx[(idx >= pd.Timestamp(start))][:horizon]
    if len(win) < 30:
        return None
    px = _price_index(dm, SP)
    nav = peak = 1.0
    dd = 0.0
    held = SP if mode == "bh" else CASH
    streak = 0
    prev_sig = None
    for day in win:
        if mode != "bh":
            sig = _above_trend(px, day, window)
            if sig is None:
                sig = True
            # conferma anti-whipsaw: il segnale deve ripetersi `confirm` giorni
            if sig == prev_sig:
                streak += 1
            else:
                streak = 1
                prev_sig = sig
            if streak >= confirm:
                if sig:
                    target = SP
                else:
                    target = _best_refuge(dm, day) if mode == "refuge" else CASH
                if target != held:
                    nav *= (1.0 - cost_bps / 1e4)  # switch = 100% turnover
                    held = target
        v = dm.loc[day].get(held)
        r = float(v) if v is not None and not pd.isna(v) else 0.0
        nav *= (1.0 + r)
        peak = max(peak, nav)
        dd = min(dd, nav / peak - 1.0)
    return {"ret": nav - 1.0, "dd": dd}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    from app.services.scoring.engine import ASSET_REGIME_DATA
    from app.services.backtest.daily_returns_matrix import build_daily_returns_matrix

    print("Loading daily matrix (cache)...")
    dm = build_daily_returns_matrix(list(ASSET_REGIME_DATA.keys()))
    print(f"  {dm.shape}\n")

    horizon = 252
    configs = [
        ("SPY buy&hold (benchmark)", dict(mode="bh")),
        ("SPY + trend MA200", dict(mode="cash", window=200)),
        ("SPY + trend MA100", dict(mode="cash", window=100)),
        ("SPY + trend MA200 conferma-5g", dict(mode="cash", window=200, confirm=5)),
        ("SPY + trend MA200 + rifugio", dict(mode="refuge", window=200)),
        ("SPY + trend MA200 + rifugio conf-5g", dict(mode="refuge", window=200, confirm=5)),
    ]

    print(f"S&P + FILTRO DI TREND — seeds {SEEDS}, n=40/cella, horizon={horizon}g, costi 10bps.")
    print("a_SP = alpha vs SPY buy&hold (+ = BATTE l'S&P).\n")

    for win_name in ["broad", "2008", "2020", "2022"]:
        print("=" * 96)
        print(f"WINDOW: {win_name}")
        print("=" * 96)
        print(f"{'config':38s} | {'ret':8s} | {'a_SP media±std':16s} | maxDD")
        print("-" * 96)
        base = {}
        for label, cfg in configs:
            per_seed, rets, dds = [], [], []
            for s in SEEDS:
                rng = random.Random(s)
                vr, vd = [], []
                for _ in range(40):
                    if win_name == "broad":
                        d = date(rng.randint(1998, 2023), rng.randint(1, 12), 1)
                    else:
                        d0, d1 = CRISIS[win_name]
                        pool = [x for x in dm.index if d0 <= x.date() <= d1]
                        d = rng.choice(pool).date()
                    r = simulate(dm, d, horizon, **cfg)
                    if r:
                        vr.append(r["ret"]); vd.append(r["dd"])
                if vr:
                    per_seed.append(float(np.mean(vr)))
                    rets.append(float(np.mean(vr))); dds.append(float(np.mean(vd)))
            if not rets:
                continue
            m_ret, m_dd = float(np.mean(rets)), float(np.mean(dds))
            if label.startswith("SPY buy&hold"):
                base = {"ret": m_ret, "per_seed": per_seed}
                print(f"{label:38s} | {m_ret*100:+6.1f}% | {'—':>16s} | {m_dd*100:6.1f}%")
                continue
            alphas = [(a - b) * 100 for a, b in zip(per_seed, base.get("per_seed", per_seed))]
            print(f"{label:38s} | {m_ret*100:+6.1f}% | {np.mean(alphas):+6.1f} ±{np.std(alphas):4.1f}pp | "
                  f"{m_dd*100:6.1f}%")
        print()


if __name__ == "__main__":
    main()
