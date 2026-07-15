"""Anti-overfitting di deploy_boost — sensibilità ai parametri + out-of-sample.

Richiesta Erik (2026-07-13): "una rosa di variabili... non dobbiamo fare overfitting
del modello, sennò diventa perfetto per i backtest ma non pronto per un futuro diverso
dal passato."

Due batterie:
  A) SENSIBILITÀ: varia UN parametro alla volta su un range → il Δ (boost−baseline) è
     un PLATEAU largo (robusto) o un PICCO stretto (overfit su un valore fortunato)?
  B) OUT-OF-SAMPLE: parametri di default valutati su periodi NON usati per sceglierli
     - broad TRAIN (1995-2010) vs broad TEST (2011-2023)
     - crisi EARLY (2008) vs crisi LATE (2020+2022)
     Se il Δ regge fuori campione → generalizza; se crolla → overfit al passato.

Nessuna ottimizzazione automatica dei parametri (sarebbe la strada dritta all'overfit):
solo verifica che il beneficio NON dipenda da una taratura fine.
"""
from __future__ import annotations

import os
import random
import sys
from datetime import date

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from horserace_backtest import run_horserace_trade  # noqa: E402

SEEDS = [1, 42]
N = 35


def _pool(kind):
    wins = {"2008": [((2007, 10), (2008, 12))],
            "late": [((2019, 12), (2020, 8)), ((2021, 12), (2022, 12))]}
    pool = []
    for (y0, m0), (y1, m1) in wins[kind]:
        y, m = y0, m0
        while (y, m) <= (y1, m1):
            pool.append(date(y, m, 1))
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return pool


def _alpha(real_matrix, wf_cache, *, boost, floor_pct=0.30, ramp_cap=2, incr=55.0,
           pool=None, yr=(1995, 2023), seed=1, n=N):
    """Alpha medio vs 60/40. boost=False → baseline pulito; True → deploy_boost coi param dati."""
    from app.services.ai_portfolio import strategist as S, sizer as SZ
    o_incr, o_ntr = S.INCREMENTAL_SCORE_THRESHOLD, SZ.dynamic_n_tranches
    kwargs = {}
    if boost:
        S.INCREMENTAL_SCORE_THRESHOLD = incr  # deadband: 55=on, 60=off
        SZ.dynamic_n_tranches = lambda c, _o=o_ntr, _k=ramp_cap: (min(_k, _o(c)) or 1) if _o(c) > 0 else 0
        kwargs = {"levers": {"floor"}, "floor_pct": floor_pct}
    try:
        rng = random.Random(seed)
        a = []
        for _ in range(n):
            d = rng.choice(pool) if pool else date(rng.randint(yr[0], yr[1]), rng.randint(1, 12), 1)
            t = run_horserace_trade(d, real_matrix, wf_cache, 12, 0.0, **kwargs)
            if t and t.alpha_vs_60_40 is not None:
                a.append(t.alpha_vs_60_40)
        return float(np.mean(a)) * 100 if a else float("nan")
    finally:
        S.INCREMENTAL_SCORE_THRESHOLD, SZ.dynamic_n_tranches = o_incr, o_ntr


def _delta(real_matrix, wf_cache, *, pool=None, yr=(1995, 2023), **boost_kw):
    """Δ medio (boost−baseline) sui SEEDS."""
    ds = []
    for s in SEEDS:
        base = _alpha(real_matrix, wf_cache, boost=False, pool=pool, yr=yr, seed=s)
        bo = _alpha(real_matrix, wf_cache, boost=True, pool=pool, yr=yr, seed=s, **boost_kw)
        ds.append(bo - base)
    return float(np.mean(ds)), float(np.std(ds))


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
    print(f"Δ = alpha_boost − alpha_baseline (pp), media±std su seed {SEEDS}, n={N}. Default: floor0.30/ramp2/deadband55\n")

    def line(tag, mean, std, marker=""):
        print(f"  {tag:34s} Δ {mean:+5.1f}±{std:4.1f}pp {marker}")

    # --- A) SENSIBILITÀ (broad, full sample) ---
    print("A) SENSIBILITÀ PARAMETRI (broad 1995-2023) — cerca PLATEAU, non PICCO")
    print("  floor_pct:")
    for f in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        m, s = _delta(real_matrix, wf_cache, floor_pct=f)
        line(f"    floor={f:.2f}", m, s, "← default" if f == 0.30 else "")
    print("  ramp_cap (max tranche):")
    for rc in [1, 2, 3, 4]:
        m, s = _delta(real_matrix, wf_cache, ramp_cap=rc)
        line(f"    ramp={rc}", m, s, "← default" if rc == 2 else "")
    print("  deadband (soglia add-tranche; 60=OFF):")
    for iv in [55.0, 57.0, 60.0]:
        m, s = _delta(real_matrix, wf_cache, incr=iv)
        line(f"    incr={iv:.0f}", m, s, "← default(on)" if iv == 55 else ("(deadband OFF)" if iv == 60 else ""))

    # --- B) OUT-OF-SAMPLE (default params) ---
    print("\nB) OUT-OF-SAMPLE (parametri default, periodi disgiunti) — il Δ generalizza?")
    m, s = _delta(real_matrix, wf_cache, yr=(1995, 2010)); line("  broad TRAIN 1995-2010", m, s)
    m, s = _delta(real_matrix, wf_cache, yr=(2011, 2023)); line("  broad TEST  2011-2023", m, s, "← mai usato per tarare")
    m, s = _delta(real_matrix, wf_cache, pool=_pool("2008")); line("  crisi EARLY (2008)", m, s)
    m, s = _delta(real_matrix, wf_cache, pool=_pool("late")); line("  crisi LATE (2020+2022)", m, s, "← crisi 'future'")


if __name__ == "__main__":
    main()
