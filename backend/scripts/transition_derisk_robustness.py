"""S41 2026-07-14 — A/B robustezza multi-seed del TRANSITION-DERISK (detector vero).

Domanda di S40: come i dati migliorano il trading PASSANDO dal regime?
Gap verificato: transition_detector.assess_transition() esiste (entropy+max_prob+
confidence, require_all_three) ma i 3 strategist live NON lo consumano; il harness
aveva solo un proxy grezzo `derisk` (confidence<0.45 OR max_prob<0.45, spara spesso).

Qui l'A/B disciplinato (regola signals-validate-real + anti-overfitting Δ multi-seed):
per OGNI finestra, Δ (arm − baseline) su N seed → media±std. Due arm:
  - derisk    = PROXY grezzo (OR permissivo)         → riferimento
  - transition = DETECTOR VERO (assess_transition)   → l'upgrade S41

Erik (gate-gusto S41): risposta alla transizione = DE-RISK GRADUALE (dimezza il
deployment quel mese, tiene le posizioni). Meno invasivo.

Metrica primaria: Δ alpha vs 60/40 (pp). Secondarie: max-DD base→arm, deployed medio
dell'arm (selettività: quanto spesso/quanto de-riska). Il detector qui NON riceve il
financial_stress_veto (non è nel wf_cache) → è un LOWER BOUND della protezione.
"""
from __future__ import annotations

import os
import sys
from datetime import date

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from horserace_backtest import run_horserace_trade  # noqa: E402
from horserace_powerset import _crisis_pool  # noqa: E402

SEEDS = [1, 5, 11, 42, 99]
WINDOWS = ["broad", "2008", "2020", "2022", "all_crisis"]

ARMS = {
    "derisk(proxy)": {"derisk": True},
    "transition(detector)": {"transition_derisk": True},
    "stress(veto-grezzo)": {"stress_derisk": True},
}


def _batch(pool, n, seed, real_matrix, wf_cache, kwargs):
    import random
    rng = random.Random(seed)
    a, dd, dep = [], [], []
    for _ in range(n):
        d = rng.choice(pool) if pool else date(rng.randint(1995, 2023), rng.randint(1, 12), 1)
        t = run_horserace_trade(d, real_matrix, wf_cache, 12, 0.0, **kwargs)
        if t and t.alpha_vs_60_40 is not None:
            a.append(t.alpha_vs_60_40)
            dd.append(t.max_drawdown)
            dep.append(t.avg_deployed)
    return (float(np.mean(a)) if a else float("nan"),
            float(np.mean(dd)) if dd else float("nan"),
            float(np.mean(dep)) if dep else float("nan"))


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
    print(f"\nΔ = alpha_arm − alpha_baseline (pp). Seeds {SEEDS}. broad n={n_broad}, crisi n={n_crisis}")
    print("baseline = strategist puro (no derisk). DD e deployed = medie dell'ARM.\n")

    for arm_name, arm_kw in ARMS.items():
        print("=" * 104)
        print(f"ARM: {arm_name}   kwargs={arm_kw}")
        print("=" * 104)
        print(f"{'window':11s} | {'Δ per seed (pp)':30s} | {'Δ media±std':13s} | {'DD base→arm':14s} | dep base→arm")
        print("-" * 104)
        for win in WINDOWS:
            pool = None if win == "broad" else _crisis_pool(win)
            n = n_broad if win == "broad" else n_crisis
            deltas, ddb_l, dda_l, depb_l, depa_l = [], [], [], [], []
            for seed in SEEDS:
                ab, ddb, depb = _batch(pool, n, seed, real_matrix, wf_cache, {})
                aa, dda, depa = _batch(pool, n, seed, real_matrix, wf_cache, arm_kw)
                deltas.append((aa - ab) * 100)
                ddb_l.append(ddb * 100); dda_l.append(dda * 100)
                depb_l.append(depb * 100); depa_l.append(depa * 100)
            dser = " ".join(f"{d:+4.1f}" for d in deltas)
            mean, std = np.mean(deltas), np.std(deltas)
            print(f"{win:11s} | {dser:30s} | {mean:+4.1f}±{std:4.1f}pp | "
                  f"{np.mean(ddb_l):5.1f}→{np.mean(dda_l):5.1f}% | "
                  f"{np.mean(depb_l):3.0f}→{np.mean(depa_l):3.0f}%")
        print()


if __name__ == "__main__":
    main()
