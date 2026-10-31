"""S56 (2026-10-31) — A/B walk-forward: PIL alle condizioni del regime trimestrale (oggi) vs annuo.

La domanda: `gdp_roc` è la variazione TRIMESTRALE del PIL (0,55% al 31/10 ≈ 2,2% annuo) mentre le
condizioni sono scritte per una crescita ANNUA (debole < 1,5%, trend 2%). Col PIL annuo il regime
del 31/10 passa da stagflazione 45,6% a reflazione 42,2%. Si cambia (= nuovo sigillo)?

CRITERIO, FISSATO PRIMA DI GUARDARE I NUMERI (può tornare NO):
  (a) il difetto che si vuole togliere deve scendere: quota delle settimane etichettate
      STAGFLAZIONE in cui il PIL reale cresceva >= 2% annuo (verità ex post) — ON < OFF;
  (b) tempismo sulle recessioni NBER a parità di base rate (stampo T20, stress = P(defl)+P(stag)):
      precisione Δ >= −1 pp a OGNI livello confrontabile, lead medio Δ >= −0,5 mesi,
      ritardo d'uscita medio Δ <= +15 giorni;
  (c) allocazione: random backtest walk-forward coi prezzi reali (stampo T10b), 15 vite × 10 anni,
      seed 42, flag di PROD: Sortino ON − OFF >= −0,05 (il gate KILL del council).
  PASS = (a) e (b) e (c). Altrimenti il flag resta OFF e non si risigilla.

Configurazione: i flag di PROD come sono (default), cambia SOLO USE_GDP_ANNUALIZED.

Uso:  python scripts/s56_gdp_annualized_ab.py [--skip-backtest]
Prima esecuzione: ~15 min per le due griglie settimanali (poi in cache per impronta dei flag).
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import date

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from regime_timeliness import nber_spans, stress_prob  # noqa: E402
from t20_walkforward_nber import BASE_RATES, _at_base_rate, _curve  # noqa: E402

from app.services.backtest.daily_regime_grid import build_weekly_grid  # noqa: E402
from app.services.backtest.walk_forward import (  # noqa: E402
    WalkForwardCache, preload_walk_forward_series,
)
from app.services.config_flags import classifier_flags_fingerprint, set_runtime_flag  # noqa: E402

S, E = date(1995, 1, 1), date(2025, 12, 31)
FLAG = "USE_GDP_ANNUALIZED"
GROWTH_OK = 2.0  # % annuo: sopra, «crescita debole» è falso


def _set(on: bool) -> None:
    set_runtime_flag(FLAG, on, persist=False)


def _grid(on: bool, series) -> pd.DataFrame:
    _set(on)
    print(f"  griglia {FLAG}={'ON ' if on else 'OFF'} · impronta {classifier_flags_fingerprint()}")
    return build_weekly_grid(series, S, E).frame


def _gdp_yoy_ex_post(series) -> pd.Series:
    """Crescita annua VERA del PIL reale nel trimestre di ogni data (verità ex post, non input)."""
    g = series["real_gdp"].dropna().copy()
    g.index = pd.to_datetime(g.index)
    yoy = (g / g.shift(4) - 1.0) * 100.0
    return yoy.dropna()


def _semantics(frame: pd.DataFrame, yoy: pd.Series) -> dict:
    probs = frame[[c for c in frame.columns if c.startswith("p_")]]
    label = probs.idxmax(axis=1).str[2:]
    g = yoy.reindex(frame.index, method="ffill")  # trimestre in corso (indice = inizio trimestre)
    stag = label == "stagflation"
    n_stag = int(stag.sum())
    wrong = int((stag & (g >= GROWTH_OK)).sum())
    return {
        "weeks": len(label), "share": label.value_counts(normalize=True).round(3).to_dict(),
        "stag_weeks": n_stag, "stag_with_growth_ok": wrong,
        "stag_wrong_rate": (wrong / n_stag) if n_stag else 0.0,
    }


def _nber(off: pd.DataFrame, on: pd.DataFrame) -> tuple[bool, list]:
    spans = [(s, e) for s, e in nber_spans(1995) if s >= pd.Timestamp(S)]
    c_off, c_on = _curve(stress_prob(off), spans), _curve(stress_prob(on), spans)
    rows = []
    for br in BASE_RATES:
        a, b = _at_base_rate(c_off, br), _at_base_rate(c_on, br)
        if a and b:
            rows.append((br, b["precision"] - a["precision"], b["lead"] - a["lead"],
                         b["exit_lag"] - a["exit_lag"], a, b))
    if not rows:
        return False, rows
    n = len(rows)
    ok = (all(r[1] >= -0.01 for r in rows)
          and sum(r[2] for r in rows) / n >= -0.5
          and sum(r[3] for r in rows) / n <= 15)
    return ok, rows


def _backtest(on: bool, wf: WalkForwardCache, real_matrix, regime_data, n=15, years=10, seed=42) -> dict:
    from scripts.random_backtest import run_simulation
    from app.services.scoring.engine import reload_calibration

    _set(on)
    reload_calibration()
    rng = random.Random(seed)
    res = []
    for i in range(n):
        start = date(rng.randint(1995, 2014), rng.randint(1, 12), 1)
        r = run_simulation(start_date=start, horizon_years=years, classifications_by_month={},
                           regime_data=regime_data, seed=seed + i, use_calm_gate=False,
                           real_returns_matrix=real_matrix, walk_forward_cache=wf)
        if r is not None:
            res.append(r)
    return {
        "n": len(res),
        "alpha_mean_pct": float(np.mean([r.alpha_total for r in res]) * 100),
        "win": sum(1 for r in res if r.alpha_total > 0),
        "sortino": float(np.mean([r.model_sortino for r in res])),
        "dd_pct": float(np.mean([r.model_max_drawdown for r in res]) * 100),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-backtest", action="store_true")
    args = ap.parse_args()

    print(f"A/B {FLAG} · finestra {S} → {E}\n")
    series = preload_walk_forward_series(S, E)
    yoy = _gdp_yoy_ex_post(series)

    off, on = _grid(False, series), _grid(True, series)
    if float((stress_prob(off) - stress_prob(on)).abs().max()) < 1e-9:
        print("⛔ griglie IDENTICHE: il flag non arriva al classificatore. Test non valido.")
        return

    # (a) semantica
    sa, sb = _semantics(off, yoy), _semantics(on, yoy)
    a_ok = sb["stag_wrong_rate"] < sa["stag_wrong_rate"]
    print("\n(a) STAGFLAZIONE CON CRESCITA >= 2% ANNUO (il difetto)")
    for name, s in (("OFF", sa), ("ON ", sb)):
        print(f"  {name}: {s['stag_with_growth_ok']}/{s['stag_weeks']} settimane di stagflazione "
              f"= {s['stag_wrong_rate']:.1%} · quote regime {s['share']}")
    print(f"  → {'✅' if a_ok else '❌'}")

    # (b) NBER
    b_ok, rows = _nber(off, on)
    print("\n(b) RECESSIONI NBER A PARITÀ DI BASE RATE")
    for br, dp, dl, dx, a, b in rows:
        print(f"  {br:>4.0%} | precisione {a['precision']:.0%}→{b['precision']:.0%} ({dp:+.0%}) | "
              f"lead {a['lead']:.1f}→{b['lead']:.1f} ({dl:+.1f} m) | uscita {a['exit_lag']:.0f}→{b['exit_lag']:.0f} ({dx:+.0f} gg)")
    print(f"  → {'✅' if b_ok else '❌'}")

    c_ok = None
    if not args.skip_backtest:
        from app.services.backtest.real_returns_matrix import build_real_returns_matrix
        from app.services.scoring.engine import ASSET_REGIME_DATA

        real_matrix = build_real_returns_matrix(list(ASSET_REGIME_DATA.keys()))
        wf = WalkForwardCache(series)
        bo = _backtest(False, wf, real_matrix, ASSET_REGIME_DATA)
        bn = _backtest(True, wf, real_matrix, ASSET_REGIME_DATA)
        d = bn["sortino"] - bo["sortino"]
        c_ok = d >= -0.05
        print("\n(c) ALLOCAZIONE (15 vite × 10 anni, prezzi reali, walk-forward)")
        for name, s in (("OFF", bo), ("ON ", bn)):
            print(f"  {name}: Sortino {s['sortino']:.3f} · alpha medio {s['alpha_mean_pct']:+.1f}% · "
                  f"vince {s['win']}/{s['n']} · DD medio {s['dd_pct']:.1f}%")
        print(f"  Δ Sortino {d:+.3f} → {'✅' if c_ok else '❌'}")

    set_runtime_flag(FLAG, None, persist=False)
    print("\n" + "=" * 70)
    verdict = a_ok and b_ok and (c_ok is not False)
    print(f"VERDETTO: {'✅ PASS' if verdict else '❌ FAIL'}"
          + ("" if c_ok is not None else " (senza (c): backtest saltato)"))


if __name__ == "__main__":
    main()
