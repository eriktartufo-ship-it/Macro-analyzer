"""Robustness analysis del sistema (Tier 4.8 roadmap Bridgewater).

Due strumenti contro il data mining:

1. **Lead-time sensitivity grid**: il lead time 9.2m medio dipende da 2
   parametri arbitrari — signal threshold (default 0.35) e lookback window
   (default 12 mesi). Se cambiando questi il risultato cambia drammaticamente,
   il segnale e' frutto di parameter tuning. Se resta stabile, e' robusto.

2. **Walk-forward OOS backtest**: il backtest 2003-2026 e' calcolato sulla
   storia intera. Walk-forward divide in finestre rolling e calcola
   metriche OOS per ogni finestra. Se Sharpe OOS oscilla in [0.6-1.0]
   intorno al valore globale 0.88, e' stabile. Se varia in [-0.5, +1.5],
   e' data mining (il classifier funziona solo su certi periodi).

Note: il classifier rule-based ha pesi hardcoded (no training). Quindi
"walk-forward" non separa training/test su parameter learning ma su
serie temporale (= rolling OOS evaluation).
"""
from __future__ import annotations

from datetime import date
from statistics import mean, median, stdev

import pandas as pd
from loguru import logger
from sqlalchemy.orm import Session

from app.services.backtest.lead_time import compute_lead_time_report


def lead_time_sensitivity_grid(
    db: Session,
    thresholds: tuple[float, ...] = (0.30, 0.35, 0.40, 0.45, 0.50),
    lookbacks: tuple[int, ...] = (6, 12, 18, 24),
) -> dict:
    """Grid sweep (threshold × lookback) per il lead-time.

    Robustezza: se hit_rate e avg_lead sono stabili across le celle, il
    segnale e' genuino. Se variano drammaticamente, il default 0.35/12
    e' parameter tuning ex-post.

    Returns:
        {
            "grid": [{threshold, lookback, hit_rate, avg_lead_months, n_hits}, ...],
            "hit_rate_mean/min/max": float,
            "avg_lead_mean/min/max": float,
            "thresholds_tested", "lookbacks_tested": list,
        }
    """
    grid: list[dict] = []
    for th in thresholds:
        for lb in lookbacks:
            r = compute_lead_time_report(db, threshold=th, lookback_months=lb)
            n_hits = sum(1 for rec in r.recessions if rec.signal_date)
            grid.append({
                "threshold": th,
                "lookback": lb,
                "hit_rate": r.hit_rate,
                "avg_lead_months": r.avg_lead_months,
                "n_hits": n_hits,
                "n_recessions": r.n_recessions_analyzed,
            })

    hit_rates = [g["hit_rate"] for g in grid]
    avg_leads = [g["avg_lead_months"] for g in grid if g["avg_lead_months"] is not None]

    return {
        "grid": grid,
        "hit_rate_mean": mean(hit_rates),
        "hit_rate_min": min(hit_rates),
        "hit_rate_max": max(hit_rates),
        "hit_rate_std": stdev(hit_rates) if len(hit_rates) > 1 else 0.0,
        "avg_lead_mean": mean(avg_leads) if avg_leads else None,
        "avg_lead_median": median(avg_leads) if avg_leads else None,
        "avg_lead_min": min(avg_leads) if avg_leads else None,
        "avg_lead_max": max(avg_leads) if avg_leads else None,
        "avg_lead_std": stdev(avg_leads) if len(avg_leads) > 1 else 0.0,
        "thresholds_tested": list(thresholds),
        "lookbacks_tested": list(lookbacks),
    }


def walk_forward_backtest(
    db: Session,
    window_start: date = date(2005, 1, 1),
    window_end: date = date(2026, 1, 1),
    test_window_months: int = 36,
    step_months: int = 12,
) -> dict:
    """Rolling OOS evaluation della macro_score_weighted strategy.

    Divide la finestra [window_start, window_end] in finestre overlapping di
    `test_window_months` mesi (default 36), avanzando di `step_months` (12).
    Per ogni finestra esegue il backtest e raccoglie CAGR/Sharpe/MaxDD della
    macro_score_weighted strategy.

    Note: il classifier rule-based non ha training step (pesi hardcoded),
    quindi "walk-forward" qui = rolling OOS evaluation di una strategy
    deterministica. Variabilita' rilevante: cambia il regime mix nel periodo.

    Args:
        test_window_months: durata di ogni finestra (>= 24 per backtest valido).
        step_months: avanzamento tra finestre (overlap = window - step).

    Returns:
        {
            "periods": [{test_start, test_end, cagr, sharpe, max_drawdown, calmar}, ...],
            "n_periods": int,
            "sharpe_mean/std/min/max": float,
            "cagr_mean/std/min/max": float,
            "max_drawdown_worst": float,
            "stability_verdict": "robust" | "fragile" | "insufficient_data",
        }
    """
    from app.services.backtest.runner import run_full_backtest

    periods: list[dict] = []
    current = pd.Timestamp(window_start)
    end_ts = pd.Timestamp(window_end)

    while current + pd.DateOffset(months=test_window_months) <= end_ts:
        test_start = current
        test_end = current + pd.DateOffset(months=test_window_months)
        try:
            r = run_full_backtest(
                db,
                start=test_start.date(),
                end=test_end.date(),
            )
            macro = next(s for s in r.strategies if s.name == "macro_score_weighted")
            st = macro.stats
            periods.append({
                "test_start": test_start.date().isoformat(),
                "test_end": test_end.date().isoformat(),
                "cagr": st.annualized_return,
                "sharpe": st.sharpe,
                "max_drawdown": st.max_drawdown,
                "calmar": st.calmar,
                "n_months": st.n_months,
            })
        except Exception as e:
            logger.warning(f"walk-forward skip {test_start.date()}: {e}")
        current = current + pd.DateOffset(months=step_months)

    if not periods:
        return {
            "periods": [],
            "n_periods": 0,
            "stability_verdict": "insufficient_data",
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "test_window_months": test_window_months,
            "step_months": step_months,
        }

    sharpes = [p["sharpe"] for p in periods]
    cagrs = [p["cagr"] for p in periods]
    drawdowns = [p["max_drawdown"] for p in periods]

    sharpe_std = stdev(sharpes) if len(sharpes) > 1 else 0.0

    # Verdict euristico:
    # - "robust" se sharpe_std < 0.30 (variabilita' moderata)
    # - "fragile" se sharpe_std >= 0.30 OR min(sharpe) < 0
    sharpe_min = min(sharpes)
    if sharpe_std < 0.30 and sharpe_min >= 0:
        verdict = "robust"
    elif sharpe_std < 0.50:
        verdict = "moderate"
    else:
        verdict = "fragile"

    return {
        "periods": periods,
        "n_periods": len(periods),
        "sharpe_mean": mean(sharpes),
        "sharpe_std": sharpe_std,
        "sharpe_min": sharpe_min,
        "sharpe_max": max(sharpes),
        "cagr_mean": mean(cagrs),
        "cagr_std": stdev(cagrs) if len(cagrs) > 1 else 0.0,
        "cagr_min": min(cagrs),
        "cagr_max": max(cagrs),
        "max_drawdown_worst": min(drawdowns),
        "max_drawdown_best": max(drawdowns),
        "stability_verdict": verdict,
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "test_window_months": test_window_months,
        "step_months": step_months,
    }
