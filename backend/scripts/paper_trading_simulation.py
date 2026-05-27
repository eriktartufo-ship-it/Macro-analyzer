"""User request 2026-05-27 — Paper trading SIMULATO con walk-forward.

Differenza vs random_backtest:
- Random backtest: rebalance MENSILE su 10 anni, ogni mese ri-classifica
- Paper trading: prediction UNA VOLTA al t0, HOLD 12 mesi, evaluate at t+12m

E' la sim onesta di un trader che apre posizione e aspetta 12 mesi.
Walk-forward classifier (indicators troncati a t0): NO FUTURE LEAK.

Per ogni simulation:
1. Pick random date 1990-2023 (lascia ≥12m forward)
2. Classify regime con walk-forward (use_momentum_pillars opt-in)
3. Compute top-5 weighted scores
4. Apply allocation (risk_parity default)
5. Compute realized cumulative return 12m forward via real_returns_matrix
6. Confronto vs 60/40, All-Weather, Permanent Portfolio benchmarks
7. Reverse engineering automatic su failures

Aggregate report:
- Win rate vs ogni benchmark
- Block-bootstrap CI sui Sortino
- Pattern detection (failures by regime, confidence, etc.)
- Actionable suggestions

Uso: `python scripts/paper_trading_simulation.py [--n-sims 30] [--seed 42] [--momentum]`
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@dataclass
class PaperTrade:
    """Singola simulation di paper trade."""
    start_date: date
    end_date: date
    regime: str
    confidence: float
    probabilities: dict[str, float]
    top5_weights: dict[str, float]
    realized_return_12m: float | None = None
    benchmark_60_40_12m: float | None = None
    benchmark_aw_12m: float | None = None
    benchmark_pp_12m: float | None = None
    alpha_vs_60_40: float | None = None
    alpha_vs_aw: float | None = None  # All-Weather
    alpha_vs_pp: float | None = None  # Permanent Portfolio
    # Reverse engineering analysis
    severity: str = "n/a"
    findings: list[str] = field(default_factory=list)


BENCHMARK_60_40 = {"us_equities_growth": 0.60, "us_bonds_long": 0.40}
BENCHMARK_ALL_WEATHER = {
    "us_equities_growth": 0.30, "us_bonds_long": 0.40,
    "us_bonds_short": 0.15, "gold": 0.075, "broad_commodities": 0.075,
}
BENCHMARK_PERMANENT = {
    "us_equities_growth": 0.25, "us_bonds_long": 0.25,
    "gold": 0.25, "cash_money_market": 0.25,
}


def _compute_cumulative_12m(matrix: pd.DataFrame, weights: dict, start: date) -> float | None:
    """Real cumulative return 12 mesi forward su portfolio weighted."""
    start_ts = pd.Timestamp(start.year, start.month, 1) + pd.offsets.MonthEnd(0)
    end_ts = start_ts + pd.DateOffset(months=12)
    mask = (matrix.index > start_ts) & (matrix.index <= end_ts)
    sub = matrix.loc[mask]
    if len(sub) < 10:  # serve almeno 10 mesi validi
        return None

    # Compound per-asset
    total = 0.0
    weight_sum = 0.0
    for asset, w in weights.items():
        if asset not in sub.columns:
            continue
        series = sub[asset].dropna()
        if len(series) < 10:
            continue
        cum = (1.0 + series).prod() - 1.0
        total += w * cum
        weight_sum += w
    if weight_sum < 0.5:
        return None
    return total / weight_sum if weight_sum > 0 else None


def _classify_severity(alpha: float | None) -> str:
    if alpha is None:
        return "no_data"
    if alpha < -0.10:
        return "severe"
    if alpha < -0.05:
        return "moderate"
    if alpha < 0:
        return "minor"
    return "winning"


def _post_mortem_synthetic(trade: PaperTrade) -> list[str]:
    """Reverse engineering findings sintetici per paper trade (no DB)."""
    findings = []
    max_prob = max(trade.probabilities.values()) if trade.probabilities else 0.0

    if trade.severity in ("severe", "moderate"):
        if max_prob < 0.40:
            findings.append(
                f"Regime boundary case (max_prob {max_prob:.2f}). "
                f"Modello aveva bassa conviction su '{trade.regime}'."
            )
        if trade.confidence < 0.30:
            findings.append(
                f"Confidence bassa ({trade.confidence:.2f}). "
                f"Sarebbe stato meglio defensive allocation."
            )
        # Regime-specific
        if trade.regime == "stagflation" and (trade.alpha_vs_60_40 or 0) < -0.05:
            findings.append(
                "Stagflation prediction underperformed: probabile late-cycle "
                "reflation (inflation alta MA growth resiliente). "
                "Check DFM nowcast in retrospect."
            )
        elif trade.regime == "deflation" and (trade.alpha_vs_60_40 or 0) < -0.05:
            findings.append(
                "Deflation prediction underperformed: GDP_COLLAPSE potrebbe "
                "aver triggered su economy resilient (no real recession)."
            )
        elif trade.regime == "reflation" and (trade.alpha_vs_60_40 or 0) < -0.05:
            findings.append(
                "Reflation underperformed: possibile stagflation embrionica "
                "(CPI > 3.5%) o regime confusion goldilocks boundary."
            )
    elif trade.severity == "winning":
        if trade.confidence > 0.45:
            findings.append(
                f"High-confidence ({trade.confidence:.2f}) regime '{trade.regime}' "
                f"vincente. Pattern conferma value della conviction."
            )

    return findings


def run_paper_trade(
    start_date: date,
    real_matrix: pd.DataFrame,
    wf_cache,
) -> PaperTrade | None:
    """Una sim: classifica con walk-forward at t0, HOLD 12m, evaluate."""
    from app.services.scoring.engine import calculate_final_scores
    from app.services.portfolio.position_sizing import risk_parity

    month_end = (pd.Timestamp(start_date.year, start_date.month, 1)
                 + pd.offsets.MonthEnd(0)).date()
    cm = wf_cache.get(month_end)
    if cm is None:
        return None

    probs = cm["probs"]
    confidence = cm.get("confidence", 0.5) or 0.5
    regime = cm["regime"]

    try:
        asset_scores = calculate_final_scores(probs)
    except Exception:
        return None

    sorted_scores = sorted(asset_scores.items(), key=lambda x: -x[1])[:5]
    # Risk parity weights from vols rolling 12m
    target_ts = pd.Timestamp(start_date.year, start_date.month, 1) + pd.offsets.MonthEnd(0)
    vols = {}
    for asset, _ in sorted_scores:
        if asset in real_matrix.columns:
            hist = real_matrix.loc[real_matrix.index <= target_ts, asset].dropna().tail(12)
            vols[asset] = float(hist.std(ddof=1)) * np.sqrt(12) if len(hist) >= 6 else 0.20
        else:
            vols[asset] = 0.20
    alloc = risk_parity({a: s for a, s in sorted_scores}, vols, top_n=len(sorted_scores))
    top5_weights = alloc.weights or {a: 1.0 / len(sorted_scores) for a, _ in sorted_scores}

    # Paper trading insight: uncertainty gate
    # Se conf < 0.30 → fallback 60/40 statico (modello incerto, no aggressive bet)
    try:
        from app.services.config_flags import use_uncertainty_gate
        if use_uncertainty_gate() and confidence < 0.30:
            top5_weights = dict(BENCHMARK_60_40)
    except Exception:
        pass

    end_date = start_date + timedelta(days=365)

    trade = PaperTrade(
        start_date=start_date,
        end_date=end_date,
        regime=regime,
        confidence=confidence,
        probabilities=probs,
        top5_weights=top5_weights,
    )

    # Realized 12m
    trade.realized_return_12m = _compute_cumulative_12m(real_matrix, top5_weights, start_date)
    trade.benchmark_60_40_12m = _compute_cumulative_12m(real_matrix, BENCHMARK_60_40, start_date)
    trade.benchmark_aw_12m = _compute_cumulative_12m(real_matrix, BENCHMARK_ALL_WEATHER, start_date)
    trade.benchmark_pp_12m = _compute_cumulative_12m(real_matrix, BENCHMARK_PERMANENT, start_date)

    if trade.realized_return_12m is not None and trade.benchmark_60_40_12m is not None:
        trade.alpha_vs_60_40 = trade.realized_return_12m - trade.benchmark_60_40_12m
    if trade.realized_return_12m is not None and trade.benchmark_aw_12m is not None:
        trade.alpha_vs_aw = trade.realized_return_12m - trade.benchmark_aw_12m
    if trade.realized_return_12m is not None and trade.benchmark_pp_12m is not None:
        trade.alpha_vs_pp = trade.realized_return_12m - trade.benchmark_pp_12m

    trade.severity = _classify_severity(trade.alpha_vs_60_40)
    trade.findings = _post_mortem_synthetic(trade)

    return trade


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-sims", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start-min-year", type=int, default=1991)
    p.add_argument("--start-max-year", type=int, default=2023)  # ≥12m forward = max 2024
    p.add_argument("--momentum", action="store_true", help="USE_MOMENTUM_PILLARS=1")
    p.add_argument("--uncertainty-gate", action="store_true",
                   help="USE_UNCERTAINTY_GATE=1 (conf < 0.30 -> 60/40 fallback)")
    args = p.parse_args()

    os.environ["USE_MOMENTUM_PILLARS"] = "1" if args.momentum else "0"
    os.environ["USE_UNCERTAINTY_GATE"] = "1" if args.uncertainty_gate else "0"

    print(f"=" * 70)
    print(f"PAPER TRADING SIMULATION — {args.n_sims} random dates, 12m hold")
    print(f"Window: {args.start_min_year}-{args.start_max_year}, seed {args.seed}")
    print(f"USE_MOMENTUM_PILLARS={'ON' if args.momentum else 'OFF'}")
    print(f"=" * 70)

    print("\nLoading real returns matrix...")
    from app.services.scoring.engine import ASSET_REGIME_DATA
    from app.services.backtest.real_returns_matrix import build_real_returns_matrix
    real_matrix = build_real_returns_matrix(list(ASSET_REGIME_DATA.keys()))
    print(f"  Matrix shape: {real_matrix.shape}")

    print("\nPre-loading FRED series for walk-forward...")
    from datetime import date as _date
    from app.services.backtest.walk_forward import (
        WalkForwardCache, preload_walk_forward_series,
    )
    series = preload_walk_forward_series(
        start_date=_date(args.start_min_year - 2, 1, 1),
        end_date=_date(args.start_max_year + 2, 12, 31),
    )
    wf_cache = WalkForwardCache(series)

    rng = random.Random(args.seed)
    trades: list[PaperTrade] = []

    print(f"\nRunning {args.n_sims} paper trades...\n")
    for i in range(args.n_sims):
        y = rng.randint(args.start_min_year, args.start_max_year)
        m = rng.randint(1, 12)
        d = date(y, m, 1)
        trade = run_paper_trade(d, real_matrix, wf_cache)
        if trade is None or trade.realized_return_12m is None:
            print(f"  Sim {i+1}: SKIP (no data)")
            continue
        trades.append(trade)
        a60 = (trade.alpha_vs_60_40 or 0) * 100
        aaw = (trade.alpha_vs_aw or 0) * 100
        app_ = (trade.alpha_vs_pp or 0) * 100
        ret = (trade.realized_return_12m or 0) * 100
        sev_marker = {"severe": "X", "moderate": "?", "minor": "-",
                      "winning": "+", "no_data": "."}.get(trade.severity, "?")
        print(f"  [{sev_marker}] Sim {i+1}: {d} {trade.regime:<12} conf={trade.confidence:.2f}  "
              f"ret={ret:+6.1f}%  a60/40={a60:+6.1f}pp  aAW={aaw:+6.1f}pp  aPP={app_:+6.1f}pp")
        if trade.findings:
            for f in trade.findings:
                print(f"         > {f}")

    if not trades:
        print("Nessuna simulation valida.")
        return

    print(f"\n{'=' * 70}")
    print("AGGREGATE METRICS (Paper trading 12m hold)")
    print(f"{'=' * 70}")

    n = len(trades)
    returns = [t.realized_return_12m for t in trades if t.realized_return_12m is not None]
    alphas_60_40 = [t.alpha_vs_60_40 for t in trades if t.alpha_vs_60_40 is not None]
    alphas_aw = [t.alpha_vs_aw for t in trades if t.alpha_vs_aw is not None]
    alphas_pp = [t.alpha_vs_pp for t in trades if t.alpha_vs_pp is not None]

    print(f"\n  Trades completati: {n}")
    print(f"  Mean realized return 12m: {np.mean(returns)*100:+.2f}% (std {np.std(returns)*100:.2f}%)")
    print(f"  Win rate vs 60/40: {sum(1 for a in alphas_60_40 if a > 0)}/{len(alphas_60_40)} "
          f"({100*sum(1 for a in alphas_60_40 if a > 0)/len(alphas_60_40):.0f}%)")
    print(f"  Win rate vs All-Weather: {sum(1 for a in alphas_aw if a > 0)}/{len(alphas_aw)} "
          f"({100*sum(1 for a in alphas_aw if a > 0)/len(alphas_aw):.0f}%)")
    print(f"  Win rate vs Permanent: {sum(1 for a in alphas_pp if a > 0)}/{len(alphas_pp)} "
          f"({100*sum(1 for a in alphas_pp if a > 0)/len(alphas_pp):.0f}%)")
    print(f"\n  Mean alpha vs 60/40: {np.mean(alphas_60_40)*100:+.2f}pp")
    print(f"  Mean alpha vs All-Weather: {np.mean(alphas_aw)*100:+.2f}pp")
    print(f"  Mean alpha vs Permanent: {np.mean(alphas_pp)*100:+.2f}pp")
    print(f"  Alpha 60/40 std: {np.std(alphas_60_40)*100:.2f}pp")

    # Block-bootstrap CI alpha vs 60/40
    try:
        from app.services.validation.bootstrap_ci import block_bootstrap_metric
        ci = block_bootstrap_metric(
            alphas_60_40, aggregator=np.mean, block_size=3, n_bootstrap=1000,
            random_state=args.seed, metric_name="alpha_vs_60_40",
        )
        print(f"\n  Block-bootstrap alpha 60/40 CI 95%: {ci.point_estimate*100:+.2f}pp "
              f"[{ci.ci_lower*100:+.2f}, {ci.ci_upper*100:+.2f}]")
        print(f"  Significantly > 0: {ci.is_significant(0.0) and ci.ci_lower > 0}")
    except Exception as e:
        print(f"  CI failed: {e}")

    print(f"\n{'=' * 70}")
    print("REVERSE ENGINEERING — Patterns identified")
    print(f"{'=' * 70}")

    # Per regime breakdown
    by_regime: dict[str, list[float]] = {}
    for t in trades:
        if t.alpha_vs_60_40 is not None:
            by_regime.setdefault(t.regime, []).append(t.alpha_vs_60_40)

    print(f"\n  Per regime:")
    for regime in ("reflation", "stagflation", "deflation", "goldilocks"):
        alphas = by_regime.get(regime, [])
        if not alphas:
            print(f"    {regime:<14} (no trades)")
            continue
        win_rate = sum(1 for a in alphas if a > 0) / len(alphas)
        mean_a = np.mean(alphas)
        print(f"    {regime:<14} n={len(alphas):<3} "
              f"win {win_rate*100:>3.0f}%  mean_a={mean_a*100:+6.2f}pp")

    # Severity distribution
    print(f"\n  Severity distribution:")
    sev_count = {}
    for t in trades:
        sev_count[t.severity] = sev_count.get(t.severity, 0) + 1
    for sev in ("winning", "minor", "moderate", "severe"):
        c = sev_count.get(sev, 0)
        print(f"    {sev:<14} {c}/{n} ({100*c/n:.0f}%)")

    # Low confidence failures
    low_conf_failures = [t for t in trades
                        if t.confidence < 0.30 and (t.alpha_vs_60_40 or 0) < -0.05]
    if low_conf_failures:
        print(f"\n  Low-confidence failures: {len(low_conf_failures)}/{n}")
        print(f"    -> Suggestion: filtra predictions con conf < 0.30 -> defensive allocation")

    # Boundary case failures
    boundary_failures = []
    for t in trades:
        max_p = max(t.probabilities.values()) if t.probabilities else 0
        if max_p < 0.40 and (t.alpha_vs_60_40 or 0) < -0.05:
            boundary_failures.append(t)
    if boundary_failures:
        print(f"  Boundary case failures (max_prob < 0.40): {len(boundary_failures)}/{n}")
        print(f"    -> Suggestion: lower calm_regime_gate threshold + defensive_transition")

    # Save JSON for downstream analysis
    out_dir = Path(__file__).resolve().parent.parent.parent / "obsidian" / "Erik" / "02_Progetti" / "Macro_Analyzer"
    if not out_dir.exists():
        out_dir = Path(__file__).resolve().parent.parent
    suffix = "_momentum" if args.momentum else ""
    json_path = out_dir / f"Paper_Trading_Report{suffix}.json"
    json_path.write_text(json.dumps({
        "metadata": {
            "n_sims": n,
            "seed": args.seed,
            "window": f"{args.start_min_year}-{args.start_max_year}",
            "momentum_active": args.momentum,
        },
        "summary": {
            "mean_return_12m": float(np.mean(returns)),
            "mean_alpha_60_40": float(np.mean(alphas_60_40)),
            "mean_alpha_aw": float(np.mean(alphas_aw)),
            "mean_alpha_pp": float(np.mean(alphas_pp)),
            "win_rate_60_40": sum(1 for a in alphas_60_40 if a > 0) / len(alphas_60_40),
            "win_rate_aw": sum(1 for a in alphas_aw if a > 0) / len(alphas_aw),
            "win_rate_pp": sum(1 for a in alphas_pp if a > 0) / len(alphas_pp),
        },
        "by_regime": {
            r: {
                "n": len(by_regime.get(r, [])),
                "win_rate": sum(1 for a in by_regime.get(r, []) if a > 0) / max(1, len(by_regime.get(r, []))),
                "mean_alpha": float(np.mean(by_regime.get(r, []))) if by_regime.get(r) else None,
            }
            for r in ("reflation", "stagflation", "deflation", "goldilocks")
        },
        "trades": [
            {
                "start": t.start_date.isoformat(),
                "regime": t.regime,
                "confidence": t.confidence,
                "realized_return_12m": t.realized_return_12m,
                "alpha_60_40": t.alpha_vs_60_40,
                "alpha_aw": t.alpha_vs_aw,
                "alpha_pp": t.alpha_vs_pp,
                "severity": t.severity,
                "findings": t.findings,
            }
            for t in trades
        ],
    }, indent=2), encoding="utf-8")
    print(f"\nJSON report saved: {json_path}")


if __name__ == "__main__":
    main()
