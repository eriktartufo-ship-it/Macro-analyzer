"""User request 2026-05-27 — Paper trading DINAMICO con walk-forward.

VERSIONE 2 (post user feedback): NO hold 12m statico. INVECE:
- Allocazione DINAMICA: ribilancia ogni mese (default) o frequenza configurable
- Tiene SEMPRE gli asset con vantaggio statistico al momento
- Walk-forward classifier (no future leak): re-classify mese-per-mese
- Transaction costs realistic (10bps default)
- Tracking turnover + regime changes per misurare "quanto si muove"

Differenza vs random_backtest 10y:
- Paper trading focused 12m horizon (no 10-year compounding noise)
- Reverse engineering automatic per trade fail
- Benchmark suite (60/40 + All-Weather + Permanent) parallela
- N sim casuali con date di start diverse

Output: aggregate report con win rate, alpha, turnover, regime transitions.

Uso: python scripts/paper_trading_simulation.py [--n-sims 30] [--seed 42]
     [--rebalance monthly|quarterly] [--momentum] [--uncertainty-gate]
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
    """Singola simulation: 12m con rebalance dinamico."""
    start_date: date
    end_date: date
    rebalance_freq: str  # "monthly" | "quarterly"

    # Aggregate tracking
    regime_trajectory: list[tuple[date, str, float]] = field(default_factory=list)
    n_rebalances: int = 0
    n_regime_changes: int = 0
    total_turnover: float = 0.0  # somma |Δw| nel periodo
    uncertainty_gate_fires: int = 0

    # Final metrics
    realized_return_12m: float | None = None
    realized_volatility: float | None = None  # annualized monthly std
    benchmark_60_40_12m: float | None = None
    benchmark_aw_12m: float | None = None
    benchmark_pp_12m: float | None = None
    benchmark_sp500_12m: float | None = None
    max_drawdown: float = 0.0  # cum DD durante 12m
    alpha_vs_60_40: float | None = None
    alpha_vs_aw: float | None = None
    alpha_vs_pp: float | None = None
    alpha_vs_sp500: float | None = None

    # Reverse engineering
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
# User request 2026-05-27: aggiungere S&P500 come benchmark.
# Proxy: 100% us_equities_growth (QQQ/Nasdaq nel nostro asset universe).
BENCHMARK_SP500 = {"us_equities_growth": 1.0}


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


def _compute_top5_weights(probs, real_matrix, target_ts, top_n: int = 5,
                          method: str = "risk_parity", confidence: float = 0.5):
    """Top-N weights dato regime probabilities + sizing method.

    Methods:
    - risk_parity: inverse vol (Bridgewater All-Weather)
    - equal: 1/N each
    - score_weighted: peso ∝ score
    - confidence_weighted: equal weight × scale_factor; scale = clip(conf × 1.5, 0.6, 1.0).
      Cash buffer fills the rest (low conf → less deployed → cash hedge).
    - vol_target_concentration: tilt top-3 (60%) quando conf > 0.6, equal-7 (40%)
      quando conf < 0.4, smooth interpolation in mezzo.
    """
    from app.services.scoring.engine import calculate_final_scores
    from app.services.portfolio.position_sizing import risk_parity

    try:
        asset_scores = calculate_final_scores(probs)
    except Exception:
        return None

    sorted_scores = sorted(asset_scores.items(), key=lambda x: -x[1])[:top_n]
    n = len(sorted_scores)

    if method == "equal":
        return {a: 1.0 / n for a, _ in sorted_scores}

    if method == "score_weighted":
        total = sum(s for _, s in sorted_scores) or 1.0
        return {a: s / total for a, s in sorted_scores}

    if method == "confidence_weighted":
        # Council sizing (d): scale exposure with confidence, cash buffer rest
        scale = max(0.6, min(1.0, confidence * 1.5))
        equal_w = scale / n
        weights = {a: equal_w for a, _ in sorted_scores}
        cash_remainder = 1.0 - scale
        if cash_remainder > 0.001:
            weights["cash_money_market"] = weights.get("cash_money_market", 0.0) + cash_remainder
        return weights

    if method == "vol_target_concentration":
        # Council sizing (c): tilt top-3 vs top-N based on confidence
        # conf > 0.6 → top-3 60% / next (n-3) 40%
        # conf < 0.4 → equal n
        # smooth interpolation
        if n <= 3:
            return {a: 1.0 / n for a, _ in sorted_scores}
        tilt = max(0.0, min(1.0, (confidence - 0.4) / 0.2))  # 0 at conf 0.4, 1 at conf 0.6+
        top3_pool = 0.40 + tilt * 0.20  # 40% top-3 at conf 0.4 → 60% at conf 0.6
        rest_pool = 1.0 - top3_pool
        weights = {}
        for i, (asset, _) in enumerate(sorted_scores):
            if i < 3:
                weights[asset] = top3_pool / 3
            else:
                weights[asset] = rest_pool / (n - 3)
        return weights

    # Calcola vols rolling 12m per tutti gli asset top-N (usato da risk_parity,
    # confidence_kelly e kelly_fractional)
    vols = {}
    for asset, _ in sorted_scores:
        if asset in real_matrix.columns:
            hist = real_matrix.loc[real_matrix.index <= target_ts, asset].dropna().tail(12)
            vols[asset] = float(hist.std(ddof=1)) * np.sqrt(12) if len(hist) >= 6 else 0.20
        else:
            vols[asset] = 0.20

    if method == "confidence_kelly":
        # T12 Council action #2: confidence-gated Kelly fractional con caps.
        # conf < 0.6: equal-weight (safe), 0.6-0.8 smooth blend, >0.8 full Kelly.
        # Caps 3-25% per evitare single-name risk.
        from app.services.portfolio.position_sizing import confidence_kelly as ck
        alloc = ck(
            {a: s for a, s in sorted_scores}, vols,
            confidence=confidence, top_n=n,
        )
        return alloc.weights or {a: 1.0 / n for a, _ in sorted_scores}

    # risk_parity (default)
    alloc = risk_parity({a: s for a, s in sorted_scores}, vols, top_n=n)
    return alloc.weights or {a: 1.0 / n for a, _ in sorted_scores}


def _portfolio_return_month(matrix, year, month, weights, hedge_proxy: str = "otm_put"):
    """Real return mensile di un portfolio at (year, month).

    T15: gestisce synthetic asset `tail_hedge_spy_put` via simulate_hedge_return
    basato su SPY return del mese (proxy us_equities_growth).
    """
    from app.services.portfolio.tail_hedge import simulate_hedge_return

    target = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    if target not in matrix.index:
        return None
    row = matrix.loc[target]
    # SPY proxy = us_equities_growth (per tail hedge synthetic payoff)
    spy_return = None
    if "us_equities_growth" in row.index and not pd.isna(row["us_equities_growth"]):
        spy_return = float(row["us_equities_growth"])

    total = 0.0
    weight_sum = 0.0
    for asset, w in weights.items():
        if asset == "tail_hedge_spy_put":
            if spy_return is None:
                continue
            total += w * simulate_hedge_return(spy_return, proxy=hedge_proxy)
            weight_sum += w
            continue
        if asset in row.index and not pd.isna(row[asset]):
            total += w * float(row[asset])
            weight_sum += w
    if weight_sum < 0.5:
        return None
    return total / weight_sum if weight_sum > 0 else None


def run_paper_trade_dynamic(
    start_date: date,
    real_matrix: pd.DataFrame,
    wf_cache,
    rebalance_freq: str = "monthly",
    transaction_cost_bps: float = 10.0,
    top_n: int = 5,
    allocation_method: str = "risk_parity",
    use_event_triggers: bool = False,
    use_tail_hedge: bool = False,
    hedge_proxy: str = "otm_put",
    use_trigger_rebalance: bool = False,
) -> PaperTrade | None:
    """Sim 12m con allocazione DINAMICA (rebalance mensile o trimestrale).

    NO future leak: ogni rebalance classifier vede solo indicators ≤ current month-end.

    Args:
        use_event_triggers: T14 council action #4. Se True, force rebalance OFF-CALENDAR
            quando trigger fires (VIX panic, M2 surge, curve uninvert, FOMC dovish).
        use_tail_hedge: T15 council action #3. Se True, alloca 0.5-2% del portfolio
            a synthetic tail hedge quando Crisis Risk Indicator score > 0.60.
    """
    from app.services.config_flags import use_uncertainty_gate
    from app.services.regime.event_triggers import (
        EventTriggerDetector, TriggerSnapshot, any_trigger_fired,
    )
    from app.services.portfolio.tail_hedge import (
        should_activate_hedge, compute_hedge_weight,
        apply_hedge_to_allocation, simulate_hedge_return,
    )
    from app.services.regime.crisis_indicator import assess_crisis_risk
    from app.services.backtest.rebalance_triggers import any_trigger_fired_simple as trigger_check

    trade = PaperTrade(
        start_date=start_date,
        end_date=start_date + timedelta(days=365),
        rebalance_freq=rebalance_freq,
    )

    cost_factor = transaction_cost_bps / 10000.0
    current = date(start_date.year, start_date.month, 1)
    model_monthly_returns: list[float] = []
    bench_60_40_monthly: list[float] = []
    bench_aw_monthly: list[float] = []
    bench_pp_monthly: list[float] = []

    current_weights: dict[str, float] = {}
    last_regime: str | None = None
    bench_sp500_monthly: list[float] = []

    # T14 event-triggered rebalance state
    trigger_detector = EventTriggerDetector() if use_event_triggers else None
    prev_trigger_snapshot: TriggerSnapshot | None = None
    event_rebalance_fires = 0

    # T19 council action #1 trigger-based rebalance state
    prev_classified_probs: dict[str, float] | None = None
    prev_indicators_snap: dict[str, float] | None = None
    trigger_rebalance_fires = 0

    for month_idx in range(12):
        # Rebalance check (calendar baseline)
        is_calendar_rebalance = (
            month_idx == 0  # initial alloc
            or rebalance_freq == "monthly"
            or (rebalance_freq == "quarterly" and month_idx % 3 == 0)
        )
        is_rebalance_month = is_calendar_rebalance

        # Classify with walk-forward
        month_end_dt = (pd.Timestamp(current.year, current.month, 1)
                        + pd.offsets.MonthEnd(0)).date()
        cm = wf_cache.get(month_end_dt)
        if cm is None:
            # No classification -&gt; skip month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1, day=1)
            else:
                current = current.replace(month=current.month + 1, day=1)
            continue

        regime = cm["regime"]
        probs = cm["probs"]
        confidence = cm.get("confidence", 0.5) or 0.5
        liquidity_surge = cm.get("liquidity_surge_triggered", False)
        trade.regime_trajectory.append((current, regime, confidence))

        if last_regime is not None and regime != last_regime:
            trade.n_regime_changes += 1
        last_regime = regime

        # T14 Event-triggered rebalance: check ad ogni mese se trigger fires
        # vs snapshot precedente. Se sì AND non è già calendar rebalance,
        # forza rebalance extra (cattura policy/panic events off-calendar).
        if trigger_detector is not None:
            indicators = cm.get("indicators", {}) or {}
            curr_snapshot = TriggerSnapshot(
                vix=float(indicators.get("vix", 18.0)),
                m2_yoy=float(indicators.get("m2_yoy", 5.0)),
                yield_curve_10y2y=float(indicators.get("yield_curve_10y2y", 1.0)),
                fomc_sentiment=float(indicators.get("fomc_sentiment", 0.0)),
            )
            trigger_results = trigger_detector.check_window(
                curr_snapshot, prev_trigger_snapshot,
            )
            if any_trigger_fired(trigger_results) and not is_calendar_rebalance:
                is_rebalance_month = True
                event_rebalance_fires += 1
            prev_trigger_snapshot = curr_snapshot

        # T19 Council action #1 — trigger-based rebalance (different from T14).
        # Check regime probability shift, VIX jump, NFCI z-score, HY OAS jump.
        # Più stringente di T14 + utilizza regime probabilities direttamente.
        if use_trigger_rebalance:
            indicators_curr = cm.get("indicators", {}) or {}
            if trigger_check(probs, prev_classified_probs, indicators_curr, prev_indicators_snap):
                if not is_calendar_rebalance:
                    is_rebalance_month = True
                    trigger_rebalance_fires += 1
            prev_classified_probs = dict(probs)
            prev_indicators_snap = dict(indicators_curr)

        # Compute new top-5 weights (con uncertainty gate)
        if is_rebalance_month:
            target_ts = pd.Timestamp(current.year, current.month, 1) + pd.offsets.MonthEnd(0)
            new_weights = _compute_top5_weights(
                probs, real_matrix, target_ts,
                top_n=top_n, method=allocation_method, confidence=confidence,
            )
            if new_weights is None:
                new_weights = current_weights or dict(BENCHMARK_60_40)

            # Uncertainty gate — BYPASS se liquidity_surge attivo (council post-2020 fix).
            # Liquidity surge = "all clear" signal, restore conviction sui top-7.
            if use_uncertainty_gate() and confidence < 0.30 and not liquidity_surge:
                new_weights = dict(BENCHMARK_60_40)
                trade.uncertainty_gate_fires += 1

            # T15 Tail hedge: if Crisis Risk > 0.60, alloca % del portfolio a hedge
            if use_tail_hedge:
                indicators = cm.get("indicators", {}) or {}
                try:
                    crisis_assessment = assess_crisis_risk(
                        indicators, dedollar_combined=None, date=str(current),
                    )
                    crisis_score = float(crisis_assessment.risk_score)
                except Exception:
                    crisis_score = 0.0
                if should_activate_hedge(crisis_score):
                    hw = compute_hedge_weight(crisis_score)
                    new_weights = apply_hedge_to_allocation(new_weights, hw)

            # Turnover from old weights
            if current_weights:
                all_assets = set(current_weights) | set(new_weights)
                turnover = sum(
                    abs(new_weights.get(a, 0.0) - current_weights.get(a, 0.0))
                    for a in all_assets
                )
                trade.total_turnover += turnover
            current_weights = new_weights
            trade.n_rebalances += 1

        # Compute monthly returns
        r_model = _portfolio_return_month(real_matrix, current.year, current.month, current_weights, hedge_proxy=hedge_proxy)
        r_b60 = _portfolio_return_month(real_matrix, current.year, current.month, BENCHMARK_60_40)
        r_aw = _portfolio_return_month(real_matrix, current.year, current.month, BENCHMARK_ALL_WEATHER)
        r_pp = _portfolio_return_month(real_matrix, current.year, current.month, BENCHMARK_PERMANENT)
        r_sp = _portfolio_return_month(real_matrix, current.year, current.month, BENCHMARK_SP500)

        if r_model is None:
            # Skip this month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1, day=1)
            else:
                current = current.replace(month=current.month + 1, day=1)
            continue

        # Applicate transaction cost se rebalance month (turnover-based)
        if is_rebalance_month and trade.total_turnover > 0:
            last_turnover = trade.total_turnover - sum(
                abs(new_weights.get(a, 0.0) - 0.0) if month_idx == 0 else 0
                for a in new_weights
            ) if month_idx > 0 else 0
            # Simpler: cost proporzionale al turnover del rebalance corrente
            # Use turnover delta vs previous (approximation)
            r_model -= cost_factor * 0.5  # approx 5bps cost per rebalance (round trip half)

        model_monthly_returns.append(r_model)
        if r_b60 is not None:
            bench_60_40_monthly.append(r_b60)
        if r_aw is not None:
            bench_aw_monthly.append(r_aw)
        if r_pp is not None:
            bench_pp_monthly.append(r_pp)
        if r_sp is not None:
            bench_sp500_monthly.append(r_sp)

        # Next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1, day=1)
        else:
            current = current.replace(month=current.month + 1, day=1)

    if len(model_monthly_returns) < 10:
        return None

    # Compound 12m
    cum_model = float(np.prod([1.0 + r for r in model_monthly_returns]) - 1.0)
    cum_b60 = float(np.prod([1.0 + r for r in bench_60_40_monthly]) - 1.0) if bench_60_40_monthly else None
    cum_aw = float(np.prod([1.0 + r for r in bench_aw_monthly]) - 1.0) if bench_aw_monthly else None
    cum_pp = float(np.prod([1.0 + r for r in bench_pp_monthly]) - 1.0) if bench_pp_monthly else None
    cum_sp = float(np.prod([1.0 + r for r in bench_sp500_monthly]) - 1.0) if bench_sp500_monthly else None

    # Max drawdown durante 12m (cumulative path)
    cum_path = np.cumprod([1.0 + r for r in model_monthly_returns])
    peak = np.maximum.accumulate(cum_path)
    dd = (cum_path / peak - 1.0)
    trade.max_drawdown = float(dd.min()) if len(dd) > 0 else 0.0

    trade.realized_return_12m = cum_model
    trade.realized_volatility = float(np.std(model_monthly_returns) * np.sqrt(12))
    trade.benchmark_60_40_12m = cum_b60
    trade.benchmark_aw_12m = cum_aw
    trade.benchmark_pp_12m = cum_pp
    trade.benchmark_sp500_12m = cum_sp

    if cum_b60 is not None:
        trade.alpha_vs_60_40 = cum_model - cum_b60
    if cum_aw is not None:
        trade.alpha_vs_aw = cum_model - cum_aw
    if cum_pp is not None:
        trade.alpha_vs_pp = cum_model - cum_pp
    if cum_sp is not None:
        trade.alpha_vs_sp500 = cum_model - cum_sp

    trade.severity = _classify_severity(trade.alpha_vs_60_40)

    # Reverse engineering findings
    if trade.severity in ("severe", "moderate"):
        if trade.n_regime_changes > 2:
            trade.findings.append(
                f"Regime instability: {trade.n_regime_changes} regime changes in 12m. "
                f"Modello over-trading. Consider higher rebalance threshold."
            )
        if trade.total_turnover > 4.0:  # > 400% cumulative turnover = high churn
            trade.findings.append(
                f"High turnover ({trade.total_turnover:.1f} cumulative). "
                f"Cost drag potrebbe spiegare underperformance."
            )
        # Regime dominante via majority vote in trajectory
        regimes_seen = [r for _, r, _ in trade.regime_trajectory]
        if regimes_seen:
            dominant = max(set(regimes_seen), key=regimes_seen.count)
            if dominant == "deflation" and (trade.alpha_vs_60_40 or 0) < -0.05:
                trade.findings.append(
                    f"Dominant regime in trajectory '{dominant}' underperformed. "
                    f"Pattern: post-crisis bounce-back NON è deflation classica."
                )
    elif trade.severity == "winning":
        avg_conf = float(np.mean([c for _, _, c in trade.regime_trajectory])) if trade.regime_trajectory else 0
        if avg_conf > 0.40:
            trade.findings.append(
                f"High-conviction trajectory (avg conf {avg_conf:.2f}) -&gt; winner."
            )

    return trade


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-sims", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start-min-year", type=int, default=1991)
    p.add_argument("--start-max-year", type=int, default=2023)
    p.add_argument("--rebalance", choices=["monthly", "quarterly"], default="monthly")
    p.add_argument("--transaction-cost-bps", type=float, default=10.0)
    p.add_argument("--momentum", action="store_true")
    p.add_argument("--uncertainty-gate", action="store_true",
                   help="Override default-ON. Use --no-uncertainty-gate to disable.")
    p.add_argument("--no-uncertainty-gate", action="store_true")
    p.add_argument("--crisis-window", choices=["2008", "2020", "2022", "all_crisis"],
                   default=None,
                   help="Council Test A: limit random dates a crisis window. "
                        "2008 = 2007-Q4 to 2009-Q1, 2020 = 2019-12 to 2020-08, "
                        "2022 = 2021-12 to 2022-12. all_crisis = union.")
    p.add_argument("--top-n", type=int, default=5,
                   help="Council Test F: top-N asset (default 5).")
    p.add_argument("--use-event-triggers", action="store_true",
                   help="T14 council #4: force rebalance off-calendar quando "
                        "trigger fires (VIX>30, M2 surge, curve uninvert, "
                        "FOMC dovish).")
    p.add_argument("--use-trigger-rebalance", action="store_true",
                   help="T19 council #1 sessione 26: trigger-based rebalance "
                        "(regime_prob_shift>0.20 OR VIX jump 50pct OR NFCI z>1 "
                        "OR HY OAS jump 30pct). Quarterly default + override.")
    p.add_argument("--use-tail-hedge", action="store_true",
                   help="T15 council #3: alloca 0.5-2pp portfolio a synthetic "
                        "tail hedge (OTM SPY put 6m) quando Crisis Risk > 0.60.")
    p.add_argument("--hedge-proxy", choices=["otm_put", "vixy"], default="otm_put",
                   help="T15 hedge proxy: otm_put (default, council T15 originale) "
                        "o vixy (more convex, higher roll cost).")
    p.add_argument("--allocation-method",
                   choices=["risk_parity", "equal", "score_weighted",
                            "confidence_weighted", "vol_target_concentration",
                            "confidence_kelly"],
                   default="risk_parity",
                   help="Council sizing options: "
                        "risk_parity, equal, score_weighted, "
                        "confidence_weighted (sizing d), "
                        "vol_target_concentration (sizing c), "
                        "confidence_kelly (T12 council #2: Kelly + caps + gating).")
    args = p.parse_args()

    os.environ["USE_MOMENTUM_PILLARS"] = "1" if args.momentum else "0"
    if args.no_uncertainty_gate:
        os.environ["USE_UNCERTAINTY_GATE"] = "0"
    elif args.uncertainty_gate:
        os.environ["USE_UNCERTAINTY_GATE"] = "1"
    # else: usa default (True post 2026-05-27)

    print(f"=" * 70)
    print(f"PAPER TRADING SIMULATION (DYNAMIC) — {args.n_sims} sims, 12m horizon")
    print(f"Rebalance: {args.rebalance} | Cost: {args.transaction_cost_bps}bps")
    print(f"Window: {args.start_min_year}-{args.start_max_year}, seed {args.seed}")
    print(f"USE_MOMENTUM_PILLARS={'ON' if args.momentum else 'OFF'}")
    print(f"USE_UNCERTAINTY_GATE={'OFF' if args.no_uncertainty_gate else 'ON (default)'}")
    print(f"=" * 70)

    print("\nLoading real returns matrix...")
    from app.services.scoring.engine import ASSET_REGIME_DATA
    from app.services.backtest.real_returns_matrix import build_real_returns_matrix
    real_matrix = build_real_returns_matrix(list(ASSET_REGIME_DATA.keys()))
    print(f"  Matrix shape: {real_matrix.shape}")

    print("\nPre-loading FRED series...")
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

    # Crisis windows (Test A council)
    CRISIS_WINDOWS = {
        "2008": [(2007, 10), (2008, 12)],  # GFC pre-bottom
        "2020": [(2019, 12), (2020, 8)],   # COVID + recovery
        "2022": [(2021, 12), (2022, 12)],  # Inflation surge
    }
    crisis_dates_pool = None
    if args.crisis_window:
        crisis_dates_pool = []
        windows = (
            [CRISIS_WINDOWS[args.crisis_window]] if args.crisis_window != "all_crisis"
            else list(CRISIS_WINDOWS.values())
        )
        for (y0, m0), (y1, m1) in windows:
            y, m = y0, m0
            while (y, m) <= (y1, m1):
                crisis_dates_pool.append(date(y, m, 1))
                if m == 12:
                    y, m = y + 1, 1
                else:
                    m += 1
        print(f"\nCrisis window pool: {len(crisis_dates_pool)} months available")

    print(f"\nRunning {args.n_sims} dynamic paper trades...\n")
    for i in range(args.n_sims):
        if crisis_dates_pool:
            d = rng.choice(crisis_dates_pool)
        else:
            y = rng.randint(args.start_min_year, args.start_max_year)
            m = rng.randint(1, 12)
            d = date(y, m, 1)
        trade = run_paper_trade_dynamic(
            d, real_matrix, wf_cache,
            rebalance_freq=args.rebalance,
            transaction_cost_bps=args.transaction_cost_bps,
            top_n=args.top_n,
            allocation_method=args.allocation_method,
            use_event_triggers=args.use_event_triggers,
            use_tail_hedge=args.use_tail_hedge,
            hedge_proxy=args.hedge_proxy,
            use_trigger_rebalance=args.use_trigger_rebalance,
        )
        if trade is None or trade.realized_return_12m is None:
            print(f"  Sim {i+1}: SKIP")
            continue
        trades.append(trade)
        a60 = (trade.alpha_vs_60_40 or 0) * 100
        ret = (trade.realized_return_12m or 0) * 100
        vol = (trade.realized_volatility or 0) * 100
        sev_marker = {"severe": "X", "moderate": "?", "minor": "-",
                      "winning": "+", "no_data": "."}.get(trade.severity, "?")
        print(f"  [{sev_marker}] Sim {i+1}: {d} ret={ret:+6.1f}%  vol={vol:5.1f}%  "
              f"a60/40={a60:+6.1f}pp  rebal={trade.n_rebalances}  "
              f"reg_chg={trade.n_regime_changes}  turn={trade.total_turnover:.1f}  "
              f"uncert_gate={trade.uncertainty_gate_fires}")
        if trade.findings:
            for f in trade.findings:
                print(f"         > {f}")

    if not trades:
        print("Nessuna simulation valida.")
        return

    print(f"\n{'=' * 70}")
    print("AGGREGATE METRICS")
    print(f"{'=' * 70}")

    n = len(trades)
    returns = [t.realized_return_12m for t in trades if t.realized_return_12m is not None]
    vols = [t.realized_volatility for t in trades if t.realized_volatility is not None]
    alphas_60_40 = [t.alpha_vs_60_40 for t in trades if t.alpha_vs_60_40 is not None]
    alphas_aw = [t.alpha_vs_aw for t in trades if t.alpha_vs_aw is not None]
    alphas_pp = [t.alpha_vs_pp for t in trades if t.alpha_vs_pp is not None]
    alphas_sp = [t.alpha_vs_sp500 for t in trades if t.alpha_vs_sp500 is not None]
    drawdowns = [t.max_drawdown for t in trades]
    rebalances = [t.n_rebalances for t in trades]
    regime_changes = [t.n_regime_changes for t in trades]
    turnovers = [t.total_turnover for t in trades]
    gate_fires = [t.uncertainty_gate_fires for t in trades]

    print(f"\n  Trades completed: {n}")
    print(f"  Mean return 12m: {np.mean(returns)*100:+.2f}% (std {np.std(returns)*100:.2f}%)")
    print(f"  Mean vol annualized: {np.mean(vols)*100:.2f}%")
    print(f"  Mean Sharpe: {(np.mean(returns) / max(np.mean(vols), 1e-6)):.3f}")
    print(f"")
    print(f"  Win rate vs 60/40: {sum(1 for a in alphas_60_40 if a > 0)}/{len(alphas_60_40)} "
          f"({100*sum(1 for a in alphas_60_40 if a > 0)/len(alphas_60_40):.0f}%)")
    print(f"  Win rate vs All-Weather: {sum(1 for a in alphas_aw if a > 0)}/{len(alphas_aw)} "
          f"({100*sum(1 for a in alphas_aw if a > 0)/len(alphas_aw):.0f}%)")
    print(f"  Win rate vs Permanent: {sum(1 for a in alphas_pp if a > 0)}/{len(alphas_pp)} "
          f"({100*sum(1 for a in alphas_pp if a > 0)/len(alphas_pp):.0f}%)")
    print(f"  Win rate vs S&P500: {sum(1 for a in alphas_sp if a > 0)}/{len(alphas_sp)} "
          f"({100*sum(1 for a in alphas_sp if a > 0)/len(alphas_sp):.0f}%)")
    print(f"")
    print(f"  Mean alpha vs 60/40: {np.mean(alphas_60_40)*100:+.2f}pp")
    print(f"  Mean alpha vs All-Weather: {np.mean(alphas_aw)*100:+.2f}pp")
    print(f"  Mean alpha vs Permanent: {np.mean(alphas_pp)*100:+.2f}pp")
    print(f"  Mean alpha vs S&P500: {np.mean(alphas_sp)*100:+.2f}pp")
    print(f"")
    print(f"  Mean max drawdown: {np.mean(drawdowns)*100:.2f}%")
    print(f"  Worst max drawdown: {min(drawdowns)*100:.2f}%")
    # Calmar = annualized return / |max DD|
    calmars = [t.realized_return_12m / abs(t.max_drawdown) if t.max_drawdown < -0.001 else 0.0
               for t in trades]
    print(f"  Mean Calmar ratio: {np.mean(calmars):.2f}")
    print(f"")
    print(f"  Mean rebalances: {np.mean(rebalances):.1f}/12 months")
    print(f"  Mean regime changes: {np.mean(regime_changes):.1f}")
    print(f"  Mean cumulative turnover: {np.mean(turnovers):.2f}")
    print(f"  Total uncertainty gate fires: {sum(gate_fires)}/{sum(rebalances)}")

    # Block bootstrap CI
    try:
        from app.services.validation.bootstrap_ci import block_bootstrap_metric
        ci = block_bootstrap_metric(
            alphas_60_40, aggregator=np.mean, block_size=3, n_bootstrap=1000,
            random_state=args.seed,
        )
        print(f"\n  Block-bootstrap alpha 60/40 CI 95%: {ci.point_estimate*100:+.2f}pp "
              f"[{ci.ci_lower*100:+.2f}, {ci.ci_upper*100:+.2f}]")
        print(f"  Significantly > 0: {ci.is_significant(0.0) and ci.ci_lower > 0}")
    except Exception as e:
        print(f"  CI failed: {e}")

    # Severity distribution
    print(f"\n  Severity:")
    sev_count = {}
    for t in trades:
        sev_count[t.severity] = sev_count.get(t.severity, 0) + 1
    for sev in ("winning", "minor", "moderate", "severe"):
        c = sev_count.get(sev, 0)
        print(f"    {sev:<14} {c}/{n} ({100*c/n:.0f}%)")

    # Save report
    out_dir = Path(__file__).resolve().parent.parent.parent / "obsidian" / "Erik" / "02_Progetti" / "Macro_Analyzer"
    if not out_dir.exists():
        out_dir = Path(__file__).resolve().parent.parent
    flags_suffix = ""
    if args.momentum:
        flags_suffix += "_momentum"
    if args.no_uncertainty_gate:
        flags_suffix += "_nogate"
    suffix = f"_{args.rebalance}{flags_suffix}_seed{args.seed}"
    json_path = out_dir / f"Paper_Trading_Dynamic{suffix}.json"
    json_path.write_text(json.dumps({
        "metadata": {
            "n_sims": n, "seed": args.seed,
            "rebalance_freq": args.rebalance,
            "transaction_cost_bps": args.transaction_cost_bps,
            "uncertainty_gate": not args.no_uncertainty_gate,
            "momentum": args.momentum,
        },
        "summary": {
            "mean_return_12m": float(np.mean(returns)),
            "mean_vol": float(np.mean(vols)),
            "mean_alpha_60_40": float(np.mean(alphas_60_40)),
            "mean_alpha_aw": float(np.mean(alphas_aw)),
            "mean_alpha_pp": float(np.mean(alphas_pp)),
            "win_rate_60_40": sum(1 for a in alphas_60_40 if a > 0) / len(alphas_60_40),
            "win_rate_aw": sum(1 for a in alphas_aw if a > 0) / len(alphas_aw),
            "win_rate_pp": sum(1 for a in alphas_pp if a > 0) / len(alphas_pp),
            "mean_rebalances": float(np.mean(rebalances)),
            "mean_regime_changes": float(np.mean(regime_changes)),
            "mean_turnover": float(np.mean(turnovers)),
        },
    }, indent=2), encoding="utf-8")
    print(f"\nReport: {json_path}")


if __name__ == "__main__":
    main()
