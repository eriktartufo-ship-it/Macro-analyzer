"""Random-start backtest simulator (10y) — confronta modello vs benchmark.

Pattern ispirato a TradingAgents decision log: simula multiple "vite" del PM
con partenze random tra 1985-2014, per 10 anni ciascuna, e misura:
- Modello: regime classifier → top-5 allocation → monthly return cumulativo
- Benchmark: 60/40 statico (US equity growth 60% + US bonds long 40%)

Output:
- Per ogni simulazione: total return, CAGR, max drawdown, Sharpe
- Aggregate: mean, std, hit rate (% simul model > benchmark)
- Errori sistemici: per quali regimi il modello fa peggio?

NB: il modello viene RI-CALCOLATO mese per mese basandosi sul DB
RegimeClassification (1027 records 1971-2026, mensili). Return proxy
basato su ASSET_REGIME_DATA + noise gaussiano scalato dalla vol.

Usage:
    python scripts/random_backtest.py [--n-simulations 10] [--horizon-years 10]
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class SimulationResult:
    start_date: date
    end_date: date
    n_months: int
    model_cumulative_return: float
    benchmark_cumulative_return: float
    model_cagr: float
    benchmark_cagr: float
    model_max_drawdown: float
    benchmark_max_drawdown: float
    model_sharpe: float
    benchmark_sharpe: float
    model_sortino: float = 0.0
    benchmark_sortino: float = 0.0
    alpha_total: float = 0.0  # model - benchmark
    regime_distribution: dict[str, int] = field(default_factory=dict)
    error_months_by_regime: dict[str, int] = field(default_factory=dict)
    calm_gate_fires: int = 0
    model_monthly_returns: list[float] = field(default_factory=list)
    benchmark_monthly_returns: list[float] = field(default_factory=list)
    extra_benchmarks_monthly: dict[str, list[float]] = field(default_factory=dict)


def _safe_float(v, default=0.0):
    if v is None:
        return default
    try:
        import math
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _compute_drawdown(cumulative_returns: list[float]) -> float:
    """Max drawdown su serie cumulative."""
    peak = 1.0
    max_dd = 0.0
    for cr in cumulative_returns:
        portfolio_value = 1.0 + cr
        if portfolio_value > peak:
            peak = portfolio_value
        dd = (portfolio_value - peak) / peak
        if dd < max_dd:
            max_dd = dd
    return max_dd


def _compute_sharpe(monthly_returns: list[float]) -> float:
    """Sharpe annualized: mean/std × √12."""
    if not monthly_returns:
        return 0.0
    arr = np.array(monthly_returns)
    if arr.std() < 1e-9:
        return 0.0
    return float(arr.mean() / arr.std() * np.sqrt(12))


def _compute_risk_parity_weights(top5, real_matrix, current_date, vol_window: int = 12):
    """Q1.4 council: pesa top-5 inversamente alla vol rolling 12m da real returns.

    Pattern Bridgewater All-Weather. Asset volatili pesati meno → DD-control.
    Fallback equal-weight se vol non calcolabile (history insufficient).
    """
    from app.services.portfolio.position_sizing import risk_parity
    target = pd.Timestamp(year=current_date.year, month=current_date.month, day=1) + pd.offsets.MonthEnd(0)
    asset_names = [a for a, _ in top5]
    vols = {}
    for asset in asset_names:
        if asset not in real_matrix.columns:
            vols[asset] = 0.20  # default 20% annualized
            continue
        hist = real_matrix.loc[real_matrix.index <= target, asset].dropna().tail(vol_window)
        if len(hist) >= 6:
            vols[asset] = float(hist.std(ddof=1)) * (12 ** 0.5)
        else:
            vols[asset] = 0.20
    scores_dict = {a: s for a, s in top5}
    alloc = risk_parity(scores_dict, vols, top_n=len(top5))
    return alloc.weights or {a: 1.0 / len(top5) for a, _ in top5}


def _compute_sortino(monthly_returns: list[float], target: float = 0.0) -> float:
    """T10: Sortino annualized = mean / downside_std * sqrt(12).

    Downside_std considera solo returns sotto target (default 0). E' la
    metrica HONEST consigliata dal council per validare flag opt-in:
    cattura DD-control senza penalizzare upside volatility.
    """
    if not monthly_returns:
        return 0.0
    arr = np.array(monthly_returns)
    downside = arr[arr < target]
    if len(downside) < 2:
        return 0.0
    dstd = downside.std(ddof=1)
    if dstd < 1e-9:
        return 0.0
    return float(arr.mean() / dstd * np.sqrt(12))


def _simulate_monthly_return(
    weights: dict[str, float],
    regime: str,
    regime_data: dict,
    rng: np.random.Generator,
) -> float:
    """Calcola return mensile dato (weights, regime, ASSET_REGIME_DATA).

    Per ogni asset:
    monthly_return = avg_return_annual / 12 + noise * vol_annual / sqrt(12)
    """
    total_return = 0.0
    for asset, weight in weights.items():
        if asset not in regime_data:
            continue
        if regime not in regime_data[asset]:
            continue
        ar_data = regime_data[asset][regime]
        avg_return = _safe_float(ar_data.get("avg_return", 0.0))
        vol = _safe_float(ar_data.get("vol", 0.10))
        # Monthly
        mu_m = avg_return / 12.0
        sigma_m = vol / np.sqrt(12)
        noise = rng.normal(0, sigma_m)
        asset_return_m = mu_m + noise
        total_return += weight * asset_return_m
    return total_return


def run_simulation(
    start_date: date,
    horizon_years: int,
    classifications_by_month: dict[tuple[int, int], dict],
    regime_data: dict,
    seed: int = 42,
    use_calm_gate: bool = False,
    real_returns_matrix=None,
    walk_forward_cache=None,
    allocation_method: str = "equal",
    transaction_cost_bps: float = 0.0,
) -> SimulationResult | None:
    """Simula partenza random per N anni.

    Args:
        use_calm_gate: T9-AUDIT-TA — se True, applica calm regime gate
            (P(goldi)>0.30 + conf>0.30 → fallback 60/40).
        real_returns_matrix: T9-TA — pd.DataFrame indexed by month-end con
            real returns 1m per asset. Se None, usa ASSET_REGIME_DATA proxy
            (CIRCULAR, magnitudes inflated). Se provided, usa prezzi reali
            yfinance + FRED CPI (HONEST).
        walk_forward_cache: T10b — WalkForwardCache instance. Se provided,
            classifica on-the-fly invocando classify_regime con indicators
            troncati a current month-end (NO future leak). Sblocca testabilita
            dei flag classifier-side (GDP_COLLAPSE, ML_BLEND, etc.). Se None,
            legge classifications pre-computed dal DB.
    """
    rng = np.random.default_rng(seed)
    n_months_target = horizon_years * 12

    model_monthly = []
    bench_monthly = []
    regime_dist: dict[str, int] = {}
    error_months: dict[str, int] = {}
    calm_gate_fires = 0

    # NEW council 2026-05-27 #2: Benchmark suite multipla per isolare edge reale.
    # Manteniamo bench primario 60/40 per back-compat metriche existing, MA
    # tracciamo anche All-Weather + Permanent + pure risk_parity per CI confronti.
    benchmark_weights = {"us_equities_growth": 0.60, "us_bonds_long": 0.40}
    benchmarks_extra = {
        # Bridgewater All-Weather conservative (raydalio's own)
        "all_weather": {
            "us_equities_growth": 0.30, "us_bonds_long": 0.40,
            "us_bonds_short": 0.15, "gold": 0.075, "broad_commodities": 0.075,
        },
        # Permanent Portfolio (Browne)
        "permanent": {
            "us_equities_growth": 0.25, "us_bonds_long": 0.25,
            "gold": 0.25, "cash_money_market": 0.25,
        },
    }
    extra_monthly: dict[str, list[float]] = {k: [] for k in benchmarks_extra}

    # CRITICAL #3 council 2026-05-27: transaction costs.
    # Costo realistic per rebalance: sum(|w_new - w_prev|) * cost_bps / 10000.
    # Council target default 10bps/leg. Per benchmark 60/40 statico, costo zero
    # (rebalance trascurabile mensile, semplificazione).
    prev_model_weights: dict[str, float] = {}
    cost_factor = transaction_cost_bps / 10000.0  # bps -> frazione

    current = start_date
    n_used = 0
    while n_used < n_months_target:
        # T10b: walk-forward classify on-the-fly se cache provided,
        # altrimenti legge DB pre-computed.
        if walk_forward_cache is not None:
            # Month-end target
            month_end = (pd.Timestamp(current.year, current.month, 1)
                         + pd.offsets.MonthEnd(0)).date()
            cm = walk_forward_cache.get(month_end)
        else:
            key = (current.year, current.month)
            cm = classifications_by_month.get(key)
        if cm is None:
            current = current + timedelta(days=31)
            current = current.replace(day=1)
            continue

        regime = cm["regime"]
        probs = cm["probs"]
        confidence = cm.get("confidence", 0.5) or 0.5
        # T10: scoring flag-aware via calculate_final_scores (rispetta env flags).
        # Fallback proxy se chiamata fallisce (back-compat).
        # T10c: passa secular_bonus dedollar quando walk-forward disponibile.
        secular_bonus_dict = None
        if walk_forward_cache is not None:
            try:
                from app.services.config_flags import use_dedollar_bonus
                if use_dedollar_bonus():
                    from app.services.backtest.walk_forward import (
                        compute_dedollar_secular_bonus_at
                    )
                    secular_bonus_dict = compute_dedollar_secular_bonus_at(
                        walk_forward_cache.series, month_end
                    )
            except Exception:
                pass
        try:
            from app.services.scoring.engine import calculate_final_scores
            asset_scores = calculate_final_scores(probs, secular_bonus=secular_bonus_dict)
        except Exception:
            asset_scores = {}
            for asset, regime_dict in regime_data.items():
                s = 0.0
                for r, p in probs.items():
                    d = regime_dict.get(r, {})
                    ret_norm = max(0.0, min(1.0,
                        (_safe_float(d.get("avg_return", -0.30)) + 0.30) / 0.60))
                    s += p * ret_norm
                asset_scores[asset] = s
        top5 = sorted(asset_scores.items(), key=lambda x: -x[1])[:5]

        # T9-AUDIT-TA: calm regime gate
        if use_calm_gate:
            from app.services.portfolio.calm_regime_gate import apply_calm_gate
            top5_dict = {a: s for a, s in top5}
            top5_weights, gate = apply_calm_gate(top5_dict, probs, confidence)
            if gate.is_fired:
                calm_gate_fires += 1
        else:
            # Q1.4 council: allocation method opzionale (equal, risk_parity, score_weighted)
            if allocation_method == "risk_parity" and real_returns_matrix is not None:
                top5_weights = _compute_risk_parity_weights(
                    top5, real_returns_matrix, current
                )
            elif allocation_method == "score_weighted":
                top5_dict = {a: s for a, s in top5}
                total = sum(top5_dict.values()) or 1.0
                top5_weights = {a: s / total for a, s in top5_dict.items()}
            else:
                top5_weights = {a: 0.20 for a, _ in top5}  # equal weight top-5

        # T10c: defensive transition mode (allocation-side) integration.
        # Quando flag attivo + classifier indica transizione confusionaria,
        # sostituisci top5 con allocazione difensiva.
        try:
            from app.services.config_flags import use_defensive_transition_mode
            if use_defensive_transition_mode():
                from app.services.regime.transition_detector import (
                    assess_transition, get_defensive_allocation
                )
                ta = assess_transition(probs, confidence)
                if ta.is_transition:
                    top5_weights = get_defensive_allocation()
        except Exception:
            pass

        # Returns (real prices se matrix disponibile, altrimenti proxy circolare)
        if real_returns_matrix is not None:
            from app.services.backtest.real_returns_matrix import get_real_return
            r_model = get_real_return(real_returns_matrix, current.year, current.month, top5_weights)
            r_bench = get_real_return(real_returns_matrix, current.year, current.month, benchmark_weights)
            # NEW #2: track extra benchmarks (best-effort, accetta None se asset missing)
            for bn, bw in benchmarks_extra.items():
                er = get_real_return(real_returns_matrix, current.year, current.month, bw)
                if er is not None:
                    extra_monthly[bn].append(er)
            if r_model is None or r_bench is None:
                # Skip month — insufficient real data coverage
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1, day=1)
                else:
                    current = current.replace(month=current.month + 1, day=1)
                continue
        else:
            r_model = _simulate_monthly_return(top5_weights, regime, regime_data, rng)
            r_bench = _simulate_monthly_return(benchmark_weights, regime, regime_data, rng)

        # CRITICAL #3: applica transaction cost al model (turnover-based).
        # Cost = Σ |w_new - w_prev| × cost_factor. Bench statico (no cost).
        if cost_factor > 0 and prev_model_weights:
            all_assets = set(prev_model_weights) | set(top5_weights)
            turnover = sum(
                abs(top5_weights.get(a, 0.0) - prev_model_weights.get(a, 0.0))
                for a in all_assets
            )
            r_model -= turnover * cost_factor
        prev_model_weights = dict(top5_weights)

        model_monthly.append(r_model)
        bench_monthly.append(r_bench)
        regime_dist[regime] = regime_dist.get(regime, 0) + 1

        # Error tracking: se model_return < benchmark_return in questo mese
        if r_model < r_bench:
            error_months[regime] = error_months.get(regime, 0) + 1

        # Next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1, day=1)
        else:
            current = current.replace(month=current.month + 1, day=1)
        n_used += 1

    if not model_monthly:
        return None

    # Cumulative returns
    model_cum = []
    bench_cum = []
    mc = 1.0
    bc = 1.0
    for r_m, r_b in zip(model_monthly, bench_monthly):
        mc *= (1.0 + r_m)
        bc *= (1.0 + r_b)
        model_cum.append(mc - 1)
        bench_cum.append(bc - 1)

    n_actual = len(model_monthly)
    years = n_actual / 12.0
    model_cagr = (mc ** (1 / years) - 1) if years > 0 else 0
    bench_cagr = (bc ** (1 / years) - 1) if years > 0 else 0
    model_dd = _compute_drawdown(model_cum)
    bench_dd = _compute_drawdown(bench_cum)
    model_sharpe = _compute_sharpe(model_monthly)
    bench_sharpe = _compute_sharpe(bench_monthly)
    model_sortino = _compute_sortino(model_monthly)
    bench_sortino = _compute_sortino(bench_monthly)

    return SimulationResult(
        start_date=start_date,
        end_date=current,
        n_months=n_actual,
        model_cumulative_return=mc - 1,
        benchmark_cumulative_return=bc - 1,
        model_cagr=model_cagr,
        benchmark_cagr=bench_cagr,
        model_max_drawdown=model_dd,
        benchmark_max_drawdown=bench_dd,
        model_sharpe=model_sharpe,
        benchmark_sharpe=bench_sharpe,
        model_sortino=model_sortino,
        benchmark_sortino=bench_sortino,
        alpha_total=mc - bc,
        regime_distribution=regime_dist,
        error_months_by_regime=error_months,
        calm_gate_fires=calm_gate_fires,
        model_monthly_returns=model_monthly,
        benchmark_monthly_returns=bench_monthly,
        extra_benchmarks_monthly=extra_monthly,
    )


def load_classifications_by_month() -> dict[tuple[int, int], dict]:
    """Carica tutte le RegimeClassification dal DB, indicizzate per (anno, mese)."""
    from app.database import SessionLocal
    from app.models import RegimeClassification
    s = SessionLocal()
    out: dict[tuple[int, int], dict] = {}
    rows = s.query(RegimeClassification).order_by(RegimeClassification.date.asc()).all()
    for row in rows:
        key = (row.date.year, row.date.month)
        if key in out:
            continue  # primo del mese vince
        out[key] = {
            "regime": row.regime,
            "probs": {
                "reflation": row.probability_reflation,
                "stagflation": row.probability_stagflation,
                "deflation": row.probability_deflation,
                "goldilocks": row.probability_goldilocks,
            },
            "confidence": row.confidence,
        }
    s.close()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-simulations", type=int, default=15)
    p.add_argument("--horizon-years", type=int, default=10)
    p.add_argument("--start-min-year", type=int, default=1985)
    p.add_argument("--start-max-year", type=int, default=2014)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--calm-gate", action="store_true",
                   help="T9-AUDIT-TA: enable calm regime gate (P(goldi)>0.30+conf>0.30 -> 60/40)")
    p.add_argument("--real-prices", action="store_true",
                   help="T9-TA: use yfinance + FRED CPI real returns instead of "
                        "ASSET_REGIME_DATA proxy. Fixes data leakage. Window auto -> 1990+.")
    p.add_argument("--walk-forward", action=argparse.BooleanOptionalAction, default=True,
                   help="T10b: classifica on-the-fly per ogni month-end (NO future "
                        "leak). Default ON post-T10c council. Disabilita con "
                        "--no-walk-forward per leggere classifications DB pre-computed.")
    p.add_argument("--allocation", choices=["equal", "risk_parity", "score_weighted"],
                   default="equal",
                   help="Q1.4 council: allocation method top-5. equal=20%% each, "
                        "risk_parity=inverse vol (Bridgewater All-Weather), "
                        "score_weighted=proportional to asset score.")
    p.add_argument("--transaction-cost-bps", type=float, default=0.0,
                   help="CRITICAL #3 council: cost per rebalance in bps "
                        "(default 0=gross). Realistic 10-20bps per leg, sum su "
                        "turnover assoluto del portfolio mensile. Council "
                        "raccomanda 10bps minimo per honesty.")
    args = p.parse_args()

    print(f"=== RANDOM-START BACKTEST SIMULATOR ===")
    print(f"Simulations: {args.n_simulations}")
    print(f"Horizon: {args.horizon_years} years")
    print(f"Random start window: {args.start_min_year}-{args.start_max_year}")
    print()

    print("Loading classifications from DB...")
    cm_by_month = load_classifications_by_month()
    print(f"Loaded {len(cm_by_month)} months ({min(cm_by_month)} to {max(cm_by_month)})")

    print("\nLoading ASSET_REGIME_DATA...")
    from app.services.scoring.engine import ASSET_REGIME_DATA

    wf_cache = None
    if args.walk_forward:
        print("\nT10b: Initializing walk-forward classifier...")
        from app.services.backtest.walk_forward import (
            WalkForwardCache, preload_walk_forward_series,
        )
        series = preload_walk_forward_series(
            start_date=date(args.start_min_year, 1, 1),
            end_date=date(args.start_max_year + args.horizon_years, 12, 31),
        )
        wf_cache = WalkForwardCache(series)
        print(f"  Pre-loaded {len(series)} FRED series for walk-forward.")

    real_matrix = None
    if args.real_prices:
        print("\nT9-TA: Loading real returns matrix (yfinance + FRED CPI)...")
        from app.services.backtest.real_returns_matrix import (
            build_real_returns_matrix, coverage_report
        )
        real_matrix = build_real_returns_matrix(list(ASSET_REGIME_DATA.keys()))
        rep = coverage_report(real_matrix)
        print(f"  Matrix shape: {real_matrix.shape}, "
              f"range {rep['date_min']} to {rep['date_max']}")
        # Auto-shift start window to 1990+ if needed
        if args.start_min_year < 1990:
            print(f"  Auto-adjust start_min_year {args.start_min_year} -> 1990 "
                  f"(real prices start)")
            args.start_min_year = 1990

    print(f"\nRunning {args.n_simulations} simulations...\n")
    rng_starts = random.Random(args.seed_base)
    results: list[SimulationResult] = []

    for i in range(args.n_simulations):
        # Random start
        start_year = rng_starts.randint(args.start_min_year, args.start_max_year)
        start_month = rng_starts.randint(1, 12)
        start_d = date(start_year, start_month, 1)
        res = run_simulation(
            start_date=start_d,
            horizon_years=args.horizon_years,
            classifications_by_month=cm_by_month,
            regime_data=ASSET_REGIME_DATA,
            seed=args.seed_base + i,
            use_calm_gate=args.calm_gate,
            real_returns_matrix=real_matrix,
            walk_forward_cache=wf_cache,
            allocation_method=args.allocation,
            transaction_cost_bps=args.transaction_cost_bps,
        )
        if res is None:
            print(f"  Sim {i+1}: SKIP (no data)")
            continue
        results.append(res)
        alpha_pct = res.alpha_total * 100
        print(
            f"  Sim {i+1}: {res.start_date} +{args.horizon_years}y  "
            f"model_CAGR={res.model_cagr*100:.1f}%  bench_CAGR={res.benchmark_cagr*100:.1f}%  "
            f"alpha={alpha_pct:+.1f}%  model_DD={res.model_max_drawdown*100:.1f}%  "
            f"bench_DD={res.benchmark_max_drawdown*100:.1f}%  "
            f"Sharpe_m={res.model_sharpe:.2f} Sharpe_b={res.benchmark_sharpe:.2f}"
        )

    # Aggregate
    print(f"\n=== AGGREGATE METRICS ({len(results)} simulations) ===")
    if not results:
        return

    model_cagrs = [r.model_cagr for r in results]
    bench_cagrs = [r.benchmark_cagr for r in results]
    alphas = [r.alpha_total for r in results]
    model_dds = [r.model_max_drawdown for r in results]
    bench_dds = [r.benchmark_max_drawdown for r in results]
    model_sharpes = [r.model_sharpe for r in results]
    bench_sharpes = [r.benchmark_sharpe for r in results]

    # Council 2026-05-17 brutal debrief: CAGR/Sharpe magnitude inflated da
    # data leakage (ASSET_REGIME_DATA usato sia per returns che per scoring →
    # tautologico). Le metriche HONEST sono DD-relative e Win rate.
    print("  --- HONEST METRICS (robust al data leakage) ---")
    win_rate = sum(1 for a in alphas if a > 0) / len(alphas)
    dd_delta = np.mean(model_dds) - np.mean(bench_dds)  # positive = better
    model_sortinos = [r.model_sortino for r in results]
    bench_sortinos = [r.benchmark_sortino for r in results]
    sortino_delta = np.mean(model_sortinos) - np.mean(bench_sortinos)
    print(f"  Win rate (alpha>0):    {sum(1 for a in alphas if a > 0)}/{len(alphas)} ({win_rate*100:.0f}%)")
    print(f"  Model max DD:          mean={np.mean(model_dds)*100:+.2f}%")
    print(f"  Benchmark max DD:      mean={np.mean(bench_dds)*100:+.2f}%")
    print(f"  DD delta (M-B):        {dd_delta*100:+.2f}pp (positive = model has shallower DD)")
    print(f"  Model Sortino:         mean={np.mean(model_sortinos):.3f}")
    print(f"  Benchmark Sortino:     mean={np.mean(bench_sortinos):.3f}")
    print(f"  Sortino delta (M-B):   {sortino_delta:+.3f} (T10 gate: >+0.05 to keep flag)")

    # Q1.5 council: Block-bootstrap CI 95% sul Sortino aggregato (model + bench)
    # da TUTTI i monthly returns cross-sims. Blocchi 12m preservano
    # autocorrelazione mensile.
    try:
        from app.services.validation.bootstrap_ci import block_bootstrap_sortino_ci
        all_model_monthly = [m for r in results for m in r.model_monthly_returns]
        all_bench_monthly = [m for r in results for m in r.benchmark_monthly_returns]
        ci_model = block_bootstrap_sortino_ci(
            all_model_monthly, block_size=12, n_bootstrap=1000, random_state=42
        )
        ci_bench = block_bootstrap_sortino_ci(
            all_bench_monthly, block_size=12, n_bootstrap=1000, random_state=42
        )
        print(f"  Block-bootstrap Sortino CI 95% (block=12m, n_boot=1000):")
        print(f"    Model:     {ci_model.point_estimate:+.3f} [{ci_model.ci_lower:+.3f}, {ci_model.ci_upper:+.3f}]")
        print(f"    Benchmark: {ci_bench.point_estimate:+.3f} [{ci_bench.ci_lower:+.3f}, {ci_bench.ci_upper:+.3f}]")
        sig = ci_model.is_significant(ci_bench.point_estimate)
        print(f"    Model significantly different from bench: {sig}")
    except Exception as e:
        print(f"  (Block-bootstrap CI failed: {e})")

    # NEW #2 council: Benchmark suite confronto
    print(f"\n=== BENCHMARK SUITE (Sortino vs Model {np.mean(model_sortinos):.3f}) ===")
    extra_aggregated: dict[str, list[float]] = {}
    for r in results:
        for bn, monthly in (r.extra_benchmarks_monthly or {}).items():
            extra_aggregated.setdefault(bn, []).extend(monthly)
    for bn, monthly in extra_aggregated.items():
        if len(monthly) < 12:
            print(f"  {bn:<18} (insufficient data: {len(monthly)} months)")
            continue
        sortino = _compute_sortino(monthly)
        try:
            ci = block_bootstrap_sortino_ci(monthly, block_size=12, n_bootstrap=500, random_state=42)
            print(f"  {bn:<18} Sortino={sortino:.3f}  CI [{ci.ci_lower:+.3f}, {ci.ci_upper:+.3f}]  N={len(monthly)}")
        except Exception:
            print(f"  {bn:<18} Sortino={sortino:.3f}")
    print(f"  Alpha consistency:     std={np.std(alphas)*100:.1f}pp (lower = more reliable)")
    print("  --- SUSPECT METRICS (CAGR/Sharpe inflated da circular returns) ---")
    print(f"  Model CAGR (suspect):  mean={np.mean(model_cagrs)*100:.2f}%, std={np.std(model_cagrs)*100:.2f}%")
    print(f"  Bench CAGR:            mean={np.mean(bench_cagrs)*100:.2f}%, std={np.std(bench_cagrs)*100:.2f}%")
    print(f"  Model Sharpe (susp):   mean={np.mean(model_sharpes):.3f}")
    print(f"  Bench Sharpe:          mean={np.mean(bench_sharpes):.3f}")
    print(f"  Alpha total (susp):    mean={np.mean(alphas)*100:.2f}%")
    print()
    if real_matrix is not None:
        print("  OK: real prices yfinance+FRED CPI - CAGR/Sharpe magnitudes")
        print("  ora trustable. Returns reali (deflazionati CPI), no circular.")
    else:
        print("  WARNING: CAGR/Sharpe/alpha magnitude usano ASSET_REGIME_DATA")
        print("  come return source AND scoring input -> circular. Use --real-prices")
        print("  flag per attivare yfinance + FRED CPI real returns.")

    # Errori per regime
    total_error_by_regime: dict[str, int] = {}
    total_months_by_regime: dict[str, int] = {}
    for r in results:
        for regime, n_err in r.error_months_by_regime.items():
            total_error_by_regime[regime] = total_error_by_regime.get(regime, 0) + n_err
        for regime, n_months in r.regime_distribution.items():
            total_months_by_regime[regime] = total_months_by_regime.get(regime, 0) + n_months

    print(f"\n=== ERRORI PER REGIME (months where model < benchmark) ===")
    for regime in ("reflation", "stagflation", "deflation", "goldilocks"):
        n_err = total_error_by_regime.get(regime, 0)
        n_tot = total_months_by_regime.get(regime, 0)
        rate = n_err / n_tot if n_tot > 0 else 0
        print(f"  {regime:<13}: {n_err}/{n_tot} months ({rate*100:.0f}% error rate)")

    # Calm gate firing stats
    total_fires = sum(r.calm_gate_fires for r in results)
    total_months = sum(r.n_months for r in results)
    print(f"\n=== CALM REGIME GATE ===")
    print(f"  Gate fires: {total_fires}/{total_months} months "
          f"({total_fires/total_months*100:.1f}% of months) "
          f"[enabled={args.calm_gate}]")


if __name__ == "__main__":
    main()
