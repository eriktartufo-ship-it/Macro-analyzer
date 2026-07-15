"""B2 (S41) — Backtest DAILY-cadence che rigioca la logica reale _decide/sizer.

Motivazione (S39b): il backtest mensile quantizza entrata/uscita a fine mese →
±7pp di rumore-fortuna e manca le V-recovery intra-mese. Qui:
  - rendimenti applicati GIORNO-PER-GIORNO (NAV/DD fedeli, cattura la V);
  - regime da griglia settimanale (ffill, no leak) — vedi daily_regime_grid;
  - CADENZA DI DECISIONE configurabile (daily/weekly/monthly) → isola l'effetto
    della cadenza dal resto.

Riusa le funzioni PURE del harness mensile (portfolio_return, drift_weights,
_apply_sim, _decide dello strategist reale). Momentum/vol qui sono DAILY.

NO FUTURE LEAK: momentum/vol usano solo giorni < day; il regime viene dalla
griglia settimanale precedente (grid_date ≤ day). I benchmark sono buy&hold sui
giorni della finestra.
"""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from horserace_backtest import (  # noqa: E402
    HorseraceTrade, portfolio_month_return, drift_weights, _apply_sim,
    BENCHMARK_60_40, BENCHMARK_SP500, CASH_ASSET,
)

CADENCE_DAYS = {"daily": 1, "weekly": 5, "monthly": 21}

# finestre di crisi (allineate al harness mensile, ma in date-range daily)
CRISIS_WINDOWS_D = {
    "2008": (date(2007, 10, 1), date(2008, 12, 31)),
    "2020": (date(2019, 12, 1), date(2020, 8, 31)),
    "2022": (date(2021, 12, 1), date(2022, 12, 31)),
}


def _momentum_daily(matrix, asset, day, threshold: float, lookback: int = 63) -> str:
    """Momentum daily: segno del cumulato degli ultimi `lookback` giorni PRIMA di day."""
    if asset not in matrix.columns:
        return "NEUTRAL"
    hist = matrix.loc[matrix.index < day, asset].dropna().tail(lookback)
    if len(hist) < lookback // 2:
        return "NEUTRAL"
    cum = float((1.0 + hist).prod() - 1.0)
    if cum > threshold:
        return "BULLISH"
    if cum < -threshold:
        return "BEARISH"
    return "NEUTRAL"


def _vol_daily(matrix, asset, day, lookback: int = 63) -> float:
    """Vol annualizzata da std daily trailing (≤ day). Default 0.20 se dati insuff."""
    if asset not in matrix.columns:
        return 0.20
    hist = matrix.loc[matrix.index <= day, asset].dropna().tail(lookback)
    return float(hist.std(ddof=1)) * np.sqrt(252) if len(hist) >= 20 else 0.20


def _bench_return_bh(matrix, days, weights) -> float:
    """Buy&hold compounded sui giorni della finestra per un mix di pesi."""
    acc = 1.0
    for day in days:
        row = matrix.loc[day]
        r, wsum = 0.0, 0.0
        for a, w in weights.items():
            if a in row.index and not pd.isna(row[a]):
                r += w * float(row[a]); wsum += w
        if wsum > 0:
            acc *= (1.0 + r / wsum)  # rinormalizza sul peso presente se asset mancante
    return acc - 1.0


def run_horserace_trade_daily(
    start_date: date, daily_matrix, regime_grid, horizon_days: int = 252,
    cadence: str = "monthly", top_n_regime: int | None = None,
    mom_threshold: float = 0.0,
) -> HorseraceTrade | None:
    """Rigioca la strategia AI a cadenza DAILY su una finestra storica.

    Args:
        start_date: inizio finestra.
        daily_matrix: output di build_daily_returns_matrix.
        regime_grid: WeeklyRegimeGrid (lookup ffill no-leak).
        horizon_days: durata in giorni di borsa.
        cadence: "daily"|"weekly"|"monthly" → ogni quanti giorni si DECIDE.
        top_n_regime: top-N asset del regime (default strategist.TOP_N_REGIME).
    """
    from app.services.ai_portfolio import strategist, sizer
    from app.services.scoring.engine import calculate_final_scores

    top_n_regime = top_n_regime or strategist.TOP_N_REGIME
    cadence_step = CADENCE_DAYS.get(cadence, 21)

    idx = daily_matrix.index
    end_approx = pd.Timestamp(start_date) + pd.Timedelta(days=int(horizon_days * 1.5) + 10)
    win = idx[(idx >= pd.Timestamp(start_date)) & (idx <= end_approx)][:horizon_days]
    if len(win) < 20:
        return None

    trade = HorseraceTrade(start_date=start_date, end_date=win[-1].date())
    positions: dict = {}
    cash_weight = 1.0
    nav = peak = 1.0
    deployed_samples: list[float] = []

    for i, day in enumerate(win):
        if i % cadence_step == 0:
            cm = regime_grid.get(day)
            if cm is not None:
                probs = cm["probs"]
                confidence = cm["confidence"]
                dominant = max(probs, key=probs.get)
                try:
                    scores = calculate_final_scores(probs)
                except Exception:
                    scores = {}
                regime_top = strategist._regime_top_n_assets(probs, top_n=top_n_regime)
                n_tranches = sizer.dynamic_n_tranches(confidence)

                raw, vols = {}, {}
                for a, sc in scores.items():
                    v = _vol_daily(daily_matrix, a, day)
                    vols[a] = v
                    raw[a] = sizer.raw_target_weight(sc, v) if a in regime_top else 0.0
                targets = sizer.normalize_and_cap(raw)

                for a, sc in scores.items():
                    target = targets.get(a, 0.0)
                    mom = _momentum_daily(daily_matrix, a, day, mom_threshold)
                    in_favor = a in regime_top
                    pos = positions.get(a)
                    mock = None
                    if pos is not None:
                        mock = SimpleNamespace(
                            estimated_pnl_pct=pos["pnl_pct"],
                            tranches_filled=pos["tranches_filled"],
                            tranches_total=pos["tranches_total"],
                            took_partial_tp=pos["took_partial_tp"],
                        )
                    action, _reason, size = strategist._decide(
                        asset=a, position=mock, score=sc, target=target,
                        confidence=confidence, momentum_sig=mom, in_favor=in_favor,
                        n_tranches_default=n_tranches, vol=vols.get(a, 0.20),
                        entry_threshold_shift=0.0,
                    )
                    _apply_sim(positions, a, action, size, sc, n_tranches, dominant, trade)

                cash_weight = max(0.0, 1.0 - sum(p["weight"] for p in positions.values()))

        deployed_samples.append(1.0 - cash_weight)

        # Applica i rendimenti daily OGNI giorno
        row = daily_matrix.loc[day]
        asset_returns = {a: (float(row[a]) if a in row.index and not pd.isna(row[a]) else None)
                         for a in positions}
        cash_return = float(row[CASH_ASSET]) if CASH_ASSET in row.index and not pd.isna(row[CASH_ASSET]) else 0.0
        pr = portfolio_month_return(positions, cash_weight, asset_returns, cash_return)
        cash_weight = drift_weights(positions, cash_weight, asset_returns, cash_return, pr)
        nav *= (1.0 + pr)
        peak = max(peak, nav)
        trade.max_drawdown = min(trade.max_drawdown, nav / peak - 1.0)
        trade.nav_path.append(nav)

    trade.realized_return = nav - 1.0
    trade.avg_deployed = float(np.mean(deployed_samples)) if deployed_samples else 0.0
    trade.min_deployed = float(np.min(deployed_samples)) if deployed_samples else 0.0

    trade.benchmark_60_40 = _bench_return_bh(daily_matrix, win, BENCHMARK_60_40)
    trade.benchmark_sp500 = _bench_return_bh(daily_matrix, win, BENCHMARK_SP500)
    trade.alpha_vs_60_40 = trade.realized_return - trade.benchmark_60_40
    trade.alpha_vs_sp500 = trade.realized_return - trade.benchmark_sp500
    a = trade.alpha_vs_60_40
    trade.severity = ("severe" if a < -0.10 else "moderate" if a < -0.05
                      else "minor" if a < 0 else "winning")
    return trade


# ---------------------------------------------------------------------------
# Runner: carica matrice daily + griglia settimanale, poi A/B cadenza
# ---------------------------------------------------------------------------
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import argparse
    import random

    p = argparse.ArgumentParser()
    p.add_argument("--n-sims", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--horizon-days", type=int, default=252)
    p.add_argument("--start-min-year", type=int, default=1998)
    p.add_argument("--start-max-year", type=int, default=2023)
    p.add_argument("--crisis-window", choices=["2008", "2020", "2022"], default=None)
    p.add_argument("--cadence", choices=["daily", "weekly", "monthly"], default="monthly")
    args = p.parse_args()

    os.environ.setdefault("USE_MOMENTUM_PILLARS", "1")
    os.environ.setdefault("USE_AI_PORTFOLIO_DEPLOY_BOOST", "0")

    from app.services.scoring.engine import ASSET_REGIME_DATA
    from app.services.backtest.daily_returns_matrix import build_daily_returns_matrix
    from app.services.backtest.daily_regime_grid import build_weekly_grid
    from app.services.backtest.walk_forward import preload_walk_forward_series

    print("Building DAILY returns matrix...")
    dm = build_daily_returns_matrix(list(ASSET_REGIME_DATA.keys()))
    print(f"  shape {dm.shape}, {dm.index.min().date()} -> {dm.index.max().date()}")
    print("Preloading FRED + building weekly regime grid (cache su disco)...")
    series = preload_walk_forward_series(start_date=date(1993, 1, 1), end_date=date(2025, 12, 31))
    grid = build_weekly_grid(series, date(1995, 1, 1), date(2025, 12, 31))
    print(f"  weekly grid points: {len(grid)}")

    pool = None
    if args.crisis_window:
        d0, d1 = CRISIS_WINDOWS_D[args.crisis_window]
        pool = [d for d in dm.index if d0 <= d.date() <= d1]

    rng = random.Random(args.seed)
    print(f"\nDAILY backtest — cadence={args.cadence}, {args.n_sims} sims, {args.horizon_days}g")
    print("=" * 80)
    trades = []
    for i in range(args.n_sims):
        if pool:
            d = rng.choice(pool).date()
        else:
            d = date(rng.randint(args.start_min_year, args.start_max_year), rng.randint(1, 12), 1)
        t = run_horserace_trade_daily(d, dm, grid, args.horizon_days, args.cadence)
        if t and t.realized_return is not None:
            trades.append(t)

    if not trades:
        print("Nessun trade valido."); return

    def _m(attr):
        v = [getattr(t, attr) for t in trades if getattr(t, attr) is not None]
        return float(np.mean(v)) if v else float("nan")

    n = len(trades)
    win60 = sum(1 for t in trades if t.alpha_vs_60_40 and t.alpha_vs_60_40 > 0)
    print(f"Trades: {n}")
    print(f"  Mean return: {_m('realized_return')*100:+.2f}%")
    print(f"  Mean deployed: {_m('avg_deployed')*100:.0f}%")
    print(f"  Win vs 60/40: {win60}/{n} ({win60/n*100:.0f}%) · mean alpha {_m('alpha_vs_60_40')*100:+.2f}pp")
    print(f"  Mean alpha vs S&P500: {_m('alpha_vs_sp500')*100:+.2f}pp")
    print(f"  Mean max drawdown: {_m('max_drawdown')*100:.2f}%")


if __name__ == "__main__":
    main()
