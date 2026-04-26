"""Performance metrics per backtest portfolio.

Tutte le metriche si applicano a Series di rendimenti mensili (decimali).
Formule standard da Sharpe (1966), Calmar (1991), max drawdown classico.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PerformanceStats:
    total_return: float          # rendimento cumulato (es. 2.5 = +250%)
    annualized_return: float     # CAGR
    annualized_volatility: float
    sharpe: float                # rf = 0
    max_drawdown: float          # negativo (es. -0.45 = -45%)
    calmar: float                # CAGR / |maxDD|
    win_rate: float              # frazione mesi con return > 0
    n_months: int
    start_date: str
    end_date: str
    final_nav: float             # NAV finale partendo da 1.0


def compute_nav(monthly_returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """Cumula i rendimenti in NAV. Index = end-of-month dates."""
    return (1 + monthly_returns.fillna(0.0)).cumprod() * initial


def compute_drawdown_series(nav: pd.Series) -> pd.Series:
    """Drawdown rolling: (nav - peak) / peak. Sempre <= 0."""
    peak = nav.cummax()
    return (nav - peak) / peak


def compute_stats(monthly_returns: pd.Series) -> PerformanceStats:
    rets = monthly_returns.dropna()
    if len(rets) < 2:
        raise ValueError("metrics: need at least 2 monthly returns")

    n = len(rets)
    nav = compute_nav(rets)
    total = float(nav.iloc[-1] - 1.0)
    years = n / 12.0
    cagr = float((nav.iloc[-1]) ** (1.0 / years) - 1.0) if years > 0 else 0.0
    vol = float(rets.std(ddof=1) * np.sqrt(12))
    sharpe = float(rets.mean() * 12 / vol) if vol > 0 else 0.0
    dd = compute_drawdown_series(nav)
    max_dd = float(dd.min())
    calmar = float(cagr / abs(max_dd)) if max_dd < -1e-9 else float("inf")
    win = float((rets > 0).mean())

    return PerformanceStats(
        total_return=total,
        annualized_return=cagr,
        annualized_volatility=vol,
        sharpe=sharpe,
        max_drawdown=max_dd,
        calmar=calmar,
        win_rate=win,
        n_months=n,
        start_date=str(rets.index.min().date()),
        end_date=str(rets.index.max().date()),
        final_nav=float(nav.iloc[-1]),
    )


def alpha_vs_benchmark(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> dict[str, float]:
    """Alpha annualized + beta + correlation rispetto al benchmark."""
    common = portfolio_returns.index.intersection(benchmark_returns.index)
    p = portfolio_returns.loc[common].dropna()
    b = benchmark_returns.loc[common].dropna()
    common2 = p.index.intersection(b.index)
    p = p.loc[common2]
    b = b.loc[common2]
    if len(p) < 12:
        return {"alpha": 0.0, "beta": 0.0, "correlation": 0.0}
    if b.std() == 0:
        beta = 0.0
    else:
        cov = float(np.cov(p, b, ddof=1)[0, 1])
        beta = cov / float(np.var(b, ddof=1))
    alpha_monthly = float(p.mean() - beta * b.mean())
    return {
        "alpha": float(alpha_monthly * 12),
        "beta": float(beta),
        "correlation": float(np.corrcoef(p, b)[0, 1]),
    }
