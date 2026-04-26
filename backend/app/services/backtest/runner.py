"""Orchestratore backtest: combina strategie + asset returns + metriche.

Funzione `run_full_backtest` esegue strategy macro + benchmark e ritorna serie
NAV + statistiche performance per il confronto.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd
from loguru import logger
from sqlalchemy.orm import Session

from app.services.backtest.metrics import (
    PerformanceStats,
    alpha_vs_benchmark,
    compute_nav,
    compute_stats,
)
from app.services.backtest.portfolio import fetch_asset_returns, run_backtest
from app.services.backtest.strategies import (
    buy_and_hold_strategy,
    regime_probs_monthly,
    score_weighted_strategy,
    sixty_forty_strategy,
    spy_only_strategy,
)


@dataclass
class StrategyResult:
    name: str
    description: str
    monthly_returns: pd.Series
    nav: pd.Series
    stats: PerformanceStats
    alpha_vs_60_40: dict[str, float] = field(default_factory=dict)


@dataclass
class FullBacktestResult:
    strategies: list[StrategyResult]
    common_start: str
    common_end: str
    n_months: int


def run_full_backtest(
    db: Session,
    start: date | None = None,
    end: date | None = None,
    top_n: int = 5,
    score_threshold: float = 30.0,
    cost_bps: float = 10.0,
    force_include_dedollar: bool | None = None,
) -> FullBacktestResult:
    """Esegue strategy macro-driven + 3 benchmark, allinea date comuni."""
    rp = regime_probs_monthly(db)
    if rp.empty:
        raise ValueError("Backtest: nessuna classification in DB. Esegui backfill prima.")

    # Asset universe = tutti gli asset class (filtrati su disponibilita Yahoo)
    from app.services.scoring.engine import ASSET_CLASSES

    asset_returns_df = fetch_asset_returns(list(ASSET_CLASSES), start=start, end=end)
    asset_returns_df = asset_returns_df.dropna(how="all")
    if asset_returns_df.empty:
        raise ValueError("Backtest: nessun rendimento asset disponibile.")

    # Determina overlap comune tra regime probs e asset returns
    common_index = rp.index.intersection(asset_returns_df.index)
    if start:
        common_index = common_index[common_index >= pd.Timestamp(start)]
    if end:
        common_index = common_index[common_index <= pd.Timestamp(end)]
    if len(common_index) < 24:
        raise ValueError(f"Backtest: overlap troppo corto ({len(common_index)} mesi)")

    rp_aligned = rp.loc[common_index]
    asset_returns_aligned = asset_returns_df.loc[common_index]

    # Strategy: macro score-weighted (shift di 1 mese per evitare lookahead)
    macro_weights_full = score_weighted_strategy(
        db, top_n=top_n, score_threshold=score_threshold,
        asset_classes=list(asset_returns_aligned.columns),
        force_include_dedollar=force_include_dedollar,
    )
    macro_weights = macro_weights_full.reindex(common_index).shift(1).dropna()

    # Riallinea returns sulla stessa finestra
    final_index = macro_weights.index
    asset_returns_final = asset_returns_aligned.loc[final_index]

    macro_run = run_backtest(macro_weights, asset_returns_final, cost_bps=cost_bps)
    macro_stats = compute_stats(macro_run.monthly_returns)

    # Benchmark: 60/40
    sf_weights = sixty_forty_strategy(final_index)
    sf_run = run_backtest(sf_weights, asset_returns_final, cost_bps=cost_bps)
    sf_stats = compute_stats(sf_run.monthly_returns)

    # Benchmark: SPY only
    spy_weights = spy_only_strategy(final_index)
    spy_run = run_backtest(spy_weights, asset_returns_final, cost_bps=cost_bps)
    spy_stats = compute_stats(spy_run.monthly_returns)

    # Benchmark: equal-weight tutti gli asset disponibili
    bh_weights = buy_and_hold_strategy(list(asset_returns_final.columns), final_index)
    bh_run = run_backtest(bh_weights, asset_returns_final, cost_bps=cost_bps)
    bh_stats = compute_stats(bh_run.monthly_returns)

    # Alpha vs 60/40 per le strategy non-benchmark
    macro_alpha = alpha_vs_benchmark(macro_run.monthly_returns, sf_run.monthly_returns)
    spy_alpha = alpha_vs_benchmark(spy_run.monthly_returns, sf_run.monthly_returns)
    bh_alpha = alpha_vs_benchmark(bh_run.monthly_returns, sf_run.monthly_returns)

    strategies = [
        StrategyResult(
            name="macro_score_weighted",
            description=f"Score-weighted top-{top_n} (threshold {score_threshold}, cost {cost_bps}bp)",
            monthly_returns=macro_run.monthly_returns,
            nav=compute_nav(macro_run.monthly_returns),
            stats=macro_stats,
            alpha_vs_60_40=macro_alpha,
        ),
        StrategyResult(
            name="60_40",
            description="60% growth equity + 40% long bonds, monthly rebalance",
            monthly_returns=sf_run.monthly_returns,
            nav=compute_nav(sf_run.monthly_returns),
            stats=sf_stats,
            alpha_vs_60_40={"alpha": 0.0, "beta": 1.0, "correlation": 1.0},
        ),
        StrategyResult(
            name="spy_buyhold",
            description="100% growth equity (SPY proxy)",
            monthly_returns=spy_run.monthly_returns,
            nav=compute_nav(spy_run.monthly_returns),
            stats=spy_stats,
            alpha_vs_60_40=spy_alpha,
        ),
        StrategyResult(
            name="equal_weight",
            description="Equal weight tutti gli asset disponibili",
            monthly_returns=bh_run.monthly_returns,
            nav=compute_nav(bh_run.monthly_returns),
            stats=bh_stats,
            alpha_vs_60_40=bh_alpha,
        ),
    ]

    logger.info(
        f"Backtest completato: {len(final_index)} mesi, "
        f"macro CAGR={macro_stats.annualized_return*100:.1f}% "
        f"vs 60/40 CAGR={sf_stats.annualized_return*100:.1f}%"
    )

    return FullBacktestResult(
        strategies=strategies,
        common_start=str(final_index.min().date()),
        common_end=str(final_index.max().date()),
        n_months=len(final_index),
    )
