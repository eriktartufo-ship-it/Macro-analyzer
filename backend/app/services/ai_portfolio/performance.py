"""AI Portfolio Performance Tracker.

Computa NAV portafoglio daily + cumulative return + DD + benchmarks 60/40
e S&P500. Stima coarse: usa ASSET_REGIME_DATA monthly returns × prob regime.

Starting NAV = $100,000. Tutte le metriche % vs starting.
"""
from __future__ import annotations

import logging
from datetime import date

from sqlalchemy.orm import Session

from app.models.ai_portfolio import (
    AiPortfolioPerformance,
    AiPortfolioPosition,
)
from app.models.daily_signals import DailySignal
from app.models.regime_classifications import RegimeClassification
from app.services.scoring.engine import ASSET_REGIME_DATA

logger = logging.getLogger(__name__)

STARTING_NAV = 100_000.0


def _estimate_daily_return(
    weights: dict[str, float],
    probabilities: dict[str, float],
) -> float:
    """Daily return stimato = weighted avg monthly_return / 21 (trading days).

    Uses ASSET_REGIME_DATA avg_return per regime, blended by regime probs.
    """
    total = 0.0
    for asset, w in weights.items():
        if w <= 0:
            continue
        asset_data = ASSET_REGIME_DATA.get(asset)
        if not asset_data:
            continue
        expected_annual = 0.0
        for regime, prob in probabilities.items():
            stats = asset_data.get(regime, {})
            expected_annual += prob * stats.get("avg_return", 0.0)
        # Daily approximation: annual / 252
        total += w * expected_annual / 252.0
    return total


def _benchmark_60_40_daily_return(probabilities: dict[str, float]) -> float:
    """60/40 = 60% us_equities_growth + 40% us_bonds_long."""
    weights = {"us_equities_growth": 0.60, "us_bonds_long": 0.40}
    return _estimate_daily_return(weights, probabilities)


def _benchmark_sp500_daily_return(probabilities: dict[str, float]) -> float:
    """S&P500 proxy = us_equities_growth 100%."""
    weights = {"us_equities_growth": 1.0}
    return _estimate_daily_return(weights, probabilities)


def snapshot(
    db: Session,
    target_date: date | None = None,
    strategy_type: str = "model_driven",
) -> AiPortfolioPerformance:
    """Compute + persist daily snapshot per strategia.

    Capital pool indipendente per strategy: ognuna parte da STARTING_NAV.
    """
    target_date = target_date or date.today()

    latest_regime = (
        db.query(RegimeClassification)
        .order_by(RegimeClassification.date.desc())
        .first()
    )
    if latest_regime is None:
        logger.warning("performance.snapshot: no regime")
        return None

    probabilities = {
        "reflation": float(latest_regime.probability_reflation or 0),
        "stagflation": float(latest_regime.probability_stagflation or 0),
        "deflation": float(latest_regime.probability_deflation or 0),
        "goldilocks": float(latest_regime.probability_goldilocks or 0),
    }

    # Positions filtered per strategy
    positions = (
        db.query(AiPortfolioPosition)
        .filter(AiPortfolioPosition.strategy_type == strategy_type)
        .all()
    )
    weights = {p.asset_class: p.current_weight_pct for p in positions}
    n_positions = len(positions)
    deployed = sum(weights.values())

    # Previous snapshot per same strategy
    prev = (
        db.query(AiPortfolioPerformance)
        .filter(AiPortfolioPerformance.strategy_type == strategy_type)
        .order_by(AiPortfolioPerformance.date.desc())
        .first()
    )
    prev_nav = prev.nav if prev else STARTING_NAV
    prev_peak = prev.peak_nav if prev else STARTING_NAV
    prev_b60 = prev.benchmark_60_40_return_pct if prev else 0.0
    prev_sp = prev.benchmark_sp500_return_pct if prev else 0.0

    # Daily returns
    portfolio_dr = _estimate_daily_return(weights, probabilities)
    b60_dr = _benchmark_60_40_daily_return(probabilities)
    sp_dr = _benchmark_sp500_daily_return(probabilities)

    new_nav = prev_nav * (1 + portfolio_dr)
    new_peak = max(prev_peak, new_nav)
    cum_ret = (new_nav / STARTING_NAV) - 1.0
    dd = (new_nav / new_peak) - 1.0  # negative or 0
    b60_cum = (1 + (prev_b60 or 0.0)) * (1 + b60_dr) - 1
    sp_cum = (1 + (prev_sp or 0.0)) * (1 + sp_dr) - 1

    # Top position
    top_asset = None
    top_w = None
    if positions:
        top = max(positions, key=lambda p: p.current_weight_pct)
        top_asset = top.asset_class
        top_w = top.current_weight_pct

    # Upsert snapshot (replace if exists for (date, strategy))
    existing = (
        db.query(AiPortfolioPerformance)
        .filter_by(date=target_date, strategy_type=strategy_type)
        .first()
    )
    if existing:
        existing.nav = new_nav
        existing.cumulative_return_pct = cum_ret
        existing.drawdown_pct = dd
        existing.peak_nav = new_peak
        existing.n_positions = n_positions
        existing.deployed_pct = deployed
        existing.top_position_asset = top_asset
        existing.top_position_weight_pct = top_w
        existing.benchmark_60_40_return_pct = b60_cum
        existing.benchmark_sp500_return_pct = sp_cum
        snap = existing
    else:
        snap = AiPortfolioPerformance(
            date=target_date,
            strategy_type=strategy_type,
            nav=new_nav,
            cumulative_return_pct=cum_ret,
            drawdown_pct=dd,
            peak_nav=new_peak,
            n_positions=n_positions,
            deployed_pct=deployed,
            top_position_asset=top_asset,
            top_position_weight_pct=top_w,
            benchmark_60_40_return_pct=b60_cum,
            benchmark_sp500_return_pct=sp_cum,
        )
        db.add(snap)
    db.commit()
    db.refresh(snap)
    return snap
