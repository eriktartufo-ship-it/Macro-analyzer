"""Deploy-boost combo per le strategie AI Portfolio — gated OFF di default.

Combo validata via `scripts/horserace_ab_matrix.py` (2026-07-13): tre leve che
"schierano meglio il capitale GIÀ approvato" (NON abbassano i gate d'ingresso),
quindi migliorano sia i mercati normali sia le crisi nei backtest:
  1. deadband  — add-tranche alla soglia di ENTRY (55) invece di INCREMENTAL (60):
     una posizione aperta a score 55-59 non resta più bloccata a 1/N.
  2. fastramp  — cap a 2 tranche (invece di 2-5): il ramp lento affamava il deploy.
  3. floor     — deployment minimo (30%): se sotto, schiera il deficit sui migliori
     per score già eleggibili (regime-favorable), cap 25% per asset.

Robustezza multi-seed (5 seed × 4 finestre, `scripts/deployboost_robustness.py`,
2026-07-13, post-audit raydalio): Δ vs 60/40 (combo − baseline) robusto e positivo,
anche cost-adjusted (10bps): broad +1.8±0.5pp, 2008 +0.6±0.3, 2020 +8.0±0.5,
all_crisis +2.0±0.3.

**2026-07-13 PROMOSSO DEFAULT-ON** (`config_flags.use_ai_portfolio_deploy_boost`).
⚠️ COSTO REALE: il max-DD ~RADDOPPIA (es. 2020 −3.6→−6.5%). ⚠️ `apply_deployment_floor`
NON applica il momentum gate → può schierare su asset regime-favorevoli ma BEARISH
(rischio noto, da quantificare). I livelli ASSOLUTI di alpha sono seed-sensibili; è
il Δ a essere stabile. Vedi [[feedback_backtest_before_touching_strategy]].
"""
from __future__ import annotations

from datetime import date

from loguru import logger
from sqlalchemy.orm import Session

from app.models.ai_portfolio import AiPortfolioPosition
from app.services.ai_portfolio import pnl_tracker
from app.services.config_flags import use_ai_portfolio_deploy_boost

RAMP_CAP = 2          # fastramp: max tranche
DEPLOY_FLOOR = 0.30   # floor: deployment minimo
FLOOR_SPREAD_K = 5    # su quanti asset spalmare il deficit
WEIGHT_CAP = 0.25     # cap per asset (coerente con sizer.WEIGHT_CAP_MAX)


def enabled() -> bool:
    return use_ai_portfolio_deploy_boost()


def cap_n_tranches(n: int) -> int:
    """fastramp: quando boost ON limita le tranche a RAMP_CAP (0 resta 0)."""
    if not enabled() or n <= 0:
        return n
    return min(RAMP_CAP, n)


def incremental_threshold(default_incr: float, entry: float) -> float:
    """deadband: quando boost ON, add-tranche alla soglia di ENTRY (non 60)."""
    return entry if enabled() else default_incr


def apply_deployment_floor(
    db: Session,
    strategy_type: str,
    scores: dict[str, float],
    eligible: set[str] | list[str],
    target_date: date,
    regime: str,
    floor: float = DEPLOY_FLOOR,
) -> int:
    """floor: se il deployment è sotto `floor`, schiera il deficit sui migliori
    per score tra gli `eligible`, cap WEIGHT_CAP per asset. No-op se boost OFF.

    Mirror live della logica `_apply_floor` testata in test_horserace_backtest.py.
    Ritorna il numero di asset toccati.
    """
    if not enabled():
        return 0
    positions = {
        p.asset_class: p
        for p in db.query(AiPortfolioPosition)
        .filter(AiPortfolioPosition.strategy_type == strategy_type)
        .all()
    }
    deployed = sum(p.current_weight_pct for p in positions.values())
    deficit = floor - deployed
    if deficit <= 0.001:
        return 0
    cands = sorted(
        [(a, scores.get(a, 0.0)) for a in eligible if scores.get(a, 0.0) > 0],
        key=lambda x: -x[1],
    )[:FLOOR_SPREAD_K]
    if not cands:
        return 0
    per = deficit / len(cands)
    n_added = 0
    for asset, sc in cands:
        pos = positions.get(asset)
        room = WEIGHT_CAP - (pos.current_weight_pct if pos else 0.0)
        add = min(per, max(0.0, room))
        if add <= 0.001:
            continue
        price = pnl_tracker.get_latest_price(asset)
        if pos is None:
            db.add(AiPortfolioPosition(
                asset_class=asset, strategy_type=strategy_type,
                target_weight_pct=add, current_weight_pct=add,
                tranches_filled=1, tranches_total=1, opened_at=target_date,
                entry_regime=regime, avg_entry_score=sc,
                avg_entry_price=price, current_price=price,
                estimated_pnl_pct=0.0, took_partial_tp=False,
                last_action="FLOOR_DEPLOY", last_action_date=target_date,
            ))
        else:
            pos.current_weight_pct += add
            pos.last_action = "FLOOR_DEPLOY"
            pos.last_action_date = target_date
        n_added += 1
    if n_added:
        logger.info(f"deploy_boost floor[{strategy_type}]: +{n_added} asset, deficit {deficit:.1%}")
    return n_added
