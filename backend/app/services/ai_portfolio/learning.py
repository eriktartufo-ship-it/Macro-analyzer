"""Post-mortem learning — il modello impara dagli errori.

Pattern: ogni volta che si chiude una posizione (FULL_CLOSE, STOP_LOSS,
TAKE_PROFIT), creiamo/aggiorniamo un record `AiPortfolioLearning` con
chiave = pattern (regime + asset_type + momentum_at_entry).

Stats: n_wins, n_losses, total_pnl, avg_pnl, win_rate, threshold_shift.

Strategist usa `apply_learnings_threshold_shift(...)` per adattare
dinamicamente la entry_threshold per quella combinazione di pattern.

Esempi pattern:
- "regime=stagflation,asset_type=factor,momentum=BULLISH"
- "regime=deflation,asset_type=sector,momentum=NEUTRAL"
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from sqlalchemy.orm import Session

from app.models.ai_portfolio import AiPortfolioDecision, AiPortfolioLearning, AiPortfolioPosition

logger = logging.getLogger(__name__)


# Quanto spingere l'entry threshold quando un pattern perde N volte
# Formula: shift = (1 - win_rate) × magnitude × min(n_total, max_n) / max_n
MAX_THRESHOLD_SHIFT = 8.0   # max +8 pt sul score threshold (55 → 63)
MIN_OBSERVATIONS = 3        # serve almeno 3 osservazioni per applicare shift
LEARNING_DECAY_DAYS = 180   # learnings più vecchi vengono down-weighted


def _asset_type(asset_class: str) -> str:
    """Categoria asset per pattern key."""
    if asset_class.startswith("sector_"):
        return "sector"
    if asset_class.startswith("factor_"):
        return "factor"
    if asset_class in ("bitcoin", "crypto_broad"):
        return "crypto"
    if asset_class in ("gold", "silver", "broad_commodities", "energy"):
        return "commodity"
    if "bonds" in asset_class or asset_class == "tips_inflation_bonds":
        return "bond"
    if "equities" in asset_class:
        return "equity"
    if asset_class == "real_estate_reits":
        return "reit"
    if asset_class == "cash_money_market":
        return "cash"
    return "other"


def build_pattern_key(regime: str, asset_class: str, momentum: str) -> str:
    """Costruisce la chiave del pattern."""
    a_type = _asset_type(asset_class)
    return f"regime={regime},asset_type={a_type},momentum={momentum}"


def _compute_threshold_shift(win_rate: float, n_total: int) -> float:
    """Threshold shift adattivo (più aggressivo se win_rate basso)."""
    if n_total < MIN_OBSERVATIONS:
        return 0.0
    # Bias verso più conservativo se win_rate < 50%
    if win_rate >= 0.5:
        # Riduce threshold se sta vincendo (entra più facilmente)
        return -min(2.0, (win_rate - 0.5) * 6)
    # Aumenta threshold proporzionalmente alla loss rate
    loss_rate = 1.0 - win_rate
    confidence_factor = min(n_total, 10) / 10.0  # da 0.3 a 1.0
    return min(MAX_THRESHOLD_SHIFT, loss_rate * MAX_THRESHOLD_SHIFT * confidence_factor)


def record_closed_position(
    db: Session,
    closed_decision: AiPortfolioDecision,
    closed_position: AiPortfolioPosition,
) -> AiPortfolioLearning:
    """Aggiorna learning quando una posizione viene chiusa (per strategy_type)."""
    pattern = build_pattern_key(
        regime=closed_position.entry_regime,
        asset_class=closed_position.asset_class,
        momentum=closed_decision.momentum_signal or "NEUTRAL",
    )
    strategy = getattr(closed_position, "strategy_type", "model_driven")

    pnl = closed_position.estimated_pnl_pct

    learning = db.query(AiPortfolioLearning).filter_by(
        pattern_key=pattern, strategy_type=strategy,
    ).first()
    if learning is None:
        learning = AiPortfolioLearning(
            pattern_key=pattern,
            strategy_type=strategy,
            n_wins=1 if pnl > 0 else 0,
            n_losses=0 if pnl > 0 else 1,
            total_pnl_pct=pnl,
            avg_pnl_pct=pnl,
            win_rate=1.0 if pnl > 0 else 0.0,
            entry_threshold_shift=0.0,  # ricalcolato sotto
            last_pattern_seen_at=closed_decision.date,
        )
        db.add(learning)
    else:
        if pnl > 0:
            learning.n_wins += 1
        else:
            learning.n_losses += 1
        n_total = learning.n_wins + learning.n_losses
        learning.total_pnl_pct += pnl
        learning.avg_pnl_pct = learning.total_pnl_pct / n_total
        learning.win_rate = learning.n_wins / n_total
        learning.last_pattern_seen_at = closed_decision.date

    n_total = learning.n_wins + learning.n_losses
    learning.entry_threshold_shift = _compute_threshold_shift(
        learning.win_rate, n_total,
    )

    logger.info(
        f"learning updated: {pattern} → wins={learning.n_wins} losses={learning.n_losses} "
        f"win_rate={learning.win_rate:.2f} avg_pnl={learning.avg_pnl_pct*100:.1f}% "
        f"shift={learning.entry_threshold_shift:+.2f}"
    )
    return learning


def get_threshold_shift_for_pattern(
    db: Session,
    regime: str,
    asset_class: str,
    momentum: str,
    strategy_type: str = "model_driven",
) -> float:
    """Restituisce lo shift threshold corrente per il pattern (0 se non esiste)."""
    pattern = build_pattern_key(regime, asset_class, momentum)
    learning = db.query(AiPortfolioLearning).filter_by(
        pattern_key=pattern, strategy_type=strategy_type,
    ).first()
    if not learning:
        return 0.0

    # Decay: learning vecchio (>180 giorni) → mezzo peso
    days_old = (date.today() - learning.last_pattern_seen_at).days
    if days_old > LEARNING_DECAY_DAYS:
        return learning.entry_threshold_shift * 0.5
    return learning.entry_threshold_shift


def list_top_learnings(
    db: Session, limit: int = 20, strategy_type: str | None = None,
) -> list[dict]:
    """Restituisce i top pattern per impatto. Filtra per strategy se passato."""
    q = db.query(AiPortfolioLearning)
    if strategy_type:
        q = q.filter(AiPortfolioLearning.strategy_type == strategy_type)
    rows = q.order_by(AiPortfolioLearning.entry_threshold_shift.desc()).limit(limit).all()
    return [
        {
            "pattern_key": r.pattern_key,
            "strategy_type": r.strategy_type,
            "n_wins": r.n_wins,
            "n_losses": r.n_losses,
            "win_rate": r.win_rate,
            "avg_pnl_pct": r.avg_pnl_pct,
            "entry_threshold_shift": r.entry_threshold_shift,
            "last_seen": str(r.last_pattern_seen_at),
            "insight_text": r.insight_text,
        }
        for r in rows
    ]
