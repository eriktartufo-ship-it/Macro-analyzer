"""Orchestrator backfill: regime + dedollar + asset scores + rolling-window prune.

Garantisce che il DB copra sempre gli ultimi N giorni (default 365) per
alimentare i grafici storici. La pulizia elimina i record più vecchi del
window per tenere il DB snello.
"""

from __future__ import annotations

from datetime import date, timedelta

from loguru import logger
from sqlalchemy.orm import Session

from app.database import engine
from app.models import DailySignal, RegimeClassification
from app.models.news_signals import NewsSignal
from app.models.secular_trends import SecularTrend
from app.services.dedollarization.backfill import backfill_dedollarization_history
from app.services.regime.backfill import backfill_regime_history
from app.services.scoring.backfill import backfill_asset_scores_history


def prune_old_records(days_to_keep: int = 365) -> dict[str, int]:
    """Elimina record più vecchi di `days_to_keep` giorni per tenere il DB snello.

    NOTA: i record con `"historical": true` nel JSON conditions_met (generati da
    `backfill_regime_history_long`) vengono preservati: sono il dataset storico
    1971+ usato per HMM training, transition matrix e backtesting. Senza questa
    eccezione, ogni daily refresh distruggerebbe il dataset di training.
    """
    cutoff = date.today() - timedelta(days=days_to_keep)
    deleted: dict[str, int] = {}
    with Session(engine) as session:
        deleted["regimes"] = (
            session.query(RegimeClassification)
            .filter(RegimeClassification.date < cutoff)
            .filter(~RegimeClassification.conditions_met.like('%"historical": true%'))
            .delete(synchronize_session=False)
        )
        deleted["daily_signals"] = (
            session.query(DailySignal)
            .filter(DailySignal.date < cutoff)
            .delete(synchronize_session=False)
        )
        deleted["secular_trends"] = (
            session.query(SecularTrend)
            .filter(SecularTrend.date < cutoff)
            .delete(synchronize_session=False)
        )
        deleted["news_signals"] = (
            session.query(NewsSignal)
            .filter(NewsSignal.date < cutoff)
            .delete(synchronize_session=False)
        )
        session.commit()

    total = sum(deleted.values())
    if total > 0:
        logger.info(f"Prune rolling window < {cutoff}: eliminati {deleted}")
    return deleted


def backfill_all(days: int = 365) -> dict:
    """Esegue in sequenza: regime → dedollar → asset scores → prune."""
    logger.info(f"=== Backfill completo ({days} giorni) ===")

    regime_stats = backfill_regime_history(days=days)
    dedollar_stats = backfill_dedollarization_history(days=days)
    asset_stats = backfill_asset_scores_history(days=days)
    pruned = prune_old_records(days_to_keep=days)

    return {
        "regime": regime_stats,
        "dedollar": dedollar_stats,
        "asset_scores": asset_stats,
        "pruned": pruned,
        "days": days,
    }


def needs_backfill(min_coverage_days: int = 300) -> bool:
    """True se il DB ha meno di `min_coverage_days` classificazioni regime.

    Usato all'avvio per decidere se lanciare il backfill automatico.
    """
    with Session(engine) as session:
        count = session.query(RegimeClassification).count()
    return count < min_coverage_days
