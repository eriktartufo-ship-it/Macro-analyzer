"""Backfill storico degli asset scores (DailySignal).

Per ogni giorno legge `RegimeClassification` + `SecularTrend(dedollarization)`
già backfillati e calcola gli score tramite `calculate_final_scores`.
Le news storiche non sono ricostruibili in modo as-of, quindi `news_signal=0.0`
per i record backfillati (la rotazione daily le aggiungerà andando avanti).
"""

from __future__ import annotations

import json
from datetime import date, timedelta

from loguru import logger
from sqlalchemy.orm import Session

from app.database import engine
from app.models import DailySignal, RegimeClassification
from app.models.secular_trends import SecularTrend
from app.services.dedollarization.scorer import calculate_secular_bonus
from app.services.scoring.engine import calculate_final_scores


def backfill_asset_scores_history(days: int = 365) -> dict:
    """Ricalcola DailySignal per ogni giorno in cui esiste un RegimeClassification.

    Dipende dai backfill di regime + dedollar già eseguiti (il secular_bonus
    richiede un SecularTrend nello stesso giorno; se manca, si usa 0.0).
    """
    today = date.today()
    start = today - timedelta(days=days)
    stats = {"written": 0, "skipped": 0, "errors": 0, "start": start, "end": today}

    with Session(engine) as session:
        regimes = (
            session.query(RegimeClassification)
            .filter(RegimeClassification.date >= start)
            .order_by(RegimeClassification.date.asc())
            .all()
        )

        # Pre-carica secular trends in un dict per evitare N query
        secular_map: dict[date, float] = {}
        secular_records = (
            session.query(SecularTrend)
            .filter(
                SecularTrend.trend_name == "dedollarization",
                SecularTrend.date >= start,
            )
            .all()
        )
        for st in secular_records:
            secular_map[st.date] = st.score  # score = combined_score

        logger.info(
            f"Asset scores backfill: {len(regimes)} regimi, "
            f"{len(secular_map)} dedollar snapshots"
        )

        for r in regimes:
            try:
                probs = {
                    "reflation": r.probability_reflation,
                    "stagflation": r.probability_stagflation,
                    "deflation": r.probability_deflation,
                    "goldilocks": r.probability_goldilocks,
                }
                dedollar_score = secular_map.get(r.date, 0.0)
                secular_bonus = calculate_secular_bonus(dedollar_score)
                scores = calculate_final_scores(
                    probs,
                    secular_bonus=secular_bonus,
                    news_signals=None,
                )

                # Upsert: elimina eventuali signal esistenti per quella data e reinserisci
                session.query(DailySignal).filter_by(date=r.date).delete()

                for asset, score in scores.items():
                    bonus = secular_bonus.get(asset, 0.0)
                    session.add(DailySignal(
                        date=r.date,
                        asset_class=asset,
                        final_score=score,
                        regime_component=score - bonus,
                        secular_trend_bonus=bonus,
                        news_signal=0.0,
                        momentum_penalty=0.0,
                        breakdown=json.dumps({
                            "regime": r.regime,
                            "probabilities": probs,
                            "backfilled": True,
                        }),
                    ))
                stats["written"] += 1

                if stats["written"] % 30 == 0:
                    session.commit()
            except Exception as e:
                logger.warning(f"Asset scores backfill errore per {r.date}: {e}")
                stats["errors"] += 1

        session.commit()

    logger.info(
        f"Asset scores backfill completato: written={stats['written']} "
        f"errors={stats['errors']}"
    )
    return stats
