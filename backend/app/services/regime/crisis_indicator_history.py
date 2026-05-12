"""Backtest storico Crisis Risk Indicator.

Applica `assess_crisis_risk` su tutti i record storici in
`RegimeClassification` per produrre una time series del livello rischio +
tipo crisi. Permette di validare visualmente che:

- Settembre-novembre 2008 → deflation_crash high/extreme
- Marzo-aprile 2020 → deflation_crash extreme (COVID crash)
- 1973-1974 → stagflation_debasement high (oil shock)
- 2022 → stagflation_debasement elevated
- 2000-2001 → bubble_goldilocks → deflation_crash transition (dotcom)

Indicatori pre-2000 sono spesso parziali (no VIX/NFCI/breakeven), e i
default neutri possono attenuare il segnale. Best signal post-2010.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from sqlalchemy.orm import Session

from app.models import RegimeClassification, SecularTrend
from app.services.regime.crisis_indicator import assess_crisis_risk


@dataclass
class CrisisHistoryPoint:
    """Singolo punto della time series crisis history."""
    date: str
    risk_level: str
    risk_score: float
    crisis_type: str
    crisis_type_confidence: float
    historical_analog: str | None
    triggers_fired: int
    triggers_total: int
    summary: str


def _load_dedollar_lookup(db: Session) -> dict[str, float]:
    """Carica tutti i dedollar combined scores indicizzati per data ISO."""
    out: dict[str, float] = {}
    records = (
        db.query(SecularTrend)
        .filter(SecularTrend.trend_name == "dedollarization")
        .all()
    )
    for rec in records:
        try:
            payload = json.loads(rec.components) if rec.components else {}
            score = payload.get("combined_score")
            if score is None and rec.score is not None:
                score = rec.score
            if score is not None:
                out[str(rec.date)] = float(score)
        except (ValueError, TypeError):
            continue
    return out


def _dedollar_for_date(date_iso: str, lookup: dict[str, float]) -> float | None:
    """Best-effort lookup dedollar: stesso giorno o nessuno (no forward fill)."""
    return lookup.get(date_iso)


def compute_crisis_history(
    db: Session,
    days: int = 365 * 5,
    limit: int | None = None,
) -> list[CrisisHistoryPoint]:
    """Calcola crisis assessment per ogni record RegimeClassification.

    Args:
        db: SQLAlchemy session.
        days: lookback window in days (default 5y).
        limit: max records (None = tutti).

    Returns:
        Lista CrisisHistoryPoint ordinata per data ascendente.
    """
    from datetime import date, timedelta

    cutoff = date.today() - timedelta(days=days)

    query = (
        db.query(RegimeClassification)
        .filter(RegimeClassification.date >= cutoff)
        .order_by(RegimeClassification.date.asc())
    )
    if limit:
        query = query.limit(limit)
    records = query.all()

    dedollar_lookup = _load_dedollar_lookup(db)

    out: list[CrisisHistoryPoint] = []
    for rec in records:
        meta = {}
        if rec.conditions_met:
            try:
                meta = json.loads(rec.conditions_met)
            except (ValueError, TypeError):
                meta = {}
        indicators = meta.get("indicators", {}) or {}
        if not indicators:
            # Senza indicators non si può valutare → skip
            continue

        ded = _dedollar_for_date(str(rec.date), dedollar_lookup)

        try:
            assessment = assess_crisis_risk(
                indicators,
                dedollar_combined=ded,
                date=str(rec.date),
            )
        except Exception:
            continue

        fired = sum(1 for t in assessment.triggers if t.get("fired"))
        out.append(CrisisHistoryPoint(
            date=assessment.date,
            risk_level=assessment.risk_level,
            risk_score=assessment.risk_score,
            crisis_type=assessment.crisis_type,
            crisis_type_confidence=assessment.crisis_type_confidence,
            historical_analog=assessment.historical_analog,
            triggers_fired=fired,
            triggers_total=len(assessment.triggers),
            summary=assessment.summary,
        ))

    return out


def aggregate_crisis_stats(points: list[CrisisHistoryPoint]) -> dict:
    """Aggrega statistiche dalla time series:
    - distribuzione livelli rischio
    - distribuzione tipi crisi
    - episodi notevoli (consecutive runs di high/extreme)
    """
    n = len(points)
    if n == 0:
        return {
            "n_points": 0,
            "level_distribution": {},
            "type_distribution": {},
            "notable_episodes": [],
        }

    level_count: dict[str, int] = {}
    type_count: dict[str, int] = {}
    for p in points:
        level_count[p.risk_level] = level_count.get(p.risk_level, 0) + 1
        type_count[p.crisis_type] = type_count.get(p.crisis_type, 0) + 1

    # Notable episodes: run consecutive di high/extreme >= 3 punti consecutivi
    episodes: list[dict] = []
    cur_start: str | None = None
    cur_type: str | None = None
    cur_len = 0
    cur_max_score = 0.0
    for p in points:
        is_elevated = p.risk_level in {"high", "extreme"}
        if is_elevated:
            if cur_start is None:
                cur_start = p.date
                cur_type = p.crisis_type
            cur_len += 1
            cur_max_score = max(cur_max_score, p.risk_score)
        else:
            if cur_start and cur_len >= 3:
                episodes.append({
                    "start": cur_start,
                    "end": p.date,
                    "duration_points": cur_len,
                    "crisis_type": cur_type,
                    "peak_risk_score": cur_max_score,
                })
            cur_start = None
            cur_type = None
            cur_len = 0
            cur_max_score = 0.0
    # Trailing episode
    if cur_start and cur_len >= 3:
        episodes.append({
            "start": cur_start,
            "end": points[-1].date,
            "duration_points": cur_len,
            "crisis_type": cur_type,
            "peak_risk_score": cur_max_score,
        })

    return {
        "n_points": n,
        "date_from": points[0].date,
        "date_to": points[-1].date,
        "level_distribution": {k: {"count": v, "pct": round(v / n, 3)} for k, v in level_count.items()},
        "type_distribution": {k: {"count": v, "pct": round(v / n, 3)} for k, v in type_count.items()},
        "notable_episodes": episodes,
    }


def serialize_points(points: list[CrisisHistoryPoint]) -> list[dict]:
    return [asdict(p) for p in points]
