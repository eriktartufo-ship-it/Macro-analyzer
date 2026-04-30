"""Orchestratore FOMC: fetch + analyze + aggregate per esposizione API."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from loguru import logger

from app.services.fomc.analyzer import FOMCAnalysis, analyze_fomc_document
from app.services.fomc.fetcher import fetch_recent_fomc_documents


@dataclass
class FOMCReport:
    analyses: list[FOMCAnalysis]
    latest_score: float | None
    avg_score_3last: float | None
    trend: str                 # "hawkening" | "dovening" | "stable" | "insufficient"
    n_documents: int


def _classify_trend(scores: list[float]) -> str:
    if len(scores) < 2:
        return "insufficient"
    # ordini crescenti per data ascendente. Confronta primo vs ultimo
    delta = scores[-1] - scores[0]
    if abs(delta) < 0.10:
        return "stable"
    return "hawkening" if delta > 0 else "dovening"


def build_fomc_report(limit: int = 6, force_refresh: bool = False) -> FOMCReport:
    """Pesca gli ultimi N documenti FOMC, li analizza, restituisce un report."""
    docs = fetch_recent_fomc_documents(limit=limit)
    if not docs:
        return FOMCReport(analyses=[], latest_score=None, avg_score_3last=None,
                          trend="insufficient", n_documents=0)

    analyses: list[FOMCAnalysis] = []
    for d in docs:
        a = analyze_fomc_document(d, force_refresh=force_refresh)
        if a is not None:
            analyses.append(a)

    if not analyses:
        return FOMCReport(analyses=[], latest_score=None, avg_score_3last=None,
                          trend="insufficient", n_documents=0)

    # Ordina per data ascendente per il calcolo del trend
    analyses_asc = sorted(analyses, key=lambda a: a.published_date)
    scores = [a.hawkish_dovish_score for a in analyses_asc]
    latest = scores[-1]
    last3 = scores[-3:] if len(scores) >= 3 else scores
    avg3 = sum(last3) / len(last3) if last3 else None
    trend = _classify_trend(scores)

    avg3_str = f"{avg3:+.2f}" if avg3 is not None else "n/a"
    logger.info(
        f"FOMC report: {len(analyses)} docs, latest={latest:+.2f}, avg3={avg3_str}, trend={trend}"
    )

    # Ordina output per data desc (il piu' recente in cima, UX-friendly)
    return FOMCReport(
        analyses=sorted(analyses, key=lambda a: a.published_date, reverse=True),
        latest_score=latest,
        avg_score_3last=avg3,
        trend=trend,
        n_documents=len(analyses),
    )


def serialize_analysis(a: FOMCAnalysis) -> dict:
    """Per JSON API. Converte date/datetime in stringhe ISO."""
    out = asdict(a)
    out["published_date"] = a.published_date.isoformat()
    out["analyzed_at"] = a.analyzed_at.isoformat()
    return out
