"""Backfill storico delle classificazioni regime.

Per ogni giorno nell'intervallo specificato, costruisce lo snapshot di
indicatori macro disponibili a quella data (tronca ciascuna serie FRED a
`as_of`) e chiama `classify_regime`. Il risultato viene upsertato su
`RegimeClassification`.

Ottimizzazione: le serie FRED necessarie vengono scaricate una sola volta;
l'iterazione per data è in memoria.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Optional

import pandas as pd
from loguru import logger
from sqlalchemy.orm import Session

from app.database import engine
from app.models import RegimeClassification
from app.services.indicators.fetcher import FredFetcher
from app.services.regime.classifier import classify_regime


# Serie FRED richieste dal classifier (core + nuovi indicatori)
_CLASSIFIER_SERIES = (
    "real_gdp",
    "cpi",
    "core_pce",
    "unrate",
    "yield_curve_10y2y",
    "yield_curve_10y3m",
    "initial_claims",
    "lei",
    "fed_funds",
    "ism_manufacturing",
    "nonfarm_payrolls",
    "industrial_production",
    "baa_spread",
    "consumer_sentiment",
)


def _build_indicators_as_of(
    series: dict[str, pd.Series],
    as_of: date,
) -> dict[str, float]:
    """Costruisce il dict indicatori usando solo osservazioni fino a `as_of`."""
    cutoff = pd.Timestamp(as_of)
    indicators: dict[str, float] = {}

    def last_before(name: str) -> Optional[float]:
        s = series.get(name)
        if s is None or s.empty:
            return None
        filtered = s[s.index <= cutoff]
        if filtered.empty:
            return None
        return float(filtered.iloc[-1])

    def roc(name: str, periods: int) -> Optional[float]:
        s = series.get(name)
        if s is None or s.empty:
            return None
        filtered = s[s.index <= cutoff]
        if len(filtered) <= periods:
            return None
        try:
            prev = float(filtered.iloc[-1 - periods])
            curr = float(filtered.iloc[-1])
            if prev == 0:
                return None
            return (curr / prev - 1) * 100
        except Exception:
            return None

    # GDP ROC (quarterly: 1 trimestre = 3 mesi)
    v = roc("real_gdp", 1)
    if v is not None:
        indicators["gdp_roc"] = v

    # PMI (monthly level)
    v = last_before("ism_manufacturing")
    if v is not None:
        indicators["pmi"] = v

    # CPI YoY (monthly: 12 osservazioni)
    v = roc("cpi", 12)
    if v is not None:
        indicators["cpi_yoy"] = v

    # Unemployment (monthly level + 3m ROC)
    v = last_before("unrate")
    if v is not None:
        indicators["unrate"] = v
    v = roc("unrate", 3)
    if v is not None:
        indicators["unrate_roc"] = v

    # Yield curves (daily level)
    v = last_before("yield_curve_10y2y")
    if v is not None:
        indicators["yield_curve_10y2y"] = v
    v = last_before("yield_curve_10y3m")
    if v is not None:
        indicators["yield_curve_10y3m"] = v

    # Initial claims (weekly: ~13 periodi = 3 mesi)
    v = roc("initial_claims", 13)
    if v is not None:
        indicators["initial_claims_roc"] = v

    # LEI (monthly, 6 periodi)
    v = roc("lei", 6)
    if v is not None:
        indicators["lei_roc"] = v

    # Fed Funds
    v = last_before("fed_funds")
    if v is not None:
        indicators["fed_funds_rate"] = v

    # --- Nuovi indicatori ---
    v = roc("core_pce", 12)
    if v is not None:
        indicators["core_pce_yoy"] = v

    v = roc("nonfarm_payrolls", 12)
    if v is not None:
        indicators["payrolls_roc_12m"] = v

    v = roc("industrial_production", 12)
    if v is not None:
        indicators["indpro_roc_12m"] = v

    v = last_before("baa_spread")
    if v is not None:
        indicators["baa_spread"] = v

    v = last_before("consumer_sentiment")
    if v is not None:
        indicators["consumer_sentiment"] = v

    return indicators


def backfill_regime_history(days: int = 365) -> dict:
    """Calcola e upserta classificazioni regime per gli ultimi `days` giorni.

    Le serie FRED necessarie vengono fetchate una sola volta; poi per ogni
    data viene costruito un snapshot indicatori troncato a quella data e
    classificato. I record esistenti per quella data vengono sostituiti.

    Returns:
        Stats: {"classified": n, "skipped": n, "errors": n, "start": date, "end": date}
    """
    fetcher = FredFetcher()
    series: dict[str, pd.Series] = {}

    logger.info(f"Backfill: pre-fetch {len(_CLASSIFIER_SERIES)} serie FRED")
    for name in _CLASSIFIER_SERIES:
        try:
            series[name] = fetcher.fetch_series(name)
        except Exception as e:
            logger.warning(f"Backfill: impossibile fetchare {name}: {e}")

    today = date.today()
    start = today - timedelta(days=days)
    stats = {"classified": 0, "skipped": 0, "errors": 0, "start": start, "end": today}

    logger.info(f"Backfill: classifico {days + 1} giorni da {start} a {today}")

    with Session(engine) as session:
        for offset in range(days, -1, -1):
            d = today - timedelta(days=offset)
            try:
                indicators = _build_indicators_as_of(series, d)
                # Se mancano i pilastri minimi (gdp/cpi/unrate) saltiamo
                if not {"gdp_roc", "cpi_yoy", "unrate"}.issubset(indicators):
                    stats["skipped"] += 1
                    continue

                result = classify_regime(indicators)

                session.query(RegimeClassification).filter_by(date=d).delete()
                session.add(RegimeClassification(
                    date=d,
                    regime=result["regime"],
                    probability_reflation=result["probabilities"]["reflation"],
                    probability_stagflation=result["probabilities"]["stagflation"],
                    probability_deflation=result["probabilities"]["deflation"],
                    probability_goldilocks=result["probabilities"]["goldilocks"],
                    confidence=result["confidence"],
                    conditions_met=json.dumps({
                        "conditions": result["conditions_detail"],
                        "indicators": indicators,
                        "dedollar_indicators": {},
                        "trajectory": {},
                        "news_sentiment": 0.0,
                        "fit_scores": result.get("fit_scores", {}),
                        "backfilled": True,
                    }),
                ))
                stats["classified"] += 1
            except Exception as e:
                logger.warning(f"Backfill errore per {d}: {e}")
                stats["errors"] += 1

        session.commit()

    logger.info(
        f"Backfill completato: classified={stats['classified']} "
        f"skipped={stats['skipped']} errors={stats['errors']}"
    )
    return stats
