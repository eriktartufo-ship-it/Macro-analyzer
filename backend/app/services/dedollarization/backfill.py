"""Backfill storico della dedollarizzazione.

Approccio: pre-fetch di tutte le serie FRED necessarie, poi per ogni giorno
tronco ogni serie a `as_of` e riuso `_prepare_dedollarization_indicators` +
`_compute_player_history` esistenti (via AsOfFetcher che finge di essere un
FredFetcher ma restituisce sempre serie tagliate a `as_of`).
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Optional

import pandas as pd
from loguru import logger
from sqlalchemy.orm import Session

from app.database import engine
from app.models.secular_trends import SecularTrend
from app.services.dedollarization.scorer import calculate_dedollarization
from app.services.indicators.fetcher import FredFetcher
from app.services.indicators.fred_codes import FRED_SERIES
from app.services.indicators.transforms import calculate_roc, calculate_zscore


# Serie FRED usate dalle pipeline dedollar + player signals + regime macro
# (regime serve per calcolare real_rate = fed_funds - cpi_yoy)
_DEDOLLAR_SERIES = (
    # Dedollar base
    "dxy_broad", "gold_price", "oil_price", "debt_gdp", "m2",
    "real_broad_dxy", "silver_price",
    # Player: SYSTEM
    "sp500", "copper_price",
    # USA hegemony
    "real_yield_10y", "interest_outlays", "tax_receipts", "gdp",
    "foreign_treasury_holdings", "yield_curve_10y2y",
    # Europe
    "italy_10y", "germany_10y", "france_10y", "chf_per_usd", "usd_per_eur",
    # Japan
    "japan_10y", "jpy_per_usd",
    # Commodity FX
    "cad_per_usd", "usd_per_aud",
    # EM
    "em_hy_oas", "em_fx_dollar_index",
    # Defense
    "defense_spending",
    # USA twin deficit / monetizzazione
    "current_account", "niip", "fed_debt_holdings",
    # Expectations
    "breakeven_5y5y", "term_premium_10y",
    # Fed liquidity
    "fed_balance_sheet", "reverse_repo",
    # BRICS+
    "cny_per_usd", "india_10y", "brazil_policy_rate",
    # Europe extra
    "ecb_balance_sheet",
    # Per real_rate
    "fed_funds", "cpi",
)


class AsOfFetcher:
    """Wrapper FredFetcher che restituisce ogni serie troncata a `as_of`.

    Riproduce l'API di FredFetcher usata da jobs._prepare_dedollarization_indicators,
    jobs._prepare_player_signals e jobs._compute_player_history, così quelle
    funzioni possono essere riusate per il backfill senza duplicazione.
    """

    def __init__(self, raw_series: dict[str, pd.Series], as_of: date):
        self.raw_series = raw_series
        self.as_of = pd.Timestamp(as_of)

    def _truncated(self, name: str) -> Optional[pd.Series]:
        s = self.raw_series.get(name)
        if s is None or s.empty:
            return None
        return s[s.index <= self.as_of]

    def fetch_series(
        self,
        series_name: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> Optional[pd.Series]:
        s = self._truncated(series_name)
        if s is None or s.empty:
            return None
        return s

    def fetch_and_transform(
        self,
        series_name: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> dict[str, pd.Series]:
        raw = self._truncated(series_name)
        if raw is None or raw.empty:
            return {"raw": pd.Series(dtype=float)}

        config = FRED_SERIES.get(series_name, {})
        freq = config.get("frequency", "monthly")
        result: dict[str, pd.Series] = {"raw": raw}

        if freq == "monthly":
            result["roc_3m"] = calculate_roc(raw, periods=3)
            result["roc_6m"] = calculate_roc(raw, periods=6)
            result["roc_12m"] = calculate_roc(raw, periods=12)
            result["zscore_12m"] = calculate_zscore(raw, window=12)
            result["zscore_36m"] = calculate_zscore(raw, window=36)
        elif freq == "quarterly":
            result["roc_3m"] = calculate_roc(raw, periods=1)
            result["roc_6m"] = calculate_roc(raw, periods=2)
            result["roc_12m"] = calculate_roc(raw, periods=4)
            result["zscore_12m"] = calculate_zscore(raw, window=4)
            result["zscore_36m"] = calculate_zscore(raw, window=12)
        elif freq == "annual":
            result["roc_3m"] = calculate_roc(raw, periods=1)
            result["roc_6m"] = calculate_roc(raw, periods=1)
            result["roc_12m"] = calculate_roc(raw, periods=1)
            result["zscore_12m"] = calculate_zscore(raw, window=3)
            result["zscore_36m"] = calculate_zscore(raw, window=5)
        else:  # daily, weekly
            result["roc_3m"] = calculate_roc(raw, periods=63)
            result["roc_6m"] = calculate_roc(raw, periods=126)
            result["roc_12m"] = calculate_roc(raw, periods=252)
            result["zscore_12m"] = calculate_zscore(raw, window=252)
            result["zscore_36m"] = calculate_zscore(raw, window=756)

        return result


def _latest_as_of(raw_series: dict[str, pd.Series], as_of: pd.Timestamp) -> dict[str, float]:
    """Equivalente a fetch_all_latest() ma tagliato a `as_of`."""
    latest: dict[str, float] = {}
    for name, s in raw_series.items():
        if s is None or s.empty:
            continue
        truncated = s[s.index <= as_of]
        if truncated.empty:
            continue
        try:
            latest[name] = float(truncated.iloc[-1])
        except Exception:
            continue
    return latest


def _macro_indicators_as_of(
    raw_series: dict[str, pd.Series],
    as_of: pd.Timestamp,
) -> dict[str, float]:
    """Calcola fed_funds_rate e cpi_yoy as-of per supportare real_rate."""
    out: dict[str, float] = {}
    ff = raw_series.get("fed_funds")
    if ff is not None:
        f = ff[ff.index <= as_of]
        if not f.empty:
            out["fed_funds_rate"] = float(f.iloc[-1])
    cpi = raw_series.get("cpi")
    if cpi is not None:
        c = cpi[cpi.index <= as_of]
        if len(c) > 12:
            try:
                out["cpi_yoy"] = (float(c.iloc[-1]) / float(c.iloc[-13]) - 1) * 100
            except Exception:
                pass
    return out


def backfill_dedollarization_history(days: int = 365) -> dict:
    """Ricalcola e upserta SecularTrend(dedollarization) per gli ultimi `days` giorni.

    Per efficienza tutte le serie FRED vengono fetchate una sola volta; per ogni
    giorno si costruisce uno snapshot as-of e si invocano le funzioni esistenti
    di preparazione + `calculate_dedollarization`.
    """
    # Import qui per evitare ciclo con jobs.py
    from app.scheduler.jobs import (
        _compute_player_history,
        _prepare_dedollarization_indicators,
    )

    fetcher = FredFetcher()
    raw: dict[str, pd.Series] = {}

    logger.info(f"Dedollar backfill: pre-fetch {len(_DEDOLLAR_SERIES)} serie FRED")
    for name in _DEDOLLAR_SERIES:
        try:
            raw[name] = fetcher.fetch_series(name)
        except Exception as e:
            logger.warning(f"Dedollar backfill: skip {name}: {e}")

    today = date.today()
    start = today - timedelta(days=days)
    stats = {"classified": 0, "skipped": 0, "errors": 0, "start": start, "end": today}

    logger.info(f"Dedollar backfill: calcolo {days + 1} giorni da {start} a {today}")

    with Session(engine) as session:
        for offset in range(days, -1, -1):
            d = today - timedelta(days=offset)
            try:
                ts = pd.Timestamp(d)
                latest = _latest_as_of(raw, ts)
                if not latest:
                    stats["skipped"] += 1
                    continue

                macro = _macro_indicators_as_of(raw, ts)
                as_of_fetcher = AsOfFetcher(raw, d)

                dedollar_indicators, player_history = _prepare_dedollarization_indicators(
                    latest, as_of_fetcher, macro
                )

                result = calculate_dedollarization(
                    dedollar_indicators, player_history=player_history
                )

                # Upsert
                session.query(SecularTrend).filter_by(
                    date=d, trend_name="dedollarization"
                ).delete()
                session.add(SecularTrend(
                    date=d,
                    trend_name="dedollarization",
                    score=result["combined_score"],
                    components=json.dumps({
                        "components": result["components"],
                        "structural": result["structural"],
                        "decade": result["decade"],
                        "twenty_year": result.get("twenty_year", {}),
                        "by_player": result.get("by_player", {}),
                        "player_history": result.get("player_history", {}),
                        "player_acceleration": result.get("player_acceleration", {}),
                        "structural_score": result["structural_score"],
                        "decade_score": result["decade_score"],
                        "twenty_year_score": result.get("twenty_year_score"),
                        "acceleration": result["acceleration"],
                        "combined_score": result["combined_score"],
                        "geopolitical_score": result.get("geopolitical_score", 0.0),
                        "raw_dedollar_indicators": dedollar_indicators,
                        "backfilled": True,
                    }),
                ))
                stats["classified"] += 1

                # Commit a blocchi per non tenere una mega-transaction
                if stats["classified"] % 30 == 0:
                    session.commit()
            except Exception as e:
                logger.warning(f"Dedollar backfill errore per {d}: {e}")
                stats["errors"] += 1

        session.commit()

    logger.info(
        f"Dedollar backfill completato: classified={stats['classified']} "
        f"skipped={stats['skipped']} errors={stats['errors']}"
    )
    return stats
