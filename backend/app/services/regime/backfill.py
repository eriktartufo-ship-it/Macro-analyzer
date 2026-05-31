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
    # Indicatori finanziari / aspettative — estendono il classifier con segnali
    # che il rule-based non usava finora. Molte di queste serie iniziano dopo
    # il 1970 (VIX dal 1990, breakeven dal 2003, NFCI dal 1971 settimanale).
    "vix",
    "nfci",
    "breakeven_10y",
    "housing_starts",
    # Council 2026-05-27 — Liquidity surge override (post TEST A 2020 disaster)
    "m2",                # M2 money stock (mensile, FRED M2SL)
    "fed_balance_sheet", # Fed WALCL (settimanale, FRED)
    # T13 forward-looking inflation features (council 2026-05-31)
    "umich_inflation_1y",  # MICH (mensile, dal 1978)
    "sticky_cpi_yoy",      # Atlanta sticky CPI (mensile, dal 1968)
    "breakeven_5y5y",      # T5YIFR (giornaliero, dal 2003)
    # T17 Labor + Credit features (council 2026-05-31 sessione 23)
    "wage_growth_atlanta", # Atlanta wage tracker (mensile, dal 1997)
    "jolts_quits_rate",    # JOLTS quits (mensile, dal 2000)
    "avg_weekly_hours",    # AWHAETP (mensile, dal 2006)
    "hy_credit_spread",    # BofA HY OAS (giornaliero, dal 1997)
    # T18 Cyclical leading features (council 2026-05-31 sessione 25)
    "heavy_truck_sales",   # HTRUCKSSAAR (mensile, dal 1967)
    "building_permits",    # PERMIT (mensile, dal 1960)
    "consumer_credit",     # TOTALSL (mensile, dal 1943)
    "durable_goods_orders", # DGORDER (mensile, dal 1992)
    # T19 Nowcast features (council 2026-05-31 sessione 26)
    "gdp_nowcast_atlanta", # GDPNOW (settimanale, dal 2014)
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

    # LEI (CFNAI 3MA: livello z-score, NON ROC. > 0 = sopra trend, < 0 = sotto)
    v = last_before("lei")
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

    # --- Nuovi indicatori finanziari / aspettative ---
    v = last_before("vix")
    if v is not None:
        indicators["vix"] = v

    v = last_before("nfci")
    if v is not None:
        indicators["nfci"] = v

    v = last_before("breakeven_10y")
    if v is not None:
        indicators["breakeven_10y"] = v

    # housing_starts e' monthly, 12 periodi = 1 anno
    v = roc("housing_starts", 12)
    if v is not None:
        indicators["housing_starts_roc_12m"] = v

    # T13 forward inflation features (level read)
    v = last_before("umich_inflation_1y")
    if v is not None:
        indicators["umich_inflation_1y"] = v
    v = last_before("sticky_cpi_yoy")
    if v is not None:
        indicators["sticky_cpi_yoy"] = v
    v = last_before("breakeven_5y5y")
    if v is not None:
        indicators["breakeven_5y5y"] = v

    # T17 Labor + Credit features
    v = last_before("wage_growth_atlanta")
    if v is not None:
        indicators["wage_growth_atlanta"] = v
    v = last_before("jolts_quits_rate")
    if v is not None:
        indicators["jolts_quits_rate"] = v
    # avg_weekly_hours YoY (12 monthly periods)
    v = roc("avg_weekly_hours", 12)
    if v is not None:
        indicators["avg_weekly_hours_yoy"] = v
    v = last_before("hy_credit_spread")
    if v is not None:
        indicators["hy_credit_spread"] = v

    # T18 Cyclical leading features (YoY transforms)
    v = roc("heavy_truck_sales", 12)
    if v is not None:
        indicators["heavy_truck_sales_yoy"] = v
    v = roc("building_permits", 12)
    if v is not None:
        indicators["building_permits_yoy"] = v
    v = roc("consumer_credit", 12)
    if v is not None:
        indicators["consumer_credit_yoy"] = v
    v = roc("durable_goods_orders", 12)
    if v is not None:
        indicators["durable_goods_orders_yoy"] = v

    # T19 Nowcast (Atlanta GDPNow level, weekly)
    v = last_before("gdp_nowcast_atlanta")
    if v is not None:
        indicators["gdp_nowcast_atlanta"] = v

    # T11 (2026-05-27): momentum derivatives per pillar acceleration-based.
    _enrich_with_momentum_derivatives(indicators, series, cutoff)

    return indicators


def _enrich_with_momentum_derivatives(
    indicators: dict[str, float],
    series: dict[str, pd.Series],
    cutoff: pd.Timestamp,
) -> None:
    """T11: arricchisce indicators dict con derivati momentum-based.

    Calcola in-place:
    - cpi_yoy_change_6m: cpi_yoy(now) - cpi_yoy(now-6m)
    - core_pce_yoy_change_6m
    - gdp_roc_change_6m: quarter-to-quarter delta GDP ROC
    - cpi_yoy_min_last_6m: min YoY% ultimi 6 mesi (sticky inflation)

    Usato sia da `_build_indicators_as_of` (backfill batch) che da scheduler
    (refresh giornaliero) per consistency.
    """

    def yoy_change_6m(name: str) -> Optional[float]:
        """cpi_yoy(now) - cpi_yoy(now - 6m). Richiede ≥ 18 osservazioni mensili."""
        s = series.get(name)
        if s is None or s.empty:
            return None
        filtered = s[s.index <= cutoff]
        if len(filtered) < 19:
            return None
        try:
            curr_yoy = (float(filtered.iloc[-1]) / float(filtered.iloc[-13]) - 1) * 100
            prev_yoy = (float(filtered.iloc[-7]) / float(filtered.iloc[-19]) - 1) * 100
            return curr_yoy - prev_yoy
        except Exception:
            return None

    v = yoy_change_6m("cpi")
    if v is not None:
        indicators["cpi_yoy_change_6m"] = v

    v = yoy_change_6m("core_pce")
    if v is not None:
        indicators["core_pce_yoy_change_6m"] = v

    # GDP quarterly: ROC è 1-period change. Per change 6m = 2 quarter diff.
    def gdp_roc_change_6m_calc() -> Optional[float]:
        s = series.get("real_gdp")
        if s is None or s.empty:
            return None
        filtered = s[s.index <= cutoff]
        if len(filtered) < 4:
            return None
        try:
            curr_roc = (float(filtered.iloc[-1]) / float(filtered.iloc[-2]) - 1) * 100
            prev_roc = (float(filtered.iloc[-3]) / float(filtered.iloc[-4]) - 1) * 100
            return curr_roc - prev_roc
        except Exception:
            return None

    v = gdp_roc_change_6m_calc()
    if v is not None:
        indicators["gdp_roc_change_6m"] = v

    # cpi_yoy_min_last_6m: minimo dei YoY% ultimi 6 mesi (sticky inflation signal)
    def cpi_yoy_min_last_6m() -> Optional[float]:
        s = series.get("cpi")
        if s is None or s.empty:
            return None
        filtered = s[s.index <= cutoff]
        if len(filtered) < 18:
            return None
        try:
            yoy_values = []
            for i in range(6):
                idx = -1 - i
                yoy = (float(filtered.iloc[idx]) / float(filtered.iloc[idx - 12]) - 1) * 100
                yoy_values.append(yoy)
            return min(yoy_values)
        except Exception:
            return None

    v = cpi_yoy_min_last_6m()
    if v is not None:
        indicators["cpi_yoy_min_last_6m"] = v

    # Council 2026-05-27 — Liquidity surge derivatives (per `USE_LIQUIDITY_SURGE_OVERRIDE`)
    # Cattura QE/stampa valuta (es. 2020 Fed +75% balance sheet in 3 mesi).
    # User insight: il V-shape recovery è LEADING-INDICATED da liquidity print,
    # NOT dai fondamentali macro tradizionali.
    def walcl_roc_3m() -> Optional[float]:
        s = None
        for key in ("fed_balance_sheet", "walcl", "WALCL"):
            cand = series.get(key)
            if cand is not None and not cand.empty:
                s = cand
                break
        if s is None:
            return None
        filtered = s[s.index <= cutoff]
        if len(filtered) < 13:
            return None
        try:
            curr = float(filtered.iloc[-1])
            prev = float(filtered.iloc[-13])  # ~3 mesi indietro (settimanale × 13)
            if prev <= 0:
                return None
            return (curr / prev - 1) * 100
        except Exception:
            return None

    v = walcl_roc_3m()
    if v is not None:
        indicators["walcl_roc_3m"] = v

    # M2 YoY: 12-month change
    def m2_yoy() -> Optional[float]:
        for key in ("m2", "m2_yoy"):
            s = series.get(key)
            if s is None or s.empty:
                continue
            filtered = s[s.index <= cutoff]
            if len(filtered) < 13:
                continue
            try:
                curr = float(filtered.iloc[-1])
                prev = float(filtered.iloc[-13])
                if prev <= 0:
                    continue
                return (curr / prev - 1) * 100
            except Exception:
                continue
        return None

    v = m2_yoy()
    if v is not None:
        indicators["m2_yoy"] = v

    # Real rate trend 3m (declining if real rate now < real rate 3m ago)
    def real_rate_decline_3m() -> Optional[float]:
        ff_s = series.get("fed_funds")
        cpi_s = series.get("cpi")
        if ff_s is None or cpi_s is None or ff_s.empty or cpi_s.empty:
            return None
        ff_filt = ff_s[ff_s.index <= cutoff]
        cpi_filt = cpi_s[cpi_s.index <= cutoff]
        if len(ff_filt) < 4 or len(cpi_filt) < 13:
            return None
        try:
            ff_now = float(ff_filt.iloc[-1])
            cpi_yoy_now = (float(cpi_filt.iloc[-1]) / float(cpi_filt.iloc[-13]) - 1) * 100
            real_rate_now = ff_now - cpi_yoy_now

            ff_3m = float(ff_filt.iloc[-4]) if len(ff_filt) >= 4 else ff_now
            cpi_yoy_3m_ago = (float(cpi_filt.iloc[-4]) / float(cpi_filt.iloc[-16]) - 1) * 100 if len(cpi_filt) >= 16 else cpi_yoy_now
            real_rate_3m_ago = ff_3m - cpi_yoy_3m_ago
            return real_rate_now - real_rate_3m_ago  # negativo = declining
        except Exception:
            return None

    v = real_rate_decline_3m()
    if v is not None:
        indicators["real_rate_change_3m"] = v


def backfill_regime_history_long(
    start_date: date,
    end_date: date | None = None,
    step_days: int = 30,
) -> dict:
    """Backfill storico a lungo raggio (anni/decadi).

    Differenza rispetto a `backfill_regime_history`:
      - accetta `start_date` esplicito (es. 1970-01-01) invece di offset in giorni
      - iterazione a passi di `step_days` (default 30 = ~mensile) per essere
        usabile su 50+ anni senza generare centinaia di migliaia di record

    La classificazione usa lo STESSO rule-based classifier corrente: i record
    storici riflettono come il classifier odierno vedrebbe quelle date. Serve
    da training per HMM e come ground-truth per la transition matrix.

    Args:
        start_date: data iniziale inclusa
        end_date: data finale (default: oggi)
        step_days: passo tra una classificazione e l'altra

    Returns:
        Stats dict identico a `backfill_regime_history`
    """
    if end_date is None:
        end_date = date.today()
    if start_date >= end_date:
        raise ValueError(f"start_date {start_date} >= end_date {end_date}")

    fetcher = FredFetcher()
    series: dict[str, pd.Series] = {}

    # Buffer di 2 anni indietro: roc 12m e initial_claims 13-period need
    # dati pre-start_date per calcolare il primo punto.
    fetch_start = start_date - timedelta(days=365 * 2)
    logger.info(
        f"Backfill storico: fetch {len(_CLASSIFIER_SERIES)} serie FRED "
        f"(fetch_buffer {fetch_start} → {end_date}, classify {start_date} → {end_date}, step={step_days}d)"
    )
    for name in _CLASSIFIER_SERIES:
        try:
            series[name] = fetcher.fetch_series(name, start_date=fetch_start, end_date=end_date)
        except Exception as e:
            logger.warning(f"Backfill: impossibile fetchare {name}: {e}")

    stats = {"classified": 0, "skipped": 0, "errors": 0,
             "start": start_date, "end": end_date, "step_days": step_days}

    with Session(engine) as session:
        d = start_date
        while d <= end_date:
            try:
                indicators = _build_indicators_as_of(series, d)
                if not {"gdp_roc", "cpi_yoy", "unrate"}.issubset(indicators):
                    stats["skipped"] += 1
                    d += timedelta(days=step_days)
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
                        "historical": True,
                    }),
                ))
                stats["classified"] += 1
                if stats["classified"] % 50 == 0:
                    session.commit()
                    logger.info(f"Backfill storico: {stats['classified']} classificati ({d})")
            except Exception as e:
                logger.warning(f"Backfill errore per {d}: {e}")
                stats["errors"] += 1

            d += timedelta(days=step_days)

        session.commit()

    logger.info(
        f"Backfill storico completato: classified={stats['classified']} "
        f"skipped={stats['skipped']} errors={stats['errors']}"
    )
    return stats


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
