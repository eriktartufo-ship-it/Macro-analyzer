"""T10b Walk-forward classifier helper per random backtest.

Council 2026-05-17 verdict (sessione 5): walk-forward è il SINGLE point of
leverage per sbloccare validation dei 4 flag UNTESTABLE
(GDP_COLLAPSE_OVERRIDE, ML_REGIME_BLEND, DEDOLLAR_BONUS, DEFENSIVE_TRANSITION).

Pattern: usa `_build_indicators_as_of(series, as_of)` già esistente in
`regime/backfill.py` per chiamare `classify_regime` mese-per-mese con
indicators troncati alla data target. Niente future leak.

Cache: le serie FRED vengono pre-fetched UNA VOLTA all'inizio del batch.
Le classifications vengono cached per (as_of_date, flag_snapshot_key) per
non ri-classificare tra varianti di flag identiche.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd
from loguru import logger

from app.services.indicators.fetcher import FredFetcher
from app.services.regime.backfill import _CLASSIFIER_SERIES, _build_indicators_as_of
from app.services.regime.classifier import classify_regime

# T10c: serie addizionali per dedollar secular_bonus computation walk-forward.
_DEDOLLAR_SERIES = ("debt_gdp", "m2_yoy")


def preload_walk_forward_series(
    start_date: date,
    end_date: date,
    fetch_buffer_years: int = 2,
) -> dict[str, pd.Series]:
    """Pre-carica tutte le serie FRED necessarie per il walk-forward.

    Args:
        start_date: prima data target del backtest.
        end_date: ultima data target del backtest.
        fetch_buffer_years: buffer retrospettivo per roc/yoy (default 2y).

    Returns:
        Dict {series_name: pd.Series indexed by date}.
    """
    fetcher = FredFetcher()
    fetch_start = start_date - timedelta(days=365 * fetch_buffer_years)
    series: dict[str, pd.Series] = {}
    all_series = _CLASSIFIER_SERIES + _DEDOLLAR_SERIES
    logger.info(f"Walk-forward: pre-fetching {len(all_series)} FRED series "
                f"({fetch_start} -> {end_date})")
    for name in all_series:
        try:
            series[name] = fetcher.fetch_series(name, start_date=fetch_start, end_date=end_date)
        except Exception as e:
            logger.warning(f"  {name}: fetch failed: {e}")
    return series


def compute_dedollar_secular_bonus_at(
    series: dict[str, pd.Series],
    as_of,
) -> dict[str, float]:
    """T10c: calcola dedollar secular_bonus a una data as_of usando indicators
    minimali derivati da serie FRED disponibili.

    Versione semplificata di `calculate_secular_bonus` per il backtest:
    usa solo gli indicators "cyclical" (debt_gdp, real_rate, m2_yoy) che sono
    disponibili come serie FRED storiche. Gli indicators strutturali (5y trends)
    non sono cruciali per il bonus, vengono lasciati a default.

    Returns:
        Dict {asset: bonus} (puo essere vuoto se nessun indicator disponibile).
    """
    from app.services.dedollarization.scorer import (
        calculate_dedollarization, calculate_secular_bonus,
    )
    cutoff = pd.Timestamp(as_of)

    def last_before(name: str):
        s = series.get(name)
        if s is None or s.empty:
            return None
        filtered = s[s.index <= cutoff]
        return float(filtered.iloc[-1]) if not filtered.empty else None

    def roc12(name: str):
        s = series.get(name)
        if s is None or s.empty:
            return None
        filtered = s[s.index <= cutoff]
        if len(filtered) < 13:
            return None
        try:
            curr = float(filtered.iloc[-1])
            prev = float(filtered.iloc[-13])
            if prev == 0:
                return None
            return (curr / prev - 1) * 100
        except Exception:
            return None

    indicators: dict[str, float] = {}
    debt = last_before("debt_gdp")
    if debt is not None:
        indicators["debt_gdp"] = debt
    # real_rate = fed_funds - cpi_yoy (approx)
    ff = last_before("fed_funds")
    cpi_yoy = roc12("cpi")
    if ff is not None and cpi_yoy is not None:
        indicators["real_rate"] = ff - cpi_yoy
    m2_y = roc12("m2_yoy")
    if m2_y is not None:
        indicators["m2_roc_12m"] = m2_y

    if not indicators:
        return {}

    try:
        result = calculate_dedollarization(indicators)
        score = result.get("combined_score", 0.0)
        return calculate_secular_bonus(score)
    except Exception:
        return {}


def classify_at(
    series: dict[str, pd.Series],
    as_of: date,
) -> Optional[dict]:
    """Classifica il regime alla data `as_of` usando solo dati pre-as_of.

    Args:
        series: pre-loaded FRED series (output di preload_walk_forward_series).
        as_of: data target.

    Returns:
        Dict con 'regime', 'probs', 'confidence' (compatibile con
        cm_by_month entries del random_backtest), o None se indicators
        insufficienti.
    """
    indicators = _build_indicators_as_of(series, as_of)
    if len(indicators) < 5:
        # Minimo indicators per classification stabile
        return None
    try:
        result = classify_regime(indicators)
    except Exception as e:
        logger.warning(f"classify_regime at {as_of} failed: {e}")
        return None

    probs = result.get("probabilities") or result.get("probs")
    if probs is None:
        return None

    return {
        "regime": result.get("regime"),
        "liquidity_surge_triggered": result.get("liquidity_surge_triggered", False),
        "gdp_collapse_triggered": result.get("gdp_collapse_triggered", False),
        "probs": {
            "reflation": float(probs.get("reflation", 0.0)),
            "stagflation": float(probs.get("stagflation", 0.0)),
            "deflation": float(probs.get("deflation", 0.0)),
            "goldilocks": float(probs.get("goldilocks", 0.0)),
        },
        "confidence": float(result.get("confidence", 0.5) or 0.5),
        # T14 — expose raw indicators per event-triggered rebalance detector
        "indicators": indicators,
    }


class WalkForwardCache:
    """Cache classifications per (as_of, configurazione-classifier) durante un batch.

    Quando si testa lo stesso flag su 15 sim, molte date target sono ricalcolate.
    Cache le riusa. Quando un flag del classifier cambia, la chiave cambia → miss.

    S42 — `flag_key` era opzionale con default "" e NESSUNO dei call-site lo passava
    (unico uso: horserace_backtest.py:271, senza argomento) → cambiare un flag del
    classifier nello stesso processo restituiva le classificazioni della config
    precedente. Ora il default è il fingerprint REALE: protegge senza che il
    chiamante debba ricordarsene. Un default che va ricordato non viene applicato.
    """

    def __init__(self, series: dict[str, pd.Series]):
        self.series = series
        self._cache: dict[tuple[date, str], Optional[dict]] = {}

    def get(self, as_of: date, flag_key: str | None = None) -> Optional[dict]:
        if flag_key is None:
            from app.services.config_flags import classifier_flags_fingerprint

            flag_key = classifier_flags_fingerprint()
        cache_key = (as_of, flag_key)
        if cache_key in self._cache:
            return self._cache[cache_key]
        cm = classify_at(self.series, as_of)
        self._cache[cache_key] = cm
        return cm

    def clear(self) -> None:
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)
