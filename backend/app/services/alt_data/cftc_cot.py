"""P1 2026-07-12 — CFTC Commitments of Traders (positioning speculativo).

Gap colmato: il dominio "positioning" aveva solo l'insider score SEC Form 4
(campionato ~10%). Il COT legacy futures-only dà il posizionamento
non-commercial (speculatori) su 4 mercati chiave, settimanale (martedì,
pubblicato venerdì, lag 3gg), GRATIS via API Socrata.

Segnale: net non-commercial in % dell'open interest, z-score vs ~3 anni.
- Estremi (|z| > 2) = crowded trade → contrarian
- Trend confermato da positioning moderato = conferma
Per quadrante: crowded short Treasury = consensus stagflation (fade-abile);
crowded long gold = debasement-bid affollato.

Fonte: publicreporting.cftc.gov dataset 6dca-aqww (Legacy Futures Only).
Verificato live 2026-07-12 (report 2026-07-07 disponibile). Nessuna API key
per volumi modesti. Cache in-memory 12h (il dato cambia 1x/settimana).

Data-first: indicators only (cot_*_z), NESSUN pillar classifier senza A/B
backtest (regola anti-hand-tuning).
"""
from __future__ import annotations

import logging
import time
from statistics import mean, stdev

import requests

logger = logging.getLogger(__name__)

_API_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"

# contract_market_name (campo Socrata) → chiave indicator
COT_MARKETS = {
    "UST 10Y NOTE": "cot_bonds10y_z",
    "GOLD": "cot_gold_z",
    "E-MINI S&P 500": "cot_equity_z",
    "USD INDEX": "cot_usd_z",
}

_HISTORY_WEEKS = 156  # ~3 anni per lo z-score
_MIN_OBS = 30

# Cache in-memory (il COT esce 1x/settimana; daily_refresh gira 1x/giorno)
_CACHE_TTL_S = 12 * 3600
_cache: tuple[float, dict[str, float]] | None = None


def compute_net_pct_oi(long_pos: float, short_pos: float, open_interest: float) -> float | None:
    """Net non-commercial in % dell'open interest. None se OI non valido."""
    try:
        oi = float(open_interest)
        if oi <= 0:
            return None
        return (float(long_pos) - float(short_pos)) / oi * 100.0
    except (TypeError, ValueError):
        return None


def zscore_last(series: list[float], min_obs: int = _MIN_OBS) -> float | None:
    """Z-score dell'ultimo valore vs la storia. None se storia corta o piatta."""
    if len(series) < min_obs:
        return None
    mu = mean(series)
    try:
        sd = stdev(series)
    except Exception:
        return None
    if sd <= 1e-9:
        return None
    return (series[-1] - mu) / sd


def _fetch_market_history(market: str, limit: int = _HISTORY_WEEKS) -> list[dict]:
    r = requests.get(
        _API_URL,
        params={
            "$select": ("report_date_as_yyyy_mm_dd,open_interest_all,"
                        "noncomm_positions_long_all,noncomm_positions_short_all"),
            "$where": f"contract_market_name='{market}'",
            "$order": "report_date_as_yyyy_mm_dd DESC",
            "$limit": limit,
        },
        timeout=45,
    )
    r.raise_for_status()
    return r.json()


def fetch_cot_zscores() -> dict[str, float]:
    """Z-score del positioning netto non-commercial per i 4 mercati COT.

    Ritorna solo le chiavi calcolabili (parziale se un mercato fallisce).
    Cache 12h. Mai eccezioni verso il chiamante.
    """
    global _cache
    now = time.time()
    if _cache is not None and now - _cache[0] < _CACHE_TTL_S:
        return dict(_cache[1])

    out: dict[str, float] = {}
    for market, key in COT_MARKETS.items():
        try:
            rows = _fetch_market_history(market)
            # rows sono DESC → riordina ASC per avere l'ultimo in coda
            series: list[float] = []
            for row in reversed(rows):
                v = compute_net_pct_oi(
                    row.get("noncomm_positions_long_all"),
                    row.get("noncomm_positions_short_all"),
                    row.get("open_interest_all"),
                )
                if v is not None:
                    series.append(v)
            z = zscore_last(series)
            if z is not None:
                out[key] = round(z, 3)
        except Exception as e:
            logger.warning(f"COT fetch {market} failed: {e}")

    if out:
        _cache = (now, dict(out))
    return out
