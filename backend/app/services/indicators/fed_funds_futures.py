"""P1 2026-07-12 — Fed funds futures REALI (CME ZQ) al posto del proxy DGS1.

Problema risolto: `fed_funds_futures_1y` in fred_codes è DGS1 (1Y Treasury),
che incorpora term premium → non è l'aspettativa pulita di policy. I futures
ZQ prezzano direttamente la media dell'EFFR nel mese del contratto:
implied_rate = 100 − price.

Segnale derivato:
- `fed_funds_implied_12m` = tasso implicito ~12 mesi avanti
- `fed_path_12m` = implied − fed_funds corrente
  (< 0 = tagli prezzati; > 0 = RIALZI prezzati — al 2026-07 il mercato prezza
  +0.4pp, cosa che DGS1 non mostra in modo pulito)

Distinzione regime (council): easing prezzato con crescita ok = reflation;
easing da panico (con stress) = deflation. Il dato è l'input, non il pillar:
nessun peso nel classifier senza A/B backtest (regola anti-hand-tuning).

Yahoo symbols: ZQ + month code + YY + ".CBT" (es. luglio 2027 = ZQN27.CBT).
Verificato live 2026-07-12: ZQN27.CBT = 95.925 → implied 4.075%.
"""
from __future__ import annotations

import logging
from datetime import date

logger = logging.getLogger(__name__)

# Month codes futures standard CME
_MONTH_CODES = ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"]


def contract_symbol(target: date) -> str:
    """Symbol Yahoo del contratto ZQ per il mese/anno target."""
    code = _MONTH_CODES[target.month - 1]
    yy = target.year % 100
    return f"ZQ{code}{yy:02d}.CBT"


def target_month(today: date, months_ahead: int = 12) -> date:
    """Primo giorno del mese `months_ahead` mesi avanti."""
    m = today.month - 1 + months_ahead
    return date(today.year + m // 12, m % 12 + 1, 1)


def implied_rate(price: float) -> float | None:
    """Tasso implicito dal prezzo ZQ (100 − price). None se fuori range sano."""
    try:
        p = float(price)
    except (TypeError, ValueError):
        return None
    rate = 100.0 - p
    # Range sanity: EFFR implicito plausibile [-1, 15]. Fuori = dato rotto.
    if rate < -1.0 or rate > 15.0:
        return None
    return rate


def fetch_implied_12m(yahoo_fetcher, today: date | None = None) -> float | None:
    """Tasso fed funds implicito ~12m avanti dai futures ZQ via Yahoo.

    Prova il contratto a 12 mesi; se il fetch fallisce (contratto illiquido o
    symbol non quotato) prova 11 e 13 mesi. None se tutti falliscono —
    NESSUN fallback a DGS1 (meglio assente che sbagliato).
    """
    today = today or date.today()
    for months in (12, 11, 13):
        symbol = contract_symbol(target_month(today, months))
        try:
            s = yahoo_fetcher.fetch(symbol)
            if s is None or s.empty:
                continue
            rate = implied_rate(float(s.dropna().iloc[-1]))
            if rate is not None:
                return rate
        except Exception as e:
            logger.debug(f"ZQ fetch {symbol} failed: {e}")
            continue
    logger.warning("Fed funds futures: nessun contratto ZQ ~12m disponibile")
    return None
