"""Storico dei bot corretto ALLA LETTURA, con S&P 500, 60/40 e MSCI ACWI dai prezzi veri.

Perché alla lettura e non riscrivendo il DB (2026-10-21): fino al fix di `market_calendar`
lo snapshot ricontava il rendimento del venerdì a ogni giro senza una chiusura nuova (30 snapshot
su 102 dal 12/07). Riscrivere lo storico in produzione sarebbe un'operazione non additiva; qui
invece i dati salvati restano com'erano e la pagina mostra la versione corretta:
- bot: il rendimento di uno snapshot senza chiusura nuova vale 0 (per gli ETF è esatto; per il
  bitcoin si perde il movimento del weekend di quei giorni: errore piccolo, dichiarato);
- benchmark: ricalcolati dai prezzi (chiusure rettificate, quindi con dividendi), non presi dalle
  colonne salvate, che avevano lo stesso difetto. 60/40 ribilanciato ogni giorno come nei bot.

Dopo il fix gli snapshot nuovi non hanno duplicati: la correzione diventa un no-op su di loro.
"""
from __future__ import annotations

import logging
from datetime import timedelta

from app.services.ai_portfolio.market_calendar import close_dates, price_asof, ref_close

logger = logging.getLogger(__name__)

STARTING_NAV = 100_000.0
TICKERS = {"sp500": "SPY", "bonds": "TLT", "acwi": "ACWI"}


def corrected_history(rows: list, fetch=None) -> tuple[list[dict], dict]:
    """rows: AiPortfolioPerformance di UNA strategia, in ordine di data, dall'inizio.
    Ritorna (righe corrette, info). Se i prezzi non arrivano, righe grezze con `corrected: False`."""
    if not rows:
        return [], {"corrected": False, "duplicates": 0}
    if fetch is None:
        from app.services.ai_portfolio import pnl_tracker

        def fetch(t, s, e):
            return pnl_tracker._fetcher().fetch(t, start_date=s, end_date=e)

    start, end = rows[0].date - timedelta(days=20), rows[-1].date
    try:
        px = {k: fetch(t, start, end) for k, t in TICKERS.items()}
    except Exception as e:  # noqa: BLE001
        logger.warning(f"history_view: prezzi non disponibili ({e}), storico grezzo")
        px = {}
    if not px or px.get("sp500") is None or px["sp500"].empty:
        return [_raw(r) for r in rows], {"corrected": False, "duplicates": 0}

    cal = close_dates(px["sp500"])
    refs = [ref_close(cal, r.date) for r in rows]
    first_idx = cal.index(refs[0]) if refs[0] in cal else 0
    base = cal[max(0, first_idx - 1)]

    # 60/40 ribilanciato ogni giorno di borsa, cumulato fino a ogni data del calendario
    cum6040, v = {base: 1.0}, 1.0
    for a, b in zip(cal[cal.index(base):], cal[cal.index(base) + 1:]):
        rs = _r(px["sp500"], a, b)
        rb = _r(px.get("bonds"), a, b)
        v *= 1 + 0.6 * (rs or 0.0) + 0.4 * (rb or 0.0)
        cum6040[b] = v

    def bench(key: str, ref):
        s = px.get(key)
        if s is None or s.empty or ref is None:
            return None
        p0, p1 = price_asof(s, base), price_asof(s, ref)
        return (p1 / p0 - 1) if (p0 and p1) else None

    out, nav, peak, prev_raw, dups = [], STARTING_NAV, STARTING_NAV, STARTING_NAV, 0
    for i, r in enumerate(rows):
        raw_r = r.nav / prev_raw - 1 if prev_raw else 0.0
        dup = i > 0 and refs[i] == refs[i - 1]
        if dup:
            dups += 1
        nav *= 1 + (0.0 if dup else raw_r)
        peak = max(peak, nav)
        prev_raw = r.nav
        out.append({
            "date": str(r.date),
            "nav": nav,
            "cumulative_return_pct": nav / STARTING_NAV - 1,
            "drawdown_pct": nav / peak - 1,
            "peak_nav": peak,
            "benchmark_sp500_return_pct": bench("sp500", refs[i]),
            "benchmark_60_40_return_pct": (cum6040.get(refs[i], v) - 1) if refs[i] else None,
            "benchmark_acwi_return_pct": bench("acwi", refs[i]),
            "price_date": str(refs[i]) if refs[i] else None,
            "duplicate_neutralized": dup,
        })
    return out, {"corrected": True, "duplicates": dups}


def _r(series, a, b):
    if series is None or series.empty:
        return None
    p0, p1 = price_asof(series, a), price_asof(series, b)
    return (p1 / p0 - 1) if (p0 and p1) else None


def _raw(r) -> dict:
    return {
        "date": str(r.date), "nav": r.nav, "cumulative_return_pct": r.cumulative_return_pct,
        "drawdown_pct": r.drawdown_pct, "peak_nav": r.peak_nav,
        "benchmark_sp500_return_pct": r.benchmark_sp500_return_pct,
        "benchmark_60_40_return_pct": r.benchmark_60_40_return_pct,
        "benchmark_acwi_return_pct": None, "price_date": None, "duplicate_neutralized": False,
    }
