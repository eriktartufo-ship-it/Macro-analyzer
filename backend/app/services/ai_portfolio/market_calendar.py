"""Calendario di mercato per i bot: QUALE chiusura è nuova, e il rendimento fra due chiusure.

🔴 Il difetto che chiude (misurato il 2026-10-21): lo scan gira ogni notte alle 00:30 UTC e lo
snapshot applicava sempre «ultima chiusura ÷ penultima». Sabato, domenica e lunedì notte
l'ultima chiusura di Wall Street è sempre quella di venerdì ⇒ il rendimento del venerdì veniva
contato TRE volte a settimana (30 snapshot su 102 dal 12/07), nei bot e nei benchmark. Il sito
dava all'S&P +5,58% dal 9/07 contro un vero +2,56%; il bot Modello +12,0% contro una stima
corretta di +7,6%. E nel weekend i bot decidevano due volte sugli stessi prezzi (tranche 1/2
sabato e 2/2 domenica al prezzo del venerdì).

Regola unica: alle 00:30 UTC del giorno D l'ultima chiusura disponibile è l'ultima con data < D.
- `has_new_close`: si decide solo se è arrivata una chiusura nuova dall'ultimo giro (weekend e
  festivi USA si saltano da soli, senza un calendario scritto a mano).
- `return_between`: rendimento fra la chiusura già contata e quella nuova; se non ce n'è una
  nuova vale 0, non la ripetizione di ieri. Il bitcoin, che quota anche nel weekend, prende il
  suo movimento vero del weekend al primo giro utile.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd


def _dates(series: pd.Series) -> list[date]:
    return sorted({pd.Timestamp(i).date() for i in series.dropna().index})


def ref_close(dates: list[date], run_date: date) -> date | None:
    """Ultima chiusura disponibile a un giro del giorno `run_date` (00:30 UTC)."""
    before = [d for d in dates if d < run_date]
    return before[-1] if before else None


def has_new_close(dates: list[date], today: date, prev_run_date: date | None) -> bool:
    """True se dall'ultimo giro è arrivata una chiusura nuova (o se non c'è un giro precedente)."""
    if prev_run_date is None:
        return True
    now, before = ref_close(dates, today), ref_close(dates, prev_run_date)
    if now is None:
        return False
    return before is None or now > before


def price_asof(series: pd.Series, d: date) -> float | None:
    s = series.dropna()
    if s.empty:
        return None
    idx = [pd.Timestamp(i).date() for i in s.index]
    vals = [float(v) for v, i in zip(s.values, idx) if i <= d]
    return vals[-1] if vals else None


def return_between(series: pd.Series, prev_run_date: date, run_date: date) -> float | None:
    """Rendimento fra la chiusura vista al giro `prev_run_date` e quella vista a `run_date`."""
    p1 = price_asof(series, run_date - timedelta(days=1))
    p0 = price_asof(series, prev_run_date - timedelta(days=1))
    if not p1 or not p0:
        return None
    return p1 / p0 - 1.0


def close_dates(series: pd.Series) -> list[date]:
    return _dates(series)


def bots_should_run(db, today: date | None = None) -> tuple[bool, str]:
    """Il giro notturno dei bot ha senso solo con una chiusura nuova di Wall Street.

    Guarda l'ultimo snapshot di performance (qualunque strategia) e le chiusure di SPY. Se il
    dato non arriva, il giro si fa comunque (fail-open dichiarato): saltare in silenzio per un
    timeout di Yahoo fermerebbe i bot senza che nessuno se ne accorga.
    """
    from app.models.ai_portfolio import AiPortfolioPerformance
    from app.services.ai_portfolio import pnl_tracker

    today = today or date.today()
    prev = (
        db.query(AiPortfolioPerformance.date)
        .filter(AiPortfolioPerformance.date < today)
        .order_by(AiPortfolioPerformance.date.desc())
        .first()
    )
    prev_date = prev[0] if prev else None
    try:
        # force_refresh: la cache su disco tollera dati vecchi di 3 giorni e potrebbe non avere
        # ancora la chiusura di venerdì; il dato fresco aggiorna anche il disco per i bot.
        s = pnl_tracker._fetcher().fetch("SPY", start_date=today - timedelta(days=20), end_date=today,
                                         force_refresh=True)
    except Exception as e:  # noqa: BLE001
        return True, f"chiusure SPY non disponibili ({e}): giro fatto comunque"
    if s is None or s.empty:
        return True, "chiusure SPY non disponibili: giro fatto comunque"
    dates = close_dates(s)
    if has_new_close(dates, today, prev_date):
        return True, f"chiusura nuova: {ref_close(dates, today)}"
    return False, (f"nessuna chiusura nuova di Wall Street dopo il {ref_close(dates, today)} "
                   f"(ultimo giro {prev_date}): i bot non decidono")
