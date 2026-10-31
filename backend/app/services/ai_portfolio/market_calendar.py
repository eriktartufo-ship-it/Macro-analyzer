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


def last_weekday_before(d: date) -> date:
    """Ultimo giorno feriale prima di `d`: la chiusura che un giro del giorno `d` deve avere.
    I festivi USA non sono esclusi (non c'è un calendario scritto a mano): quel giorno si
    riscarica una volta in più e si usa l'ultima chiusura disponibile."""
    p = d - timedelta(days=1)
    while p.weekday() >= 5:
        p -= timedelta(days=1)
    return p


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


def return_between(series: pd.Series, from_close: date, to_close: date) -> float | None:
    """Rendimento fra due DATE DI CHIUSURA (quella già contata e quella nuova).

    2026-10-22: prima prendeva le date dei GIRI e ricalcolava la chiusura «vista allora» sui dati di
    oggi. Con un dato arrivato in ritardo quella ricostruzione è sbagliata e lascia un buco (un
    rendimento mai contato). Ora la chiusura già contata è quella salvata nello snapshot."""
    if to_close <= from_close:
        return 0.0
    p1 = price_asof(series, to_close)
    p0 = price_asof(series, from_close)
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
        db.query(AiPortfolioPerformance)
        .filter(AiPortfolioPerformance.date < today)
        .order_by(AiPortfolioPerformance.date.desc())
        .first()
    )
    prev_date = prev.date if prev else None
    prev_price_date = getattr(prev, "price_date", None) if prev else None
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
    now = ref_close(dates, today)
    # la chiusura già contata è quella SALVATA dall'ultimo giro; per gli snapshot vecchi (senza
    # price_date) si ricostruisce, come prima
    done = prev_price_date or (ref_close(dates, prev_date) if prev_date else None)
    if now is not None and (done is None or now > done):
        return True, f"chiusura nuova: {now} (già contata: {done})"
    return False, (f"nessuna chiusura nuova di Wall Street dopo il {done} "
                   f"(ultimo giro {prev_date}, ultima disponibile {now}): i bot non decidono")
