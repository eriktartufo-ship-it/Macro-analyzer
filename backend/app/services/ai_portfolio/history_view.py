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

2026-10-31 — RENDIMENTI PERSI (stessa logica: si corregge alla lettura, il DB resta com'è).
Dal 23/10 la cache Yahoo delle 00:00 non aveva la chiusura nuova e lo snapshot di Modello e LLM
contava rendimento esattamente 0 su una chiusura nuova (6 giri su 8). Qui quei giri si ricalcolano
dai prezzi veri, coi pesi che il bot aveva quel giorno ricostruiti dal libro reale (`rb_movements`
registra ogni variazione di peso del bot, anche floor e tetto). Se i pesi ricostruiti non quadrano
con le posizioni di adesso non si ricalcola niente e `info` lo dice.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

from app.services.ai_portfolio.market_calendar import close_dates, price_asof, ref_close

logger = logging.getLogger(__name__)

STARTING_NAV = 100_000.0
TICKERS = {"sp500": "SPY", "bonds": "TLT", "acwi": "ACWI"}
CASH_TICKER = "BIL"  # come `performance.CASH_OVERNIGHT_TICKER`
# Variazioni sotto `_EPS_W` (0,05%) del libro reale non diventano movimenti: in un mese si
# accumulano poco. 0,5% di peso su una seduta da ±2% sposta il rendimento di 0,01%.
WEIGHT_TOLERANCE = 0.005


def corrected_history(rows: list, fetch=None, weights: dict | None = None,
                      cash_overnight: bool = True) -> tuple[list[dict], dict]:
    """rows: AiPortfolioPerformance di UNA strategia, in ordine di data, dall'inizio.
    weights: {data del giro: {asset: peso}} (`bot_weights_by_day`); senza, i rendimenti persi
    restano 0 e `info["lost_returns"]` li conta.
    Ritorna (righe corrette, info). Se i prezzi non arrivano, righe grezze con `corrected: False`."""
    if not rows:
        return [], {"corrected": False, "duplicates": 0}
    if fetch is None:
        from app.services.ai_portfolio import pnl_tracker

        def fetch(t, s, e):
            return pnl_tracker._fetcher().fetch(t, start_date=s, end_date=e, need_through=e)

    start, end = rows[0].date - timedelta(days=20), rows[-1].date
    try:
        px = {k: fetch(t, start, end) for k, t in TICKERS.items()}
    except Exception as e:  # noqa: BLE001
        logger.warning(f"history_view: prezzi non disponibili ({e}), storico grezzo")
        px = {}
    if not px or px.get("sp500") is None or px["sp500"].empty:
        return [_raw(r) for r in rows], {"corrected": False, "duplicates": 0}

    cal = close_dates(px["sp500"])
    # la chiusura salvata dallo snapshot vince sulla ricostruzione dal calendario (2026-10-22)
    refs = [getattr(r, "price_date", None) or ref_close(cal, r.date) for r in rows]
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
    lost, recomputed = 0, 0
    for i, r in enumerate(rows):
        raw_r = r.nav / prev_raw - 1 if prev_raw else 0.0
        dup = i > 0 and refs[i] == refs[i - 1]
        if dup:
            dups += 1
        redone = False
        # chiusura nuova ma rendimento ESATTAMENTE 0 con posizioni aperte: era la cache, non il mercato
        if (i > 0 and not dup and raw_r == 0.0 and refs[i] and refs[i - 1] and refs[i] > refs[i - 1]
                and (getattr(r, "n_positions", 0) or 0) > 0):
            lost += 1
            w = _weights_on(weights, r.date) if weights else None
            rr = _lost_return(w, refs[i - 1], refs[i], fetch, cash_overnight) if w else None
            if rr is not None:
                raw_r, redone = rr, True
                recomputed += 1
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
            "return_recomputed": redone,
        })
    return out, {"corrected": True, "duplicates": dups, "lost_returns": lost, "recomputed": recomputed}


_LOST_MEMO: dict[tuple, float] = {}


def _weights_on(weights: dict, d: date) -> dict[str, float] | None:
    """Pesi del bot dopo il giro del giorno `d` (ultimo giorno con movimenti <= d)."""
    days = [k for k in weights if k <= d]
    return weights[max(days)] if days else None


def _lost_return(w: dict[str, float], a: date, b: date, fetch, cash_overnight: bool) -> float | None:
    """Rendimento del bot fra due chiusure coi pesi `w` (+ liquidità all'overnight, come lo
    snapshot). Memorizzato: sono giri passati, i prezzi non cambiano più."""
    from app.services.ai_portfolio.momentum import asset_to_ticker

    key = (tuple(sorted(w.items())), a, b, cash_overnight)
    if key in _LOST_MEMO:
        return _LOST_MEMO[key]
    total = 0.0
    for asset, wt in w.items():
        t = asset_to_ticker(asset)
        r = _r(fetch(t, a - timedelta(days=20), b), a, b) if t else None
        if r is None:
            logger.warning(f"history_view: rendimento perso non ricalcolabile ({asset}, {a}->{b})")
            return None
        total += wt * r
    if cash_overnight:
        idle = max(0.0, 1.0 - sum(w.values()))
        if idle > 0:
            total += idle * (_r(fetch(CASH_TICKER, a - timedelta(days=20), b), a, b) or 0.0)
    _LOST_MEMO[key] = total
    return total


def bot_weights_by_day(db, strategy: str) -> tuple[dict[date, dict[str, float]] | None, str]:
    """Pesi del bot dopo il giro di ogni giorno, dal libro reale: somma dei `weight_delta_pct` dei
    movimenti fino a quel giorno. Quadratura: la somma di TUTTI i movimenti deve dare le posizioni
    di adesso (entro `WEIGHT_TOLERANCE`); se non torna → (None, motivo)."""
    from app.models.ai_portfolio import AiPortfolioPosition
    from app.models.real_book import RbMovement

    movs = (db.query(RbMovement)
            .filter(RbMovement.strategy_type == strategy, RbMovement.side.in_(("buy", "sell")))
            .order_by(RbMovement.decided_on, RbMovement.id).all())
    if not movs:
        return None, "nessun movimento nel libro reale"
    cur: dict[str, float] = {}
    by_day: dict[date, dict[str, float]] = {}
    for mv in movs:
        cur[mv.asset_class] = cur.get(mv.asset_class, 0.0) + (mv.weight_delta_pct or 0.0)
        by_day[mv.decided_on] = {a: x for a, x in cur.items() if x > 1e-9}
    now = {p.asset_class: p.current_weight_pct for p in
           db.query(AiPortfolioPosition).filter(AiPortfolioPosition.strategy_type == strategy).all()}
    gap = max((abs(cur.get(a, 0.0) - now.get(a, 0.0)) for a in set(cur) | set(now)), default=0.0)
    if gap > WEIGHT_TOLERANCE:
        return None, f"i pesi ricostruiti non quadrano con le posizioni (scarto massimo {gap:.4f})"
    return by_day, ""


def corrected_history_for(db, strategy: str, rows: list | None = None) -> tuple[list[dict], dict]:
    """Unico ingresso per chi mostra lo storico dei bot (equity curve, classifica, libro reale)."""
    from app.models.ai_portfolio import AiPortfolioPerformance
    from app.services import config_flags

    if rows is None:
        rows = (db.query(AiPortfolioPerformance)
                .filter(AiPortfolioPerformance.strategy_type == strategy)
                .order_by(AiPortfolioPerformance.date).all())
    weights, why = bot_weights_by_day(db, strategy)
    out, info = corrected_history(rows, weights=weights,
                                  cash_overnight=config_flags.use_ai_portfolio_cash_overnight())
    if weights is None and info.get("lost_returns"):
        info["recompute_skipped"] = why
    return out, info


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
