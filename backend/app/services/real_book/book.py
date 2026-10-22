"""Libro reale — orchestrazione col DB: copia i bot, esegue all'apertura, accredita le
cedole, misura il valore a fine giornata.

Due momenti al giorno (scheduler):
- `run_after_scan` — subito dopo che i 3 bot hanno deciso (di notte, prima dell'apertura):
  confronta lo stato di ogni bot con l'ultimo stato copiato e crea gli ORDINI in attesa,
  con il loro perché. Non esegue niente: il prezzo d'apertura di oggi non esiste ancora.
- `run_after_close` — dopo la chiusura di Xetra/Milano: esegue gli ordini in attesa al
  prezzo d'APERTURA della prima seduta valida (una decisione di sabato si esegue lunedì
  alle 09:00), accredita le cedole staccate, scrive il valore del libro alla chiusura.

🔴 Perché si copia lo STATO e non le decisioni: il pavimento di investimento
(`deploy_boost.apply_deployment_floor`) cambia le posizioni dei bot SENZA scrivere una
decisione. Un libro che seguisse solo `ai_portfolio_decisions` lo perderebbe in silenzio.
Le decisioni servono solo per il perché. E ogni cambio di peso che non ha una causa
riconoscibile diventa un movimento `riallineamento`, visibile, invece di un buco.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from typing import Iterable

from sqlalchemy.orm import Session

from app.models.ai_portfolio import AiPortfolioDecision, AiPortfolioPerformance, AiPortfolioPosition
from app.models.real_book import RbBook, RbBotSnapshot, RbMovement, RbNav
from app.models.regime_classifications import RegimeClassification
from app.services.real_book import ledger as L
from app.services.real_book import narrator as N
from app.services.real_book.instruments import (
    BENCHMARK_ACWI_KEY,
    BENCHMARK_BOND_KEY,
    BENCHMARK_KEY,
    INSTRUMENTS,
    instrument_for,
    label_for,
)
from app.services.real_book.market import (
    Market,
    YahooMarket,
    close_on,
    first_bar_on_or_after,
    last_close_on_or_before,
    open_time_utc,
)

logger = logging.getLogger(__name__)

STRATEGIES = ("model_driven", "llm_driven", "risk_managed_sp")
_EPS_W = 0.0005  # sotto lo 0,05% di peso un cambio non è un'operazione
_CLOSE_KIND = {"STOP_LOSS": "stop_loss", "TAKE_PROFIT": "take_profit", "FULL_CLOSE": "full_close"}

# Indicatori che il diario può citare (se presenti nel giorno), con nome leggibile.
# 🔴 Il perché cita i VALORI, mai le soglie delle condizioni del classificatore: le condizioni
# hanno un punteggio sfumato, e il 21/10/2026 `inflation_high` ("CPI YoY > 4%") contava al 63%
# con il CPI a 3,72%. Scrivere "spinto da: CPI > 4%" sarebbe stato falso, e la guardia sui
# numeri non l'avrebbe fermato (il 4 stava nei fatti).
_INDICATORI = {
    "cpi_yoy": ("inflazione USA", "%"),
    "gdp_roc": ("crescita del PIL USA", "%"),
    "unrate": ("disoccupazione USA", "%"),
    "consumer_sentiment": ("fiducia dei consumatori", ""),
    "core_pce_yoy": ("inflazione core PCE", "%"),
    "fed_funds_rate": ("tassi Fed", "%"),
    "yield_curve_10y2y": ("curva dei tassi 10-2 anni", " punti"),
    "vix": ("VIX", ""),
}


# ---------------------------------------------------------------------------
# Fatti (la materia prima del perché)
# ---------------------------------------------------------------------------

def regime_facts(db: Session, on: date) -> dict:
    rc = (
        db.query(RegimeClassification)
        .filter(RegimeClassification.date <= on)
        .order_by(RegimeClassification.date.desc())
        .first()
    )
    if rc is None:
        return {}
    probs = {
        "reflation": rc.probability_reflation or 0.0,
        "stagflation": rc.probability_stagflation or 0.0,
        "deflation": rc.probability_deflation or 0.0,
        "goldilocks": rc.probability_goldilocks or 0.0,
    }
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])
    dom, alt = ranked[0], ranked[1]
    out = {
        "regime": dom[0],
        "regime_it": N.REGIME_IT.get(dom[0], dom[0]),
        "prob_regime_pct": round(dom[1] * 100, 1),
        "regime_alternativo_it": N.REGIME_IT.get(alt[0], alt[0]),
        "prob_alternativo_pct": round(alt[1] * 100, 1),
        "fiducia_pct": round((rc.confidence or 0.0) * 100, 0),
        "data_regime": str(rc.date),
    }
    try:
        payload = json.loads(rc.conditions_met or "{}")
    except Exception:
        payload = {}
    ind = payload.get("indicators") or {}
    out["indicatori"] = [
        {"nome": label, "valore": round(float(ind[k]), 2), "unita": unit}
        for k, (label, unit) in _INDICATORI.items() if isinstance(ind.get(k), (int, float))
    ]
    return out


def _thresholds() -> dict:
    from app.services.ai_portfolio import deploy_boost
    from app.services.ai_portfolio import strategist as S

    return {
        "soglia_ingresso": S.ENTRY_SCORE_THRESHOLD,
        "soglia_uscita": S.EXIT_SCORE_THRESHOLD,
        "stop_pct": round(S.STOP_LOSS_PCT * 100, 0),
        "tp_parziale_pct": round(S.PARTIAL_TP_PCT * 100, 0),
        "tp_pct": round(S.FULL_TP_PCT * 100, 0),
        "floor_pct": round(deploy_boost.DEPLOY_FLOOR * 100, 0),
    }


def _decision_for(db: Session, strategy: str, asset: str, actions: Iterable[str],
                  since: date | None = None) -> AiPortfolioDecision | None:
    q = db.query(AiPortfolioDecision).filter(
        AiPortfolioDecision.strategy_type == strategy,
        AiPortfolioDecision.asset_class == asset,
        AiPortfolioDecision.action.in_(list(actions)),
    )
    if since is not None:
        q = q.filter(AiPortfolioDecision.date >= since)
    return q.order_by(AiPortfolioDecision.date.desc(), AiPortfolioDecision.id.desc()).first()


def _facts(strategy: str, asset: str, kind: str, side: str, regime: dict,
           dec: AiPortfolioDecision | None, tranche: int, total: int, dw: float,
           pos: AiPortfolioPosition | None = None) -> dict:
    f = {
        "strategy": strategy,
        "strategia_nome": N.STRATEGY_IT.get(strategy, strategy),
        "asset": asset,
        "asset_nome": label_for(asset),
        "evento": kind,
        "lato": "acquisto" if side == "buy" else "vendita",
        "tranche": tranche,
        "tranche_totali": total,
        "peso_variazione_pct": round(dw * 100, 2),
        **{k: v for k, v in regime.items() if k != "regime"},
        **_thresholds(),
    }
    if dec is not None:
        f["punteggio"] = round(dec.score or 0.0, 1)
        f["momentum"] = N.MOMENTUM_IT.get(dec.momentum_signal or "", dec.momentum_signal)
        if dec.vol_60d:
            f["volatilita_60g_pct"] = round(dec.vol_60d * 100, 1)
        if dec.estimated_pnl_pct is not None and side == "sell":
            f["pnl_pct"] = round(dec.estimated_pnl_pct * 100, 1)
        f["motivo_del_bot"] = dec.reason_short
        f["data_decisione_bot"] = str(dec.date)
    if pos is not None and kind == "presa_in_carico":
        f["aperta_il"] = str(pos.opened_at)
        f["regime_di_ingresso_it"] = N.REGIME_IT.get(pos.entry_regime, pos.entry_regime)
        f["punteggio_medio_ingresso"] = round(pos.avg_entry_score or 0.0, 1)
    return f


# ---------------------------------------------------------------------------
# Stato del libro (ricavato SEMPRE dai movimenti)
# ---------------------------------------------------------------------------

def ledger_state(db: Session, strategy: str, initial: float, until: date | None = None) -> L.LedgerState:
    state = L.LedgerState(cash=initial)
    q = db.query(RbMovement).filter(RbMovement.strategy_type == strategy, RbMovement.status == "filled")
    if until is not None:
        q = q.filter(RbMovement.fill_date <= until)
    for mv in q.order_by(RbMovement.fill_date, RbMovement.id).all():
        if mv.side == "buy":
            L.apply_buy(state, mv.asset_class, mv.instrument_key,
                        L.Fill(int(mv.quantity), mv.fill_price_eur or 0.0, mv.gross_eur, mv.spread_eur, mv.fee_eur))
        elif mv.side == "sell" and mv.asset_class in state.lots:
            L.apply_sell(state, mv.asset_class,
                         L.Fill(int(mv.quantity), mv.fill_price_eur or 0.0, mv.gross_eur, mv.spread_eur, mv.fee_eur))
            L.replay_tax(state, mv.tax_eur or 0.0, mv.zainetto_used_eur or 0.0, mv.zainetto_added_eur or 0.0,
                         mv.fill_date.year)
        elif mv.side == "dividend":
            L.apply_dividend(state, mv.gross_eur, mv.tax_eur or 0.0)
    return state


def _book(db: Session, strategy: str) -> RbBook | None:
    return db.query(RbBook).filter(RbBook.strategy_type == strategy).first()


def _latest_bot_nav(db: Session, strategy: str, on: date | None = None) -> float | None:
    q = db.query(AiPortfolioPerformance).filter(AiPortfolioPerformance.strategy_type == strategy)
    if on is not None:
        q = q.filter(AiPortfolioPerformance.date <= on)
    row = q.order_by(AiPortfolioPerformance.date.desc()).first()
    return row.nav if row else None


# ---------------------------------------------------------------------------
# 1. Copia del bot → ordini in attesa
# ---------------------------------------------------------------------------

def mirror(db: Session, strategy: str, market: Market, now: datetime) -> list[RbMovement]:
    today = now.date()
    book = _book(db, strategy)
    first_run = book is None
    if first_run:
        book = RbBook(strategy_type=strategy, started_on=today,
                      initial_capital_eur=L.INITIAL_CAPITAL_EUR,
                      bot_nav_start=_latest_bot_nav(db, strategy, today))
        db.add(book)
        db.flush()

    bot = {p.asset_class: p for p in db.query(AiPortfolioPosition)
           .filter(AiPortfolioPosition.strategy_type == strategy).all()}
    snaps = {s.asset_class: s for s in db.query(RbBotSnapshot)
             .filter(RbBotSnapshot.strategy_type == strategy).all()}
    state = ledger_state(db, strategy, book.initial_capital_eur)
    pending = db.query(RbMovement).filter(RbMovement.strategy_type == strategy,
                                          RbMovement.status == "pending").all()

    prices: dict[str, float] = {}

    def est_price(key: str) -> float | None:
        if key not in prices:
            p = last_close_on_or_before(market, INSTRUMENTS[key].yahoo, today)
            if p:
                prices[key] = p
        return prices.get(key)

    for lot in state.lots.values():
        if lot.qty > 0:
            est_price(lot.instrument_key)
    nav_est, _ = L.nav(state, prices)
    avail = state.cash - sum(
        (mv.quantity * (mv.est_price_eur or 0.0) * (1 + INSTRUMENTS[mv.instrument_key].half_spread) + L.FEE_PER_ORDER_EUR)
        for mv in pending if mv.side == "buy"
    )
    regime = regime_facts(db, today)
    created: list[RbMovement] = []

    def new_mv(**kw) -> RbMovement:
        mv = RbMovement(strategy_type=strategy, decided_on=today, decided_at=now, **kw)
        db.add(mv)
        created.append(mv)
        return mv

    def sell(asset: str, kind: str, fraction: float | None, dec, tranche=0, total=0, dw=0.0):
        inst = instrument_for(asset)
        if inst is None:
            return
        for mv in pending:
            if mv.side == "buy" and mv.asset_class == asset and mv.status == "pending":
                mv.status = "cancelled"
                mv.note = "annullato prima dell'apertura: il bot ha cambiato idea"
        lot = state.lots.get(asset)
        qty = L.sell_qty(lot.qty if lot else 0, fraction)
        if qty <= 0:
            return
        facts = _facts(strategy, asset, kind, "sell", regime, dec, tranche, total, dw)
        new_mv(asset_class=asset, instrument_key=inst.key, side="sell", kind=kind,
               tranche_index=tranche, tranches_total=total, weight_delta_pct=dw,
               sell_fraction=fraction, quantity=qty, est_price_eur=est_price(inst.key),
               decision_id=dec.id if dec else None, facts_json=json.dumps(facts, ensure_ascii=False),
               narrative=N.why_template(facts), narrative_source="template")

    def buy(asset: str, kind: str, dw: float, dec, tranche: int, total: int, pos=None):
        nonlocal avail
        inst = instrument_for(asset)
        if inst is None:
            logger.warning(f"real_book: nessuno strumento per {asset}")
            return
        facts = _facts(strategy, asset, kind, "buy", regime, dec, tranche, total, dw, pos)
        p = est_price(inst.key)
        qty, note = (0, "prezzo non disponibile") if not p else L.size_buy(dw * nav_est, p, inst.half_spread, avail)
        if qty > 0 and p:
            avail -= qty * p * (1 + inst.half_spread) + L.FEE_PER_ORDER_EUR
        new_mv(asset_class=asset, instrument_key=inst.key, side="buy", kind=kind,
               status="pending" if qty > 0 else "skipped",
               tranche_index=tranche, tranches_total=total, weight_delta_pct=dw,
               quantity=qty, est_price_eur=p, decision_id=dec.id if dec else None,
               facts_json=json.dumps(facts, ensure_ascii=False),
               narrative=N.why_template(facts), narrative_source="template",
               exit_plan=N.exit_plan(facts), note=note)

    open_actions = ("OPEN_TRANCHE",)
    for asset in sorted(set(bot) | set(snaps)):
        pos, snap = bot.get(asset), snaps.get(asset)
        if snap is not None and (pos is None or pos.opened_at != snap.opened_at):
            dec = _decision_for(db, strategy, asset, _CLOSE_KIND, since=snap.last_action_date)
            sell(asset, _CLOSE_KIND.get(dec.action, "full_close") if dec else "riallineamento", None, dec,
                 dw=-snap.current_weight_pct)
            snap = None
        if pos is None:
            continue
        if snap is None:
            if first_run:
                kind, dec = "presa_in_carico", _decision_for(db, strategy, asset, open_actions)
            elif pos.last_action == "FLOOR_DEPLOY":
                kind, dec = "floor", None
            else:
                kind, dec = "tranche", _decision_for(db, strategy, asset, open_actions, since=pos.last_action_date)
            buy(asset, kind, pos.current_weight_pct, dec, pos.tranches_filled, pos.tranches_total, pos)
            continue
        dw = pos.current_weight_pct - snap.current_weight_pct
        if pos.tranches_filled > snap.tranches_filled and dw > _EPS_W:
            dec = _decision_for(db, strategy, asset, open_actions, since=pos.last_action_date)
            buy(asset, "tranche", dw, dec, pos.tranches_filled, pos.tranches_total)
        elif pos.tranches_filled < snap.tranches_filled and dw < -_EPS_W:
            dec = _decision_for(db, strategy, asset, ("CLOSE_TRANCHE",), since=snap.last_action_date)
            sell(asset, "close_tranche", 1 - pos.current_weight_pct / snap.current_weight_pct, dec,
                 pos.tranches_filled, pos.tranches_total, dw)
        elif dw > _EPS_W and pos.last_action == "FLOOR_DEPLOY":
            buy(asset, "floor", dw, None, pos.tranches_filled, pos.tranches_total)
        elif dw > _EPS_W:
            buy(asset, "riallineamento", dw, None, pos.tranches_filled, pos.tranches_total)
        elif dw < -_EPS_W and pos.last_action == "CAP_TRIM":
            dec = _decision_for(db, strategy, asset, ("CAP_TRIM",), since=pos.last_action_date)
            sell(asset, "cap_trim", 1 - pos.current_weight_pct / snap.current_weight_pct, dec,
                 pos.tranches_filled, pos.tranches_total, dw)
        elif dw < -_EPS_W:
            sell(asset, "riallineamento", 1 - pos.current_weight_pct / snap.current_weight_pct, None, dw=dw)

    # Rete di sicurezza: quote in portafoglio di un asset che il bot non ha (né aveva).
    for asset, lot in state.lots.items():
        if lot.qty > 0 and asset not in bot and asset not in snaps:
            sell(asset, "riallineamento", None, None)

    db.query(RbBotSnapshot).filter(RbBotSnapshot.strategy_type == strategy).delete()
    for p in bot.values():
        db.add(RbBotSnapshot(strategy_type=strategy, asset_class=p.asset_class,
                             current_weight_pct=p.current_weight_pct, tranches_filled=p.tranches_filled,
                             tranches_total=p.tranches_total, opened_at=p.opened_at,
                             last_action=p.last_action, last_action_date=p.last_action_date))
    db.flush()
    return created


# ---------------------------------------------------------------------------
# 2. Esecuzione all'apertura
# ---------------------------------------------------------------------------

def fill_pending(db: Session, strategy: str, market: Market, today: date) -> int:
    book = _book(db, strategy)
    if book is None:
        return 0
    pend = db.query(RbMovement).filter(RbMovement.strategy_type == strategy,
                                       RbMovement.status == "pending").all()
    ready = []
    for mv in pend:
        # deciso DOPO l'apertura di quel giorno ⇒ la prima apertura utile è il giorno dopo
        # (2026-10-22: un giro rilanciato a mano a metà mattina non può comprare al prezzo delle 09:00)
        start = mv.decided_on
        if mv.decided_at and mv.decided_at > open_time_utc(mv.decided_on):
            start = mv.decided_on + timedelta(days=1)
        bar = first_bar_on_or_after(market, INSTRUMENTS[mv.instrument_key].yahoo, start, today)
        if bar is not None:
            ready.append((bar, mv))
    # in ordine di seduta; nella stessa seduta prima le vendite (liberano cassa)
    ready.sort(key=lambda bm: (bm[0][0], 0 if bm[1].side == "sell" else 1, bm[1].id))
    n = 0
    for (d, o, _c), mv in ready:
        state = ledger_state(db, strategy, book.initial_capital_eur, until=d)
        inst = INSTRUMENTS[mv.instrument_key]
        if mv.side == "sell":
            lot = state.lots.get(mv.asset_class)
            q = L.sell_qty(lot.qty if lot else 0, mv.sell_fraction)
            f = L.execute_sell(q, o, inst.half_spread)
            if f is None:
                mv.status, mv.note = "cancelled", "niente da vendere all'apertura"
                continue
            lot, realized = L.apply_sell(state, mv.asset_class, f)
            mv.realized_pnl_eur = realized
            mv.tax_eur, mv.zainetto_used_eur, mv.zainetto_added_eur = L.tax_on_realized(
                state, realized, inst.tax_class, inst.tax_rate, d.year)
        else:
            f = L.execute_buy(int(mv.quantity), o, inst.half_spread, state.cash)
            if f is None:
                mv.status, mv.note = "skipped", "liquidità insufficiente all'apertura"
                continue
            lot = L.apply_buy(state, mv.asset_class, mv.instrument_key, f)
        mv.status = "filled"
        mv.quantity = f.qty
        mv.fill_date, mv.fill_at = d, open_time_utc(d)
        mv.fill_price_eur, mv.gross_eur, mv.fee_eur, mv.spread_eur = f.price, f.gross, f.fee, f.spread
        mv.cash_after_eur = state.cash
        mv.position_qty_after, mv.avg_price_after_eur = lot.qty, lot.avg_price
        if f.note:
            mv.note = f"{mv.note}; {f.note}" if mv.note else f.note
        opens = {mv.instrument_key: o}
        for other in state.lots.values():
            if other.qty > 0 and other.instrument_key not in opens:
                p = last_close_on_or_before(market, INSTRUMENTS[other.instrument_key].yahoo, d)
                if p:
                    opens[other.instrument_key] = p
        total, _ = L.nav(state, opens)
        mv.weight_after_pct = (lot.qty * o / total) if total > 0 else None
        db.flush()
        n += 1
    return n


# ---------------------------------------------------------------------------
# 3. Cedole
# ---------------------------------------------------------------------------

def _qty_at(db: Session, strategy: str, asset: str, before: date) -> float:
    q = 0.0
    for mv in db.query(RbMovement).filter(RbMovement.strategy_type == strategy,
                                          RbMovement.asset_class == asset,
                                          RbMovement.status == "filled",
                                          RbMovement.fill_date < before).all():
        if mv.side == "buy":
            q += mv.quantity
        elif mv.side == "sell":
            q -= mv.quantity
    return q


def credit_dividends(db: Session, strategy: str, market: Market, today: date, now: datetime) -> int:
    book = _book(db, strategy)
    if book is None:
        return 0
    held = {(mv.asset_class, mv.instrument_key) for mv in db.query(RbMovement).filter(
        RbMovement.strategy_type == strategy, RbMovement.side == "buy", RbMovement.status == "filled").all()}
    n = 0
    for asset, key in sorted(held):
        inst = INSTRUMENTS[key]
        if not inst.distributes:
            continue
        divs = market.dividends(inst.yahoo, book.started_on, today)
        for ex_date, per_share in divs.items():
            exists = db.query(RbMovement).filter(
                RbMovement.strategy_type == strategy, RbMovement.asset_class == asset,
                RbMovement.side == "dividend", RbMovement.fill_date == ex_date).first()
            if exists:
                continue
            qty = _qty_at(db, strategy, asset, ex_date)
            if qty <= 0:
                continue
            amount = qty * float(per_share)
            tax = amount * inst.tax_rate
            db.add(RbMovement(
                strategy_type=strategy, asset_class=asset, instrument_key=key, side="dividend",
                kind="dividend", status="filled", decided_on=ex_date, decided_at=now,
                fill_date=ex_date, fill_at=open_time_utc(ex_date), quantity=qty,
                fill_price_eur=float(per_share), gross_eur=amount, tax_eur=tax,
                narrative=(f"Stacco cedola su {label_for(asset)}: {N._n(amount, 2)} € lordi, "
                           f"{N._n(amount - tax, 2)} € netti in cassa dopo la ritenuta del "
                           f"{N._n(inst.tax_rate * 100, 1)}%."),
                narrative_source="template", facts_json="{}",
                note=f"{N._n(qty, 0)} quote × {N._n(float(per_share), 4)} € per quota",
            ))
            n += 1
    db.flush()
    return n


# ---------------------------------------------------------------------------
# 4. Valore del libro alla chiusura
# ---------------------------------------------------------------------------

def _start_open(market: Market, key: str, start: date, today: date) -> float | None:
    first = first_bar_on_or_after(market, INSTRUMENTS[key].yahoo, start, today)
    return first[1] if first else None


def _net_gain(g: float, rate: float) -> float:
    """Guadagno dopo le tasse se si vendesse oggi (una perdita resta perdita)."""
    return g * (1 - rate) if g > 0 else g


def benchmarks(market: Market, book: RbBook, today: date) -> dict[str, float | None]:
    """S&P 500, MSCI ACWI e 60/40 in euro, comprati all'apertura del primo giorno e tenuti.
    Netto = venduti oggi in regime amministrato (azioni 26%, Treasury 12,5%, cedole tassate)."""
    out: dict[str, float | None] = {k: None for k in (
        "sp_gross", "sp_net", "acwi_gross", "acwi_net", "6040_gross", "6040_net")}
    for attr, key in (("benchmark_start_price_eur", BENCHMARK_KEY), ("acwi_start_price_eur", BENCHMARK_ACWI_KEY),
                      ("bond_start_price_eur", BENCHMARK_BOND_KEY)):
        if getattr(book, attr) is None:
            setattr(book, attr, _start_open(market, key, book.started_on, today))

    def gross(key: str, start_px: float | None) -> float | None:
        c = close_on(market, INSTRUMENTS[key].yahoo, today) or last_close_on_or_before(market, INSTRUMENTS[key].yahoo, today)
        return (c / start_px - 1) if (c and start_px) else None

    sp, acwi = gross(BENCHMARK_KEY, book.benchmark_start_price_eur), gross(BENCHMARK_ACWI_KEY, book.acwi_start_price_eur)
    if sp is not None:
        out["sp_gross"], out["sp_net"] = sp, _net_gain(sp, 0.26)
    if acwi is not None:
        out["acwi_gross"], out["acwi_net"] = acwi, _net_gain(acwi, 0.26)
    bond = INSTRUMENTS[BENCHMARK_BOND_KEY]
    bond_price = gross(BENCHMARK_BOND_KEY, book.bond_start_price_eur)
    if sp is not None and bond_price is not None and book.bond_start_price_eur:
        divs = market.dividends(bond.yahoo, book.started_on, today)
        coupon = float(sum(v for d, v in divs.items() if d > book.started_on)) / book.bond_start_price_eur
        out["6040_gross"] = 0.6 * sp + 0.4 * (bond_price + coupon)
        out["6040_net"] = 0.6 * _net_gain(sp, 0.26) + 0.4 * (_net_gain(bond_price, bond.tax_rate)
                                                             + coupon * (1 - bond.tax_rate))
    return out


def snapshot_nav(db: Session, strategy: str, market: Market, today: date) -> RbNav | None:
    book = _book(db, strategy)
    bench = INSTRUMENTS[BENCHMARK_KEY]
    if book is None or close_on(market, bench.yahoo, today) is None:
        return None  # mercato chiuso o dato non ancora uscito: nessun valore inventato
    state = ledger_state(db, strategy, book.initial_capital_eur, until=today)
    prices: dict[str, float] = {}
    for lot in state.lots.values():
        if lot.qty > 0 and lot.instrument_key not in prices:
            yahoo = INSTRUMENTS[lot.instrument_key].yahoo
            p = close_on(market, yahoo, today) or last_close_on_or_before(market, yahoo, today)
            if p:
                prices[lot.instrument_key] = p
    total, invested = L.nav(state, prices)
    tax_info = {k: (i.tax_class, i.tax_rate) for k, i in INSTRUMENTS.items()}
    latent = L.latent_tax(state, prices, tax_info, today.year)
    bm = benchmarks(market, book, today)
    bot_now = _latest_bot_nav(db, strategy, today)
    bot_ret = (bot_now / book.bot_nav_start - 1) if (bot_now and book.bot_nav_start) else None

    row = db.query(RbNav).filter(RbNav.strategy_type == strategy, RbNav.date == today).first()
    if row is None:
        row = RbNav(strategy_type=strategy, date=today, nav_eur=total, cash_eur=state.cash, invested_eur=invested)
        db.add(row)
    row.nav_eur, row.cash_eur, row.invested_eur = total, state.cash, invested
    row.n_positions = sum(1 for lot in state.lots.values() if lot.qty > 0)
    row.return_gross_pct = (total + state.taxes) / book.initial_capital_eur - 1
    row.return_net_pct = (total - latent) / book.initial_capital_eur - 1
    row.taxes_cum_eur, row.latent_tax_eur = state.taxes, latent
    row.zainetto_eur = L.zainetto_available(state, today.year)
    row.bot_return_pct = bot_ret
    row.bench_sp_gross_pct, row.bench_sp_net_pct = bm["sp_gross"], bm["sp_net"]
    row.bench_acwi_gross_pct, row.bench_acwi_net_pct = bm["acwi_gross"], bm["acwi_net"]
    row.bench_6040_gross_pct, row.bench_6040_net_pct = bm["6040_gross"], bm["6040_net"]
    row.fees_cum_eur, row.spread_cum_eur, row.dividends_cum_eur = state.fees, state.spread, state.dividends
    positions = []
    for asset, lot in sorted(state.lots.items()):
        if lot.qty <= 0:
            continue
        p = prices.get(lot.instrument_key)
        value = lot.qty * p if p else lot.cost_eur
        positions.append({
            "asset_class": asset, "instrument_key": lot.instrument_key, "qty": lot.qty,
            "price_eur": p, "value_eur": value, "avg_price_eur": lot.avg_price,
            "pnl_pct": (p / lot.avg_price - 1) if (p and lot.avg_price) else None,
            "weight_pct": value / total if total > 0 else None,
        })
    row.positions_json = json.dumps(positions)
    db.flush()
    return row


# ---------------------------------------------------------------------------
# Giri dello scheduler
# ---------------------------------------------------------------------------

def _narrate(created: list[RbMovement]) -> None:
    from app.services.config_flags import use_real_book_llm_narrator

    todo = [mv for mv in created if mv.side in ("buy", "sell") and mv.narrative]
    if not todo:
        return
    if not use_real_book_llm_narrator():
        return
    items = [{"id": mv.id, "facts": json.loads(mv.facts_json), "draft": mv.narrative,
              "forbidden": N.forbidden_tokens(mv.instrument_key)} for mv in todo]
    out: dict[str, tuple[str, str]] = {}
    for i in range(0, len(items), 10):
        out.update(N.llm_why_batch(items[i:i + 10]))
    for mv in todo:
        text, source = out.get(str(mv.id), ("", "template (LLM non disponibile)"))
        if text:
            mv.narrative, mv.narrative_source = text, source
        else:
            mv.narrative_source = source


def run_after_scan(db: Session, market: Market | None = None, now: datetime | None = None,
                   narrate: bool = True) -> dict:
    market = market or YahooMarket()
    now = now or datetime.utcnow()
    summary = {}
    for strategy in STRATEGIES:
        try:
            created = mirror(db, strategy, market, now)
            filled = fill_pending(db, strategy, market, now.date())
            if narrate:
                _narrate(created)
            db.commit()
            summary[strategy] = {"nuovi_ordini": len(created), "eseguiti": filled}
        except Exception as e:
            db.rollback()
            logger.exception(f"real_book run_after_scan {strategy} fallito: {e}")
            summary[strategy] = {"errore": str(e)}
    return summary


def run_after_close(db: Session, market: Market | None = None, today: date | None = None,
                    now: datetime | None = None) -> dict:
    market = market or YahooMarket()
    now = now or datetime.utcnow()
    today = today or now.date()
    summary = {}
    for strategy in STRATEGIES:
        try:
            filled = fill_pending(db, strategy, market, today)
            divs = credit_dividends(db, strategy, market, today, now)
            row = snapshot_nav(db, strategy, market, today)
            db.commit()
            summary[strategy] = {"eseguiti": filled, "cedole": divs,
                                 "nav": round(row.nav_eur, 2) if row else None}
        except Exception as e:
            db.rollback()
            logger.exception(f"real_book run_after_close {strategy} fallito: {e}")
            summary[strategy] = {"errore": str(e)}
    return summary
