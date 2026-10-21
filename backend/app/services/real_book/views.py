"""Cosa vede il sito del libro reale: pubblico per asset class, dettaglio strumento solo admin.

Scelta Erik 2026-10-21: il diario è pubblico ma SENZA strumenti. Un «ho comprato X perché…»
con ticker e ISIN su un sito pubblico può configurarsi come raccomandazione d'investimento
(regolamento MAR). Quindi nome, ticker, ISIN, quantità e prezzo unitario (che da solo
identifica lo strumento) escono solo con il token admin. Il perché è scritto per asset class
già alla fonte (template e guardia del narratore), quindi vale per entrambe le viste.
"""
from __future__ import annotations

import json
from datetime import date

from sqlalchemy.orm import Session

from app.models.real_book import RbBook, RbBotSnapshot, RbMovement, RbNav
from app.services.real_book.book import STRATEGIES
from app.services.real_book.instruments import INSTRUMENTS, label_for

KIND_IT = {
    "tranche": "Tranche",
    "presa_in_carico": "Presa in carico",
    "floor": "Investimento minimo",
    "close_tranche": "Presa di profitto parziale",
    "full_close": "Chiusura",
    "stop_loss": "Stop loss",
    "take_profit": "Presa di profitto",
    "riallineamento": "Riallineamento",
    "dividend": "Cedola",
}
STATUS_IT = {"pending": "in attesa dell'apertura", "filled": "eseguito",
             "cancelled": "annullato", "skipped": "non eseguito"}


def _instrument(key: str) -> dict | None:
    inst = INSTRUMENTS.get(key)
    if inst is None:
        return None
    return {"key": inst.key, "name": inst.name, "ticker": inst.ticker, "isin": inst.isin,
            "exchange": inst.exchange, "kind": inst.kind}


def movement_view(mv: RbMovement, admin: bool) -> dict:
    cost = None
    if mv.realized_pnl_eur is not None and mv.gross_eur:
        cost = mv.gross_eur - mv.fee_eur - mv.realized_pnl_eur
    out = {
        "id": mv.id,
        "strategy": mv.strategy_type,
        "asset_class": mv.asset_class,
        "asset_label": label_for(mv.asset_class),
        "side": mv.side,
        "kind": mv.kind,
        "kind_label": KIND_IT.get(mv.kind, mv.kind),
        "status": mv.status,
        "status_label": STATUS_IT.get(mv.status, mv.status),
        "tranche_index": mv.tranche_index,
        "tranches_total": mv.tranches_total,
        "decided_on": str(mv.decided_on),
        "decided_at": mv.decided_at.isoformat() + "Z" if mv.decided_at else None,
        "fill_date": str(mv.fill_date) if mv.fill_date else None,
        "fill_at": mv.fill_at.isoformat() + "Z" if mv.fill_at else None,
        "gross_eur": mv.gross_eur,
        "fee_eur": mv.fee_eur,
        "spread_eur": mv.spread_eur,
        "weight_after_pct": mv.weight_after_pct,
        "realized_pnl_eur": mv.realized_pnl_eur,
        "realized_pnl_pct": (mv.realized_pnl_eur / cost) if cost else None,
        "narrative": mv.narrative,
        "narrative_source": mv.narrative_source,
        "exit_plan": mv.exit_plan,
    }
    if admin:
        out.update({
            "instrument": _instrument(mv.instrument_key),
            "quantity": mv.quantity,
            "est_price_eur": mv.est_price_eur,
            "fill_price_eur": mv.fill_price_eur,
            "avg_price_after_eur": mv.avg_price_after_eur,
            "position_qty_after": mv.position_qty_after,
            "cash_after_eur": mv.cash_after_eur,
            "note": mv.note,
        })
    return out


def journal(db: Session, strategy: str | None, limit: int, admin: bool) -> list[dict]:
    q = db.query(RbMovement)
    if strategy:
        q = q.filter(RbMovement.strategy_type == strategy)
    rows = q.order_by(RbMovement.decided_at.desc(), RbMovement.id.desc()).limit(limit).all()
    return [movement_view(mv, admin) for mv in rows]


def _concentration(positions: list[dict]) -> list[dict]:
    """Strumenti su cui il libro è arrivato da più asset class (es. energy + sector_energy)."""
    by_inst: dict[str, list[dict]] = {}
    for p in positions:
        by_inst.setdefault(p["instrument_key"], []).append(p)
    out = []
    for key, ps in by_inst.items():
        if len(ps) > 1:
            out.append({
                "assets": [label_for(p["asset_class"]) for p in ps],
                "weight_pct": sum(p.get("weight_pct") or 0 for p in ps),
                "instrument_key": key,
            })
    return out


def overview(db: Session, admin: bool) -> dict:
    books = {b.strategy_type: b for b in db.query(RbBook).all()}
    result = {}
    for strategy in STRATEGIES:
        book = books.get(strategy)
        if book is None:
            result[strategy] = None
            continue
        nav = (db.query(RbNav).filter(RbNav.strategy_type == strategy)
               .order_by(RbNav.date.desc()).first())
        snaps = {s.asset_class: s for s in db.query(RbBotSnapshot)
                 .filter(RbBotSnapshot.strategy_type == strategy).all()}
        positions = json.loads(nav.positions_json) if nav else []
        conc = _concentration(positions)
        pos_view = []
        for p in positions:
            snap = snaps.get(p["asset_class"])
            v = {
                "asset_class": p["asset_class"],
                "asset_label": label_for(p["asset_class"]),
                "value_eur": p.get("value_eur"),
                "weight_pct": p.get("weight_pct"),
                "pnl_pct": p.get("pnl_pct"),
                "tranches_filled": snap.tranches_filled if snap else None,
                "tranches_total": snap.tranches_total if snap else None,
            }
            if admin:
                v.update({"instrument": _instrument(p["instrument_key"]), "qty": p.get("qty"),
                          "price_eur": p.get("price_eur"), "avg_price_eur": p.get("avg_price_eur")})
            pos_view.append(v)
        if not admin:
            for c in conc:
                c.pop("instrument_key", None)
        pending = db.query(RbMovement).filter(RbMovement.strategy_type == strategy,
                                              RbMovement.status == "pending").count()
        result[strategy] = {
            "started_on": str(book.started_on),
            "initial_capital_eur": book.initial_capital_eur,
            "nav": None if nav is None else {
                "date": str(nav.date), "nav_eur": nav.nav_eur, "cash_eur": nav.cash_eur,
                "invested_eur": nav.invested_eur, "cumulative_return_pct": nav.cumulative_return_pct,
                "bot_return_pct": nav.bot_return_pct, "benchmark_return_pct": nav.benchmark_return_pct,
                "reality_gap_pct": (nav.cumulative_return_pct - nav.bot_return_pct)
                if nav.bot_return_pct is not None else None,
                "fees_cum_eur": nav.fees_cum_eur, "spread_cum_eur": nav.spread_cum_eur,
                "dividends_cum_eur": nav.dividends_cum_eur, "n_positions": nav.n_positions,
            },
            "positions": pos_view,
            "concentration": conc,
            "pending_orders": pending,
        }
    return {"strategies": result, "admin": admin, "benchmark_label": "S&P 500 in euro"}


def nav_series(db: Session, strategy: str, days: int) -> list[dict]:
    from datetime import timedelta

    since = date.today() - timedelta(days=days)
    rows = (db.query(RbNav).filter(RbNav.strategy_type == strategy, RbNav.date >= since)
            .order_by(RbNav.date).all())
    return [{"date": str(r.date), "nav_eur": r.nav_eur, "cumulative_return_pct": r.cumulative_return_pct,
             "bot_return_pct": r.bot_return_pct, "benchmark_return_pct": r.benchmark_return_pct} for r in rows]
