"""Tetto per STRUMENTO, non solo per asset (2026-10-21).

Il difetto: `energy` e `sector_energy` sono due asset dello scoring con lo stesso prezzo (XLE).
Il tetto del 25% valeva per asset, quindi il bot Modello arrivava a 9,1% + 20,8% = 29,8% sullo
stesso ETF senza vederlo. L'ha fatto emergere il libro reale, dove i due asset diventano lo
stesso strumento.

La guardia sta sull'ESITO, a fine giro, e vale per ogni bot qualunque strada abbia preso
(ingresso, tranche, pavimento di investimento): nessuno strumento sopra il tetto. Se succede, i
membri del gruppo si riducono in proporzione e ogni riduzione è una decisione `CAP_TRIM` col suo
motivo, così il libro reale la copia come vendita spiegata.
"""
from __future__ import annotations

from datetime import date

from loguru import logger
from sqlalchemy.orm import Session

from app.models.ai_portfolio import AiPortfolioDecision, AiPortfolioPosition
from app.services.ai_portfolio.sizer import WEIGHT_CAP_MAX


def instrument_groups() -> dict[str, list[str]]:
    """Ticker → asset che lo condividono (solo i gruppi con almeno due asset)."""
    from app.services.prices.asset_universe import ASSET_TICKERS

    by: dict[str, list[str]] = {}
    for asset, cfg in ASSET_TICKERS.items():
        if cfg.get("ticker"):
            by.setdefault(str(cfg["ticker"]), []).append(asset)
    return {t: sorted(a) for t, a in by.items() if len(a) > 1}


def enforce_instrument_cap(db: Session, strategy_type: str, target_date: date, regime: str,
                           cap: float = WEIGHT_CAP_MAX) -> int:
    """Riduce i gruppi oltre il tetto. Ritorna quante posizioni ha ridotto."""
    db.flush()  # le posizioni aperte in questo giro devono esserci
    positions = {p.asset_class: p for p in db.query(AiPortfolioPosition)
                 .filter(AiPortfolioPosition.strategy_type == strategy_type).all()}
    n = 0
    for ticker, members in instrument_groups().items():
        held = [positions[a] for a in members if a in positions and positions[a].current_weight_pct > 0]
        total = sum(p.current_weight_pct for p in held)
        if len(held) < 2 or total <= cap + 1e-6:
            continue
        scale = cap / total
        nomi = " + ".join(p.asset_class for p in held)
        for p in held:
            new_w = p.current_weight_pct * scale
            cut = p.current_weight_pct - new_w
            db.add(AiPortfolioDecision(
                date=target_date, asset_class=p.asset_class, strategy_type=strategy_type,
                action="CAP_TRIM", size_pct=cut, tranche_index=0, tranches_total=p.tranches_total,
                score=p.avg_entry_score, confidence=0.0, regime=regime, momentum_signal="NEUTRAL",
                estimated_pnl_pct=p.estimated_pnl_pct,
                reason_short=(f"tetto per strumento: {nomi} sono lo stesso ETF ({ticker}), insieme "
                              f"{total * 100:.1f}% > {cap * 100:.0f}%: ridotta da {p.current_weight_pct * 100:.1f}% "
                              f"a {new_w * 100:.1f}%"),
            ))
            p.current_weight_pct = new_w
            p.target_weight_pct = min(p.target_weight_pct, new_w)
            p.last_action = "CAP_TRIM"
            p.last_action_date = target_date
            n += 1
        logger.info(f"instrument_cap[{strategy_type}]: {nomi} ({ticker}) {total:.1%} → {cap:.0%}")
    return n
