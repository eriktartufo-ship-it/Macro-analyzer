"""Bot "S&P a rischio gestito" — 3a strategia dell'horserace (2026-07-15).

RICHIESTA ERIK: "un trader che si concentra nel battere s&p500 smorzando le discese,
stando più aggressivo capendo dove stanno andando i soldi".

⚠️ PROMESSA ONESTA (il nome NON è "batti-S&P", e c'è un motivo misurato):
`scripts/sp_trend_edge.py` (5 seed × 40 sim, dati reali, costi 10bps), RIFATTO il 2026-10-21 sull'S&P
500 vero (SPY dal 1993, ^GSPC prima). Fino a quel giorno il bot e il suo backtest usavano
`us_equities_growth` = QQQ, cioè il Nasdaq-100, etichettato «SPY»: i numeri vecchi erano del Nasdaq.

  configurazione dal vivo = MA200 + rifugio (il migliore fra Treasury lunghi, oro, cash)
              | ret ampia | α vs S&P        | DD ampia | 2008: α / DD          | 2020 α  | 2022 α
  S&P b&h     |  +10.8%   |      —          | −14.2%   |    —  / −43.3%        |   —     |   —
  MA200→cash  |   +7.2%   | −3.6 ± 1.5pp    | −10.2%   | +24.2pp / −6.6%       | −5.9pp  | −5.5pp
  MA200→rifug |   +8.5%   | −2.4 ± 1.9pp    | −12.6%   | +29.3pp / −20.6%      | −9.6pp  | −6.2pp

È un'ASSICURAZIONE: paga ~2-4 punti l'anno nei periodi normali e incassa quando arriva un 2008.
Perde nelle V-recovery (2020: esce dopo il crollo e rientra tardi). ⚠️ Il 2022, che sul Nasdaq
sembrava a favore (+2/+4pp), sull'S&P è a sfavore. Variante non adottata, solo misurata: rifugio con
conferma a 5 giorni = +0,1 ± 2,4pp sull'ampia e +33pp nel 2008 (cambiarla è una decisione di Erik).

Perché esiste comunque: è un profilo di rischio DIVERSO dal model_driven (che è
macro-difensivo e schiera ~35%). Qui si sta sul beta azionario e si esce solo
quando il trend gira.

Meccanica:
  - SPY sopra la MA(window) → 100% us_equities_sp500 (S&P 500);
  - sotto → REFUGE_CASH: 100% cash · REFUGE_BEST: il migliore fra bond/oro/cash per
    momentum trailing (= "seguire dove si spostano i soldi", intuizione Erik: nel
    2022 ha aggiunto +4.0 vs +2.2 del solo cash, ma con DD peggiore).
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from app.models.ai_portfolio import AiPortfolioDecision, AiPortfolioPosition
from app.models.regime_classifications import RegimeClassification
from app.services.ai_portfolio import pnl_tracker
from app.services.ai_portfolio.market_calendar import last_weekday_before
from app.services.prices.yahoo_fetcher import YahooFetcher

logger = logging.getLogger(__name__)

STRATEGY_TYPE = "risk_managed_sp"

# 2026-10-21: era "us_equities_growth" (QQQ = Nasdaq-100) per un errore di etichetta che stava
# anche nel backtest. Ora è l'S&P 500 vero (SPY), fuori dall'universo dello scoring.
SP_ASSET = "us_equities_sp500"
CASH_ASSET = "cash_money_market"
REFUGE_CANDIDATES = ("us_bonds_long", "gold", "cash_money_market")

TREND_WINDOW = 200          # MA200: broad −3.2pp vs MA100 −7.5pp (MA100 troppo nervosa)
CONFIRM_DAYS = 1            # conferma anti-whipsaw (5g: broad −2.5 ma 2008 +10.3 vs +12.3)
REFUGE_LOOKBACK = 63
USE_BEST_REFUGE = True      # False = solo cash (protegge di più, rende di meno)


# ---------------------------------------------------------------------------
# Logica PURA (testabile senza rete/DB)
# ---------------------------------------------------------------------------

def is_above_trend(prices: pd.Series, window: int = TREND_WINDOW) -> Optional[bool]:
    """Il prezzo è sopra la sua MA(window)? None se storia insufficiente."""
    s = prices.dropna()
    if len(s) < window:
        return None
    ma = s.tail(window).mean()
    if pd.isna(ma):
        return None
    return float(s.iloc[-1]) > float(ma)


def pick_refuge(
    price_history: dict[str, pd.Series],
    lookback: int = REFUGE_LOOKBACK,
    use_best: bool = USE_BEST_REFUGE,
) -> str:
    """Dove andare quando l'S&P è sotto trend.

    use_best=False → sempre cash (massima protezione del DD).
    use_best=True  → il migliore per momentum fra bond/oro/cash: è l'intuizione di
    Erik ("i soldi si spostano da qualche altra parte") resa operativa. Rende di
    più ma protegge meno (2008 DD −24.7% vs −13.9% del cash).
    """
    if not use_best:
        return CASH_ASSET
    best, best_r = CASH_ASSET, None
    for a in REFUGE_CANDIDATES:
        s = price_history.get(a)
        if s is None or len(s.dropna()) < lookback + 1:
            continue
        s = s.dropna()
        r = float(s.iloc[-1] / s.iloc[-(lookback + 1)] - 1.0)
        if best_r is None or r > best_r:
            best, best_r = a, r
    return best


def decide_target(
    price_history: dict[str, pd.Series],
    window: int = TREND_WINDOW,
    use_best_refuge: bool = USE_BEST_REFUGE,
) -> tuple[Optional[str], str]:
    """Ritorna (asset_target, motivo). `None` = DATI INSUFFICIENTI, non decidere.

    ⚠️ Il target None NON è "vai in cash": è "non lo so". Sono cose diverse e
    confonderle è un bug (bugtracker 2026-07-15: il primo scan in prod prese una
    serie corta, cadde su "cash prudenziale" e sembrava una scelta di de-risk
    mentre era un dato mancante). Su None il chiamante deve SALTARE il giorno e
    lasciare le posizioni invariate, non aprire cash.
    """
    sp = price_history.get(SP_ASSET)
    if sp is None or sp.empty:
        return (None, "prezzi S&P non disponibili -> nessuna decisione (giorno saltato)")
    above = is_above_trend(sp, window)
    if above is None:
        n = len(sp.dropna())
        return (None, f"storia insufficiente per MA{window} ({n} giorni) "
                      f"-> nessuna decisione (giorno saltato)")
    if above:
        return (SP_ASSET, f"S&P sopra MA{window}: resto sul beta azionario")
    refuge = pick_refuge(price_history, use_best=use_best_refuge)
    return (refuge, f"S&P sotto MA{window} -> de-risk su {refuge}")


# ---------------------------------------------------------------------------
# Daily scan (stessa interfaccia degli altri strategist)
# ---------------------------------------------------------------------------

def _fetch_prices() -> dict[str, pd.Series]:
    """Serve solo l'S&P + i candidati rifugio: 4 fetch invece di 24.

    Finestra 3 anni: la MA200 richiede 200 giorni di BORSA (~290 di calendario) e
    una finestra stretta lascia zero margine per festivi/buchi/dati parziali. In
    prod il primo scan (2026-07-15) prese una serie corta e il bot cadde nel
    fallback "cash prudenziale" — che sembrava una decisione ma era un dato mancante.
    """
    fetcher = YahooFetcher()
    end = date.today()
    start = end - timedelta(days=3 * 365)
    out: dict[str, pd.Series] = {}
    from app.services.ai_portfolio import momentum as _m
    for asset in (SP_ASSET, *REFUGE_CANDIDATES):
        ticker = _m.asset_to_ticker(asset)
        if not ticker:
            continue
        try:
            # 2026-10-31: l'ultima seduta deve esserci (la cache delle 00:00 poteva non averla)
            s = fetcher.fetch(ticker, start_date=start, end_date=end, need_through=last_weekday_before(end))
            if s is not None and not s.empty:
                out[asset] = s.dropna()
        except Exception as e:
            logger.warning(f"risk_managed_sp price fetch failed {asset}: {e}")
    return out


def daily_scan(db: Session, target_date: date | None = None) -> dict:
    """Tiene UNA posizione al 100%: S&P se sopra trend, altrimenti il rifugio."""
    target_date = target_date or date.today()

    latest = (
        db.query(RegimeClassification)
        .order_by(RegimeClassification.date.desc())
        .first()
    )
    regime = latest.regime if latest else "unknown"

    price_history = _fetch_prices()
    target_asset, reason = decide_target(price_history)

    # Dati insufficienti != "vai in cash": non si scommette su un dato mancante.
    # Si salta il giorno lasciando le posizioni invariate (fail-safe, non fail-silent).
    if target_asset is None:
        logger.warning(f"risk_managed_sp: {reason}")
        return {"skipped": reason, "n_decisions": 0,
                "n_open_tranches": 0, "n_closures": 0}

    positions = (
        db.query(AiPortfolioPosition)
        .filter(AiPortfolioPosition.strategy_type == STRATEGY_TYPE)
        .all()
    )
    summary = {"n_decisions": 0, "n_open_tranches": 0, "n_closures": 0,
               "target": target_asset, "reason": reason}

    current = positions[0] if positions else None
    if current is not None:
        px = pnl_tracker.get_latest_price(current.asset_class)
        if px is not None and current.avg_entry_price:
            current.current_price = px
            current.estimated_pnl_pct = pnl_tracker.compute_pnl_pct(
                current.avg_entry_price, px)

    if current is not None and current.asset_class == target_asset:
        db.add(AiPortfolioDecision(
            date=target_date, asset_class=target_asset, strategy_type=STRATEGY_TYPE,
            action="HOLD", size_pct=0.0, tranche_index=0, tranches_total=1,
            score=50.0, confidence=1.0, regime=regime,
            momentum_signal="BULLISH" if target_asset == SP_ASSET else "NEUTRAL",
            estimated_pnl_pct=current.estimated_pnl_pct, reason_short=reason,
        ))
        summary["n_decisions"] += 1
        db.commit()
        summary["n_positions_after"] = 1
        return summary

    # switch: chiudi la vecchia, apri la nuova al 100%
    for p in positions:
        db.add(AiPortfolioDecision(
            date=target_date, asset_class=p.asset_class, strategy_type=STRATEGY_TYPE,
            action="FULL_CLOSE", size_pct=0.0, tranche_index=0, tranches_total=1,
            score=50.0, confidence=1.0, regime=regime, momentum_signal="NEUTRAL",
            estimated_pnl_pct=p.estimated_pnl_pct,
            reason_short=f"switch -> {target_asset}: {reason}",
        ))
        db.delete(p)
        summary["n_closures"] += 1
        summary["n_decisions"] += 1

    entry_price = pnl_tracker.get_latest_price(target_asset)
    db.add(AiPortfolioPosition(
        asset_class=target_asset, strategy_type=STRATEGY_TYPE,
        target_weight_pct=1.0, current_weight_pct=1.0,
        tranches_filled=1, tranches_total=1,
        opened_at=target_date, entry_regime=regime, avg_entry_score=50.0,
        avg_entry_price=entry_price, current_price=entry_price,
        estimated_pnl_pct=0.0, took_partial_tp=False,
        last_action="OPEN_TRANCHE", last_action_date=target_date,
    ))
    db.add(AiPortfolioDecision(
        date=target_date, asset_class=target_asset, strategy_type=STRATEGY_TYPE,
        action="OPEN_TRANCHE", size_pct=1.0, tranche_index=1, tranches_total=1,
        score=50.0, confidence=1.0, regime=regime,
        momentum_signal="BULLISH" if target_asset == SP_ASSET else "NEUTRAL",
        reason_short=reason,
    ))
    summary["n_open_tranches"] += 1
    summary["n_decisions"] += 1
    db.commit()
    summary["n_positions_after"] = 1
    return summary
