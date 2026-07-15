"""AI Portfolio Strategist — daily scan + decisioni autonome.

User request 2026-06-10: trigger combo regime + score + momentum (MA50+RSI).
Capitale $100k paper, tranche dinamiche, posizioni a 0 ammesse.

Logic per asset:
1. Se NO posizione attiva:
   - Skip se confidence < 0.30 (uncertainty gate)
   - Skip se score < 55 (entry threshold)
   - Skip se NOT in regime top-N (regime favorevole)
   - Skip se momentum != BULLISH
   - Apri tranche 1/N (size = target/N)
2. Se HAS posizione attiva:
   - FULL_CLOSE se score < 45 (signal collapse)
   - FULL_CLOSE se PNL < -10% (stop loss)
   - CLOSE_TRANCHE (partial TP) se PNL > +15% AND not yet took_partial_tp
   - FULL_CLOSE se PNL > +30% (full take profit)
   - FULL_CLOSE se regime shift adverse (out of top-N)
   - OPEN_TRANCHE incrementale se tranches_filled < tranches_total AND score > 60
   - HOLD altrimenti

Take profit/loss: simulati su `estimated_pnl_pct` computato come delta score
+ vol-weighted drift dal momento di entry. Realistic but coarse (no real price tracking).
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from app.models.ai_portfolio import (
    AiPortfolioDecision,
    AiPortfolioPosition,
)
from app.models.daily_signals import DailySignal
from app.models.regime_classifications import RegimeClassification
from app.services.ai_portfolio import sizer, momentum, pnl_tracker, learning, deploy_boost

STRATEGY_TYPE = "model_driven"
from app.services.prices.yahoo_fetcher import YahooFetcher
from app.services.scoring.engine import ASSET_CLASSES, ASSET_REGIME_DATA

logger = logging.getLogger(__name__)

# Thresholds (decisioni council sessione 26-06-10)
ENTRY_SCORE_THRESHOLD = 55.0
EXIT_SCORE_THRESHOLD = 45.0
INCREMENTAL_SCORE_THRESHOLD = 60.0
STOP_LOSS_PCT = -0.10
PARTIAL_TP_PCT = 0.15
FULL_TP_PCT = 0.30
TOP_N_REGIME = 12  # top-12 asset per regime corrente = "favorevole"


def _regime_top_n_assets(probabilities: dict[str, float], top_n: int = TOP_N_REGIME) -> set[str]:
    """Computa top-N asset per regime corrente usando ASSET_REGIME_DATA."""
    dominant_regime = max(probabilities, key=probabilities.get)
    rankings = []
    for asset, regime_data in ASSET_REGIME_DATA.items():
        stats = regime_data.get(dominant_regime, {})
        # Composite rank: hit_rate (50%) + sharpe (50%)
        score = stats.get("hit_rate", 0) * 0.5 + max(0, stats.get("sharpe", 0)) * 0.5
        rankings.append((asset, score))
    rankings.sort(key=lambda x: -x[1])
    return {a for a, _ in rankings[:top_n]}


def _refresh_real_pnl(position: AiPortfolioPosition) -> float:
    """Fase 2: PnL reale da Yahoo daily prices invece di stima.

    Strategia:
    - Se avg_entry_price assente (positions create pre-fase-2): cattura prezzo corrente
      e azzera PnL (start clean tracking da qui in poi)
    - Se presente: PnL = (current - entry) / entry
    """
    current_price = pnl_tracker.get_latest_price(position.asset_class)
    if current_price is None:
        return position.estimated_pnl_pct  # fallback: invariato

    if position.avg_entry_price is None or position.avg_entry_price == 0:
        # Backfill: posizioni pre-fase2 senza prezzo entry
        position.avg_entry_price = current_price
        position.current_price = current_price
        return 0.0

    position.current_price = current_price
    pnl = pnl_tracker.compute_pnl_pct(position.avg_entry_price, current_price)
    return pnl


def _fetch_price_history_for_universe() -> dict[str, pd.Series]:
    """Fetch 6m daily prices per tutti gli asset universe (Yahoo cache 24h)."""
    from datetime import timedelta

    fetcher = YahooFetcher()
    out: dict[str, pd.Series] = {}
    end = date.today()
    start = end - timedelta(days=200)  # 6.5 mesi per coprire 60d vol + 50d MA
    for asset in ASSET_CLASSES:
        ticker = momentum.asset_to_ticker(asset)
        if not ticker:
            continue
        try:
            s = fetcher.fetch(ticker, start_date=start, end_date=end)
            if s is not None and not s.empty:
                out[asset] = s.dropna()
        except Exception as e:
            logger.warning(f"price fetch failed for {asset}/{ticker}: {e}")
    return out


def daily_scan(
    db: Session,
    target_date: date | None = None,
) -> dict:
    """Esegue scan AI Portfolio: classifier latest → decisioni → DB updates.

    Returns:
        Summary dict: {n_decisions, n_open_tranches, n_closures, n_positions_after}
    """
    target_date = target_date or date.today()

    # 1. Latest regime + scores
    latest_regime = (
        db.query(RegimeClassification)
        .order_by(RegimeClassification.date.desc())
        .first()
    )
    if latest_regime is None:
        logger.warning("daily_scan: no regime classification in DB")
        return {"error": "no regime"}

    confidence = float(latest_regime.confidence or 0.5)
    probabilities = {
        "reflation": float(latest_regime.probability_reflation or 0),
        "stagflation": float(latest_regime.probability_stagflation or 0),
        "deflation": float(latest_regime.probability_deflation or 0),
        "goldilocks": float(latest_regime.probability_goldilocks or 0),
    }
    dominant_regime = max(probabilities, key=probabilities.get)

    signals = (
        db.query(DailySignal)
        .filter(DailySignal.date == latest_regime.date)
        .all()
    )
    scoreboard = {s.asset_class: float(s.final_score) for s in signals}

    # 2. Compute regime favorable top-N
    regime_top = _regime_top_n_assets(probabilities, top_n=TOP_N_REGIME)

    # 3. Fetch price history per universe
    logger.info("daily_scan: fetching prices universe...")
    price_history = _fetch_price_history_for_universe()

    # 4. Compute target weights raw per asset (positivi)
    raw_weights: dict[str, float] = {}
    asset_vols: dict[str, Optional[float]] = {}
    for asset in ASSET_CLASSES:
        score = scoreboard.get(asset, 0.0)
        sig, vol = momentum.get_asset_signal_and_vol(asset, price_history)
        asset_vols[asset] = vol
        # Solo asset con score>0 considerati per sizing
        raw_weights[asset] = sizer.raw_target_weight(score, vol) if asset in regime_top else 0.0

    targets_normalized = sizer.normalize_and_cap(raw_weights)
    # fastramp (deploy-boost): cap tranche a 2 quando il flag è ON
    n_tranches_default = deploy_boost.cap_n_tranches(sizer.dynamic_n_tranches(confidence))

    # 5. Load existing positions (filter per strategy) + PnL reale
    existing_positions = {}
    for p in (
        db.query(AiPortfolioPosition)
        .filter(AiPortfolioPosition.strategy_type == STRATEGY_TYPE)
        .all()
    ):
        p.estimated_pnl_pct = _refresh_real_pnl(p)
        existing_positions[p.asset_class] = p

    # 6. Decide per ogni asset
    summary = {"n_decisions": 0, "n_open_tranches": 0, "n_closures": 0}

    for asset in ASSET_CLASSES:
        score = scoreboard.get(asset, 0.0)
        target = targets_normalized.get(asset, 0.0)
        sig, vol = momentum.get_asset_signal_and_vol(asset, price_history)
        in_favor = asset in regime_top
        position = existing_positions.get(asset)

        # Fase 2: threshold dinamico da learning post-mortem
        threshold_shift = learning.get_threshold_shift_for_pattern(
            db, regime=dominant_regime, asset_class=asset, momentum=sig,
            strategy_type=STRATEGY_TYPE,
        )

        action, reason, size = _decide(
            asset=asset,
            position=position,
            score=score,
            target=target,
            confidence=confidence,
            momentum_sig=sig,
            in_favor=in_favor,
            n_tranches_default=n_tranches_default,
            vol=vol,
            entry_threshold_shift=threshold_shift,
        )

        # Log decision if non-trivial
        if action != "HOLD" or position is not None:
            decision = AiPortfolioDecision(
                date=target_date,
                asset_class=asset,
                strategy_type=STRATEGY_TYPE,
                action=action,
                size_pct=size,
                tranche_index=(position.tranches_filled + 1) if (position and action == "OPEN_TRANCHE") else (1 if action == "OPEN_TRANCHE" else 0),
                tranches_total=(position.tranches_total if position else n_tranches_default),
                score=score,
                confidence=confidence,
                regime=dominant_regime,
                momentum_signal=sig,
                vol_60d=vol,
                estimated_pnl_pct=(position.estimated_pnl_pct if position else None),
                reason_short=reason,
            )
            db.add(decision)
            summary["n_decisions"] += 1

        # Apply action to position
        _apply(db, asset, position, action, target, size, score, dominant_regime, n_tranches_default, target_date, summary)

    # floor (deploy-boost): se sotto il minimo, schiera il deficit sui top del regime
    deploy_boost.apply_deployment_floor(
        db, STRATEGY_TYPE, scoreboard, regime_top, target_date, dominant_regime,
    )

    db.commit()

    summary["n_positions_after"] = (
        db.query(AiPortfolioPosition)
        .filter(AiPortfolioPosition.strategy_type == STRATEGY_TYPE)
        .count()
    )
    return summary


def _decide(
    asset: str,
    position: Optional[AiPortfolioPosition],
    score: float,
    target: float,
    confidence: float,
    momentum_sig: str,
    in_favor: bool,
    n_tranches_default: int,
    vol: Optional[float],
    entry_threshold_shift: float = 0.0,
) -> tuple[str, str, float]:
    """Returns (action, reason_short, size_pct).

    Fase 2: entry_threshold_shift adatta dinamicamente la entry threshold in base
    ai learnings post-mortem accumulati per quel pattern.
    """
    effective_entry_threshold = ENTRY_SCORE_THRESHOLD + entry_threshold_shift
    if position is None:
        # Entry rules
        if confidence < 0.30:
            return ("SKIP_NO_SIGNAL", "confidence below uncertainty gate", 0.0)
        if score < effective_entry_threshold:
            shift_str = f" (learned shift {entry_threshold_shift:+.1f})" if entry_threshold_shift != 0 else ""
            return ("SKIP_NO_SIGNAL", f"score {score:.1f} < entry threshold {effective_entry_threshold:.1f}{shift_str}", 0.0)
        if not in_favor:
            return ("SKIP_NO_SIGNAL", "asset not in current regime top-12", 0.0)
        if momentum_sig != "BULLISH":
            return ("SKIP_NO_SIGNAL", f"momentum {momentum_sig} (need BULLISH)", 0.0)
        if target <= 0:
            return ("SKIP_NO_SIGNAL", "computed target weight 0", 0.0)
        if n_tranches_default <= 0:
            return ("SKIP_NO_SIGNAL", "n_tranches=0 (low confidence)", 0.0)
        # OPEN first tranche
        first_size = target / n_tranches_default
        return ("OPEN_TRANCHE", f"entry signal score {score:.1f} momentum BULLISH regime favorable", first_size)

    # Already has position
    pnl = position.estimated_pnl_pct

    if score < EXIT_SCORE_THRESHOLD:
        return ("FULL_CLOSE", f"score {score:.1f} < exit threshold (signal collapse)", 0.0)
    if pnl <= STOP_LOSS_PCT:
        return ("STOP_LOSS", f"stop loss PnL {pnl*100:.1f}% (max {STOP_LOSS_PCT*100:.0f}%)", 0.0)
    if pnl >= FULL_TP_PCT:
        return ("TAKE_PROFIT", f"full TP PnL {pnl*100:.1f}%", 0.0)
    if pnl >= PARTIAL_TP_PCT and not position.took_partial_tp:
        # Partial TP = chiudi 1 tranche
        return ("CLOSE_TRANCHE", f"partial TP PnL {pnl*100:.1f}%", 0.0)
    if not in_favor:
        return ("FULL_CLOSE", "regime shift adverse (out of top-12)", 0.0)

    # Incremental tranche — deadband (deploy-boost): soglia add = ENTRY quando ON
    incr_threshold = deploy_boost.incremental_threshold(INCREMENTAL_SCORE_THRESHOLD, ENTRY_SCORE_THRESHOLD)
    if position.tranches_filled < position.tranches_total and score >= incr_threshold:
        size = target / position.tranches_total
        return ("OPEN_TRANCHE", f"score {score:.1f} strong, add tranche {position.tranches_filled + 1}/{position.tranches_total}", size)

    return ("HOLD", "no trigger fired", 0.0)


def _apply(
    db: Session,
    asset: str,
    position: Optional[AiPortfolioPosition],
    action: str,
    target: float,
    size: float,
    score: float,
    regime: str,
    n_tranches_default: int,
    target_date: date,
    summary: dict,
) -> None:
    """Mutate DB based on action."""
    if action == "OPEN_TRANCHE":
        # Fase 2: cattura prezzo Yahoo all'entry
        entry_price = pnl_tracker.get_latest_price(asset)
        if position is None:
            new_pos = AiPortfolioPosition(
                asset_class=asset,
                strategy_type=STRATEGY_TYPE,
                target_weight_pct=target,
                current_weight_pct=size,
                tranches_filled=1,
                tranches_total=n_tranches_default,
                opened_at=target_date,
                entry_regime=regime,
                avg_entry_score=score,
                avg_entry_price=entry_price,
                current_price=entry_price,
                estimated_pnl_pct=0.0,
                took_partial_tp=False,
                last_action="OPEN_TRANCHE",
                last_action_date=target_date,
            )
            db.add(new_pos)
            summary["n_open_tranches"] += 1
        else:
            position.current_weight_pct += size
            tranches_before = position.tranches_filled
            position.tranches_filled += 1
            position.target_weight_pct = target
            # Update avg entry score weighted
            n = position.tranches_filled
            position.avg_entry_score = (position.avg_entry_score * (n - 1) + score) / n
            # Fase 2: VWAP entry price (tranche aggiuntiva)
            if entry_price is not None:
                position.avg_entry_price = pnl_tracker.update_avg_entry_price(
                    position.avg_entry_price, tranches_before, entry_price,
                )
                position.current_price = entry_price
            position.last_action = "OPEN_TRANCHE"
            position.last_action_date = target_date
            summary["n_open_tranches"] += 1

    elif action == "CLOSE_TRANCHE":
        if position is None:
            return
        per_tranche = position.current_weight_pct / max(position.tranches_filled, 1)
        position.current_weight_pct -= per_tranche
        position.tranches_filled = max(0, position.tranches_filled - 1)
        position.took_partial_tp = True
        position.last_action = "CLOSE_TRANCHE"
        position.last_action_date = target_date
        if position.current_weight_pct <= 0.001:
            db.delete(position)
            summary["n_closures"] += 1
        else:
            summary["n_closures"] += 1

    elif action in ("FULL_CLOSE", "STOP_LOSS", "TAKE_PROFIT"):
        if position is not None:
            # Fase 2: record post-mortem learning prima di cancellare
            try:
                # Costruisci stub decision per learning input
                last_decision = AiPortfolioDecision(
                    date=target_date,
                    asset_class=asset,
                    strategy_type=STRATEGY_TYPE,
                    action=action,
                    momentum_signal="NEUTRAL",  # fallback; learning usa entry_regime
                    score=score,
                    confidence=0.0,
                    regime=regime,
                    reason_short="auto-closed",
                )
                learning.record_closed_position(db, last_decision, position)
            except Exception as e:
                logger.warning(f"learning recording failed for {asset}: {e}")
            db.delete(position)
            summary["n_closures"] += 1
