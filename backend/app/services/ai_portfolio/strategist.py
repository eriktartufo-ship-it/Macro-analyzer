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
from app.services.ai_portfolio import sizer, momentum
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


def _estimate_pnl_update(
    position: AiPortfolioPosition,
    current_score: float,
    current_vol: Optional[float],
) -> float:
    """Aggiorna estimated PnL: drift = (current_score - entry_score)/100 * 0.5
    + vol-noise. Coarse but bounded.
    """
    score_delta = (current_score - position.avg_entry_score) / 100.0
    return position.estimated_pnl_pct + score_delta * 0.05  # 5% sensitivity per score delta


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
    n_tranches_default = sizer.dynamic_n_tranches(confidence)

    # 5. Load existing positions + aggiorna PnL stimato per ciascuna
    existing_positions = {}
    for p in db.query(AiPortfolioPosition).all():
        curr_score = scoreboard.get(p.asset_class, 0.0)
        curr_vol = None  # popolato dopo nella fase 6 via momentum.get_asset_signal_and_vol
        p.estimated_pnl_pct = _estimate_pnl_update(p, curr_score, curr_vol)
        existing_positions[p.asset_class] = p

    # 6. Decide per ogni asset
    summary = {"n_decisions": 0, "n_open_tranches": 0, "n_closures": 0}

    for asset in ASSET_CLASSES:
        score = scoreboard.get(asset, 0.0)
        target = targets_normalized.get(asset, 0.0)
        sig, vol = momentum.get_asset_signal_and_vol(asset, price_history)
        in_favor = asset in regime_top
        position = existing_positions.get(asset)

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
        )

        # Log decision if non-trivial
        if action != "HOLD" or position is not None:
            decision = AiPortfolioDecision(
                date=target_date,
                asset_class=asset,
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

    db.commit()

    summary["n_positions_after"] = db.query(AiPortfolioPosition).count()
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
) -> tuple[str, str, float]:
    """Returns (action, reason_short, size_pct)."""
    if position is None:
        # Entry rules
        if confidence < 0.30:
            return ("SKIP_NO_SIGNAL", "confidence below uncertainty gate", 0.0)
        if score < ENTRY_SCORE_THRESHOLD:
            return ("SKIP_NO_SIGNAL", f"score {score:.1f} < entry threshold {ENTRY_SCORE_THRESHOLD}", 0.0)
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

    # Incremental tranche
    if position.tranches_filled < position.tranches_total and score >= INCREMENTAL_SCORE_THRESHOLD:
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
        if position is None:
            new_pos = AiPortfolioPosition(
                asset_class=asset,
                target_weight_pct=target,
                current_weight_pct=size,
                tranches_filled=1,
                tranches_total=n_tranches_default,
                opened_at=target_date,
                entry_regime=regime,
                avg_entry_score=score,
                estimated_pnl_pct=0.0,
                took_partial_tp=False,
                last_action="OPEN_TRANCHE",
                last_action_date=target_date,
            )
            db.add(new_pos)
            summary["n_open_tranches"] += 1
        else:
            position.current_weight_pct += size
            position.tranches_filled += 1
            position.target_weight_pct = target
            # Update avg entry score weighted
            n = position.tranches_filled
            position.avg_entry_score = (position.avg_entry_score * (n - 1) + score) / n
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
            db.delete(position)
            summary["n_closures"] += 1
