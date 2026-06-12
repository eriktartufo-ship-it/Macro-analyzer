"""Data-Driven Strategist — decisioni rule-based su raw indicators FRED.

Filosofia: NON usa il classifier regime né lo scoreboard ASSET_REGIME_DATA.
Decide direttamente da soglie hardcoded sugli indicators e i loro ROC/Z-score.

Trigger types (council 2026-06-12):
- RISK_OFF: VIX z>2σ, HY_OAS>6, yield_curve<-0.5 → defensive (gold, bonds_long, cash)
- RISK_ON: M2_yoy>5%, NFCI<-0.2, VIX z<-1σ → growth (equities, sector_tech)
- INFLATION_HEDGE: CPI_yoy>4 + CPI_chg_6m>0.5 + wage>4% → sector_energy, gold, TIPS
- DEFLATION_HEDGE: GDP_roc<0, claims_rising, breakeven<1.5 → bonds_long, cash, gold
- HOUSING_BOOM: building_permits_yoy>+10, mortgage_rates_falling → REITs, sector_financials
- INDUSTRIAL_EXPANSION: heavy_truck_yoy>+5, durable_goods>+5, indpro>+2 → sector_industrials, factor_value

Asset scoring data-driven:
Per ogni asset, sommiamo "bonus" da trigger fired che lo favoriscono e "malus"
da trigger fired che lo penalizzano. Score 0-100 risultante.

Stesso scheletro strategist.py: top-N regime favorable, entry/exit triggers,
DCA tranche, stop loss/take profit. Differenza = source of truth dei scores.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from app.models.ai_portfolio import AiPortfolioDecision, AiPortfolioPosition
from app.models.regime_classifications import RegimeClassification
from app.services.ai_portfolio import learning, momentum, pnl_tracker, sizer
from app.services.scoring.engine import ASSET_CLASSES

logger = logging.getLogger(__name__)


STRATEGY_TYPE = "data_driven"
ENTRY_SCORE_THRESHOLD = 55.0
EXIT_SCORE_THRESHOLD = 45.0
INCREMENTAL_SCORE_THRESHOLD = 60.0
STOP_LOSS_PCT = -0.10
PARTIAL_TP_PCT = 0.15
FULL_TP_PCT = 0.30


# ============================================================================
# RULE ENGINE — soglie hardcoded da statistiche storiche FRED 1990-2024
# ============================================================================

def _safe(indicators: dict, key: str, default: float = 0.0) -> float:
    v = indicators.get(key)
    if v is None:
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


# Asset groups per trigger mapping
ASSET_GROUPS = {
    "DEFENSIVE": {"us_bonds_long", "us_bonds_short", "tips_inflation_bonds", "gold",
                  "silver", "cash_money_market", "sector_utilities", "sector_staples",
                  "sector_healthcare"},
    "GROWTH_EQUITIES": {"us_equities_growth", "sector_technology", "factor_momentum",
                       "factor_quality", "em_equities"},
    "VALUE_EQUITIES": {"us_equities_value", "factor_value", "sector_financials"},
    "INFLATION_HEDGE": {"gold", "silver", "broad_commodities", "energy",
                       "sector_energy", "tips_inflation_bonds", "real_estate_reits"},
    "DEFLATION_HEDGE": {"us_bonds_long", "us_bonds_short", "cash_money_market", "gold"},
    "CYCLICAL": {"sector_energy", "sector_financials", "sector_technology",
                 "factor_momentum", "factor_value", "us_equities_value",
                 "real_estate_reits"},
    "CRYPTO_RISK_ON": {"bitcoin", "crypto_broad"},
    "INTERNATIONAL": {"international_dm_equities", "em_equities"},
}


def evaluate_data_driven_scores(
    indicators: dict[str, float],
) -> dict[str, float]:
    """Compute asset score 0-100 from raw indicators only.

    Logic: each asset starts at neutral 50. Triggers fired apply +/- bonus.
    """
    # Extract raw indicators with safe defaults
    vix = _safe(indicators, "vix", 18.0)
    nfci = _safe(indicators, "nfci", 0.0)
    hy_oas = _safe(indicators, "hy_credit_spread", 4.0)
    cpi = _safe(indicators, "cpi_yoy", 2.5)
    cpi_chg_6m = _safe(indicators, "cpi_yoy_change_6m", 0.0)
    core_pce = _safe(indicators, "core_pce_yoy", 2.0)
    gdp_roc = _safe(indicators, "gdp_roc", 1.0)
    gdp_chg_6m = _safe(indicators, "gdp_roc_change_6m", 0.0)
    yield_curve = _safe(indicators, "yield_curve_10y2y", 1.0)
    m2_yoy = _safe(indicators, "m2_yoy", 5.0)
    fed_funds = _safe(indicators, "fed_funds_rate", 3.0)
    breakeven_10y = _safe(indicators, "breakeven_10y", 2.0)
    unrate = _safe(indicators, "unrate", 4.5)
    unrate_roc = _safe(indicators, "unrate_roc", 0.0)
    claims_roc = _safe(indicators, "initial_claims_roc", 0.0)
    sentiment = _safe(indicators, "consumer_sentiment", 75.0)
    truck_yoy = _safe(indicators, "heavy_truck_sales_yoy", 0.0)
    permits_yoy = _safe(indicators, "building_permits_yoy", 0.0)
    durable_yoy = _safe(indicators, "durable_goods_orders_yoy", 0.0)
    consumer_credit_yoy = _safe(indicators, "consumer_credit_yoy", 4.0)
    wage_growth = _safe(indicators, "wage_growth_atlanta", 3.0)
    sticky_cpi = _safe(indicators, "sticky_cpi_yoy", 2.5)
    insider = _safe(indicators, "insider_score", 0.0)

    # Fire trigger flags
    triggers: dict[str, bool] = {}
    triggers["RISK_OFF"] = (
        vix > 25.0 or hy_oas > 6.0 or yield_curve < -0.5 or nfci > 0.5
    )
    triggers["RISK_OFF_EXTREME"] = vix > 35.0 or hy_oas > 8.0
    triggers["RISK_ON"] = (
        m2_yoy > 5.0 and nfci < -0.2 and vix < 18.0
    )
    triggers["INFLATION_HEDGE"] = (
        cpi > 3.5 and cpi_chg_6m > 0.3 and wage_growth > 3.5
    )
    triggers["INFLATION_PEAK"] = (
        cpi > 4.0 and core_pce > 3.5 and sticky_cpi > 4.0
    )
    triggers["DEFLATION"] = (
        gdp_roc < 0.5 and claims_roc > 5.0 and breakeven_10y < 1.8
    )
    triggers["DEFLATION_SEVERE"] = (
        gdp_roc < -1.0 and vix > 25.0 and breakeven_10y < 1.5
    )
    triggers["HOUSING_BOOM"] = permits_yoy > 5.0
    triggers["HOUSING_BUST"] = permits_yoy < -20.0
    triggers["INDUSTRIAL_EXPANSION"] = (
        truck_yoy > 5.0 and durable_yoy > 3.0
    )
    triggers["INDUSTRIAL_CONTRACTION"] = (
        truck_yoy < -10.0 or durable_yoy < -5.0
    )
    triggers["FED_RESTRICTIVE"] = fed_funds > 4.5 and gdp_chg_6m < -0.3
    triggers["FED_ACCOMMODATIVE"] = fed_funds < 2.5 and m2_yoy > 6.0
    triggers["LABOR_TIGHT"] = unrate < 4.0 and unrate_roc < 0
    triggers["LABOR_WEAKENING"] = unrate_roc > 0.3 or claims_roc > 8.0
    triggers["YIELD_CURVE_INVERTED"] = yield_curve < 0
    triggers["YIELD_CURVE_NORMALIZING"] = yield_curve > 1.5
    triggers["CONSUMER_STRONG"] = sentiment > 80.0 and consumer_credit_yoy > 4.0
    triggers["CONSUMER_WEAK"] = sentiment < 60.0
    triggers["INSIDER_BUYING"] = insider > 0.3
    triggers["INSIDER_SELLING"] = insider < -0.3

    # Map trigger fired → asset score adjustments
    scores: dict[str, float] = {a: 50.0 for a in ASSET_CLASSES}

    def boost(group_name: str, delta: float):
        for a in ASSET_GROUPS.get(group_name, set()):
            if a in scores:
                scores[a] += delta

    if triggers["RISK_OFF_EXTREME"]:
        boost("DEFENSIVE", +18); boost("GROWTH_EQUITIES", -18); boost("CRYPTO_RISK_ON", -25)
    elif triggers["RISK_OFF"]:
        boost("DEFENSIVE", +12); boost("GROWTH_EQUITIES", -10); boost("CRYPTO_RISK_ON", -15)

    if triggers["RISK_ON"]:
        boost("GROWTH_EQUITIES", +12); boost("CYCLICAL", +8); boost("CRYPTO_RISK_ON", +10)
        boost("DEFENSIVE", -5)

    if triggers["INFLATION_HEDGE"]:
        boost("INFLATION_HEDGE", +15)
        if "us_bonds_long" in scores:
            scores["us_bonds_long"] -= 12
    if triggers["INFLATION_PEAK"]:
        boost("INFLATION_HEDGE", +8)
        if "sector_energy" in scores: scores["sector_energy"] += 8
        if "gold" in scores: scores["gold"] += 6

    if triggers["DEFLATION"]:
        boost("DEFLATION_HEDGE", +15)
        boost("GROWTH_EQUITIES", -8)
    if triggers["DEFLATION_SEVERE"]:
        boost("DEFLATION_HEDGE", +10)
        boost("CYCLICAL", -15)
        boost("CRYPTO_RISK_ON", -20)

    if triggers["HOUSING_BOOM"]:
        if "real_estate_reits" in scores: scores["real_estate_reits"] += 12
        if "sector_financials" in scores: scores["sector_financials"] += 8
    if triggers["HOUSING_BUST"]:
        if "real_estate_reits" in scores: scores["real_estate_reits"] -= 15
        if "sector_financials" in scores: scores["sector_financials"] -= 10

    if triggers["INDUSTRIAL_EXPANSION"]:
        if "factor_value" in scores: scores["factor_value"] += 8
        if "sector_energy" in scores: scores["sector_energy"] += 6
        boost("VALUE_EQUITIES", +5)
    if triggers["INDUSTRIAL_CONTRACTION"]:
        if "factor_value" in scores: scores["factor_value"] -= 8
        boost("CYCLICAL", -10)

    if triggers["FED_RESTRICTIVE"]:
        if "us_bonds_long" in scores: scores["us_bonds_long"] += 6
        if "cash_money_market" in scores: scores["cash_money_market"] += 8
        boost("GROWTH_EQUITIES", -6)
    if triggers["FED_ACCOMMODATIVE"]:
        boost("GROWTH_EQUITIES", +8); boost("CRYPTO_RISK_ON", +12)
        if "real_estate_reits" in scores: scores["real_estate_reits"] += 6

    if triggers["LABOR_TIGHT"]:
        boost("GROWTH_EQUITIES", +4)
        if "sector_staples" in scores: scores["sector_staples"] += 3
    if triggers["LABOR_WEAKENING"]:
        boost("DEFENSIVE", +6); boost("GROWTH_EQUITIES", -5)

    if triggers["YIELD_CURVE_INVERTED"]:
        boost("DEFENSIVE", +4)
    if triggers["YIELD_CURVE_NORMALIZING"]:
        boost("CYCLICAL", +6)

    if triggers["CONSUMER_STRONG"]:
        boost("GROWTH_EQUITIES", +4); boost("CYCLICAL", +4)
        if "real_estate_reits" in scores: scores["real_estate_reits"] += 3
    if triggers["CONSUMER_WEAK"]:
        if "sector_staples" in scores: scores["sector_staples"] += 4
        boost("CRYPTO_RISK_ON", -8)

    if triggers["INSIDER_BUYING"]:
        boost("GROWTH_EQUITIES", +3); boost("VALUE_EQUITIES", +3)
    if triggers["INSIDER_SELLING"]:
        boost("DEFENSIVE", +3); boost("GROWTH_EQUITIES", -3)

    # Clip 0-100
    return {a: max(0.0, min(100.0, s)) for a, s in scores.items()}, triggers


# ============================================================================
# DAILY SCAN — analoga a strategist.py ma usa data-driven scores
# ============================================================================

def _fetch_price_history_for_universe() -> dict[str, pd.Series]:
    """Identico a strategist.py: serie prezzi Yahoo per universe."""
    from datetime import timedelta
    from app.services.prices.yahoo_fetcher import YahooFetcher

    fetcher = YahooFetcher()
    out: dict[str, pd.Series] = {}
    end = date.today()
    start = end - timedelta(days=200)
    for asset in ASSET_CLASSES:
        ticker = momentum.asset_to_ticker(asset)
        if not ticker:
            continue
        try:
            s = fetcher.fetch(ticker, start_date=start, end_date=end)
            if s is not None and not s.empty:
                out[asset] = s.dropna()
        except Exception as e:
            logger.warning(f"data_strategist price fetch failed for {asset}: {e}")
    return out


def _refresh_real_pnl(position: AiPortfolioPosition) -> float:
    """Idem strategist.py."""
    current_price = pnl_tracker.get_latest_price(position.asset_class)
    if current_price is None:
        return position.estimated_pnl_pct
    if position.avg_entry_price is None or position.avg_entry_price == 0:
        position.avg_entry_price = current_price
        position.current_price = current_price
        return 0.0
    position.current_price = current_price
    return pnl_tracker.compute_pnl_pct(position.avg_entry_price, current_price)


def daily_scan(
    db: Session,
    target_date: date | None = None,
) -> dict:
    """Daily scan strategia data-driven."""
    target_date = target_date or date.today()

    # Indicators dal latest RegimeClassification.conditions_met JSON
    import json as _json
    latest = (
        db.query(RegimeClassification)
        .order_by(RegimeClassification.date.desc())
        .first()
    )
    if latest is None:
        return {"error": "no regime classification (need indicators source)"}

    try:
        payload = _json.loads(latest.conditions_met) if latest.conditions_met else {}
    except Exception:
        payload = {}
    indicators = payload.get("indicators", {}) or {}

    # Score data-driven + fired triggers
    scores, triggers = evaluate_data_driven_scores(indicators)
    fired_names = [t for t, fired in triggers.items() if fired]
    fired_summary = ", ".join(fired_names[:6]) if fired_names else "no major triggers"

    # Confidence proxy: numero trigger fired / 10 (capped)
    confidence = min(0.95, max(0.30, len(fired_names) / 10.0))
    n_tranches_default = sizer.dynamic_n_tranches(confidence)

    # Price history per momentum
    price_history = _fetch_price_history_for_universe()

    # Target weights
    raw_weights: dict[str, float] = {}
    for asset in ASSET_CLASSES:
        score = scores.get(asset, 50.0)
        sig, vol = momentum.get_asset_signal_and_vol(asset, price_history)
        raw_weights[asset] = sizer.raw_target_weight(score, vol)
    targets_normalized = sizer.normalize_and_cap(raw_weights)

    # Existing positions per data_driven
    existing = {
        p.asset_class: p
        for p in db.query(AiPortfolioPosition)
        .filter(AiPortfolioPosition.strategy_type == STRATEGY_TYPE)
        .all()
    }
    for p in existing.values():
        p.estimated_pnl_pct = _refresh_real_pnl(p)

    summary = {"n_decisions": 0, "n_open_tranches": 0, "n_closures": 0}

    for asset in ASSET_CLASSES:
        score = scores.get(asset, 50.0)
        target = targets_normalized.get(asset, 0.0)
        sig, vol = momentum.get_asset_signal_and_vol(asset, price_history)
        position = existing.get(asset)

        threshold_shift = learning.get_threshold_shift_for_pattern(
            db, regime=latest.regime, asset_class=asset, momentum=sig,
            strategy_type=STRATEGY_TYPE,
        )

        action, reason, size = _decide(
            position=position,
            score=score,
            target=target,
            confidence=confidence,
            momentum_sig=sig,
            n_tranches_default=n_tranches_default,
            entry_threshold_shift=threshold_shift,
            fired_summary=fired_summary,
        )

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
                regime=latest.regime,  # solo per logging context
                momentum_signal=sig,
                vol_60d=vol,
                estimated_pnl_pct=(position.estimated_pnl_pct if position else None),
                reason_short=reason,
            )
            db.add(decision)
            summary["n_decisions"] += 1

        _apply(db, asset, position, action, target, size, score,
               latest.regime, n_tranches_default, target_date, summary)

    db.commit()
    summary["n_positions_after"] = (
        db.query(AiPortfolioPosition)
        .filter(AiPortfolioPosition.strategy_type == STRATEGY_TYPE)
        .count()
    )
    summary["triggers_fired"] = fired_names
    return summary


def _decide(
    position: Optional[AiPortfolioPosition],
    score: float,
    target: float,
    confidence: float,
    momentum_sig: str,
    n_tranches_default: int,
    entry_threshold_shift: float = 0.0,
    fired_summary: str = "",
) -> tuple[str, str, float]:
    effective_entry = ENTRY_SCORE_THRESHOLD + entry_threshold_shift
    if position is None:
        if confidence < 0.30:
            return ("SKIP_NO_SIGNAL", "data confidence below uncertainty gate", 0.0)
        if score < effective_entry:
            return ("SKIP_NO_SIGNAL", f"data score {score:.1f} < threshold {effective_entry:.1f}", 0.0)
        if momentum_sig != "BULLISH":
            return ("SKIP_NO_SIGNAL", f"momentum {momentum_sig} (need BULLISH)", 0.0)
        if target <= 0 or n_tranches_default <= 0:
            return ("SKIP_NO_SIGNAL", "target weight 0", 0.0)
        first_size = target / n_tranches_default
        return ("OPEN_TRANCHE", f"data entry: score {score:.1f}, triggers: {fired_summary[:60]}", first_size)

    pnl = position.estimated_pnl_pct
    if score < EXIT_SCORE_THRESHOLD:
        return ("FULL_CLOSE", f"data score collapse {score:.1f}", 0.0)
    if pnl <= STOP_LOSS_PCT:
        return ("STOP_LOSS", f"stop loss {pnl*100:.1f}%", 0.0)
    if pnl >= FULL_TP_PCT:
        return ("TAKE_PROFIT", f"full TP {pnl*100:.1f}%", 0.0)
    if pnl >= PARTIAL_TP_PCT and not position.took_partial_tp:
        return ("CLOSE_TRANCHE", f"partial TP {pnl*100:.1f}%", 0.0)
    if position.tranches_filled < position.tranches_total and score >= INCREMENTAL_SCORE_THRESHOLD:
        size = target / position.tranches_total
        return ("OPEN_TRANCHE", f"data score strong {score:.1f}, add tranche {position.tranches_filled+1}/{position.tranches_total}", size)
    return ("HOLD", "no trigger", 0.0)


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
    if action == "OPEN_TRANCHE":
        entry_price = pnl_tracker.get_latest_price(asset)
        if position is None:
            db.add(AiPortfolioPosition(
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
            ))
            summary["n_open_tranches"] += 1
        else:
            position.current_weight_pct += size
            n_before = position.tranches_filled
            position.tranches_filled += 1
            position.target_weight_pct = target
            n = position.tranches_filled
            position.avg_entry_score = (position.avg_entry_score * (n - 1) + score) / n
            if entry_price is not None:
                position.avg_entry_price = pnl_tracker.update_avg_entry_price(
                    position.avg_entry_price, n_before, entry_price,
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

    elif action in ("FULL_CLOSE", "STOP_LOSS", "TAKE_PROFIT"):
        if position is not None:
            try:
                last_decision = AiPortfolioDecision(
                    date=target_date,
                    asset_class=asset,
                    strategy_type=STRATEGY_TYPE,
                    action=action,
                    momentum_signal="NEUTRAL",
                    score=score,
                    confidence=0.0,
                    regime=regime,
                    reason_short="auto-closed",
                )
                learning.record_closed_position(db, last_decision, position)
            except Exception as e:
                logger.warning(f"data learning record failed: {e}")
            db.delete(position)
            summary["n_closures"] += 1
