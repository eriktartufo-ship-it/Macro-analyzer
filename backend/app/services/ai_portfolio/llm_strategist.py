"""LLM-Driven Strategist — Gemini decide cosa comprare/vendere.

3a strategia horserace (sessione 2026-06-13). NON è il commentary del
llm_reasoner: qui Gemini è il DECISORE. Ogni giorno:

1. Riceve in input:
   - regime classifier + confidence + probabilità
   - 25 indicators FRED chiave + ROC
   - score Model-Driven per ogni asset (cosa dice il classifier-based)
   - score Data-Driven per ogni asset (cosa dicono le rule-based su indicators)
   - posizioni aperte con PnL reale e giorni held
   - learnings post-mortem accumulati per strategia llm_driven

2. Ritorna JSON strutturato:
   {"decisions": [
       {"asset": "us_equities_growth", "action": "OPEN_TRANCHE",
        "score": 65, "confidence": 0.7, "size_pct": 4.5, "reason": "..."},
       ...
   ]}

3. Le decisioni sono validate/clampate + applicate via stessa pipeline degli altri
   strategist (open/close/stop/tp/incremental). Pool $100k separato, learning
   post-mortem isolato.

Fallback: se Gemini fail → no-op (nessuna decisione, posizioni invariate).
Mai crash, mai fake data: la 3a strategia salta semplicemente la giornata.

Costo: 1 call Gemini/day = trascurabile.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import date
from typing import Optional

import pandas as pd
import requests
from sqlalchemy.orm import Session

from app.models.ai_portfolio import AiPortfolioDecision, AiPortfolioPosition
from app.models.daily_signals import DailySignal
from app.models.regime_classifications import RegimeClassification
from app.services.ai_portfolio import data_strategist, learning, momentum, pnl_tracker, sizer
from app.services.llm import settings as llm_settings
from app.services.scoring.engine import ASSET_CLASSES

logger = logging.getLogger(__name__)


STRATEGY_TYPE = "llm_driven"
_PROMPT_VERSION = "v1-decider"

# Stesse soglie hard di safety (Gemini suggerisce ma noi clampiamo)
MAX_SIZE_PCT_PER_TRANCHE = 0.08  # 8% max per tranche, anti-hallucination
MIN_CONFIDENCE = 0.30
STOP_LOSS_PCT = -0.10
FULL_TP_PCT = 0.30

VALID_ACTIONS = {
    "OPEN_TRANCHE", "CLOSE_TRANCHE", "FULL_CLOSE",
    "STOP_LOSS", "TAKE_PROFIT", "HOLD", "SKIP_NO_SIGNAL",
}

_GEMINI_API_URL_TPL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/{model}:generateContent"
)


_SYSTEM_PROMPT = """Sei un portfolio manager senior di un hedge fund discreazionale.
Decidi tu cosa fare, NON ripeti meccanicamente quello che dicono i due sistemi
algorithmici (Model-Driven e Data-Driven) che ti mostro: usali come INPUT, ma
hai libertà di dissentire.

**LINGUA**: campo `reason` SEMPRE in italiano professionale, breve (max 150 char).

**UNIVERSE FISSO** (24 asset). Puoi decidere solo su questi:
us_equities_growth, us_equities_value, us_bonds_long, us_bonds_short,
tips_inflation_bonds, gold, silver, cash_money_market, broad_commodities,
energy, real_estate_reits, international_dm_equities, em_equities,
bitcoin, crypto_broad, sector_technology, sector_energy, sector_financials,
sector_utilities, sector_staples, sector_healthcare, factor_momentum,
factor_quality, factor_value

**AZIONI VALIDE**:
- OPEN_TRANCHE: apri/aggiungi tranche (specifica size_pct 1.0-8.0)
- CLOSE_TRANCHE: chiudi parziale (partial TP)
- FULL_CLOSE: chiudi tutto (regime adverse o tesi rotta)
- STOP_LOSS / TAKE_PROFIT: solo se PnL conferma
- HOLD: mantieni posizione esistente invariata
- SKIP_NO_SIGNAL: nessun signal (default per asset senza posizione)

**REGOLE FERREE**:
1. NON aprire più di 8% size per tranche
2. NON suggerire OPEN per asset con confidence < 0.30 nel tuo giudizio
3. STOP_LOSS è automatico a -10% PnL (tu segnalalo solo se vuoi forzarlo prima)
4. TAKE_PROFIT è automatico a +30% PnL
5. Massimo 12 OPEN_TRANCHE per giorno (evita overtrading)
6. Per asset SENZA posizione e SENZA OPEN: ometti dall'output (NON serve SKIP esplicito)

**OUTPUT JSON STRETTO** (NO markdown):
{
  "thesis": "1 frase italiana riassuntiva della tua tesi macro oggi (max 250 char)",
  "decisions": [
    {"asset": "us_equities_growth", "action": "OPEN_TRANCHE", "score": 65,
     "confidence": 0.7, "size_pct": 4.5, "reason": "tech leadership + momentum confermato"},
    ...
  ]
}

Decidi come Soros: aggressivo dove hai conviction, prudente dove i due sistemi
sotto disagree."""


def _compute_input_hash(
    target_date: date,
    regime: str,
    confidence: float,
    n_indicators: int,
    n_positions: int,
    model: str,
) -> str:
    canonical = {
        "date": str(target_date),
        "regime": regime,
        "confidence": round(confidence, 3),
        "n_indicators": n_indicators,
        "n_positions": n_positions,
        "model": model,
        "prompt_version": _PROMPT_VERSION,
    }
    return hashlib.md5(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


def _build_user_prompt(
    target_date: date,
    regime: str,
    probabilities: dict,
    confidence: float,
    indicators: dict,
    model_scoreboard: dict[str, float],
    data_scoreboard: dict[str, float],
    positions: list[AiPortfolioPosition],
    learnings: list[dict],
) -> str:
    probs_str = ", ".join(
        f"{k}={v:.0%}" for k, v in sorted(probabilities.items(), key=lambda x: -x[1])
    )

    # Indicators chiave (top 18, già normalizzati)
    key_inds = [
        "gdp_roc", "gdp_roc_change_6m", "cpi_yoy", "cpi_yoy_change_6m", "core_pce_yoy",
        "sticky_cpi_yoy", "wage_growth_atlanta", "unrate", "unrate_roc",
        "initial_claims_roc", "yield_curve_10y2y", "fed_funds_rate", "m2_yoy",
        "vix", "hy_credit_spread", "nfci", "consumer_sentiment",
        "heavy_truck_sales_yoy", "building_permits_yoy", "durable_goods_orders_yoy",
        "breakeven_10y",
    ]
    ind_lines = []
    for k in key_inds:
        v = indicators.get(k)
        if v is not None:
            try:
                ind_lines.append(f"  - {k}: {float(v):.3f}")
            except (ValueError, TypeError):
                pass

    # Score comparison side-by-side (solo top divergenti + favorevoli)
    score_lines = []
    for asset in ASSET_CLASSES:
        m = model_scoreboard.get(asset, 0.0)
        d = data_scoreboard.get(asset, 50.0)
        if max(m, d) >= 50 or abs(m - d) > 15:
            divergence = "‼" if abs(m - d) > 20 else " "
            score_lines.append(f"  {divergence} {asset}: model={m:.1f} | data={d:.1f}")

    # Positions
    pos_lines = []
    for p in positions:
        days = p.days_held if hasattr(p, "days_held") else "?"
        pos_lines.append(
            f"  - {p.asset_class}: peso {p.current_weight_pct*100:.1f}%, "
            f"PnL {p.estimated_pnl_pct*100:+.1f}%, tranches {p.tranches_filled}/{p.tranches_total}, "
            f"days {days}, entry_regime {p.entry_regime}"
        )

    # Learnings (proprio della strategia llm_driven, solo con shift)
    learn_lines = []
    for L in learnings[:5]:
        if L.get("entry_threshold_shift", 0) == 0:
            continue
        learn_lines.append(
            f"  - {L['pattern_key']}: {L['n_wins']}W/{L['n_losses']}L "
            f"(win_rate {L['win_rate']:.0%}, shift {L['entry_threshold_shift']:+.1f})"
        )

    return f"""## CONTESTO PORTFOLIO LLM-DRIVEN — {target_date}

**Regime classifier**: {regime.upper()} (confidence {confidence:.0%})
**Probabilità**: {probs_str}

## INDICATORS MACRO (FRED, raw)
{chr(10).join(ind_lines) if ind_lines else "  (nessun indicator)"}

## SCORE COMPARISON: Model-Driven vs Data-Driven
(asset interessanti, ‼ = divergenza > 20 punti)
{chr(10).join(score_lines[:30]) if score_lines else "  (nessun asset interessante)"}

## TUE POSIZIONI APERTE ({len(positions)})
{chr(10).join(pos_lines) if pos_lines else "  (nessuna posizione)"}

## TUOI LEARNINGS POST-MORTEM (solo llm_driven, top 5 con shift attivi)
{chr(10).join(learn_lines) if learn_lines else "  (nessun pattern con shift accumulato)"}

## RICHIESTA

Decidi cosa fare oggi. Vedi sopra cosa suggeriscono i 2 sistemi algoritmici
(Model + Data). Sei libero di concordare, dissentire, o aprire asset che NON
hanno score >55 in nessuno dei due se la tua tesi macro lo giustifica.

Output JSON: { "thesis": "...", "decisions": [...] }
"""


def _call_gemini(prompt: str, api_key: str, model: str) -> Optional[str]:
    url = _GEMINI_API_URL_TPL.format(model=model)
    try:
        resp = requests.post(
            url,
            params={"key": api_key},
            json={
                "contents": [
                    {"parts": [{"text": f"{_SYSTEM_PROMPT}\n\n---\n\n{prompt}"}]},
                ],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 6000,
                    "responseMimeType": "application/json",
                    "thinkingConfig": {"thinkingBudget": 0},
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            return None
        parts = candidates[0].get("content", {}).get("parts") or []
        return "".join(p.get("text", "") for p in parts).strip() or None
    except Exception as e:
        logger.warning(f"LLM Strategist Gemini call failed (model={model}): {e}")
        return None


def _fetch_price_history_for_universe() -> dict[str, pd.Series]:
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
            logger.warning(f"llm_strategist price fetch failed for {asset}: {e}")
    return out


def _refresh_real_pnl(position: AiPortfolioPosition) -> float:
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
    """Daily scan strategia LLM-driven (Gemini decisore)."""
    target_date = target_date or date.today()

    # 1. Regime + indicators
    latest = (
        db.query(RegimeClassification)
        .order_by(RegimeClassification.date.desc())
        .first()
    )
    if latest is None:
        return {"error": "no regime classification"}

    regime = latest.regime
    confidence = float(latest.confidence or 0.5)
    probabilities = {
        "reflation": float(latest.probability_reflation or 0),
        "stagflation": float(latest.probability_stagflation or 0),
        "deflation": float(latest.probability_deflation or 0),
        "goldilocks": float(latest.probability_goldilocks or 0),
    }
    indicators = {}
    try:
        if latest.conditions_met:
            payload = json.loads(latest.conditions_met)
            indicators = payload.get("indicators", {}) or {}
    except Exception:
        pass

    # 2. Model scoreboard (da DailySignal)
    signals = (
        db.query(DailySignal)
        .filter(DailySignal.date == latest.date)
        .all()
    )
    model_scoreboard = {s.asset_class: float(s.final_score) for s in signals}

    # 3. Data scoreboard (re-evaluating rule engine — cheap)
    data_scoreboard, _ = data_strategist.evaluate_data_driven_scores(indicators)

    # 4. Positions LLM-driven + PnL reale
    existing = {
        p.asset_class: p
        for p in db.query(AiPortfolioPosition)
        .filter(AiPortfolioPosition.strategy_type == STRATEGY_TYPE)
        .all()
    }
    for p in existing.values():
        p.estimated_pnl_pct = _refresh_real_pnl(p)
        # Augment days_held attribute on the fly per prompt
        try:
            p.days_held = (target_date - p.opened_at).days
        except Exception:
            p.days_held = 0

    # 5. Learnings llm_driven
    learnings = learning.list_top_learnings(db, limit=10, strategy_type=STRATEGY_TYPE)

    # 6. Hash + skip se invariato (anti double-call)
    model = llm_settings.get_model()
    h = _compute_input_hash(target_date, regime, confidence, len(indicators), len(existing), model)

    # Check se già fatto oggi
    already_today = (
        db.query(AiPortfolioDecision)
        .filter(
            AiPortfolioDecision.date == target_date,
            AiPortfolioDecision.strategy_type == STRATEGY_TYPE,
        )
        .first()
    )
    if already_today:
        return {"skipped": "already scanned today", "hash": h}

    api_key = llm_settings.get_api_key()
    if not api_key:
        logger.warning("LLM Strategist: no Gemini key, skipping day")
        return {"skipped": "no api key"}

    # 7. Prompt + call
    prompt = _build_user_prompt(
        target_date=target_date,
        regime=regime,
        probabilities=probabilities,
        confidence=confidence,
        indicators=indicators,
        model_scoreboard=model_scoreboard,
        data_scoreboard=data_scoreboard,
        positions=list(existing.values()),
        learnings=learnings,
    )
    raw = _call_gemini(prompt, api_key=api_key, model=model)
    if raw is None:
        return {"skipped": "llm call failed"}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning(f"LLM Strategist invalid JSON: {e}, raw: {raw[:200]}")
        return {"skipped": "invalid llm json"}

    raw_decisions = parsed.get("decisions") or []
    if not isinstance(raw_decisions, list):
        return {"skipped": "decisions not list"}

    # 8. Price history per momentum + sizing
    price_history = _fetch_price_history_for_universe()

    summary = {
        "n_decisions": 0, "n_open_tranches": 0, "n_closures": 0,
        "thesis": str(parsed.get("thesis", ""))[:300],
    }

    # Cap totale OPEN al giorno (anti-overtrading)
    MAX_OPEN_PER_DAY = 12
    opens_today = 0

    for raw_d in raw_decisions[:50]:
        if not isinstance(raw_d, dict):
            continue
        asset = str(raw_d.get("asset", "")).strip()
        action = str(raw_d.get("action", "")).strip().upper()
        if asset not in ASSET_CLASSES or action not in VALID_ACTIONS:
            continue

        # Clamping
        try:
            llm_score = float(raw_d.get("score", 50))
        except (ValueError, TypeError):
            llm_score = 50.0
        llm_score = max(0.0, min(100.0, llm_score))

        try:
            llm_conf = float(raw_d.get("confidence", 0.5))
        except (ValueError, TypeError):
            llm_conf = 0.5
        llm_conf = max(0.0, min(1.0, llm_conf))

        try:
            size_pct = float(raw_d.get("size_pct", 0)) / 100.0  # arrivava in %, normalizziamo
        except (ValueError, TypeError):
            size_pct = 0.0
        size_pct = max(0.0, min(MAX_SIZE_PCT_PER_TRANCHE, size_pct))

        reason = str(raw_d.get("reason", ""))[:300]
        position = existing.get(asset)
        sig, vol = momentum.get_asset_signal_and_vol(asset, price_history)

        # Safety rails
        if action == "OPEN_TRANCHE":
            if opens_today >= MAX_OPEN_PER_DAY:
                continue
            if llm_conf < MIN_CONFIDENCE:
                continue
            if size_pct <= 0:
                continue
            opens_today += 1
        elif action in ("CLOSE_TRANCHE", "FULL_CLOSE", "STOP_LOSS", "TAKE_PROFIT"):
            if position is None:
                continue  # niente da chiudere
        # HOLD/SKIP_NO_SIGNAL: log per audit ma niente effetto stato

        # Log decision
        dec = AiPortfolioDecision(
            date=target_date,
            asset_class=asset,
            strategy_type=STRATEGY_TYPE,
            action=action,
            size_pct=size_pct,
            tranche_index=(position.tranches_filled + 1) if (position and action == "OPEN_TRANCHE") else (1 if action == "OPEN_TRANCHE" else 0),
            tranches_total=(position.tranches_total if position else 3),
            score=llm_score,
            confidence=llm_conf,
            regime=regime,
            momentum_signal=sig,
            vol_60d=vol,
            estimated_pnl_pct=(position.estimated_pnl_pct if position else None),
            reason_short=reason,
        )
        db.add(dec)
        summary["n_decisions"] += 1

        _apply(db, asset, position, action, size_pct, llm_score, regime, target_date, summary)

    db.commit()
    summary["n_positions_after"] = (
        db.query(AiPortfolioPosition)
        .filter(AiPortfolioPosition.strategy_type == STRATEGY_TYPE)
        .count()
    )
    return summary


def _apply(
    db: Session,
    asset: str,
    position: Optional[AiPortfolioPosition],
    action: str,
    size: float,
    score: float,
    regime: str,
    target_date: date,
    summary: dict,
) -> None:
    if action == "OPEN_TRANCHE":
        entry_price = pnl_tracker.get_latest_price(asset)
        if position is None:
            db.add(AiPortfolioPosition(
                asset_class=asset,
                strategy_type=STRATEGY_TYPE,
                target_weight_pct=size * 3,  # implicit target = 3 tranche
                current_weight_pct=size,
                tranches_filled=1,
                tranches_total=3,
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
            position.tranches_filled = min(position.tranches_filled + 1, position.tranches_total)
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
                    reason_short="llm-closed",
                )
                learning.record_closed_position(db, last_decision, position)
            except Exception as e:
                logger.warning(f"llm learning record failed: {e}")
            db.delete(position)
            summary["n_closures"] += 1
