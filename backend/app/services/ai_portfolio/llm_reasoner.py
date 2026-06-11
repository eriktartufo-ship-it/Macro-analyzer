"""LLM Reasoner per AI Portfolio — spiega decisioni + critica regime.

Fase 2 sessione 2026-06-11. Gemini chiamato 1x/giorno con:
- regime classifier output (regime + probs + confidence)
- decisioni del giorno (action, asset, score, momentum)
- positions correnti (asset, target%, PnL)
- learnings attivi (pattern + win_rate)

Output JSON italiano:
- daily_summary: 2-3 frasi
- decisions_reasoning: {asset: spiegazione_breve}
- regime_classifier_says: regime classifier dominante
- ai_agrees: high|medium|low
- ai_alternative_regime: regime alternativo se ai_agrees != high
- regime_challenge_reasoning: perché il classifier potrebbe sbagliare

Cache: 1 row per data in `ai_portfolio_reasoning`. Hash di (date + n_decisions
+ regime + confidence) per skip se nulla è cambiato.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import date, datetime, timezone
from typing import Optional

import requests
from sqlalchemy.orm import Session

from app.models.ai_portfolio import (
    AiPortfolioDecision,
    AiPortfolioPosition,
    AiPortfolioReasoning,
)
from app.models.regime_classifications import RegimeClassification
from app.services.llm import settings as llm_settings

logger = logging.getLogger(__name__)


_PROMPT_VERSION = "v1-italian"
_GEMINI_API_URL_TPL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/{model}:generateContent"
)

_SYSTEM_PROMPT = """Sei un analista hedge fund senior che fa peer-review delle
decisioni di un sistema AI di portfolio management.

**LINGUA OBBLIGATORIA: ITALIANO.** Tutto l'output in italiano professionale.
Termini tecnici inglesi (CPI, GDP, S&P500, YoY, MA50, RSI) sono ammessi.

Compito tripudo:
1. Spiega in 2-3 frasi le decisioni più importanti del giorno (top OPEN + top CLOSE)
2. Per ogni asset con decisione non-HOLD, scrivi UNA frase breve italiana di
   spiegazione (max 100 char)
3. **METTI IN DISCUSSIONE IL REGIME**: il classifier dice X, ma tu lo trovi
   coerente con i dati? Considera scenari alternativi. Sii brutale se i dati
   suggeriscono qualcos'altro.

Output JSON schema (NO markdown fences):
{
  "daily_summary": "2-3 frasi italiane riassuntive",
  "decisions_reasoning": {"asset_class_1": "1 frase breve", ...},
  "regime_classifier_says": "stagflation|reflation|deflation|goldilocks",
  "ai_agrees": "high|medium|low",
  "ai_alternative_regime": null oppure stringa regime alternativo,
  "regime_challenge_reasoning": "se ai_agrees != high spiega perché in italiano (2-3 frasi); altrimenti null"
}

NON inventare. Cita SOLO i numeri presenti nell'input. Sii incisivo come Soros
in una nota interna."""


def _compute_input_hash(
    target_date: date,
    regime: str,
    confidence: float,
    n_decisions: int,
    decisions_actions: list[str],
    n_positions: int,
    model: str,
) -> str:
    """Hash sull'input per cache invalidation."""
    canonical = {
        "date": str(target_date),
        "regime": regime,
        "confidence": round(confidence, 3),
        "n_decisions": n_decisions,
        "decisions_actions_sorted": sorted(decisions_actions),
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
    decisions: list[AiPortfolioDecision],
    positions: list[AiPortfolioPosition],
    indicators: dict,
    learnings: list[dict],
) -> str:
    probs_str = ", ".join(f"{k}={v:.2%}" for k, v in sorted(probabilities.items(), key=lambda x: -x[1]))

    # Decisions resumé (priorità non-HOLD)
    relevant = [d for d in decisions if d.action not in ("HOLD", "SKIP_NO_SIGNAL")]
    skipped = [d for d in decisions if d.action == "SKIP_NO_SIGNAL"]
    dec_lines = []
    for d in relevant[:20]:
        dec_lines.append(
            f"  - {d.asset_class}: {d.action} (score {d.score:.1f}, mom {d.momentum_signal}, "
            f"conf {d.confidence:.2%}, reason: {d.reason_short})"
        )
    skipped_summary = f"  + {len(skipped)} asset skipped (sotto threshold o senza signal)" if skipped else ""

    # Positions
    pos_lines = []
    for p in positions[:15]:
        pnl_str = f"PnL {p.estimated_pnl_pct*100:+.1f}%"
        pos_lines.append(
            f"  - {p.asset_class}: peso {p.current_weight_pct*100:.1f}%/target {p.target_weight_pct*100:.1f}%, "
            f"tranches {p.tranches_filled}/{p.tranches_total}, {pnl_str}, entry regime {p.entry_regime}"
        )

    # Indicators chiave (top 10 utili)
    key_inds = [
        "gdp_roc", "cpi_yoy", "core_pce_yoy", "unrate", "yield_curve_10y2y",
        "vix", "consumer_sentiment", "fed_funds_rate", "hy_credit_spread",
        "wage_growth_atlanta",
    ]
    ind_lines = [f"  - {k}: {indicators[k]:.3f}" for k in key_inds if k in indicators and indicators[k] is not None]

    # Learnings (top 5 con shift assoluto > 0)
    learn_lines = []
    for L in learnings[:5]:
        if L["entry_threshold_shift"] == 0:
            continue
        learn_lines.append(
            f"  - {L['pattern_key']}: {L['n_wins']}W/{L['n_losses']}L "
            f"(win_rate {L['win_rate']:.0%}, avg_pnl {L['avg_pnl_pct']*100:+.1f}%, "
            f"threshold_shift {L['entry_threshold_shift']:+.1f})"
        )

    return f"""## CONTESTO PORTFOLIO AI — {target_date}

Regime classifier: **{regime.upper()}** (confidence {confidence:.2%})
Probabilità: {probs_str}

## INDICATORS MACRO CHIAVE (FRED)
{chr(10).join(ind_lines) if ind_lines else "  (nessun indicator disponibile)"}

## DECISIONI DI OGGI ({len(relevant)} non-trivial)
{chr(10).join(dec_lines) if dec_lines else "  (nessuna decisione non-trivial)"}
{skipped_summary}

## POSIZIONI ATTUALI ({len(positions)})
{chr(10).join(pos_lines) if pos_lines else "  (nessuna posizione aperta)"}

## LEARNINGS POST-MORTEM (pattern con shift attivi)
{chr(10).join(learn_lines) if learn_lines else "  (nessun pattern ha ancora prodotto shift)"}

## RICHIESTA

1. Riassumi in 2-3 frasi italiane cosa è successo oggi
2. Per OGNI decisione non-HOLD scrivi 1 frase italiana di spiegazione (campo `decisions_reasoning`)
3. METTI IN DISCUSSIONE il regime classifier: leggi gli indicators sopra e dimmi
   se sei d'accordo (high), perplesso (medium), o in disaccordo (low). Se non
   high, proponi `ai_alternative_regime` e spiega in `regime_challenge_reasoning`.

Output JSON puro, italiano, no markdown."""


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
                    "temperature": 0.2,
                    "maxOutputTokens": 3000,
                    "responseMimeType": "application/json",
                    "thinkingConfig": {"thinkingBudget": 0},
                },
            },
            timeout=90,
        )
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            return None
        parts = candidates[0].get("content", {}).get("parts") or []
        return "".join(p.get("text", "") for p in parts).strip() or None
    except Exception as e:
        logger.warning(f"AI Portfolio LLM call failed (model={model}): {e}")
        return None


def generate_reasoning(
    db: Session,
    target_date: date | None = None,
    force_refresh: bool = False,
) -> Optional[AiPortfolioReasoning]:
    """Genera o cache-restituisce reasoning + regime challenge per la data."""
    target_date = target_date or date.today()

    # Latest regime
    latest = (
        db.query(RegimeClassification)
        .order_by(RegimeClassification.date.desc())
        .first()
    )
    if latest is None:
        return None

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

    # Decisions
    decisions = (
        db.query(AiPortfolioDecision)
        .filter(AiPortfolioDecision.date == target_date)
        .all()
    )
    positions = db.query(AiPortfolioPosition).all()

    # Learnings (top 10 per shift assoluto)
    from app.services.ai_portfolio import learning as L
    learnings = L.list_top_learnings(db, limit=10)

    # Hash + cache check
    model = llm_settings.get_model()
    h = _compute_input_hash(
        target_date,
        regime,
        confidence,
        len(decisions),
        [d.action for d in decisions],
        len(positions),
        model,
    )

    existing = db.query(AiPortfolioReasoning).filter_by(date=target_date).first()
    if existing and not force_refresh and existing.data_hash == h:
        return existing

    api_key = llm_settings.get_api_key()
    if not api_key:
        logger.warning("AI Portfolio reasoning: no Gemini API key, skipping")
        return existing  # fallback stale or None

    prompt = _build_user_prompt(
        target_date=target_date,
        regime=regime,
        probabilities=probabilities,
        confidence=confidence,
        decisions=decisions,
        positions=positions,
        indicators=indicators,
        learnings=learnings,
    )
    raw = _call_gemini(prompt, api_key=api_key, model=model)
    if raw is None:
        return existing

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning(f"AI Portfolio reasoning: invalid JSON: {e}, raw: {raw[:200]}")
        return existing

    # Validate + clamp
    daily_summary = str(parsed.get("daily_summary", ""))[:1000]
    dec_reasoning = parsed.get("decisions_reasoning", {}) or {}
    if not isinstance(dec_reasoning, dict):
        dec_reasoning = {}
    dec_reasoning_clipped = {
        str(k)[:64]: str(v)[:300] for k, v in dec_reasoning.items()
    }
    ai_agrees = str(parsed.get("ai_agrees", "high")).lower()
    if ai_agrees not in ("high", "medium", "low"):
        ai_agrees = "high"
    alt_regime = parsed.get("ai_alternative_regime")
    if alt_regime is not None:
        alt_regime = str(alt_regime).lower()
        if alt_regime not in ("reflation", "stagflation", "deflation", "goldilocks"):
            alt_regime = None
    regime_challenge = parsed.get("regime_challenge_reasoning")
    regime_challenge = str(regime_challenge)[:800] if regime_challenge else None

    # Upsert
    if existing:
        existing.data_hash = h
        existing.daily_summary = daily_summary
        existing.decisions_reasoning_json = json.dumps(dec_reasoning_clipped)
        existing.regime_classifier_says = regime
        existing.ai_agrees = ai_agrees
        existing.ai_alternative_regime = alt_regime
        existing.regime_challenge_reasoning = regime_challenge
        existing.provider = model
        out = existing
    else:
        out = AiPortfolioReasoning(
            date=target_date,
            data_hash=h,
            daily_summary=daily_summary,
            decisions_reasoning_json=json.dumps(dec_reasoning_clipped),
            regime_classifier_says=regime,
            ai_agrees=ai_agrees,
            ai_alternative_regime=alt_regime,
            regime_challenge_reasoning=regime_challenge,
            provider=model,
        )
        db.add(out)
    db.commit()
    db.refresh(out)
    return out


def latest_reasoning(db: Session) -> Optional[AiPortfolioReasoning]:
    return (
        db.query(AiPortfolioReasoning)
        .order_by(AiPortfolioReasoning.date.desc())
        .first()
    )
