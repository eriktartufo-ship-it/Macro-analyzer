"""Advisor buy-timing (NO trading) per oro/argento/BTC — Gemini con fallback Groq.

Obiettivo: aiutare a capire se è un momento favorevole per ACCUMULARE (DCA),
non segnali di trading. Output = semaforo per asset (green/yellow/red) +
spiegazione in italiano semplice, per un lettore non tecnico.

Contesto passato all'LLM (tutto già disponibile nel Macro Analyzer):
- regime corrente + indicatori dal JSON conditions_met dell'ultima classificazione
  (stesso pattern di ai_portfolio/llm_strategist)
- stats di mercato calcolate qui: spot vs media 200gg, variazione 1y,
  gold/silver ratio

Cadenza: job settimanale (lunedì) + refresh manuale rate-limited dall'UI.
Graceful degradation: senza API key o senza regime → None, mai 500.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import requests
from loguru import logger
from sqlalchemy.orm import Session

from app.config import settings
from app.models import PtAdvice, RegimeClassification
from app.services.llm import settings as llm_settings

_GEMINI_URL_TPL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
_GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_MODEL = "llama-3.3-70b-versatile"

ADVICE_ASSETS = ("gold", "silver", "bitcoin")
_LIGHTS = {"green", "yellow", "red"}

# Indicatori più rilevanti per metalli preziosi + BTC (subset del set completo)
_RELEVANT_INDICATORS = [
    "cpi_yoy", "core_pce_yoy", "breakeven_10y", "umich_inflation_1y",
    "fed_funds_rate", "fed_funds_implied_12m", "fed_path_12m",
    "net_liquidity_usd_bn", "net_liquidity_roc_3m", "m2_yoy",
    "yield_curve_10y2y", "term_premium_10y",
    "vix", "move_index", "nfci",
    "cot_gold_z", "cot_usd_z",
    "copper_gold_ratio", "gold_oil_ratio",
    "dedollar_combined",
]

_SYSTEM_PROMPT = (
    "Sei un analista macro. Aiuti un risparmiatore NON esperto che accumula "
    "oro fisico, argento e Bitcoin nel lungo periodo (niente trading, mai leva). "
    "Per ogni asset dai un semaforo di ACCUMULO: green=momento favorevole per "
    "comprare, yellow=neutro/comprare con moderazione, red=caro o rischioso, "
    "meglio aspettare. Linguaggio semplice, niente jargon, frasi brevi. "
    "Rispondi SOLO JSON."
)


def _market_stats(db: Session) -> dict:
    """Spot vs 200dma e var 1y per gold/silver/BTC + gold/silver ratio."""
    stats: dict[str, dict] = {}
    try:
        from datetime import date

        from app.services.prices.yahoo_fetcher import YahooFetcher

        fetcher = YahooFetcher()
        end = date.today()
        start = end - timedelta(days=420)
        series = {}
        for key, ticker in (("gold", "GC=F"), ("silver", "SI=F"), ("bitcoin", "BTC-USD")):
            try:
                s = fetcher.fetch(ticker, start, end)
                if s is None or len(s) < 30:
                    continue
                series[key] = s
                last = float(s.iloc[-1])
                ma200 = float(s.tail(200).mean())
                year_ago = float(s.iloc[0])
                stats[key] = {
                    "spot_usd": round(last, 2),
                    "vs_200dma_pct": round((last / ma200 - 1) * 100, 1),
                    "change_1y_pct": round((last / year_ago - 1) * 100, 1),
                }
            except Exception:
                continue
        if "gold" in series and "silver" in series:
            stats["gold_silver_ratio"] = round(
                float(series["gold"].iloc[-1]) / float(series["silver"].iloc[-1]), 1,
            )
    except Exception as e:
        logger.warning(f"pt advisor market stats failed: {e}")
    return stats


def _macro_context(db: Session) -> dict | None:
    latest = (
        db.query(RegimeClassification)
        .order_by(RegimeClassification.date.desc())
        .first()
    )
    if latest is None:
        return None
    indicators = {}
    try:
        if latest.conditions_met:
            indicators = json.loads(latest.conditions_met).get("indicators", {}) or {}
    except Exception:
        pass
    return {
        "regime": latest.regime,
        "confidence": float(latest.confidence or 0.5),
        "date": latest.date.isoformat(),
        "indicators": {
            k: indicators[k] for k in _RELEVANT_INDICATORS if indicators.get(k) is not None
        },
    }


def _build_prompt(macro: dict, market: dict) -> str:
    ind_lines = "\n".join(f"- {k}: {float(v):.3f}" for k, v in macro["indicators"].items())
    return f"""## CONTESTO MACRO ({macro['date']})
Regime classifier: {macro['regime'].upper()} (confidence {macro['confidence']:.0%})

## INDICATORI (valori correnti)
{ind_lines or "(non disponibili)"}

## MERCATO
{json.dumps(market, indent=1)}

## RICHIESTA
Per ognuno di: gold, silver, bitcoin → semaforo di ACCUMULO di lungo periodo.
Il lettore compra poco alla volta ogni mese e vuole solo sapere se questo è un
momento buono, normale o cattivo per la sua prossima tranche.

Output JSON:
{{
 "macro_summary": "2-3 frasi semplici sul quadro macro attuale (max 400 char)",
 "assets": {{
  "gold": {{"light": "green|yellow|red", "headline": "1 frase (max 90 char)",
            "reasons": ["2-4 motivi, 1 frase ciascuno, citando i numeri sopra"],
            "what_to_watch": "1 cosa concreta da tenere d'occhio"}},
  "silver": {{...}},
  "bitcoin": {{...}}
 }}
}}"""


def _call_gemini(prompt: str) -> str | None:
    api_key = llm_settings.get_api_key()
    if not api_key:
        return None
    try:
        resp = requests.post(
            _GEMINI_URL_TPL.format(model=llm_settings.get_model()),
            params={"key": api_key},
            json={
                "contents": [{"parts": [{"text": f"{_SYSTEM_PROMPT}\n\n---\n\n{prompt}"}]}],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 2000,
                    "responseMimeType": "application/json",
                    "thinkingConfig": {"thinkingBudget": 0},
                },
            },
            timeout=60,
        )
        resp.raise_for_status()
        candidates = resp.json().get("candidates") or []
        if not candidates:
            return None
        parts = candidates[0].get("content", {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts).strip()
        return text or None
    except Exception as e:
        logger.warning(f"pt advisor gemini failed: {e}")
        return None


def _call_groq(prompt: str) -> str | None:
    if not settings.groq_api_key:
        return None
    try:
        resp = requests.post(
            _GROQ_API_URL,
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
            json={
                "model": _GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 2000,
                "response_format": {"type": "json_object"},
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.warning(f"pt advisor groq failed: {e}")
        return None


def _safe_parse_json(raw: str | None) -> dict | None:
    if not raw:
        return None
    import re

    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def validate_advice(data: dict) -> dict | None:
    """Clamping/default sui campi LLM. None se manca la sostanza."""
    if not isinstance(data, dict):
        return None
    assets_in = data.get("assets") or {}
    assets_out = {}
    for key in ADVICE_ASSETS:
        a = assets_in.get(key)
        if not isinstance(a, dict):
            continue
        light = str(a.get("light", "yellow")).lower()
        if light not in _LIGHTS:
            light = "yellow"
        reasons = a.get("reasons", [])
        if not isinstance(reasons, list):
            reasons = []
        assets_out[key] = {
            "light": light,
            "headline": str(a.get("headline", ""))[:120],
            "reasons": [str(r)[:200] for r in reasons][:4],
            "what_to_watch": str(a.get("what_to_watch", ""))[:200],
        }
    if not assets_out:
        return None
    return {
        "macro_summary": str(data.get("macro_summary", ""))[:500],
        "assets": assets_out,
    }


def generate_advice(db: Session, trigger: str = "scheduled") -> PtAdvice | None:
    """Genera e persiste un nuovo advice. None se contesto/LLM non disponibili."""
    macro = _macro_context(db)
    if macro is None:
        logger.warning("pt advisor: nessuna regime classification, skip")
        return None
    market = _market_stats(db)
    prompt = _build_prompt(macro, market)

    raw = _call_gemini(prompt)
    provider = "gemini"
    if raw is None:
        raw = _call_groq(prompt)
        provider = "groq"
    advice = validate_advice(_safe_parse_json(raw) or {})
    if advice is None:
        logger.warning("pt advisor: nessun output LLM valido")
        return None

    advice["regime"] = macro["regime"]
    advice["regime_confidence"] = macro["confidence"]
    advice["market"] = market
    row = PtAdvice(
        trigger=trigger,
        provider=provider,
        content_json=json.dumps(advice, ensure_ascii=False),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    logger.info(f"pt advisor: advice #{row.id} generato via {provider} ({trigger})")
    return row


def latest_advice(db: Session) -> dict | None:
    row = db.query(PtAdvice).order_by(PtAdvice.generated_at.desc()).first()
    if row is None:
        return None
    try:
        content = json.loads(row.content_json)
    except Exception:
        return None
    return {
        "id": row.id,
        "generated_at": row.generated_at.isoformat(),
        "trigger": row.trigger,
        "provider": row.provider,
        **content,
    }


def can_manual_refresh(db: Session, min_interval: timedelta = timedelta(hours=1)) -> bool:
    """Rate-limit del bottone: max 1 refresh manuale/ora."""
    row = (
        db.query(PtAdvice)
        .filter(PtAdvice.trigger == "manual")
        .order_by(PtAdvice.generated_at.desc())
        .first()
    )
    return row is None or datetime.utcnow() - row.generated_at >= min_interval


def weekly_advice_job() -> None:
    """Job APScheduler: advice settimanale (lunedì)."""
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        generate_advice(db, trigger="scheduled")
    except Exception as e:
        logger.error(f"pt weekly advice job failed: {e}")
    finally:
        db.close()
