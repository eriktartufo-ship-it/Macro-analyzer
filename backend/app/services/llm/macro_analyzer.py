"""LLM Macro Analysis service — Gemini 2.5 Flash.

User request 2026-05-31: pannello fisso con analisi macro da LLM, stile dedollar,
no re-fetch ogni reload (cache disco TTL 24h).

Pattern: clone di `services/fomc/analyzer.py` con prompt tuned per macro overall.

Input: regime corrente + probabilities + confidence + key indicators + top-10 scoreboard.
Output JSON structured:
- headline (1 frase incisiva)
- regime_thesis (2-3 frasi sul perché del regime)
- key_drivers (3-5 driver chiari)
- risks (3-5 rischi tail)
- opportunities (3-5 opportunità asset/posizioni)
- time_horizon (es. "3-6 mesi")
- confidence_qualitative (low/medium/high)
- provider + cached_at metadata

Cache: file JSON disco `.cache/macro_llm/<date>_<regime>.json` TTL 24h.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests

from app.services.llm import settings as llm_settings

logger = logging.getLogger(__name__)

# Cache directory (relative to backend root)
_CACHE_DIR = Path(__file__).resolve().parents[3] / ".cache" / "macro_llm"
_CACHE_TTL_HOURS = 24 * 7  # 7 giorni TTL safety, ma cache invalidata anche da hash dati

_GEMINI_API_URL_TPL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/{model}:generateContent"
)

# Versione del prompt — incrementa quando cambi il prompt per invalidare la cache esistente
_PROMPT_VERSION = "v2-italian"

_SYSTEM_PROMPT = """Sei un analista macro senior di un hedge fund top-tier (Bridgewater/Citadel style).
Stile: caveman density, NO convenevoli, NO disclaimer generici, NO "consigli finanziari".
Tono: professionale ma incisivo, come Soros o Dalio in una nota interna.

**LINGUA OBBLIGATORIA: ITALIANO.** Tutti i campi (headline, regime_thesis, key_drivers,
risks, opportunities, time_horizon) DEVONO essere scritti in italiano corretto e
professionale. Termini tecnici inglesi acronimi sono ammessi (CPI, GDP, PCE, FOMC,
S&P500, MA, YoY) ma frasi e ragionamenti SEMPRE in italiano. Per `confidence_qualitative`
usa SOLO i valori `low`/`medium`/`high` (sono enum, non testo).

Devi produrre un'analisi macro JSON-only basata SOLO sui dati forniti.
NON inventare dati. NON citare numeri non presenti nell'input.
Cita SOLO ciò che vedi negli indicators + scoreboard.

Output JSON schema:
{
  "headline": "1 frase incisiva in italiano (max 100 char)",
  "regime_thesis": "2-3 frasi in italiano: perché il classifier vede questo regime, contesto storico breve",
  "key_drivers": ["driver 1 in italiano con numero specifico", "driver 2", "driver 3", "driver 4", "driver 5"],
  "risks": ["rischio 1 in italiano", "rischio 2", "rischio 3"],
  "opportunities": ["opportunità 1 in italiano (asset/posizione)", "opp 2", "opp 3"],
  "time_horizon": "X-Y mesi/trimestri (in italiano)",
  "confidence_qualitative": "low|medium|high"
}

NO markdown fences. Output JSON puro. Lingua: ITALIANO obbligatorio."""


def _compute_data_hash(
    regime: str,
    probabilities: dict[str, float],
    confidence: float,
    indicators: dict[str, float],
    scoreboard: dict[str, float],
    model: str,
) -> str:
    """MD5 hash dei dati che influenzano l'output LLM.

    Se i dati sono uguali → stesso hash → cache hit (no Gemini call).
    Cambio API → MAYBE different output → diverso modello incluso nel hash.

    Round numeri a 3 decimali per evitare invalidazioni spurie da floating-point.
    """
    def _round_dict(d: dict, ndigits: int = 3) -> dict:
        out = {}
        for k, v in d.items():
            if isinstance(v, (int, float)):
                out[k] = round(float(v), ndigits)
            else:
                out[k] = v
        return out

    canonical = {
        "regime": regime,
        "probabilities": _round_dict(probabilities, 3),
        "confidence": round(float(confidence), 3),
        "indicators": _round_dict(indicators, 3),
        "scoreboard": _round_dict(scoreboard, 1),  # 1 decimal sufficient per asset score
        "model": model,
        "prompt_version": _PROMPT_VERSION,  # cambio prompt → invalidazione automatica cache
    }
    canonical_json = json.dumps(canonical, sort_keys=True)
    return hashlib.md5(canonical_json.encode("utf-8")).hexdigest()


def _cache_key(data_hash: str) -> str:
    """Cache filename basato sull'hash dei dati (NON su date+regime).

    Vantaggio: se nello stesso giorno i dati cambiano (es. refresh midday),
    nuovo hash → nuovo cache. Se dati identici per giorni → 1 sola call LLM.
    """
    return f"{data_hash}.json"


def _cache_path(key: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / key


def _read_cache(key: str) -> Optional[dict]:
    """Read cache if exists + not expired."""
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        cached_at_str = data.get("cached_at")
        if cached_at_str:
            cached_at = datetime.fromisoformat(cached_at_str)
            if cached_at.tzinfo is None:
                cached_at = cached_at.replace(tzinfo=timezone.utc)
            age = datetime.now(timezone.utc) - cached_at
            if age > timedelta(hours=_CACHE_TTL_HOURS):
                logger.info(f"macro_llm cache expired: {key} (age {age})")
                return None
        return data
    except Exception as e:
        logger.warning(f"macro_llm cache read failed for {key}: {e}")
        return None


def _write_cache(key: str, payload: dict) -> None:
    try:
        path = _cache_path(key)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"macro_llm cache write failed for {key}: {e}")


def _build_prompt(
    regime: str,
    probabilities: dict,
    confidence: float,
    indicators: dict,
    top_scores: list[tuple[str, float]],
    fit_scores: dict | None = None,
) -> str:
    """Costruisce il prompt user per Gemini."""
    probs_str = ", ".join(f"{k}={v:.2%}" for k, v in sorted(probabilities.items(), key=lambda x: -x[1]))
    indicators_str = "\n".join(
        f"  {k}: {v:.3f}" if isinstance(v, (int, float)) else f"  {k}: {v}"
        for k, v in sorted(indicators.items())
        if v is not None
    )
    scores_str = "\n".join(f"  {asset}: {score:.1f}" for asset, score in top_scores[:10])
    fit_str = ""
    if fit_scores:
        fit_str = "\nFit scores raw (consistency con regime, max 1.0):\n" + "\n".join(
            f"  {r}: {s:.3f}" for r, s in sorted(fit_scores.items(), key=lambda x: -x[1])
        )

    return f"""## REGIME CORRENTE (classifier macro 4 quadranti)

Regime dominante: **{regime.upper()}**
Confidence: {confidence:.2%}
Probabilità: {probs_str}
{fit_str}

## INDICATORS MACRO (dati FRED ultimi disponibili)

{indicators_str}

## TOP-10 ASSET SCORES (regime corrente, 0-100 scale)

{scores_str}

## RICHIESTA

Produci analisi JSON come da schema. Riferimenti specifici a numeri sopra.
Non aggiungere disclaimer. Non inventare dati. Sii incisivo come un analista
che scrive una nota interna al CIO."""


def _call_gemini(prompt: str, api_key: str, model: str) -> Optional[str]:
    """Chiama Gemini con api_key + model passati esplicitamente.

    Args:
        prompt: user prompt completo
        api_key: Gemini API key (non None — caller verifica)
        model: model name (es. "gemini-2.5-flash")
    """
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
                    "maxOutputTokens": 2000,
                    "responseMimeType": "application/json",
                    "thinkingConfig": {"thinkingBudget": 0},
                },
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            return None
        parts = candidates[0].get("content", {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts).strip()
        return text or None
    except Exception as e:
        logger.warning(f"Gemini macro analysis call failed (model={model}): {e}")
        return None


def _validate_response(raw: dict) -> dict:
    """Valida + clamp campi response. Defaults per missing fields."""
    return {
        "headline": str(raw.get("headline", "Analisi non disponibile"))[:200],
        "regime_thesis": str(raw.get("regime_thesis", ""))[:600],
        "key_drivers": [str(x)[:200] for x in (raw.get("key_drivers") or [])][:5],
        "risks": [str(x)[:200] for x in (raw.get("risks") or [])][:5],
        "opportunities": [str(x)[:200] for x in (raw.get("opportunities") or [])][:5],
        "time_horizon": str(raw.get("time_horizon", "3-6 mesi"))[:50],
        "confidence_qualitative": str(raw.get("confidence_qualitative", "medium"))[:20],
    }


def get_macro_analysis(
    regime: str,
    probabilities: dict[str, float],
    confidence: float,
    indicators: dict[str, float],
    scoreboard: dict[str, float],
    classification_date: str,
    fit_scores: dict[str, float] | None = None,
    force_refresh: bool = False,
) -> Optional[dict]:
    """Entry point: ritorna analisi LLM da cache o fresh fetch.

    Args:
        regime: regime dominante stringa
        probabilities: dict {regime: prob}
        confidence: 0-1
        indicators: dict raw indicators
        scoreboard: dict {asset: score}
        classification_date: 'YYYY-MM-DD' (chiave cache)
        fit_scores: dict raw fit per regime (opzionale)
        force_refresh: bypass cache

    Returns:
        dict con analysis + metadata o None se LLM unavailable + nessuna cache.
    """
    # Resolve runtime settings (api_key + model)
    api_key = llm_settings.get_api_key()
    model = llm_settings.get_model()

    # Hash dei dati: se identici → cache hit (no LLM call)
    data_hash = _compute_data_hash(
        regime, probabilities, confidence, indicators, scoreboard, model,
    )
    cache_key = _cache_key(data_hash)

    # Cache hit (data hash match)
    if not force_refresh:
        cached = _read_cache(cache_key)
        if cached is not None:
            cached["cache_hit"] = True
            return cached

    # Cache miss → necessaria chiamata LLM. Verifica API key.
    if not api_key:
        # Last resort: prova ad usare qualsiasi cache (anche stale) per non lasciare UI vuota
        cached = _read_cache(cache_key)
        if cached is not None:
            cached["cache_hit"] = True
            cached["stale"] = True
            return cached
        return None

    top_scores = sorted(scoreboard.items(), key=lambda x: -x[1])[:15]
    prompt = _build_prompt(regime, probabilities, confidence, indicators, top_scores, fit_scores)
    raw = _call_gemini(prompt, api_key=api_key, model=model)
    if raw is None:
        # LLM call failed → return stale cache se esiste
        cached = _read_cache(cache_key)
        if cached is not None:
            cached["cache_hit"] = True
            cached["stale"] = True
            return cached
        return None

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning(f"LLM returned invalid JSON: {e}, raw: {raw[:200]}")
        return None

    validated = _validate_response(parsed)
    payload = {
        **validated,
        "regime": regime,
        "classification_date": classification_date,
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "provider": model,
        "data_hash": data_hash,
        "cache_hit": False,
    }
    _write_cache(cache_key, payload)
    return payload
