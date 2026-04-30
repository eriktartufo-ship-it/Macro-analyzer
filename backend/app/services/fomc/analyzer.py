"""LLM analyzer per FOMC statements/minutes.

Provider stack: Claude (preferito) → Groq Llama 3.3 70B (fallback).
Estrae:
  - hawkish_dovish_score: -1.0 (very dovish, easing) → +1.0 (very hawkish, tightening)
  - confidence: 0..1 quanto il testo e' chiaro nella direzione
  - key_topics: lista di temi macro citati (inflation, employment, growth, ecc.)
  - forward_guidance: stringa breve sul forward path
  - regime_implication: dict con nudge per i 4 regimi (-0.2..+0.2 per regime)
  - summary: 2-3 frasi in italiano

I documenti sono in inglese, l'output e' strutturato JSON. Cache aggressiva: la
stessa coppia (url, llm_version) non viene ri-analizzata.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import requests
from loguru import logger

from app.config import settings
from app.services.fomc.fetcher import FOMCDocument


_ANALYSIS_CACHE_ROOT = Path(__file__).resolve().parents[3] / ".cache" / "fomc_analysis"

_CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
_CLAUDE_MODEL = "claude-haiku-4-5-20251001"  # fast + cheap, sufficiente per FOMC analysis

_GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_MODEL = "llama-3.3-70b-versatile"


_SYSTEM_PROMPT = """You are a senior macro economist at a major investment bank.
Analyze the following FOMC document (statement or minutes) and produce a structured assessment.

Provide JSON with EXACTLY these fields (no markdown, no commentary):
{
  "hawkish_dovish_score": <float -1.0 to +1.0>,
  "confidence": <float 0.0 to 1.0>,
  "key_topics": [<3-6 short topic strings>],
  "forward_guidance": "<one-sentence summary of forward path>",
  "regime_implication": {
    "reflation": <float -0.2 to +0.2>,
    "stagflation": <float -0.2 to +0.2>,
    "deflation": <float -0.2 to +0.2>,
    "goldilocks": <float -0.2 to +0.2>
  },
  "summary": "<2-3 sentences in Italian summarizing the key macro takeaway>"
}

Definitions:
- hawkish_dovish_score: -1.0 = very dovish (easing bias, dovish forward guidance, focus on labor market weakness),
  0.0 = neutral, +1.0 = very hawkish (tightening bias, focus on persistent inflation, hawkish forward guidance).
- confidence: 1.0 if the document explicitly signals a direction, 0.5 if mixed/ambiguous, 0.0 if no signal.
- key_topics: short snake_case strings like "inflation_persistent", "labor_softening", "balance_sheet", "tariffs", "growth_moderating".
- regime_implication: how this FOMC stance shifts probability for each macro regime. Hawkish + sticky inflation = +stagflation/-goldilocks.
  Dovish + slowing growth = +deflation/+goldilocks. Sum across regimes ~ 0 (it's a relative shift).
- summary: in italian, 2-3 sentences max, focus on what this means for asset allocation.

Strip boilerplate (website headers, navigation menus). Focus on substance: rate decisions, economic projections, forward guidance language."""


@dataclass
class FOMCAnalysis:
    url: str
    doc_type: str
    published_date: date
    title: str
    hawkish_dovish_score: float       # -1..+1
    confidence: float                  # 0..1
    key_topics: list[str]
    forward_guidance: str
    regime_implication: dict[str, float]
    summary: str
    provider: str                      # "claude" | "groq"
    analyzed_at: datetime


def _safe_parse_json(raw: str) -> dict | None:
    """Tenta parsing JSON tollerante a markdown fences ed extra text."""
    if not raw:
        return None
    # Rimuovi markdown code fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())
    # Trova primo blocco {...}
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _validate_analysis(data: dict, doc: FOMCDocument, provider: str) -> FOMCAnalysis:
    """Estrae campi con default safe + clamping."""
    score = float(data.get("hawkish_dovish_score", 0.0))
    score = max(-1.0, min(1.0, score))

    conf = float(data.get("confidence", 0.5))
    conf = max(0.0, min(1.0, conf))

    topics = data.get("key_topics", [])
    if not isinstance(topics, list):
        topics = []
    topics = [str(t)[:60] for t in topics][:8]

    guidance = str(data.get("forward_guidance", ""))[:300]

    impl = data.get("regime_implication", {}) or {}
    regime_impl = {}
    for r in ("reflation", "stagflation", "deflation", "goldilocks"):
        v = float(impl.get(r, 0.0)) if isinstance(impl.get(r), (int, float)) else 0.0
        regime_impl[r] = max(-0.2, min(0.2, v))

    summary = str(data.get("summary", ""))[:600]

    return FOMCAnalysis(
        url=doc.url,
        doc_type=doc.doc_type,
        published_date=doc.published_date,
        title=doc.title,
        hawkish_dovish_score=score,
        confidence=conf,
        key_topics=topics,
        forward_guidance=guidance,
        regime_implication=regime_impl,
        summary=summary,
        provider=provider,
        analyzed_at=datetime.now(),
    )


def _truncate_text(text: str, max_chars: int = 18000) -> str:
    """Tronca preservando l'inizio (statement/minutes hanno il succo all'inizio)."""
    return text[:max_chars] if len(text) > max_chars else text


def _call_claude(prompt: str, system: str = _SYSTEM_PROMPT) -> str | None:
    if not settings.anthropic_api_key:
        return None
    try:
        resp = requests.post(
            _CLAUDE_API_URL,
            headers={
                "x-api-key": settings.anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": _CLAUDE_MODEL,
                "max_tokens": 1500,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        # Estrai testo dalla risposta Claude (lista content blocks)
        content = data.get("content", [])
        if not content:
            return None
        return content[0].get("text", "") if isinstance(content[0], dict) else None
    except Exception as e:
        logger.warning(f"Claude API call failed: {e}")
        return None


def _call_groq(prompt: str, system: str = _SYSTEM_PROMPT) -> str | None:
    if not settings.groq_api_key:
        return None
    try:
        resp = requests.post(
            _GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {settings.groq_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": _GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 1500,
                "response_format": {"type": "json_object"},
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.warning(f"Groq API call failed: {e}")
        return None


def _cache_path(url: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9]", "_", url)[-80:]
    return _ANALYSIS_CACHE_ROOT / f"{safe}.json"


def _load_cache(url: str) -> FOMCAnalysis | None:
    p = _cache_path(url)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return FOMCAnalysis(
            url=data["url"], doc_type=data["doc_type"],
            published_date=date.fromisoformat(data["published_date"]),
            title=data["title"],
            hawkish_dovish_score=float(data["hawkish_dovish_score"]),
            confidence=float(data["confidence"]),
            key_topics=list(data.get("key_topics", [])),
            forward_guidance=str(data.get("forward_guidance", "")),
            regime_implication=dict(data.get("regime_implication", {})),
            summary=str(data.get("summary", "")),
            provider=str(data.get("provider", "unknown")),
            analyzed_at=datetime.fromisoformat(data["analyzed_at"]),
        )
    except Exception as e:
        logger.warning(f"FOMC cache load failed for {url}: {e}")
        return None


def _save_cache(analysis: FOMCAnalysis) -> None:
    _ANALYSIS_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    p = _cache_path(analysis.url)
    payload = asdict(analysis)
    payload["published_date"] = analysis.published_date.isoformat()
    payload["analyzed_at"] = analysis.analyzed_at.isoformat()
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def analyze_fomc_document(doc: FOMCDocument, force_refresh: bool = False) -> FOMCAnalysis | None:
    """Analizza un documento FOMC con LLM. Usa cache su disco per evitare re-call."""
    if not force_refresh:
        cached = _load_cache(doc.url)
        if cached:
            return cached

    if not doc.text or len(doc.text) < 200:
        logger.warning(f"FOMC: documento troppo corto per analisi ({len(doc.text)} chars)")
        return None

    prompt = (
        f"Document type: {doc.doc_type}\n"
        f"Title: {doc.title}\n"
        f"Published: {doc.published_date}\n\n"
        f"--- Document text ---\n{_truncate_text(doc.text)}"
    )

    # Prova Claude per primo, poi Groq
    raw = None
    provider = "unknown"
    if settings.anthropic_api_key:
        raw = _call_claude(prompt)
        provider = "claude"
    if not raw and settings.groq_api_key:
        raw = _call_groq(prompt)
        provider = "groq"
    if not raw:
        logger.warning(f"FOMC analyzer: nessun provider LLM disponibile o tutti falliti")
        return None

    parsed = _safe_parse_json(raw)
    if not parsed:
        logger.warning(f"FOMC analyzer: JSON parsing fallito, raw head: {raw[:200]}")
        return None

    analysis = _validate_analysis(parsed, doc, provider)
    _save_cache(analysis)
    logger.info(
        f"FOMC analyzed [{provider}]: {doc.published_date} {doc.doc_type} "
        f"hawkish/dovish={analysis.hawkish_dovish_score:+.2f} confidence={analysis.confidence:.2f}"
    )
    return analysis
