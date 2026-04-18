"""News sentiment scorer via Groq API (Llama 3.3 70B).

Prende un batch di titoli/notizie e chiede al LLM di:
  1. Valutare il sentiment macro (bearish → bullish)
  2. Stimare la rilevanza per l'analisi macro
  3. Identificare quali asset class sono impattati e come
"""

import json
from typing import Any

import requests
from loguru import logger

from app.config import settings
from app.services.scoring.engine import ASSET_CLASSES

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a macro-economic analyst. Given a list of recent financial news headlines, analyze each one and provide:

1. **sentiment**: Score from -1.0 (very bearish for global economy) to +1.0 (very bullish)
2. **relevance**: Score from 0.0 (irrelevant to macro analysis) to 1.0 (highly relevant)
3. **summary**: One sentence explanation in English
4. **affected_assets**: Dict of asset classes impacted, with score from -1.0 (very negative for that asset) to +1.0 (very positive)

Available asset classes: us_equities_growth, us_equities_value, international_dm_equities, em_equities, us_bonds_short, us_bonds_long, tips_inflation_bonds, gold, silver, broad_commodities, energy, real_estate_reits, cash_money_market, bitcoin, crypto_broad

Respond with a JSON array. Example:
[
  {
    "headline": "Fed raises rates by 50bps",
    "sentiment": -0.4,
    "relevance": 0.95,
    "summary": "Aggressive rate hike signals tighter monetary conditions ahead",
    "affected_assets": {"us_bonds_long": -0.8, "us_equities_growth": -0.5, "gold": -0.3, "cash_money_market": 0.4}
  }
]

Only include asset classes that are meaningfully affected (|score| > 0.1).
If a headline is not related to economics/finance, set relevance to 0 and skip affected_assets.
Respond ONLY with valid JSON, no markdown."""


def parse_llm_response(response: Any) -> dict:
    """Parsa e valida la risposta del LLM per un singolo headline.

    Returns:
        {sentiment, relevance, summary, affected_assets}
    """
    if response is None or not isinstance(response, dict):
        return {
            "sentiment": 0.0,
            "relevance": 0.0,
            "summary": "",
            "affected_assets": {},
        }

    sentiment = float(response.get("sentiment", 0.0))
    relevance = float(response.get("relevance", 0.0))
    summary = str(response.get("summary", ""))
    affected = response.get("affected_assets", {})

    # Clamp
    sentiment = max(-1.0, min(1.0, sentiment))
    relevance = max(0.0, min(1.0, relevance))

    # Filtra asset validi e clamp scores
    valid_assets = {}
    if isinstance(affected, dict):
        for asset, score in affected.items():
            if asset in ASSET_CLASSES:
                valid_assets[asset] = max(-1.0, min(1.0, float(score)))

    return {
        "sentiment": sentiment,
        "relevance": relevance,
        "summary": summary,
        "affected_assets": valid_assets,
    }


def score_news_batch(headlines: list[dict[str, str]]) -> list[dict]:
    """Invia un batch di headlines a Groq per scoring.

    Args:
        headlines: Lista di {title, url, source, date}

    Returns:
        Lista di scored news {headline, sentiment, relevance, summary, affected_assets}
    """
    if not headlines:
        return []

    if not settings.groq_api_key:
        logger.warning("GROQ_API_KEY non configurata, skip news scoring")
        return []

    # Prepara il prompt con i titoli
    titles_text = "\n".join(
        f"{i+1}. [{h['source']}] {h['title']}"
        for i, h in enumerate(headlines[:20])  # Max 20 per batch
    )

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {settings.groq_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Analyze these headlines:\n\n{titles_text}"},
                ],
                "temperature": 0.1,
                "max_tokens": 4000,
                "response_format": {"type": "json_object"},
            },
            timeout=30,
        )
        resp.raise_for_status()

        content = resp.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)

        # Groq con response_format json_object restituisce un dict wrapper
        # Cerchiamo la lista di risultati dentro qualsiasi chiave
        items: list = []
        if isinstance(parsed, list):
            items = parsed
        elif isinstance(parsed, dict):
            # Cerca la prima chiave che contiene una lista
            for key in ("results", "headlines", "analysis", "news", "items"):
                if key in parsed and isinstance(parsed[key], list):
                    items = parsed[key]
                    break
            if not items:
                # Prova qualsiasi valore che sia una lista
                for val in parsed.values():
                    if isinstance(val, list) and len(val) > 0:
                        items = val
                        break
                if not items:
                    # Singolo oggetto
                    items = [parsed]

        results = []
        for item in items:
            if isinstance(item, dict):
                results.append(parse_llm_response(item))

        logger.info(f"Groq scored {len(results)} headlines (from {len(items)} items)")
        return results

    except requests.exceptions.RequestException as e:
        logger.error(f"Errore Groq API: {e}")
        return []
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.error(f"Errore parsing risposta Groq: {e}")
        return []


def aggregate_signals(scored_news: list[dict]) -> dict[str, float]:
    """Aggrega i segnali di tutte le notizie in un singolo score per asset.

    Formula: media pesata per relevance, scalata a [-5, +5].
    Il segnale decade con il tempo (notizie recenti pesano di più).

    Returns:
        Dict {asset_class: signal} dove signal è in [-5, +5]
    """
    if not scored_news:
        return {asset: 0.0 for asset in ASSET_CLASSES}

    # Accumula per asset: (somma_pesata, somma_pesi)
    accum: dict[str, list[float]] = {asset: [0.0, 0.0] for asset in ASSET_CLASSES}

    for news in scored_news:
        relevance = news.get("relevance", 0.0)
        if relevance < 0.1:
            continue

        affected = news.get("affected_assets", {})
        for asset, impact in affected.items():
            if asset in accum:
                accum[asset][0] += impact * relevance
                accum[asset][1] += relevance

    # Calcola media pesata e scala a [-5, +5]
    signals: dict[str, float] = {}
    for asset in ASSET_CLASSES:
        total_weighted, total_weight = accum[asset]
        if total_weight > 0:
            avg = total_weighted / total_weight  # [-1, +1]
            # Scala: il segnale è più forte se ci sono più notizie concordi
            strength = min(1.0, total_weight / 3.0)  # 3+ notizie = full strength
            signals[asset] = round(avg * strength * 5.0, 2)  # scala a [-5, +5]
        else:
            signals[asset] = 0.0

    # Clamp finale
    for asset in signals:
        signals[asset] = max(-5.0, min(5.0, signals[asset]))

    return signals
