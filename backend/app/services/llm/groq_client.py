"""Client Groq UNICO del Macro Analyzer (2026-10-21).

🔴 Il difetto che chiude: il nome del modello era scritto a mano in TRE file (punteggio delle
notizie, ripiego dell'analisi FOMC, ripiego dell'advisor dei portafogli). Groq ha ritirato
`llama-3.3-70b-versatile` ⇒ 404 `model_not_found` a ogni chiamata, e per almeno un mese il sito
non ha valutato una sola notizia, senza un errore visibile. È la stessa forma del difetto Gemini
chiuso nella S48 (`services/llm/settings`): un nome di modello in un posto che nessuno aggiorna.

Ora: un posto solo per il modello (`GROQ_MODEL`, default sotto) e per i parametri, e un 404 di
modello ritirato finisce nei log con un messaggio che dice COSA fare, non un generico «failed».

Sostituto scelto il 2026-10-21 su un banco col compito vero (20 notizie di oggi, prompt reale):
gpt-oss-120b 3,8 s · gpt-oss-20b 2,1 s · qwen3.8-27b 5,0 s — tutti JSON valido, 20/20. Scelto il
più capace dei tre; la qualità dei punteggi NON è misurata su un golden set (DA MISURARE).
"""
from __future__ import annotations

from typing import Optional

import requests
from loguru import logger

from app.config import settings

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"


def get_model() -> str:
    return (settings.groq_model or "").strip() or DEFAULT_MODEL


def request_body(system: str, user: str, max_tokens: int, temperature: float = 0.1,
                 model: Optional[str] = None) -> dict:
    model = model or get_model()
    body: dict = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if model.startswith("openai/gpt-oss"):
        # modelli che ragionano: i token del ragionamento contano in max_tokens; "low" basta per
        # un JSON di punteggi e tiene bassa la latenza (3,8 s misurati su 20 notizie)
        body["reasoning_effort"] = "low"
    return body


def chat_json(system: str, user: str, max_tokens: int, temperature: float = 0.1,
              timeout: int = 60, label: str = "groq") -> Optional[str]:
    """Contenuto JSON della risposta, o None (degradazione morbida: il chiamante non crasha)."""
    if not settings.groq_api_key:
        return None
    model = get_model()
    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {settings.groq_api_key}", "Content-Type": "application/json"},
            json=request_body(system, user, max_tokens, temperature, model),
            timeout=timeout,
        )
        if resp.status_code == 404 and "model_not_found" in resp.text:
            logger.error(f"{label}: Groq non ha più il modello `{model}` (ritirato). "
                         f"Imposta GROQ_MODEL con uno di GET https://api.groq.com/openai/v1/models")
            return None
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:  # noqa: BLE001
        logger.warning(f"{label}: chiamata Groq fallita ({model}): {e}")
        return None
