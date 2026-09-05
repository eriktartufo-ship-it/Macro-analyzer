"""LLM runtime settings — Gemini API key + model configurable from UI.

User request 2026-06: cambiare la API key e il modello Gemini per la lettura
macro senza redeploy.

Storage: `.cache/llm_settings.json` su disco (process-local + persistent).

Resolution priority:
1. Runtime override (file `.cache/llm_settings.json`) → highest priority
2. Env var (`GEMINI_API_KEY_MACRO`, `GEMINI_MODEL_MACRO`) → fallback
3. Global config (`settings.gemini_api_key`, default model) → last resort

Available models: vedi `AVAILABLE_MODELS` sotto — è l'unica lista, tenerne una
seconda qui in prosa significa averne una sbagliata (lo era: elencava 2.5-flash
come default quando il default era già 3.5-flash).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from app.config import settings as global_settings

logger = logging.getLogger(__name__)

_SETTINGS_PATH = Path(__file__).resolve().parents[3] / ".cache" / "llm_settings.json"

# `gemini-3.8-flash` dal 2026-09-05. Era `gemini-3.5-flash`. Qui il cambio è
# facile da difendere perché NON è un compromesso: 3.8-flash è più recente,
# stable dal 2026-09-02, e costa **la metà** del 3.5-flash che sostituisce —
# $0.75/$3.75 per Mtok (promo fino al 31/12/2026, poi $1.50/$7.50) contro
# $1.50/$9.00 del 3.5, che non ha promo. Fonte: ai.google.dev/gemini-api/docs/pricing
# letta il 2026-09-05.
#
# ⚠️ La qualità è quella DICHIARATA da Google (batte 3.7 su tutti i benchmark
# pubblicati, primo su DeepSWE): sono benchmark di coding/agentic, non di lettura
# macro. Per Macro sono comunque il proxy più vicino al lavoro vero (ragionamento
# su testo), a differenza di Ormio dove il lavoro è OCR di calligrafia e quei
# benchmark non dicono nulla. Se l'analisi macro peggiora, il rollback è una riga
# qui oppure il selettore modello nel pannello (che vince su questo default).
DEFAULT_MODEL = "gemini-3.8-flash"

# Modelli Gemini disponibili (doc ai.google.dev/gemini-api/docs/models, 2026-09-05).
# Selezione: text reasoning models, no image/video/audio/embedding.
AVAILABLE_MODELS = [
    # Gemini 3.x — generazione corrente
    "gemini-3.8-flash",          # stable dal 02/09/2026, most intelligent (DEFAULT)
    "gemini-3.7-flash",          # stable, generazione precedente
    "gemini-3.6-flash",          # stable, in esecuzione su Ormio
    "gemini-3.5-flash",          # stable, ex-default (più caro del 3.8: $1.50/$9.00)
    "gemini-3.5-flash-lite",     # stable, reduced cost
    "gemini-3.1-flash-lite",     # stable, reduced cost
    "gemini-3.1-pro-preview",    # preview, max capability
    # Gemini 2.5 — generazione precedente, ancora stabile + free tier
    "gemini-2.5-flash",          # stable, il più economico dei non-lite ($0.30/$2.50)
    "gemini-2.5-flash-lite",     # cheapest, budget-friendly
    "gemini-2.5-pro",            # most advanced 2.5
]


def is_gemini_3(model: Optional[str] = None) -> bool:
    """True se il modello è della linea 3.x (regole di config diverse dalla 2.x)."""
    return (model or get_model()).lower().startswith("gemini-3")


def generation_config(
    *,
    max_output_tokens: int,
    model: Optional[str] = None,
    json_output: bool = True,
    thinking: str = "low",
    temperature: float = 0.2,
) -> dict:
    """Costruisce il `generationConfig` corretto per il modello — UNICO scrittore.

    Prima di questa funzione ogni call-site LLM del progetto teneva la propria
    copia di `{"temperature": X, "thinkingConfig": {"thinkingBudget": 0}}`.
    Sette copie della stessa decisione = sette posti dove sbagliarla al prossimo
    cambio di modello. Ora l'intenzione la dichiara il chiamante
    (quanto output, JSON o prosa, quanto deve pensare) e la TRADUZIONE nei
    parametri giusti per quella famiglia di modelli sta qui.

    Le due regole della linea 3.x, dalla doc ufficiale letta il 2026-09-05:

    🔴 **`temperature`/`topP`/`topK` sono deprecati e IGNORATI.**
    `ai.google.dev/gemini-api/docs/whats-new-gemini-3.5` dice di rimuoverli da
    tutte le richieste perché *"Gemini 3's reasoning capabilities are optimized
    for the default settings"*. Il guasto è che **non danno errore**: la
    richiesta torna 200 e il valore viene buttato via (i valori fuori range,
    quelli sì, danno 400). Quindi una `temperature` lasciata lì non è
    innocua — è una leva che qualcuno crede accesa. Per il determinismo la doc
    indica l'unico sostituto vero: *"define a system instruction with explicit
    rules for your specific use case"*, cioè si scrive nel PROMPT, non qui.

    🔴 **`thinkingBudget` → `thinkingLevel`**, e su 3.8 Flash il livello
    `minimal` **non esiste più** (su 3.6 sì): lì è un errore di validazione
    prima di generare un token. Poiché `thinkingBudget: 0` è esattamente
    "non pensare", portarlo su 3.8 avrebbe spento l'analisi LLM in silenzio —
    tutti i call-site catturano l'eccezione e tornano `None`, quindi nessun
    crash e nessuno se ne accorge.

    Sui modelli 2.x nulla di tutto questo vale: `temperature` funziona davvero e
    `thinkingBudget` è il parametro giusto. Per questo la funzione RAMIFICA
    invece di uniformare: uniformare vorrebbe dire rompere il ramo 2.x.
    """
    cfg: dict = {"maxOutputTokens": max_output_tokens}
    if json_output:
        cfg["responseMimeType"] = "application/json"

    if is_gemini_3(model):
        # Niente temperature/topP/topK: sarebbero accettati e ignorati.
        cfg["thinkingConfig"] = {"thinkingLevel": thinking}
    else:
        cfg["temperature"] = temperature
        cfg["thinkingConfig"] = {"thinkingBudget": 0}
    return cfg


def _read_settings_file() -> dict:
    if not _SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"llm settings read failed: {e}")
        return {}


def _write_settings_file(data: dict) -> None:
    _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_api_key() -> Optional[str]:
    """Resolve API key: runtime → env → config."""
    file_data = _read_settings_file()
    if file_data.get("gemini_api_key"):
        return file_data["gemini_api_key"]
    env_key = os.getenv("GEMINI_API_KEY_MACRO")
    if env_key:
        return env_key
    return global_settings.gemini_api_key


def get_model() -> str:
    """Resolve model: runtime → env → default."""
    file_data = _read_settings_file()
    if file_data.get("gemini_model"):
        return file_data["gemini_model"]
    env_model = os.getenv("GEMINI_MODEL_MACRO")
    if env_model:
        return env_model
    return DEFAULT_MODEL


def set_api_key(key: Optional[str]) -> None:
    """Set runtime API key (None to clear)."""
    data = _read_settings_file()
    if key:
        data["gemini_api_key"] = key
    else:
        data.pop("gemini_api_key", None)
    _write_settings_file(data)


def set_model(model: Optional[str]) -> None:
    """Set runtime model (None to clear)."""
    data = _read_settings_file()
    if model:
        if model not in AVAILABLE_MODELS:
            logger.warning(f"unknown model '{model}', accepting anyway (may fail at API call)")
        data["gemini_model"] = model
    else:
        data.pop("gemini_model", None)
    _write_settings_file(data)


def get_current_state() -> dict:
    """Return masked state for GET endpoint (no raw API key)."""
    key = get_api_key()
    masked = None
    if key:
        if len(key) > 12:
            masked = f"{key[:6]}...{key[-4:]}"
        else:
            masked = "***"
    return {
        "api_key_set": bool(key),
        "api_key_masked": masked,
        "model": get_model(),
        "model_default": DEFAULT_MODEL,
        "models_available": AVAILABLE_MODELS,
        "source_api_key": _api_key_source(),
        "source_model": _model_source(),
    }


def _api_key_source() -> str:
    if _read_settings_file().get("gemini_api_key"):
        return "runtime"
    if os.getenv("GEMINI_API_KEY_MACRO"):
        return "env_macro"
    if global_settings.gemini_api_key:
        return "env_global"
    return "none"


def _model_source() -> str:
    if _read_settings_file().get("gemini_model"):
        return "runtime"
    if os.getenv("GEMINI_MODEL_MACRO"):
        return "env_macro"
    return "default"
