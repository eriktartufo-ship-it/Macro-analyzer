"""LLM runtime settings — Gemini API key + model configurable from UI.

User request 2026-06: cambiare la API key e il modello Gemini per la lettura
macro senza redeploy.

Storage: `.cache/llm_settings.json` su disco (process-local + persistent).

Resolution priority:
1. Runtime override (file `.cache/llm_settings.json`) → highest priority
2. Env var (`GEMINI_API_KEY_MACRO`, `GEMINI_MODEL_MACRO`) → fallback
3. Global config (`settings.gemini_api_key`, default model) → last resort

Available models (Gemini family, 2026):
- gemini-2.5-flash (default, fast + cheap)
- gemini-2.5-pro (more capable, slower)
- gemini-2.5-flash-lite (cheapest)
- gemini-3.1-pro-preview (preview)
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

DEFAULT_MODEL = "gemini-2.5-flash"

AVAILABLE_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite",
    "gemini-3.1-pro-preview",
]


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
