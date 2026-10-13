"""Accesso admin del Macro Analyzer (S51, 2026-10-13).

Gli endpoint che CAMBIANO qualcosa (flag del modello, chiave e modello Gemini, reset del portafoglio
AI, refresh, backfill, retrain, rigenerazione di analisi LLM a pagamento) vogliono
`Authorization: Bearer <MACRO_ADMIN_TOKEN>`. Le letture restano pubbliche.

Fail-closed: un server senza token configurato rifiuta (503), non si apre.
Il censimento che impedisce a un endpoint nuovo di nascere scoperto e' in tests/test_accesso_admin.py.
"""

from __future__ import annotations

import hmac

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings

_bearer = HTTPBearer(auto_error=False)


def _valido(supplied: str) -> bool:
    expected = settings.macro_admin_token
    return bool(expected) and hmac.compare_digest(expected.encode("utf-8"), supplied.encode("utf-8"))


def require_admin(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> None:
    """Dependency FastAPI per gli endpoint che scrivono."""
    if not settings.macro_admin_token:
        raise HTTPException(status_code=503, detail="MACRO_ADMIN_TOKEN non configurato sul server")
    if credentials is None:
        raise HTTPException(status_code=401, detail="Serve il token admin (Impostazioni → Accesso admin)")
    if not _valido(credentials.credentials):
        raise HTTPException(status_code=403, detail="Token admin non valido")


def is_admin(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> bool:
    """Per le letture con un effetto a pagamento (force_refresh di un'analisi LLM)."""
    return credentials is not None and _valido(credentials.credentials)


def require_admin_for_regeneration(force_refresh: bool, admin: bool) -> None:
    if force_refresh and not admin:
        raise HTTPException(status_code=403, detail="Rigenerare un'analisi LLM richiede il token admin")
