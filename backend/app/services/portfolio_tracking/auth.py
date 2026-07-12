"""Auth leggera per Portfolio Tracking — 2 utenti da env, token HMAC stdlib.

Scelte (coerenti con l'uso famigliare, zero nuove dipendenze):
- Utenti e password in env `PT_USERS` (il .env sta solo sul VPS, mai nel repo).
- Token = base64url(payload JSON) + "." + base64url(HMAC-SHA256(secret, payload)).
  Niente JWT lib: hmac/hashlib/base64 stdlib bastano per questo perimetro.
- Sessione 30 giorni con re-issue rolling da GET /pt/me (lezione RGV: sessione
  rolling, e MAI cambiare PT_SECRET in prod).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings

TOKEN_TTL_SECONDS = 30 * 24 * 3600  # 30 giorni
ROLLING_REISSUE_BEFORE = 20 * 24 * 3600  # re-issue se restano < 20 giorni

_bearer = HTTPBearer(auto_error=False)


@dataclass
class PtUser:
    username: str
    display_name: str
    password: str


def parse_users(raw: str | None = None) -> dict[str, PtUser]:
    """Parsa PT_USERS: "user:Display:pw,user2:pw2" → {username: PtUser}.

    3 campi = username:display:password, 2 campi = username:password.
    Entry malformate vengono ignorate (graceful, come il resto del progetto).
    """
    raw = settings.pt_users if raw is None else raw
    users: dict[str, PtUser] = {}
    for entry in (raw or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":")
        if len(parts) == 3:
            username, display, password = parts
        elif len(parts) == 2:
            username, password = parts
            display = username.capitalize()
        else:
            continue
        username = username.strip().lower()
        if username and password:
            users[username] = PtUser(username, display.strip() or username.capitalize(), password)
    return users


def _b64e(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64d(data: str) -> bytes:
    pad = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + pad)


def _sign(payload: bytes, secret: str) -> str:
    return _b64e(hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).digest())


def mint_token(username: str, secret: str | None = None, now: float | None = None) -> str:
    secret = secret if secret is not None else settings.pt_secret
    now = time.time() if now is None else now
    payload = json.dumps(
        {"u": username, "exp": int(now + TOKEN_TTL_SECONDS)}, separators=(",", ":"),
    ).encode("utf-8")
    return f"{_b64e(payload)}.{_sign(payload, secret)}"


def verify_token(token: str, secret: str | None = None, now: float | None = None) -> str | None:
    """Ritorna lo username se il token è valido e non scaduto, altrimenti None."""
    secret = secret if secret is not None else settings.pt_secret
    if not secret or not token or "." not in token:
        return None
    try:
        payload_b64, sig = token.rsplit(".", 1)
        payload = _b64d(payload_b64)
        if not hmac.compare_digest(_sign(payload, secret), sig):
            return None
        data = json.loads(payload)
        now = time.time() if now is None else now
        if float(data.get("exp", 0)) < now:
            return None
        return str(data["u"])
    except Exception:
        return None


def token_remaining_seconds(token: str) -> float:
    try:
        payload = _b64d(token.rsplit(".", 1)[0])
        return float(json.loads(payload).get("exp", 0)) - time.time()
    except Exception:
        return 0.0


def check_login(username: str, password: str) -> PtUser | None:
    """Verifica credenziali (confronto constant-time sulla password)."""
    user = parse_users().get((username or "").strip().lower())
    if not user:
        return None
    if hmac.compare_digest(user.password.encode("utf-8"), (password or "").encode("utf-8")):
        return user
    return None


def require_pt_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> PtUser:
    """Dependency FastAPI: Bearer token → PtUser (401 se invalido)."""
    if credentials is None:
        raise HTTPException(status_code=401, detail="Token mancante")
    username = verify_token(credentials.credentials)
    if not username:
        raise HTTPException(status_code=401, detail="Sessione scaduta, rifai il login")
    user = parse_users().get(username)
    if not user:
        raise HTTPException(status_code=401, detail="Utente non più attivo")
    return user


def require_pt_admin(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    """Dependency admin (push Claude Code): Bearer == PT_ADMIN_TOKEN."""
    expected = settings.pt_admin_token
    if not expected:
        raise HTTPException(status_code=503, detail="PT_ADMIN_TOKEN non configurato")
    supplied = credentials.credentials if credentials else ""
    if not hmac.compare_digest(expected.encode("utf-8"), supplied.encode("utf-8")):
        raise HTTPException(status_code=403, detail="Admin token non valido")
