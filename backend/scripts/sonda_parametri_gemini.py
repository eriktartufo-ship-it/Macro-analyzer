"""Chiede AL MODELLO quali parametri accetta, invece di dedurlo dalla doc.

Perche' esiste (2026-09-05): Macro e' passata a `gemini-3.8-flash` e manda
`thinkingConfig.thinkingLevel` al posto del legacy `thinkingBudget`. La doc dice che
e' il parametro giusto e che su 3.8 il livello `minimal` non esiste piu' — ma "la doc
lo dice" e "il modello lo accetta" sono due cose diverse, e qui sbagliare non produce
un errore visibile: tutti i call-site catturano l'eccezione e tornano `None`, quindi
l'analisi LLM si spegnerebbe in silenzio.

Serve anche a Ormio: la lettura multipla deve tornare a VARIARE fra una lettura e
l'altra, e per farlo le serve una leva che il modello ascolti davvero. Qui si scopre
quali sono.

Uso:  python backend/scripts/sonda_parametri_gemini.py [modello ...]
Legge la chiave da GEMINI_API_KEY (env o `.env` alla radice del progetto).
Non scrive niente e non tocca il DB: manda prompt da pochi token a Google.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx

RADICE = Path(__file__).resolve().parents[2]
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"

# Ogni caso e' un pezzo di `generationConfig` da provare da solo, cosi' un rifiuto
# nomina UN parametro invece di una combinazione.
CASI: dict[str, dict] = {
    "nudo (controllo)": {},
    "thinkingLevel=low": {"thinkingConfig": {"thinkingLevel": "low"}},
    "thinkingLevel=medium": {"thinkingConfig": {"thinkingLevel": "medium"}},
    "thinkingLevel=high": {"thinkingConfig": {"thinkingLevel": "high"}},
    "thinkingLevel=minimal": {"thinkingConfig": {"thinkingLevel": "minimal"}},
    "thinkingBudget=0 (legacy)": {"thinkingConfig": {"thinkingBudget": 0}},
    "temperature=0.2": {"temperature": 0.2},
    "temperature=9.9 (fuori range)": {"temperature": 9.9},
    "mediaResolution=high": {"mediaResolution": "MEDIA_RESOLUTION_HIGH"},
    "mediaResolution=medium": {"mediaResolution": "MEDIA_RESOLUTION_MEDIUM"},
}


def chiave() -> str:
    k = os.getenv("GEMINI_API_KEY", "").strip()
    if k:
        return k
    envf = RADICE / ".env"
    if envf.exists():
        for riga in envf.read_text(encoding="utf-8").splitlines():
            if riga.startswith("GEMINI_API_KEY="):
                return riga.split("=", 1)[1].strip().strip("\"'")
    raise SystemExit("GEMINI_API_KEY non trovata (ne' in env ne' in .env)")


def main() -> None:
    modelli = sys.argv[1:] or ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-2.5-flash"]
    k = chiave()
    with httpx.Client(timeout=90.0) as http:
        for m in modelli:
            print(f"\n=== {m} ===")
            for nome, extra in CASI.items():
                cfg = {"maxOutputTokens": 16, **extra}
                try:
                    r = http.post(
                        ENDPOINT.format(m=m),
                        params={"key": k},
                        json={
                            "contents": [{"parts": [{"text": "Rispondi solo: ok"}]}],
                            "generationConfig": cfg,
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  {nome:32s} eccezione {type(exc).__name__}")
                    continue
                if r.status_code == 200:
                    print(f"  {nome:32s} 200 ACCETTATO")
                else:
                    try:
                        msg = r.json().get("error", {}).get("message", "")
                    except Exception:  # noqa: BLE001
                        msg = r.text
                    print(f"  {nome:32s} {r.status_code} RIFIUTATO — {msg[:120]}")


if __name__ == "__main__":
    main()
