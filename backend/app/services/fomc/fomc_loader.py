"""Loader del FOMC analysis più recente per uso nel meta-classifier.

Recupera l'ultimo documento analizzato dalla cache `.cache/fomc_analysis/*.json`
ed estrae regime_implication + confidence + published_date per applicare il
nudge alle regime probabilities (Tier 1.2 roadmap Bridgewater).

Strategia: legge direttamente i file cache (no nuova LLM call). Se il fetcher
RSS ha già scaricato e analizzato i documenti recenti, qui troviamo l'output
cached.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from loguru import logger

# Riusa lo stesso path della cache FOMC analysis di analyzer.py
_FOMC_CACHE_ROOT = Path(__file__).resolve().parents[3] / ".cache" / "fomc_analysis"


def latest_fomc_nudge() -> dict[str, Any] | None:
    """Ritorna l'ultimo FOMC analysis cached come dict per il nudge.

    Output:
        {
            "nudge": {regime: float},      # regime_implication
            "confidence": float,
            "doc_date": date,
            "doc_type": str,
            "url": str,
        }

    Ritorna None se la cache è vuota o nessun file parsabile.
    """
    if not _FOMC_CACHE_ROOT.exists():
        return None

    candidates: list[tuple[date, dict]] = []
    for path in _FOMC_CACHE_ROOT.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            published = data.get("published_date")
            if not published:
                continue
            doc_date = date.fromisoformat(published)
            candidates.append((doc_date, data))
        except Exception as e:
            logger.warning(f"latest_fomc_nudge: skip {path.name} ({e})")
            continue

    if not candidates:
        return None

    # Più recente per pubblicazione
    candidates.sort(key=lambda kv: kv[0], reverse=True)
    doc_date, data = candidates[0]

    impl = data.get("regime_implication") or {}
    if not isinstance(impl, dict):
        return None

    return {
        "nudge": {k: float(v) for k, v in impl.items() if isinstance(v, (int, float))},
        "confidence": float(data.get("confidence", 0.0)),
        "doc_date": doc_date,
        "doc_type": str(data.get("doc_type", "unknown")),
        "url": str(data.get("url", "")),
    }
