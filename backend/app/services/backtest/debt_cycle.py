"""DEBT CYCLE THERMOMETER (S46) — termometro del ciclo LUNGO del debito (Dalio).

Orizzonte DIVERSO dall'orologio del ciclo breve: l'orologio gira in mesi (business
cycle), questo si muove in ANNI (deleveraging cycle, ~50-75 anni). Va letto come
CONTESTO STRUTTURALE, non come timing (verdetto finance_council 2026-07-16). Per
rispondere onestamente a "siamo a fine ciclo?" si leggono i DUE insieme, mai fusi in
una lancetta sola (velocita' di rotazione incompatibili -> lettura falsa).

Composito 0-1 (0 = inizio ciclo del debito / bassa leva, 1 = tardo ciclo / alta leva
+ repressione), da 3 componenti STRUTTURALI lente:
  - debito pubblico / GDP (livello di leva del sistema)
  - bilancio Fed / GDP (quanta monetizzazione gia' usata)
  - term premium 10Y (compensation del mercato: bassa/negativa = repressione tardo-ciclo)

⚠️ ETICHETTA (obbligatoria, council): composito EURISTICO, calibrazione DA VERIFICARE
sui dati storici. Contesto, NON un segnale di timing. Nessun conto alla rovescia.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from loguru import logger

# ancore di normalizzazione (dichiarate, non derivate: convenzione di lettura).
# score = clamp((val - lo) / (hi - lo), 0, 1). Per il term premium l'asse e' INVERTITO
# (basso/negativo = tardo-ciclo -> score alto).
_ANCHORS = {
    "debt_gdp":   {"lo": 60.0,  "hi": 130.0, "invert": False,
                   "label": "Debito pubblico / GDP", "unit": "%", "w": 0.40},
    "fedbs_gdp":  {"lo": 6.0,   "hi": 35.0,  "invert": False,
                   "label": "Bilancio Fed / GDP", "unit": "%", "w": 0.35},
    "term_premium": {"lo": -1.0, "hi": 1.0,  "invert": True,
                     "label": "Term premium 10Y", "unit": "%", "w": 0.25},
}


def _sub_score(key: str, val: float) -> float:
    a = _ANCHORS[key]
    s = (val - a["lo"]) / (a["hi"] - a["lo"])
    s = max(0.0, min(1.0, s))
    if a["invert"]:
        s = 1.0 - s
    return s


def _stage(composite: float) -> tuple[str, str]:
    if composite < 0.34:
        return "early", "Inizio ciclo del debito"
    if composite < 0.67:
        return "mid", "Ciclo del debito maturo"
    return "late", "Tardo ciclo del debito / alta leva"


def compute_debt_thermometer(components: dict[str, Optional[float]]) -> dict:
    """Composito 0-1 dai valori grezzi delle componenti (PURA, testabile).

    Args:
        components: {debt_gdp, fedbs_gdp, term_premium} -> valore grezzo o None.
            None = componente non disponibile (esclusa + peso rinormalizzato sulle
            presenti, e segnalata: mai esclusione silenziosa).

    Ritorna: {composite, stage, stage_label, components:[...], used, missing}.
    """
    rows = []
    used, missing = [], []
    wsum = 0.0
    acc = 0.0
    for key, a in _ANCHORS.items():
        val = components.get(key)
        if val is None:
            missing.append(key)
            rows.append({"key": key, "label": a["label"], "unit": a["unit"],
                         "value": None, "score": None, "available": False})
            continue
        s = _sub_score(key, float(val))
        rows.append({"key": key, "label": a["label"], "unit": a["unit"],
                     "value": round(float(val), 2), "score": round(s, 4),
                     "available": True})
        used.append(key)
        acc += a["w"] * s
        wsum += a["w"]

    composite = round(acc / wsum, 4) if wsum > 1e-9 else None
    stage, stage_label = _stage(composite) if composite is not None else ("na", "Dati insufficienti")
    return {
        "composite": composite,
        "stage": stage,
        "stage_label": stage_label,
        "components": rows,
        "used": used,
        "missing": missing,
    }


def fetch_debt_components(as_of: Optional[date] = None) -> dict[str, Optional[float]]:
    """Ultimo valore delle 3 componenti da FRED (degradazione graziosa: una serie che
    fallisce -> None, non crash)."""
    from app.services.indicators.fetcher import FredFetcher

    fetcher = FredFetcher()

    def _last(name: str) -> Optional[float]:
        try:
            s = fetcher.fetch_series(name).dropna()
            if as_of is not None:
                import pandas as pd
                s = s[s.index <= pd.Timestamp(as_of)]
            return float(s.iloc[-1]) if len(s) else None
        except Exception as e:
            logger.warning(f"debt_cycle: fetch {name} failed: {e}")
            return None

    debt_gdp = _last("debt_gdp")            # gia' in %
    walcl = _last("fed_balance_sheet")      # milioni USD
    gdp = _last("gdp")                       # miliardi USD
    tp = _last("acm_term_premium_10y")      # %

    fedbs_gdp = None
    if walcl is not None and gdp is not None and gdp > 0:
        fedbs_gdp = (walcl / 1000.0) / gdp * 100.0  # milioni->miliardi / GDP miliardi

    return {"debt_gdp": debt_gdp, "fedbs_gdp": fedbs_gdp, "term_premium": tp}


def debt_thermometer(as_of: Optional[date] = None) -> dict:
    """Termometro completo (fetch + composito). Wrapper per l'endpoint."""
    comp = fetch_debt_components(as_of)
    out = compute_debt_thermometer(comp)
    out["as_of"] = (as_of or date.today()).isoformat()
    return out
