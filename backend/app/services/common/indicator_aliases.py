"""T9-AUDIT-BUG-FIX: alias indicators canonicalization (DRY).

Council bug hunt 2026-05-16 ha scoperto naming mismatch fra:
- Fetcher live (popolato `RegimeClassification.conditions_met.indicators`):
  usa nomi tipo `fed_funds_rate`, `baa_spread`, `yield_curve_10y2y`,
  `initial_claims_roc`, `payrolls_roc_12m`, `indpro_roc_12m`,
  `housing_starts_roc_12m`, `term_premium_10y`.
- ML classifier + historical_episodes + sub_regimes + asset_conditional +
  financial_stress_veto: usano nomi "vecchi" tipo `fed_funds`, `baa_10y_spread`,
  `yield_curve_spread`, `claims_roc`, `payrolls_yoy`, `indpro_yoy`,
  `housing_starts_yoy`, `acm_term_premium`.

Risultato pre-fix: 8/19 ML features in default + financial-stress veto mai
attivo + sub_regimes/asset_conditional con stress sottostimato.

Soluzione: single source of truth `INDICATOR_ALIASES` + helper `safe_indicator()`.
"""
from __future__ import annotations

from typing import Any
import math


# Single source of truth: nome canonico → tuple di possibili alias nel JSON.
INDICATOR_ALIASES: dict[str, tuple[str, ...]] = {
    "fed_funds": ("fed_funds_rate", "fed_funds"),
    "baa_10y_spread": ("baa_spread", "baa_10y_spread"),
    "yield_curve_spread": ("yield_curve_10y2y", "yield_curve_spread"),
    "claims_roc": ("initial_claims_roc", "claims_roc"),
    "indpro_yoy": ("indpro_roc_12m", "indpro_yoy"),
    "payrolls_yoy": ("payrolls_roc_12m", "payrolls_yoy"),
    "housing_starts_yoy": ("housing_starts_roc_12m", "housing_starts_yoy"),
    "acm_term_premium": ("term_premium_10y", "acm_term_premium"),
}


def safe_indicator(
    indicators: dict[str, Any],
    canonical_key: str,
    default: float = 0.0,
) -> float:
    """Lookup indicator con alias fallback + NaN/Inf safe.

    Cerca:
    1. Direct `canonical_key` (es. "fed_funds")
    2. Tutti gli alias mappati (es. "fed_funds_rate")
    3. Default
    """
    def _parse(v: Any) -> float | None:
        if v is None:
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        if math.isnan(f) or math.isinf(f):
            return None
        return f

    # Try canonical key first
    if canonical_key in indicators:
        result = _parse(indicators[canonical_key])
        if result is not None:
            return result

    # Try aliases
    for alias in INDICATOR_ALIASES.get(canonical_key, ()):
        if alias != canonical_key and alias in indicators:
            result = _parse(indicators[alias])
            if result is not None:
                return result

    return default


def derive_pmi_proxy(indicators: dict[str, Any]) -> float:
    """T9-AUDIT-BUG-4: NAPM (ISM PMI) discontinuato su FRED.

    Priorità (P1 2026-07-12):
    1. `pmi` reale se presente (episodi storici del training ML lo hanno).
    2. **Survey Fed regionali** (Empire + Philly + Dallas, diffusion index):
       sono LEADING ed escono PRIMA dell'ISM (correlazione col PMI ~0.85).
       Mapping documentato: `pmi ≈ 50 + composite/3`, clamp [35, 65]
       (composite +30 → pmi 60; -30 → 40 — coerente con range storico ISM).
    3. Fallback legacy: `pmi ≈ 50 + 5 × indpro_yoy` clamp [30, 65] —
       indpro è COINCIDENT, distrugge il timing: usato solo se le survey
       non sono disponibili (es. scenari storici del backfill senza survey).
    """
    if "pmi" in indicators:
        v = safe_indicator(indicators, "pmi", default=50.0)
        if v != 50.0:  # se non default, usa valore reale
            return v

    # P1: survey regionali leading (media delle disponibili)
    surveys = [
        indicators.get(k)
        for k in ("empire_fed_activity", "philly_fed_activity", "dallas_fed_activity")
    ]
    valid = []
    for s in surveys:
        try:
            f = float(s)
        except (TypeError, ValueError):
            continue
        if not (math.isnan(f) or math.isinf(f)):
            valid.append(f)
    if valid:
        composite = sum(valid) / len(valid)
        return max(35.0, min(65.0, 50.0 + composite / 3.0))

    indpro_yoy = safe_indicator(indicators, "indpro_yoy", default=0.0)
    proxy = 50.0 + 5.0 * indpro_yoy
    return max(30.0, min(65.0, proxy))
