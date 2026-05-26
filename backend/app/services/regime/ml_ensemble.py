"""Tier 8.3 — Ensemble blend rule-based + ML regime classifier.

Pattern: il rule-based è stabile su scenari noti ma confonde recessioni
disinflazionistiche (2000, 1990) con stagflation. Il ML cattura
deflation crashes ma confonde 2022 (CPI 8% + gdp 1.5) con deflation.

Soluzione: blend pesato `final = (1-alpha) * rule + alpha * ml`.
- alpha basso (0.2-0.3): ML aggiunge "spinta" disambiguante senza dominare.
- alpha alto (0.5+): ML domina, rischio overfit su 18 episodi.

Default alpha=0.4: compromesso che migliora 2000 (+30pp deflation) e
mantiene 2022 corretto (rule-based dice stag, ML dice defl, blend = stag).

Lazy training del modello al primo uso, cached in module.
"""
from __future__ import annotations

from typing import Any

from app.services.regime.ml_classifier import (
    MLModelBundle,
    REGIMES,
    predict_regime as ml_predict_regime,
    train_classifier,
)


# Cache module-level: trained una volta al primo uso, ri-trainabile via reset_cache.
_CACHED_BUNDLE: MLModelBundle | None = None


def get_or_train_bundle() -> MLModelBundle:
    """Restituisce il bundle ML cached; trainalo se non esistente."""
    global _CACHED_BUNDLE
    if _CACHED_BUNDLE is None:
        _CACHED_BUNDLE = train_classifier()
    return _CACHED_BUNDLE


def reset_cache() -> None:
    """Forza re-training (es. dopo update episodi storici)."""
    global _CACHED_BUNDLE
    _CACHED_BUNDLE = None


def _apply_hard_constraints(
    ml_probs: dict[str, float],
    indicators: dict[str, Any],
) -> dict[str, float]:
    """T9-AUDIT-BUG-FIX: hard constraints post-ML per evitare predictions
    inconsistenti con i fondamentali macro.

    Bug scoperto 2026-05-16: ML predice deflation 73% in scenario CPI 3.95%.
    Causa: tree-based models sono soft constraint, vedono pattern (sentiment
    basso + lei neg + gdp basso) e ignorano vincoli definizionali (CPI alto
    è incompatibile con deflation per definizione).

    Constraint applicati:
    1. CPI > 3.0% AND breakeven_10y > 2.0 → deflation cap 0.20 (impossibile defla)
    2. CPI < 2.0% → stagflation cap 0.15 (impossibile stagflazione)
    3. gdp_roc > 3.0% → deflation cap 0.15 (no recession se gdp forte)
    """
    def _safe(key: str, default: float) -> float:
        v = indicators.get(key, default)
        if v is None:
            return default
        try:
            import math
            f = float(v)
            if math.isnan(f) or math.isinf(f):
                return default
            return f
        except (TypeError, ValueError):
            return default

    cpi = _safe("cpi_yoy", 2.5)
    breakeven = _safe("breakeven_10y", 2.0)
    gdp = _safe("gdp_roc", 2.0)

    out = dict(ml_probs)

    # Constraint 1: CPI alto + breakeven > 2.0 = NO deflation
    if cpi > 3.0 and breakeven > 2.0:
        cap = 0.20
        if out.get("deflation", 0) > cap:
            excess = out["deflation"] - cap
            out["deflation"] = cap
            # Ridistribuisci excess pro-quota su reflation/stagflation/goldilocks
            non_defl = [r for r in out if r != "deflation"]
            tot_other = sum(out[r] for r in non_defl)
            if tot_other > 0:
                for r in non_defl:
                    out[r] += excess * (out[r] / tot_other)

    # Constraint 2: CPI basso = NO stagflation
    if cpi < 2.0:
        cap = 0.15
        if out.get("stagflation", 0) > cap:
            excess = out["stagflation"] - cap
            out["stagflation"] = cap
            non_stag = [r for r in out if r != "stagflation"]
            tot_other = sum(out[r] for r in non_stag)
            if tot_other > 0:
                for r in non_stag:
                    out[r] += excess * (out[r] / tot_other)

    # Constraint 3: gdp forte = NO deflation grave
    if gdp > 3.0:
        cap = 0.15
        if out.get("deflation", 0) > cap:
            excess = out["deflation"] - cap
            out["deflation"] = cap
            non_defl = [r for r in out if r != "deflation"]
            tot_other = sum(out[r] for r in non_defl)
            if tot_other > 0:
                for r in non_defl:
                    out[r] += excess * (out[r] / tot_other)

    # Renormalize (safety)
    total = sum(out.values())
    if total > 0:
        out = {k: v / total for k, v in out.items()}
    return out


def blend_with_rule_based(
    rule_based_probs: dict[str, float],
    indicators: dict[str, Any],
    alpha: float = 0.25,
) -> tuple[dict[str, float], dict[str, float]]:
    """Blend rule-based con ML prediction.

    Args:
        rule_based_probs: output di `classify_regime(...)["probabilities"]`.
        indicators: dict indicators per ML inference.
        alpha: peso ML ∈ [0, 1]. 0=solo rule, 1=solo ML.

    Returns:
        (blended_probs, ml_probs_constrained): tuple per audit trail.
        ml_probs_constrained = ml output dopo hard constraints applicati.
    """
    bundle = get_or_train_bundle()
    ml_probs_raw = ml_predict_regime(indicators, bundle)

    # T9-AUDIT-BUG-FIX: hard constraints macro pre-blend
    ml_probs = _apply_hard_constraints(ml_probs_raw, indicators)

    # Assicura tutti i regimi presenti in entrambi
    blended: dict[str, float] = {}
    for r in REGIMES:
        rb = rule_based_probs.get(r, 0.0)
        ml = ml_probs.get(r, 0.0)
        blended[r] = (1.0 - alpha) * rb + alpha * ml

    # Renormalize Σ=1
    total = sum(blended.values())
    if total > 0:
        blended = {k: v / total for k, v in blended.items()}
    return blended, ml_probs
