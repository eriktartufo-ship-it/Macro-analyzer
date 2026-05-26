"""Tier 6.2 — Regime continuo 2D.

Filosofia: i 4 quadranti discreti (reflation/stagflation/deflation/goldilocks)
perdono sfumature. La realtà macro è continua: "55% reflation + 45% goldilocks"
è un punto preciso nel piano (growth × inflation), non una probabilistic blob.

Quadranti:
- reflation = (+growth, +inflation)
- stagflation = (-growth, +inflation)
- deflation = (-growth, -inflation)
- goldilocks = (+growth, -inflation)

Due viste:
1. **Probability-derived** (`compute_from_probs`): coordinate in [-1, +1] come
   weighted average dei quadranti.
2. **Indicator-derived** (`compute_from_indicators`): z-score di gdp_roc e
   cpi_yoy, più ricco ma dipende da indicators puntuali.

API non rompe back-compat: `probabilities` resta autoritativo. `regime_2d` è
info layer aggiuntivo per visualizzazione (scatter map traiettoria).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Quadrante signed coordinates per `compute_from_probs`.
# Reflation: (+,+) — Stagflation: (-,+) — Deflation: (-,-) — Goldilocks: (+,-)
_QUADRANT_COORDS: dict[str, tuple[float, float]] = {
    "reflation": (+1.0, +1.0),
    "stagflation": (-1.0, +1.0),
    "deflation": (-1.0, -1.0),
    "goldilocks": (+1.0, -1.0),
}


@dataclass
class Regime2DAssessment:
    """Coordinate continue + nearest quadrant + distance map."""
    growth_z: float
    inflation_z: float
    nearest_quadrant: str
    distance_to_quadrants: dict[str, float]
    source: str  # 'probs' | 'indicators' | 'blended'


def compute_from_probs(probs: dict[str, float]) -> tuple[float, float]:
    """Calcola (growth, inflation) ∈ [-1, +1] come avg pesato sui quadranti.

    growth = +P(refl) - P(stag) - P(defl) + P(gold)
    inflation = +P(refl) + P(stag) - P(defl) - P(gold)

    Se probs vuoto o tutte 0, returns (0, 0).
    """
    growth = 0.0
    inflation = 0.0
    for regime, p in probs.items():
        coords = _QUADRANT_COORDS.get(regime)
        if coords is None:
            continue
        gx, iy = coords
        growth += p * gx
        inflation += p * iy
    return (round(growth, 4), round(inflation, 4))


def compute_from_indicators(indicators: dict[str, Any]) -> tuple[float, float]:
    """Calcola z-score (growth, inflation) da indicators puntuali.

    Z-score semplice (target centered):
    - growth_z = (gdp_roc - 2.0) / 1.5   # 2% trend USA, sigma ~1.5%
    - inflation_z = (cpi_yoy - 2.0) / 1.5   # Fed target 2%, sigma ~1.5%

    Clamp [-3, +3] per evitare outlier domini visual.
    """
    gdp = _safe_float(indicators.get("gdp_roc"), 2.0)
    cpi = _safe_float(indicators.get("cpi_yoy"), 2.0)

    growth_z = max(-3.0, min(3.0, (gdp - 2.0) / 1.5))
    inflation_z = max(-3.0, min(3.0, (cpi - 2.0) / 1.5))
    return (round(growth_z, 4), round(inflation_z, 4))


def _safe_float(val: Any, default: float) -> float:
    if val is None:
        return default
    try:
        v = float(val)
    except (TypeError, ValueError):
        return default
    import math
    if math.isnan(v) or math.isinf(v):
        return default
    return v


def _euclidean_to_quadrant(point: tuple[float, float], quadrant: str) -> float:
    qx, qy = _QUADRANT_COORDS[quadrant]
    dx = point[0] - qx
    dy = point[1] - qy
    return (dx * dx + dy * dy) ** 0.5


def nearest_quadrant(growth: float, inflation: float) -> tuple[str, dict[str, float]]:
    """Quadrante con minor distanza euclidea + mappa distanze."""
    distances = {
        q: round(_euclidean_to_quadrant((growth, inflation), q), 4)
        for q in _QUADRANT_COORDS
    }
    nearest = min(distances, key=distances.get)
    return nearest, distances


def assess_regime_2d(
    probs: dict[str, float] | None = None,
    indicators: dict[str, Any] | None = None,
    blend: float = 0.5,
) -> Regime2DAssessment:
    """Assess 2D regime: probabilità-based, indicator-based, o blend.

    Args:
        probs: regime probabilities (preferito se presente)
        indicators: dict di indicators per z-score view
        blend: peso per indicator view se entrambi forniti (0=all probs, 1=all indicators)

    Behaviour:
        - Solo probs → source='probs'
        - Solo indicators → source='indicators'
        - Entrambi → source='blended' con weighted avg
    """
    has_probs = bool(probs)
    has_indicators = bool(indicators)

    if has_probs and not has_indicators:
        growth, inflation = compute_from_probs(probs)
        source = "probs"
    elif has_indicators and not has_probs:
        growth, inflation = compute_from_indicators(indicators)
        # Normalize z [-3,+3] → [-1,+1] per coerenza con coord quadranti
        growth = max(-1.0, min(1.0, growth / 2.0))
        inflation = max(-1.0, min(1.0, inflation / 2.0))
        source = "indicators"
    elif has_probs and has_indicators:
        gp, ip = compute_from_probs(probs)
        gi, ii = compute_from_indicators(indicators)
        # Normalize indicator z-score per blend coerente
        gi_norm = max(-1.0, min(1.0, gi / 2.0))
        ii_norm = max(-1.0, min(1.0, ii / 2.0))
        growth = round((1 - blend) * gp + blend * gi_norm, 4)
        inflation = round((1 - blend) * ip + blend * ii_norm, 4)
        source = "blended"
    else:
        growth, inflation = 0.0, 0.0
        source = "empty"

    nq, dists = nearest_quadrant(growth, inflation)
    return Regime2DAssessment(
        growth_z=growth,
        inflation_z=inflation,
        nearest_quadrant=nq,
        distance_to_quadrants=dists,
        source=source,
    )
