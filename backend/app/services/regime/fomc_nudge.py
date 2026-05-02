"""FOMC regime nudge applicato alle probabilities (Tier 1.2 roadmap Bridgewater).

Il dict `regime_implication` estratto da Gemini (es. {"reflation": -0.05,
"stagflation": +0.05, ...}) entra come post-correzione alle probabilities
del meta-classifier:

    P_final(r) = renormalize( max(0, P_meta(r) + nudge(r) · confidence · decay) )

Componenti:
- `compute_decay_factor`: half-life temporale (default 30d) per evitare che
  uno statement vecchio influenzi il regime per sempre.
- `apply_fomc_nudge`: applica il nudge a una distribuzione, scalando per
  confidence + decay, con renormalizzazione finale.
"""
from __future__ import annotations

from datetime import date

from app.services.regime.classifier import REGIMES


def compute_decay_factor(age_days: int, half_life_days: int = 30) -> float:
    """Decadimento esponenziale half-life.

    factor = 0.5 ^ (age_days / half_life_days)

    - age=0: factor=1.0
    - age=half_life: factor=0.5
    - age=2*half_life: factor=0.25
    - age negativo (futuro): factor=1.0 (clamp, no boost)
    """
    if age_days <= 0:
        return 1.0
    if half_life_days <= 0:
        return 1.0
    return 0.5 ** (age_days / half_life_days)


def apply_fomc_nudge(
    probabilities: dict[str, float],
    nudge: dict[str, float] | None,
    confidence: float,
    doc_date: date,
    half_life_days: int = 30,
    *,
    today: date | None = None,
) -> dict[str, float]:
    """Applica il nudge FOMC alle regime probabilities.

    Args:
        probabilities: P_meta(r) input, dict {regime: prob}.
        nudge: regime_implication dict (es. {"reflation": -0.05, ...}). Range
            tipico ±0.20 (output Gemini clamped). Se None → no-op.
        confidence: 0..1, output Gemini su quanto il segnale del documento è
            chiaro. Suppress weak signals.
        doc_date: data di pubblicazione del documento FOMC. Usato per decay.
        half_life_days: half-life del decay temporale (default 30d).
        today: override per test (default = date.today()).

    Returns:
        P_final(r) post-nudge, dict {regime: prob} con Σ=1.
    """
    if nudge is None:
        return dict(probabilities)

    today_d = today or date.today()
    age = (today_d - doc_date).days
    decay = compute_decay_factor(age, half_life_days=half_life_days)
    scale = confidence * decay

    if scale <= 0:
        return dict(probabilities)

    # Apply nudge × scale + clamp negativi a 0
    adjusted = {}
    for r in REGIMES:
        delta = float(nudge.get(r, 0.0)) * scale
        adjusted[r] = max(0.0, float(probabilities.get(r, 0.0)) + delta)

    # Renormalize Σ=1
    total = sum(adjusted.values())
    if total <= 0:
        return {r: 1.0 / len(REGIMES) for r in REGIMES}
    return {r: v / total for r, v in adjusted.items()}
