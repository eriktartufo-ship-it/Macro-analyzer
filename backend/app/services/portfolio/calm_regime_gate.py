"""T9-AUDIT-TA: Calm regime gate.

Council finding (random backtest 2026-05-16): goldilocks 54% error rate
(modello fa peggio di 60/40 in regime calmo). Pattern strutturale:

> raydalio: "In goldilocks il benchmark 60/40 è già near-optimal.
> Top-5 equal-weight DEVE diluire growth con asset inferiori. Math."
>
> buffett: "Quando l'oceano è calmo, smetti di sterzare."
>
> bughunter: "Tactical rotation in regime calmo = drag da turnover
> senza upside. Modello fa over-trading."

Soluzione: se il regime è chiaramente goldilocks (alta confidence),
**bypass tactical allocation** e fallback a 60/40 statico.

Threshold:
- P(goldilocks) > 0.55 AND confidence > 0.65 → CALM GATE FIRED

Output:
- `should_use_60_40()`: bool → True se fallback consigliato
- `apply_calm_gate(top5_scores, probs, confidence)`: trasforma top5 in 60/40
  se gate triggered, altrimenti restituisce top5 invariato.
"""
from __future__ import annotations

from dataclasses import dataclass


# Calm 60/40 default (council reco: us_equities_growth + us_bonds_long)
CALM_ALLOCATION_60_40: dict[str, float] = {
    "us_equities_growth": 0.60,
    "us_bonds_long": 0.40,
}


@dataclass
class CalmGateAssessment:
    is_fired: bool
    goldilocks_prob: float
    confidence: float
    description: str


def assess_calm_gate(
    probs: dict[str, float],
    confidence: float,
    goldilocks_threshold: float = 0.30,
    confidence_threshold: float = 0.30,
) -> CalmGateAssessment:
    """Verifica se attivare il calm regime gate.

    Args:
        probs: regime probabilities (deve sommare ~1.0)
        confidence: classifier confidence 0-1
        goldilocks_threshold: minimo P(goldilocks) per gate (default 0.30,
            recalibrato 2026-05-17 dopo audit DB: max P(goldi)=0.418).
        confidence_threshold: minimo confidence (default 0.30, recalibrato
            2026-05-17 dopo finding empirico: classifier ha anti-correlazione
            strutturale tra P(goldi) e confidence — 0.45 dava 1/1025 records).

    Returns:
        CalmGateAssessment con is_fired flag e rationale.
    """
    p_goldi = probs.get("goldilocks", 0.0)
    is_fired = p_goldi > goldilocks_threshold and confidence > confidence_threshold

    if is_fired:
        desc = (
            f"CALM GATE FIRED: goldilocks {p_goldi:.0%} > {goldilocks_threshold:.0%} "
            f"AND confidence {confidence:.0%} > {confidence_threshold:.0%}. "
            f"Tactical rotation disabilitata, fallback a 60/40 statico."
        )
    else:
        desc = (
            f"NO calm gate: goldilocks={p_goldi:.0%} confidence={confidence:.0%}. "
            f"Tactical allocation regime-based attiva."
        )

    return CalmGateAssessment(
        is_fired=is_fired,
        goldilocks_prob=round(p_goldi, 4),
        confidence=round(confidence, 4),
        description=desc,
    )


def apply_calm_gate(
    top5_scores: dict[str, float],
    probs: dict[str, float],
    confidence: float,
) -> tuple[dict[str, float], CalmGateAssessment]:
    """Applica calm gate. Se fired, sostituisce top5 con 60/40 statico.

    Args:
        top5_scores: dict {asset: score} top-5 corrente.
        probs: regime probs.
        confidence: classifier confidence.

    Returns:
        (final_allocation_weights, assessment).
        Se gate fired: weights = CALM_ALLOCATION_60_40.
        Altrimenti: weights = equal-weight top-5 del input.
    """
    assessment = assess_calm_gate(probs, confidence)

    if assessment.is_fired:
        return CALM_ALLOCATION_60_40.copy(), assessment

    # Equal-weight top-5
    if not top5_scores:
        return {}, assessment
    n = len(top5_scores)
    weights = {asset: 1.0 / n for asset in top5_scores}
    return weights, assessment
