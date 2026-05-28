"""Tier 9 — Defensive transition mode detector.

Filosofia (Buffett + Taleb + raydalio): "Quando non sai cosa fare, non fare
nulla. Cash è una posizione." Meglio missare un rally del 5% che subire un
drawdown del 30% perché il modello era confuso.

Pattern: i grandi investitori riconoscono i momenti di **bassa conviction**
e si ritirano in posizioni difensive (cash, short bonds, gold) finché il
segnale non torna chiaro.

Detector logic:
- max(probs) < 0.45 → nessun regime domina chiaramente
- shannon_entropy(probs) > 1.40 → incertezza alta (max=log2(4)=2.0)
- confidence < 0.40 → modello non sicuro

Se almeno 2 di 3 trigger fired → TRANSITION DETECTED.

Allocazione difensiva (All-Weather conservative):
- cash_money_market: 35%
- us_bonds_short: 25%
- us_bonds_long: 15%
- gold: 15%
- tips_inflation_bonds: 10%
"""
from __future__ import annotations

from dataclasses import dataclass


import math


DEFENSIVE_ALLOCATION: dict[str, float] = {
    "cash_money_market": 0.35,
    "us_bonds_short": 0.25,
    "us_bonds_long": 0.15,
    "gold": 0.15,
    "tips_inflation_bonds": 0.10,
}


@dataclass
class TransitionAssessment:
    """Esito detection transizione + razionale."""
    is_transition: bool
    triggers_fired: list[str]
    triggers_count: int
    max_prob: float
    entropy: float
    confidence: float
    description: str


def shannon_entropy(probs: dict[str, float]) -> float:
    """Shannon entropy base 2. Max = log2(4) = 2.0 per 4 regimi uniformi."""
    e = 0.0
    for p in probs.values():
        if p > 1e-9:
            e -= p * math.log2(p)
    return round(e, 4)


def assess_transition(
    probs: dict[str, float],
    confidence: float,
    max_prob_threshold: float = 0.38,
    entropy_threshold: float = 1.55,
    confidence_threshold: float = 0.30,
    require_all_three: bool = True,
    financial_stress_vetoed: bool = False,
) -> TransitionAssessment:
    """Valuta se il regime corrente è in transizione confusionaria.

    **T9-AUDIT-FIX**: soglie strette + require 3/3 (era 2/3).
    Pre-T9: 78% transition detected OOS (troppo permissivo).
    Post-T9: atteso 25-30% transition rate.

    Args:
        probs: regime probabilities (deve sommare ~1.0)
        confidence: confidence del classifier 0-1
        max_prob_threshold: sotto questo, no dominant regime (default 0.38, era 0.45)
        entropy_threshold: sopra questo, alta incertezza (default 1.55, era 1.40)
        confidence_threshold: sotto questo, modello non sicuro (default 0.30, era 0.40)
        require_all_three: se True (default), serve TUTTI 3 trigger fired.
            Se False, basta 2/3 (legacy permissivo).

    Returns:
        TransitionAssessment con dettaglio trigger fired.
    """
    triggers: list[str] = []

    max_p = max(probs.values()) if probs else 0.0
    if max_p < max_prob_threshold:
        triggers.append(
            f"max_prob={max_p:.3f} < {max_prob_threshold} (no dominant regime)"
        )

    ent = shannon_entropy(probs)
    if ent > entropy_threshold:
        triggers.append(
            f"entropy={ent:.3f} > {entropy_threshold} (high uncertainty)"
        )

    if confidence < confidence_threshold:
        triggers.append(
            f"confidence={confidence:.3f} < {confidence_threshold} (model unsure)"
        )

    # T9-AUDIT-FIX: financial-stress veto FORZA transition (bypass require_all_three).
    # Cattura 2018 Q4 / 2023 SVB dove regime classifier dice goldilocks/reflation
    # ma market+credit stress reale → defensive allocation obbligata.
    if financial_stress_vetoed:
        triggers.append("financial_stress_veto active (forced transition)")

    min_triggers = 3 if require_all_three else 2
    is_transition = financial_stress_vetoed or len(triggers) >= min_triggers

    if is_transition:
        reason = (
            "veto-forced" if financial_stress_vetoed
            else f"{len(triggers)}/3 trigger"
        )
        desc = (
            f"Transizione rilevata ({reason}). "
            f"Suggerimento difensivo: cash 35% + bonds 40% + gold 15% + TIPS 10%."
        )
    else:
        desc = (
            f"Regime stabile ({len(triggers)}/3 trigger). "
            f"Allocazione regime-based affidabile."
        )

    return TransitionAssessment(
        is_transition=is_transition,
        triggers_fired=triggers,
        triggers_count=len(triggers),
        max_prob=round(max_p, 4),
        entropy=ent,
        confidence=round(confidence, 4),
        description=desc,
    )


def get_defensive_allocation() -> dict[str, float]:
    """Restituisce l'allocazione difensiva All-Weather conservative.

    Σ weights = 1.0. Cash heavy + bonds + gold + TIPS.
    """
    return dict(DEFENSIVE_ALLOCATION)


# `blend_with_defensive` rimosso 2026-05-28 (code audit): orphan, solo test, zero
# consumer in produzione. routes.py:2028 chiama get_defensive_allocation() raw.
