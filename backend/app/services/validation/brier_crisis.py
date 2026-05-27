"""T9-AUDIT P1-C — Brier condizionale per regime-crisi.

Council finding (soros): "La metrica giusta non è accuracy, è **calibrazione
condizionale al regime di crisi** (Brier su 1973, 2008, 2022). Se quella è
<0.10 sei già a posto."

Filosofia: accuracy media (73.5% LOO) include sia "easy" goldilocks/reflation
sia "hard" crisis episodes. Per un investor reale CONTA solo la calibrazione
sui regime-crisi (dove l'allocation sbagliata costa -30% drawdown).

5 episodi crisis benchmark:
- 1973-75 OPEC stagflation
- 1979 Volcker shock peak
- 2008-09 GFC deflation crash
- 2020 Q1 COVID crash
- 2022 inflation surge

Per ognuno calcoliamo Brier score: Σ (P_predicted_r - P_true_r)² / N_regimes
dove P_true_r = 1 se r è il regime vero, 0 altrimenti.

Brier interpretation:
- 0.00 = perfetto
- 0.10 = molto buono
- 0.20 = accettabile
- 0.30+ = scarso

Council goal: Brier_crisis ≤ 0.10.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.services.regime.historical_episodes import HISTORICAL_EPISODES


REGIMES = ("reflation", "stagflation", "deflation", "goldilocks")

# Names dei 5 crisis episodes nel dataset (council labels noi)
CRISIS_EPISODE_NAMES = (
    "1973-75 OPEC",
    "1979 Volcker shock",
    "2008-09 GFC",
    "2020 Q1 COVID",
    "2022 inflation surge",
)

# NEW #6 council 2026-05-27 — IMF/BIS systemic crises (independent labels).
# Selection-bias-resistant set: ogni nome match parziale su HISTORICAL_EPISODES.
# Source: IMF Systemic Banking Crises Database (Laeven & Valencia 2018) +
# BIS Systemic Stress dataset 1970-2024.
INDEPENDENT_CRISIS_EPISODES = (
    "1973-75",        # OPEC (IMF/BIS confirmed)
    "1979",           # Volcker peak (BIS stress)
    "1981-82",        # Volcker recession (IMF labeled "double dip")
    "1987",           # Black Monday (BIS systemic)
    "1990-91",        # S&L crisis + Iraq war recession (IMF)
    "1997-98",        # Asian + LTCM (IMF flagship)
    "2000-02",        # Dotcom (BIS systemic)
    "2008-09",        # GFC (IMF/BIS)
    "2011",           # European debt crisis (IMF Greece/PIIGS)
    "2020",           # COVID (BIS pandemic stress)
    "2022",           # Inflation surge (BIS confirmed)
)


@dataclass
class CrisisBrierResult:
    """Singolo episodio crisis."""
    name: str
    true_regime: str
    predicted_regime: str
    predicted_probs: dict[str, float]
    brier_score: float  # 0=perfetto, 1=peggio
    is_correct: bool


@dataclass
class CrisisBrierAggregate:
    """Aggregato sui 5 crisis episodes."""
    n_episodes: int
    mean_brier: float
    max_brier: float
    accuracy: float  # % corretti
    is_well_calibrated: bool  # mean Brier <= 0.10
    per_episode: list[CrisisBrierResult]
    diagnostic: str


def _compute_brier(predicted_probs: dict[str, float], true_regime: str) -> float:
    """Multiclass Brier score: Σ (p_r - y_r)² / N."""
    return sum(
        (predicted_probs.get(r, 0.0) - (1.0 if r == true_regime else 0.0)) ** 2
        for r in REGIMES
    ) / len(REGIMES)


def assess_crisis_brier(
    predict_fn,
    use_independent_labels: bool = False,
) -> CrisisBrierAggregate:
    """Valuta Brier condizionale sui crisis episodes.

    Args:
        predict_fn: callable `indicators -> probs_dict` (es. wrapper di
            classify_regime o predict_regime).
        use_independent_labels: NEW #6 council 2026-05-27. Se True, usa
            INDEPENDENT_CRISIS_EPISODES (IMF/BIS labeled, 11 episodes,
            selection-bias-resistant). Default False (back-compat, 5 episodes).

    Returns:
        CrisisBrierAggregate.
    """
    # Trova crisis episodes nel dataset
    crisis_names = INDEPENDENT_CRISIS_EPISODES if use_independent_labels else CRISIS_EPISODE_NAMES
    crisis_eps = []
    for name_match in crisis_names:
        for ep in HISTORICAL_EPISODES:
            if name_match in ep.name:
                crisis_eps.append(ep)
                break

    if not crisis_eps:
        return CrisisBrierAggregate(
            n_episodes=0, mean_brier=0.0, max_brier=0.0, accuracy=0.0,
            is_well_calibrated=False, per_episode=[],
            diagnostic="No crisis episodes found in dataset",
        )

    per_episode = []
    correct = 0
    briers = []
    for ep in crisis_eps:
        probs = predict_fn(ep.indicators)
        predicted = max(probs, key=probs.get)
        brier = _compute_brier(probs, ep.true_regime)
        is_correct = predicted == ep.true_regime
        if is_correct:
            correct += 1
        briers.append(brier)
        per_episode.append(CrisisBrierResult(
            name=ep.name,
            true_regime=ep.true_regime,
            predicted_regime=predicted,
            predicted_probs={k: round(v, 3) for k, v in probs.items()},
            brier_score=round(brier, 4),
            is_correct=is_correct,
        ))

    mean_brier = sum(briers) / len(briers)
    max_brier = max(briers)
    accuracy = correct / len(crisis_eps)
    is_well_calibrated = mean_brier <= 0.10

    if is_well_calibrated:
        diagnostic = (
            f"WELL-CALIBRATED: mean Brier={mean_brier:.3f} (≤0.10 target). "
            f"Modello CALIBRATO sui regime di crisi → utile per allocation reale."
        )
    elif mean_brier <= 0.20:
        diagnostic = (
            f"ACCEPTABLE: mean Brier={mean_brier:.3f} (≤0.20). "
            f"Modello accettabile ma con margine miglioramento sui crisis."
        )
    else:
        diagnostic = (
            f"POOR CALIBRATION: mean Brier={mean_brier:.3f} (>0.20). "
            f"Modello inaffidabile sui crisis events — review urgente."
        )

    return CrisisBrierAggregate(
        n_episodes=len(crisis_eps),
        mean_brier=round(mean_brier, 4),
        max_brier=round(max_brier, 4),
        accuracy=round(accuracy, 3),
        is_well_calibrated=is_well_calibrated,
        per_episode=per_episode,
        diagnostic=diagnostic,
    )
