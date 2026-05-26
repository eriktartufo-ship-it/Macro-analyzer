"""T9-AUDIT P0: Circuit-breakers per advisory mode.

Council finding (raydalio + taleb + buffett):
"OK per advisory con 3 circuit-breaker mandatori, NO per autonomous."

3 circuit-breaker:

1. **Low confidence freeze**: se `max_prob < 0.45` → "regime incerto",
   no top-N raccomandazioni, mostra solo defensive.
2. **OOD detector**: Mahalanobis distance da training distribution. Se
   `mahalanobis > 99th percentile` → "out-of-distribution",
   ridurre conviction × 0.5.
3. **Consecutive wrong tracker**: se 2 predizioni consecutive sbagliate
   (post-mortem mensile), freeze advisory finché review umano.

Output per ogni circuit-breaker:
- `is_triggered: bool`
- `severity: 'info' | 'warning' | 'critical'`
- `description`
- `recommendation`: cosa fare quando triggered
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.services.regime.historical_episodes import HISTORICAL_EPISODES


@dataclass
class CircuitBreakerSignal:
    """Singolo segnale circuit-breaker."""
    name: str
    is_triggered: bool
    severity: str  # info | warning | critical
    metric_value: float | None
    threshold: float | None
    description: str
    recommendation: str


# Indicators chiave per OOD detection (subset stabile cross-time)
_OOD_INDICATORS = (
    "gdp_roc", "cpi_yoy", "unrate", "vix",
    "fed_funds", "breakeven_10y", "baa_10y_spread",
)


def _compute_training_distribution_stats() -> tuple[np.ndarray, np.ndarray]:
    """Estrae mean + cov inverse dei training indicators per Mahalanobis.

    Cached al primo uso. Returns (mean_vec, cov_inv_matrix).

    T9-AUDIT-BUG-FIX (bughunter HIGH-2): alias util per match live indicators.
    Pre-fix: `indicators.get("fed_funds")` su HISTORICAL_EPISODES funziona (training
    set usa "fed_funds") MA su live indicators (con `fed_funds_rate`) usa default →
    Mahalanobis su solo 4-5 dimensioni efficaci.
    """
    from app.services.common.indicator_aliases import safe_indicator
    rows = []
    for ep in HISTORICAL_EPISODES:
        row = [safe_indicator(ep.indicators, key, default=0.0) for key in _OOD_INDICATORS]
        rows.append(row)
    X = np.array(rows, dtype=np.float64)
    mean_vec = X.mean(axis=0)
    cov = np.cov(X.T)
    # Regularize per evitare singolarità su sample piccolo
    cov_reg = cov + np.eye(len(_OOD_INDICATORS)) * 1e-4
    try:
        cov_inv = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        cov_inv = np.eye(len(_OOD_INDICATORS))
    return mean_vec, cov_inv


# Module-level cache (ricomputato se HISTORICAL_EPISODES cambia)
_TRAINING_STATS: tuple[np.ndarray, np.ndarray] | None = None
_OOD_99TH_PERCENTILE: float | None = None


def _get_training_stats() -> tuple[np.ndarray, np.ndarray]:
    global _TRAINING_STATS
    if _TRAINING_STATS is None:
        _TRAINING_STATS = _compute_training_distribution_stats()
    return _TRAINING_STATS


def _compute_ood_threshold() -> float:
    """Calcola 99th percentile Mahalanobis distance del training set.

    T9-AUDIT-BUG-FIX (bughunter HIGH-2): usa safe_indicator alias-aware.
    """
    from app.services.common.indicator_aliases import safe_indicator
    global _OOD_99TH_PERCENTILE
    if _OOD_99TH_PERCENTILE is not None:
        return _OOD_99TH_PERCENTILE
    mean_vec, cov_inv = _get_training_stats()
    distances = []
    for ep in HISTORICAL_EPISODES:
        row = np.array([safe_indicator(ep.indicators, k, default=0.0) for k in _OOD_INDICATORS])
        diff = row - mean_vec
        d = float(np.sqrt(diff @ cov_inv @ diff))
        distances.append(d)
    _OOD_99TH_PERCENTILE = float(np.percentile(distances, 99))
    return _OOD_99TH_PERCENTILE


def reset_ood_cache() -> None:
    """Invalida cache (chiamato se HISTORICAL_EPISODES viene esteso)."""
    global _TRAINING_STATS, _OOD_99TH_PERCENTILE
    _TRAINING_STATS = None
    _OOD_99TH_PERCENTILE = None


def compute_mahalanobis_distance(indicators: dict[str, Any]) -> float:
    """Mahalanobis distance di indicators da training distribution.

    T9-AUDIT-BUG-FIX (bughunter HIGH-2): usa safe_indicator alias-aware.
    Pre-fix: 3/7 dimensioni live sempre = training mean (silent default
    su `fed_funds`/`baa_10y_spread`/`yield_curve_spread`).
    """
    from app.services.common.indicator_aliases import safe_indicator
    mean_vec, cov_inv = _get_training_stats()
    row = []
    for i, k in enumerate(_OOD_INDICATORS):
        v_f = safe_indicator(indicators, k, default=float(mean_vec[i]))
        row.append(v_f)
    row_arr = np.array(row)
    diff = row_arr - mean_vec
    return float(np.sqrt(diff @ cov_inv @ diff))


def check_low_confidence(
    max_prob: float,
    threshold: float = 0.45,
) -> CircuitBreakerSignal:
    """CB1: low confidence freeze. max_prob < 0.45 → no top-N raccomandazioni."""
    is_trig = max_prob < threshold
    return CircuitBreakerSignal(
        name="low_confidence",
        is_triggered=is_trig,
        severity="warning" if is_trig else "info",
        metric_value=max_prob,
        threshold=threshold,
        description=(
            f"max_prob={max_prob:.3f} {'<' if is_trig else '>='} {threshold}"
        ),
        recommendation=(
            "Regime classifier non sicuro. Mostra solo defensive allocation, "
            "no top-5 specific picks."
            if is_trig else
            "Confidence sufficiente. Top-N raccomandazioni affidabili."
        ),
    )


def check_ood_distribution(indicators: dict[str, Any]) -> CircuitBreakerSignal:
    """CB2: OOD detector. Mahalanobis distance > 99th percentile training → flag."""
    distance = compute_mahalanobis_distance(indicators)
    threshold = _compute_ood_threshold()
    is_trig = distance > threshold
    return CircuitBreakerSignal(
        name="ood_distribution",
        is_triggered=is_trig,
        severity="critical" if is_trig else "info",
        metric_value=distance,
        threshold=threshold,
        description=(
            f"Mahalanobis={distance:.2f} {'>' if is_trig else '<='} "
            f"threshold_99th={threshold:.2f}"
        ),
        recommendation=(
            "Indicators correnti FUORI dalla distribuzione training. "
            "Modello mai testato in queste condizioni. "
            "Riduci conviction × 0.5 + defensive heavy."
            if is_trig else
            "Indicators dentro distribuzione training (in-sample-like)."
        ),
    )


def check_consecutive_wrong(
    recent_predictions_correct: list[bool],
    threshold: int = 2,
) -> CircuitBreakerSignal:
    """CB3: 2+ predizioni consecutive sbagliate → freeze advisory.

    Args:
        recent_predictions_correct: lista bool ordinata recent-first.
            Es. [False, False, True, ...] = ultime 2 wrong.
        threshold: minimo consecutive wrong per trigger.
    """
    # Conta consecutive wrong da inizio
    n_wrong = 0
    for correct in recent_predictions_correct:
        if not correct:
            n_wrong += 1
        else:
            break
    is_trig = n_wrong >= threshold
    return CircuitBreakerSignal(
        name="consecutive_wrong",
        is_triggered=is_trig,
        severity="critical" if is_trig else "info",
        metric_value=float(n_wrong),
        threshold=float(threshold),
        description=(
            f"Consecutive wrong: {n_wrong} (threshold {threshold})"
        ),
        recommendation=(
            f"FREEZE advisory. Review umano necessario prima di nuove "
            f"raccomandazioni. Possibile regime change non catturato."
            if is_trig else
            "Track record recente accettabile."
        ),
    )


def evaluate_all_circuit_breakers(
    max_prob: float,
    indicators: dict[str, Any],
    recent_predictions_correct: list[bool] | None = None,
) -> dict[str, Any]:
    """Valuta tutti i 3 circuit-breaker in un colpo.

    Returns:
        {
            "signals": list[CircuitBreakerSignal],
            "any_triggered": bool,
            "any_critical": bool,
            "advisory_mode": "normal" | "uncertain" | "frozen",
            "summary": str,
        }
    """
    signals = [
        check_low_confidence(max_prob),
        check_ood_distribution(indicators),
    ]
    if recent_predictions_correct is not None:
        signals.append(check_consecutive_wrong(recent_predictions_correct))

    any_trig = any(s.is_triggered for s in signals)
    any_crit = any(s.is_triggered and s.severity == "critical" for s in signals)

    if any_crit:
        mode = "frozen"
        summary = "FROZEN: critical circuit-breaker triggered. No new advisory."
    elif any_trig:
        mode = "uncertain"
        summary = "UNCERTAIN: warning triggered. Defensive only + caveats."
    else:
        mode = "normal"
        summary = "NORMAL: all circuit-breakers green. Advisory affidabile."

    return {
        "signals": signals,
        "any_triggered": any_trig,
        "any_critical": any_crit,
        "advisory_mode": mode,
        "summary": summary,
    }
