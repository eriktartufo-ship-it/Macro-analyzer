"""Tier 5.1 LOOP CLOSURE — Crisis modulation post-classifier.

Modula le probabilities di regime in base al `crisis_type` + `risk_score`
prodotti da `assess_crisis_risk`. È un post-processing chirurgico che
chiude il loop tra crisis indicator (oggi vista parallela) e regime
ufficiale.

Pipeline:
1. classify_regime → probs base
2. (Opt-in) apply_crisis_modulation(probs, indicators, dedollar) → probs modulate
3. Save / serve

**Filosofia**: il Crisis Indicator vede dimensioni di stress che il
classifier 4-quadranti non distingue (deflation 2008 vs deflation mild
2024 sono entrambe "deflation"). Modulation traduce questo "secondo
asse" in shift di probability.

**Regole shift** (opt-in via `use_crisis_modulation()`):

- `risk_score < 0.3` → no-op (segnale debole, no modulation).
- `deflation_crash` & risk>=0.3: deflation += α·risk_score
- `stagflation_debasement` & risk>=0.3: stagflation += α·risk_score
- `bubble_goldilocks` & risk>=0.3: deflation += 0.5·α·risk_score,
  stagflation += 0.5·α·risk_score (preludio mean-reversion downside).
  Sottrae da reflation + goldilocks proporzionalmente.

Alpha default = 0.05 → max shift singolo ±5pp.

Renormalizzazione finale Σ=1.0.
"""
from __future__ import annotations

from app.services.config_flags import use_crisis_modulation
from app.services.regime.crisis_indicator import assess_crisis_risk

_ALPHA_DEFAULT = 0.05
_RISK_THRESHOLD = 0.30  # default: sotto questa soglia → no-op
_RISK_THRESHOLD_BUBBLE = 0.05  # bubble per design ha pochi trigger
                               # (vix_compressed da solo basta come segnale)


def apply_crisis_modulation(
    probs: dict[str, float],
    indicators: dict[str, float],
    dedollar_combined: float | None = None,
    alpha: float = _ALPHA_DEFAULT,
) -> dict[str, float]:
    """Modula probs in base al crisis_type + risk_score.

    Args:
        probs: probabilities pre-modulation (somma ~1.0).
        indicators: indicators dict per assess_crisis_risk.
        dedollar_combined: dedollar score 0-1 (opzionale).
        alpha: max shift per regime (default 0.05).

    Returns:
        Nuovo dict probs modulate (somma = 1.0). Se flag OFF o risk basso,
        ritorna copy di probs invariata.
    """
    if not use_crisis_modulation():
        return dict(probs)
    if not probs:
        return dict(probs)

    try:
        assessment = assess_crisis_risk(indicators, dedollar_combined=dedollar_combined)
    except Exception:
        # Graceful: se crisis assessment fallisce, no-op
        return dict(probs)

    risk = float(assessment.risk_score)
    crisis_type = assessment.crisis_type
    # Bubble è "calma sospetta": pochi trigger ma signal early-warning forte.
    # Threshold più bassa per non perdere il segnale (es. 2000 pre-burst,
    # 2021 SPAC mania entrambi avevano risk_score basso).
    threshold = _RISK_THRESHOLD_BUBBLE if crisis_type == "bubble_goldilocks" else _RISK_THRESHOLD
    if risk < threshold:
        return dict(probs)
    shift_amount = alpha * risk  # max alpha quando risk=1.0

    modulated = dict(probs)
    if crisis_type == "deflation_crash":
        _shift_toward(modulated, target="deflation", amount=shift_amount)
    elif crisis_type == "stagflation_debasement":
        _shift_toward(modulated, target="stagflation", amount=shift_amount)
    elif crisis_type == "bubble_goldilocks":
        # Preludio mean-reversion: split equamente verso defl + stag
        half = shift_amount * 0.5
        _shift_toward(modulated, target="deflation", amount=half)
        _shift_toward(modulated, target="stagflation", amount=half)
    else:
        # no_crisis o tipi non gestiti → no-op
        return dict(probs)

    # Renormalizza Σ=1.0 (cleanup eventuali drift floating point)
    total = sum(modulated.values())
    if total <= 0:
        return dict(probs)
    return {k: max(0.0, v) / total for k, v in modulated.items()}


def _shift_toward(probs: dict[str, float], target: str, amount: float) -> None:
    """Sposta `amount` di probability mass dagli altri regimi verso `target`.

    Sottrae proporzionalmente dagli altri (rispetta i pesi relativi).
    Modifica `probs` in-place. Clamping floor 0.0 sui contribuenti.
    """
    if target not in probs:
        return
    others = {k: v for k, v in probs.items() if k != target}
    others_sum = sum(others.values())
    if others_sum <= 0:
        # Tutti zero: aggiungi al target ma non oltre 1.0
        probs[target] = min(1.0, probs[target] + amount)
        return

    # Quanto possiamo realmente prelevare: min(amount, others_sum * 0.99)
    # (lasciamo un floor per evitare di azzerare completamente gli altri)
    actual = min(amount, others_sum * 0.99)
    probs[target] = probs[target] + actual

    # Sottrai proporzionalmente dagli altri
    factor = (others_sum - actual) / others_sum
    for k in others:
        probs[k] = others[k] * factor
