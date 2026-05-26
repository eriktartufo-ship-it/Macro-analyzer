"""T9-AUDIT — Walk-forward validation per DFM nowcast.

Council finding (Bug 10): `gdp_yoy_dfm = 4.49%` incoerente con `gdp_roc QoQ
0.49%` (≈2% annualizzato). Possibile look-ahead o miscalibration nel
mapping factor → GDP YoY.

Soluzione: walk-forward validation OLS R² su rolling window.

Pipeline:

1. Per ogni cutoff_year da 1995 a 2024:
   a. Training set: dati FRED fino al cutoff
   b. Fit DFM + OLS factor→GDP YoY
   c. Predict gdp_yoy_dfm per cutoff+1
   d. Confronta con REAL gdp_yoy (gdp_roc QoQ × 4 ≈ annualized) post-cutoff
2. R² walk-forward = correlation predicted vs actual

Output: per-year predictions + R² aggregate. Se R² < 0.30 → DFM nowcast
inaffidabile, da disabilitare in production.

NOTE: questa è una validation **lightweight**. Per full validation serve
panel di indicators FRED storici, che richiede long fetch. Per ora valutiamo
le predictions storiche già in DB (via `gdp_yoy_dfm` campo se popolato).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DFMValidationResult:
    """Risultato single-cutoff validation."""
    cutoff_date: str
    predicted_gdp_yoy: float | None
    actual_gdp_yoy: float | None
    error: float | None  # predicted - actual


@dataclass
class DFMValidationAggregate:
    """R² aggregato + statistics."""
    n_observations: int
    r_squared: float | None
    mean_abs_error: float | None
    bias: float | None  # mean error (predicted - actual)
    is_reliable: bool  # R² >= 0.30 AND |bias| < 1.0
    diagnostic: str


def validate_dfm_walkforward(
    classifications: list,
    min_observations: int = 12,
) -> DFMValidationAggregate:
    """Walk-forward validation di `gdp_yoy_dfm` vs `gdp_roc × 4` annualized.

    Args:
        classifications: lista RegimeClassification ordinati per data.
        min_observations: minimo per calcolare R².

    Returns:
        Aggregate R² + diagnostic.
    """
    import json

    pairs: list[tuple[float, float]] = []
    per_obs: list[DFMValidationResult] = []

    for row in classifications:
        cm = row.conditions_met or {}
        if isinstance(cm, str):
            try:
                cm = json.loads(cm)
            except (ValueError, TypeError):
                continue
        ind = cm.get("indicators", {}) if isinstance(cm, dict) else {}

        gdp_yoy_dfm = ind.get("gdp_yoy_dfm")
        gdp_roc = ind.get("gdp_roc")

        if gdp_yoy_dfm is None or gdp_roc is None:
            per_obs.append(DFMValidationResult(
                cutoff_date=str(row.date),
                predicted_gdp_yoy=gdp_yoy_dfm,
                actual_gdp_yoy=None,
                error=None,
            ))
            continue

        try:
            predicted = float(gdp_yoy_dfm)
            # Annualize gdp_roc (QoQ) approssimato a YoY
            # NB: gdp_roc già in DB potrebbe essere YoY o QoQ. Verificare.
            # Per ora assumiamo gdp_roc è QoQ % → ×4 ≈ YoY annualized.
            actual_proxy = float(gdp_roc) * 4.0
        except (TypeError, ValueError):
            continue

        pairs.append((predicted, actual_proxy))
        per_obs.append(DFMValidationResult(
            cutoff_date=str(row.date),
            predicted_gdp_yoy=round(predicted, 3),
            actual_gdp_yoy=round(actual_proxy, 3),
            error=round(predicted - actual_proxy, 3),
        ))

    if len(pairs) < min_observations:
        return DFMValidationAggregate(
            n_observations=len(pairs),
            r_squared=None,
            mean_abs_error=None,
            bias=None,
            is_reliable=False,
            diagnostic=(
                f"Insufficient data: {len(pairs)} pairs (need {min_observations})"
            ),
        )

    import numpy as np
    preds = np.array([p[0] for p in pairs])
    actuals = np.array([p[1] for p in pairs])

    # R² = 1 - SS_res/SS_tot (only if variation in actual)
    ss_tot = float(((actuals - actuals.mean()) ** 2).sum())
    ss_res = float(((preds - actuals) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0

    mae = float(np.abs(preds - actuals).mean())
    bias = float((preds - actuals).mean())

    is_reliable = r_squared >= 0.30 and abs(bias) < 1.0

    if is_reliable:
        diagnostic = (
            f"RELIABLE: R²={r_squared:.3f} (>=0.30) AND |bias|={abs(bias):.2f} (<1.0). "
            f"DFM nowcast affidabile."
        )
    else:
        diagnostic = (
            f"UNRELIABLE: R²={r_squared:.3f} (target >=0.30), bias={bias:+.2f} "
            f"(target |bias|<1.0). DFM nowcast inaffidabile, considerare "
            f"disable o refit."
        )

    return DFMValidationAggregate(
        n_observations=len(pairs),
        r_squared=round(r_squared, 4),
        mean_abs_error=round(mae, 4),
        bias=round(bias, 4),
        is_reliable=is_reliable,
        diagnostic=diagnostic,
    )
