"""Tier 8.4 — Forecast probabilità regime futuro prossimi 3-6 mesi.

Approccio: usa transition matrix empirica calcolata dalla storia
`RegimeClassification` per propagare le probabilities correnti N step avanti.

P(regime_{t+k} | regime_t) = (T^k)[regime_t, regime_{t+k}]

dove T è la transition matrix mensile (4x4) stimata da count empirici
dei passaggi mese-su-mese in DB.

Output:
- forecast_horizon_months: lista [1, 3, 6]
- probs_t0: regime corrente
- probs_t+1, t+3, t+6: distribuzione propagata
- expected_regime per ogni horizon
- regime_volatility (Shannon entropy della distribuzione)

Combinato con T8.3 ensemble: forecast diventa più affidabile perché parte
da probs già blendate rule+ML.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sqlalchemy.orm import Session

from app.models import RegimeClassification


REGIMES = ["reflation", "stagflation", "deflation", "goldilocks"]


@dataclass
class RegimeForecast:
    """Forecast struct per N step avanti."""
    horizon_months: int
    probs: dict[str, float]
    expected_regime: str
    entropy: float  # Shannon entropy, max = log2(4) = 2.0 (uniforme)
    is_degenerate: bool = False  # True se entropy > 1.95 (no signal utile)


def estimate_transition_matrix(
    db: Session,
    min_records: int = 50,
) -> np.ndarray:
    """Stima transition matrix monthly 4x4 da DB classifications storiche.

    Algoritmo:
    1. Ordina classifications by date.
    2. Per ogni coppia (t, t+1) consecutiva, increment count[regime_t][regime_t+1].
    3. Normalize riga: P(j|i) = count[i][j] / sum(count[i]).
    4. Fallback uniforme se < min_records.

    Returns:
        T: ndarray (4, 4), T[i][j] = P(regime_j | regime_i).
    """
    rows = (
        db.query(RegimeClassification)
        .order_by(RegimeClassification.date.asc())
        .all()
    )
    if len(rows) < min_records:
        # Fallback: uniform transition con self-persistence
        T = np.full((4, 4), 0.1)
        np.fill_diagonal(T, 0.7)
        return T

    counts = np.zeros((4, 4), dtype=np.float64)
    regime_idx = {r: i for i, r in enumerate(REGIMES)}

    prev_regime: str | None = None
    for row in rows:
        if row.regime not in regime_idx:
            prev_regime = None
            continue
        if prev_regime is not None:
            i = regime_idx[prev_regime]
            j = regime_idx[row.regime]
            counts[i][j] += 1.0
        prev_regime = row.regime

    # Normalizza righe + safety floor (evita prob 0 esatto)
    T = np.zeros_like(counts)
    for i in range(4):
        row_sum = counts[i].sum()
        if row_sum > 0:
            T[i] = counts[i] / row_sum
        else:
            T[i] = np.full(4, 0.25)
        # Floor minimo per ogni transition (evita propagazione degenerata)
        T[i] = np.maximum(T[i], 0.005)
        T[i] = T[i] / T[i].sum()
    return T


def shannon_entropy(probs: dict[str, float]) -> float:
    """Shannon entropy in base 2. Max = log2(4) ≈ 2.0 = uniforme."""
    e = 0.0
    for p in probs.values():
        if p > 1e-9:
            e -= p * np.log2(p)
    return float(round(e, 4))


def forecast_horizons(
    current_probs: dict[str, float],
    transition_matrix: np.ndarray,
    horizons: tuple[int, ...] = (1, 3, 6),
) -> list[RegimeForecast]:
    """Propaga current_probs N step avanti via transition matrix.

    p_{t+k} = p_t @ T^k

    Args:
        current_probs: dict {regime: prob} che somma a ~1.
        transition_matrix: 4x4 numpy array.
        horizons: tuple di mesi per i quali generare forecast.
    """
    # Vector dall'ordine REGIMES
    p_vec = np.array([current_probs.get(r, 0.0) for r in REGIMES])
    s = p_vec.sum()
    if s > 0:
        p_vec = p_vec / s

    out: list[RegimeForecast] = []
    T = transition_matrix
    T_powered = np.eye(4)
    last_horizon = 0

    for h in sorted(horizons):
        # Eleva T alla potenza (h - last_horizon)
        for _ in range(h - last_horizon):
            T_powered = T_powered @ T
        last_horizon = h
        forecasted = p_vec @ T_powered
        forecasted = forecasted / forecasted.sum()
        probs_dict = {r: float(round(forecasted[i], 4)) for i, r in enumerate(REGIMES)}
        ent = shannon_entropy(probs_dict)
        # Degenerate detection: entropy > 1.95 ≈ uniforme (no signal utile)
        is_degenerate = ent > 1.95
        expected = "uniform" if is_degenerate else max(probs_dict, key=probs_dict.get)
        out.append(RegimeForecast(
            horizon_months=h,
            probs=probs_dict,
            expected_regime=expected,
            entropy=ent,
            is_degenerate=is_degenerate,
        ))
    return out


def asset_allocation_forecast(
    forecast_horizons_list: list[RegimeForecast],
    score_fn,  # callable: (asset, regime) -> score 0-100
    asset_classes: list[str],
    top_n: int = 5,
) -> dict[int, list[tuple[str, float]]]:
    """Per ogni horizon, calcola top-N asset weighted by forecasted regime probs.

    Args:
        forecast_horizons_list: output di `forecast_horizons`.
        score_fn: funzione (asset, regime) → score (es. _asset_regime_score
            o _asset_regime_score_percentile).
        asset_classes: lista 15 asset class.
        top_n: top-N da restituire.

    Returns:
        {horizon_months: [(asset, weighted_score), ...]}
    """
    out: dict[int, list[tuple[str, float]]] = {}
    for fc in forecast_horizons_list:
        scores = {}
        for asset in asset_classes:
            s = sum(fc.probs.get(r, 0.0) * score_fn(asset, r) for r in REGIMES)
            scores[asset] = round(s, 2)
        top = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
        out[fc.horizon_months] = top
    return out
