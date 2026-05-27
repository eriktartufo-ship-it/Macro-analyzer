"""Council 2026-05-27 NEW #3 — Drift detection PSI/KL divergence.

Concept drift = la distribuzione degli indicators live cambia rispetto a
quella del training set. Esempio: AI capex 2024+ porta IT productivity boom
che il classifier (fine-tuned su pre-2024) NON ha mai visto.

Quando questo accade, il modello CONTINUA a fare predictions ma con
underlying distribution shift → confidence può rimanere alta ma accuracy
crolla silently.

Soluzione: PSI (Population Stability Index) + KL divergence comparison tra:
- Distribution training: indicators all'epoca della labeled episodes
- Distribution live: indicators ultimi 90gg (rolling window)

Threshold action (industria-standard):
- PSI < 0.10: no significant drift
- 0.10 <= PSI < 0.25: moderate drift, monitor
- PSI >= 0.25: significant drift, model retrain raccomandato

API:
- `compute_psi(reference, current, bins=10)`: scalare PSI per una feature
- `detect_drift(indicators_history, current_indicators)`: full report multi-feature
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class FeatureDriftResult:
    """Drift result per singola feature."""
    feature_name: str
    psi: float
    kl_divergence: float
    drift_level: str  # "none" | "moderate" | "significant"
    reference_n: int
    current_n: int
    reference_mean: float
    current_mean: float


@dataclass
class DriftReport:
    """Aggregate drift report cross-features."""
    n_features: int
    max_psi: float
    mean_psi: float
    drift_features: list[str]  # features con PSI >= 0.25
    by_feature: list[FeatureDriftResult]
    diagnostic: str


def compute_psi(
    reference: list[float] | np.ndarray,
    current: list[float] | np.ndarray,
    bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Population Stability Index tra due distributions.

    PSI = Σ (current_pct - ref_pct) × ln(current_pct / ref_pct)

    Args:
        reference: distribution training set.
        current: distribution rolling recente.
        bins: numero quantili (default 10).
        eps: floor per evitare log(0).

    Returns:
        Scalare PSI. < 0.10 = stable, < 0.25 = moderate, >= 0.25 = significant.
    """
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]

    if len(ref) < 10 or len(cur) < 10:
        return 0.0  # insufficient data

    # Quantili da reference (binning consistente)
    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.quantile(ref, quantiles)
    # Edges devono essere unique (evita divisioni vuote)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        return 0.0  # all values identical, no drift detectable

    # Conta per bin
    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    cur_counts, _ = np.histogram(cur, bins=bin_edges)

    ref_pct = ref_counts / max(1, ref_counts.sum())
    cur_pct = cur_counts / max(1, cur_counts.sum())

    ref_pct = np.where(ref_pct < eps, eps, ref_pct)
    cur_pct = np.where(cur_pct < eps, eps, cur_pct)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


def compute_kl_divergence(
    reference: list[float] | np.ndarray,
    current: list[float] | np.ndarray,
    bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """KL divergence KL(current || reference). Misura asimmetrica.

    KL = Σ current_p × log(current_p / ref_p)

    Args:
        reference, current: distributions.
        bins: histogram bins.
        eps: floor log.

    Returns:
        KL >= 0. = 0 quando current == reference.
    """
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]

    if len(ref) < 10 or len(cur) < 10:
        return 0.0

    # Shared bin edges (min/max combined)
    all_vals = np.concatenate([ref, cur])
    edges = np.linspace(all_vals.min(), all_vals.max(), bins + 1)

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    ref_pct = ref_counts / max(1, ref_counts.sum())
    cur_pct = cur_counts / max(1, cur_counts.sum())

    ref_pct = np.where(ref_pct < eps, eps, ref_pct)
    cur_pct = np.where(cur_pct < eps, eps, cur_pct)

    kl = float(np.sum(cur_pct * np.log(cur_pct / ref_pct)))
    return kl


def _drift_level(psi: float) -> str:
    if psi >= 0.25:
        return "significant"
    if psi >= 0.10:
        return "moderate"
    return "none"


def detect_drift(
    reference: dict[str, list[float]],
    current: dict[str, list[float]],
    bins: int = 10,
) -> DriftReport:
    """Multi-feature drift report.

    Args:
        reference: dict feature_name -> lista valori training distribution.
        current: dict feature_name -> lista valori rolling recente.
        bins: histogram bins.

    Returns:
        DriftReport completo con per-feature breakdown.
    """
    results: list[FeatureDriftResult] = []
    drift_features: list[str] = []

    for feature in reference:
        ref = reference[feature]
        cur = current.get(feature, [])
        if not cur:
            continue
        psi = compute_psi(ref, cur, bins=bins)
        kl = compute_kl_divergence(ref, cur, bins=bins)
        level = _drift_level(psi)
        if level == "significant":
            drift_features.append(feature)

        ref_arr = [v for v in ref if not (v is None or math.isnan(v))]
        cur_arr = [v for v in cur if not (v is None or math.isnan(v))]
        results.append(FeatureDriftResult(
            feature_name=feature,
            psi=round(psi, 4),
            kl_divergence=round(kl, 4),
            drift_level=level,
            reference_n=len(ref_arr),
            current_n=len(cur_arr),
            reference_mean=round(float(np.mean(ref_arr)), 4) if ref_arr else 0.0,
            current_mean=round(float(np.mean(cur_arr)), 4) if cur_arr else 0.0,
        ))

    if not results:
        return DriftReport(
            n_features=0, max_psi=0.0, mean_psi=0.0,
            drift_features=[], by_feature=[],
            diagnostic="No features to compare.",
        )

    psis = [r.psi for r in results]
    max_psi = max(psis)
    mean_psi = sum(psis) / len(psis)

    if drift_features:
        diagnostic = (
            f"SIGNIFICANT DRIFT detected on {len(drift_features)} feature(s): "
            f"{', '.join(drift_features)}. Max PSI={max_psi:.3f}. "
            f"Model retrain raccomandato."
        )
    elif max_psi >= 0.10:
        diagnostic = (
            f"Moderate drift (max PSI={max_psi:.3f}). Monitor closely. "
            f"No retrain necessary yet."
        )
    else:
        diagnostic = (
            f"No significant drift (max PSI={max_psi:.3f} < 0.10). "
            f"Model distribution stable."
        )

    return DriftReport(
        n_features=len(results),
        max_psi=round(max_psi, 4),
        mean_psi=round(mean_psi, 4),
        drift_features=drift_features,
        by_feature=results,
        diagnostic=diagnostic,
    )
