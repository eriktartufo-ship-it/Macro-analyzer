"""Monte Carlo simulator: traiettorie regime + asset scores con bande di incertezza.

A differenza di `transition_matrix.project_probabilities` che fa moltiplicazione
matriciale (proiezione media analitica, no incertezza), qui simuliamo N traiettorie
discrete campionando dalla matrice di transizione empirica. Il risultato e' la
**distribuzione completa** ad ogni step, da cui estraiamo quantili (p10/p25/p50/p75/p90)
per ogni regime → cone chart visivo dell'incertezza.

Per ciascun path:
  - Stato iniziale: campionato dalla distribuzione corrente (multinomial)
  - Step t+1: campionato da Categorical(transition_matrix[state_t, :])

Per gli asset scores: per ogni path applichiamo il scoring engine ad ogni step
(distribuzione = 1-hot sullo stato), poi aggreghiamo per asset/step.

NOTA: 1-hot e' una semplificazione utile per il cono asset; un'alternativa piu'
fine e' fare scoring sulla distribuzione marginale (gia' fatto da
project_probabilities). Manteniamo 1-hot perche' restituisce naturalmente la
varianza (path con regime diverso → asset score diverso → spread misurabile).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger
from sqlalchemy.orm import Session

from app.services.regime.classifier import REGIMES
from app.services.regime.transition_matrix import compute_transition_matrix
from app.services.scoring.engine import ASSET_CLASSES, calculate_final_scores

DEFAULT_N_PATHS = 500
DEFAULT_STEPS = 12        # 12 step a horizon 30d default = ~1 anno
DEFAULT_HORIZON = 30


@dataclass
class RegimeBand:
    regime: str
    median: list[float]   # P50 per ogni step (incluso step 0 = current)
    p10: list[float]
    p25: list[float]
    p75: list[float]
    p90: list[float]
    mean: list[float]


@dataclass
class AssetBand:
    asset: str
    median: list[float]
    p10: list[float]
    p25: list[float]
    p75: list[float]
    p90: list[float]
    mean: list[float]


@dataclass
class MonteCarloResult:
    n_paths: int
    n_steps: int
    horizon_days: int
    initial_distribution: dict[str, float]
    step_dates_offsets: list[int]     # giorni dal punto corrente per ogni step
    regime_bands: list[RegimeBand]
    asset_bands: list[AssetBand]
    transition_matrix_observations: int
    notes: list[str]


def _sample_initial_states(
    initial: dict[str, float], n_paths: int, rng: np.random.Generator,
) -> np.ndarray:
    """Campiona n_paths stati iniziali dalla distribuzione corrente."""
    probs = np.array([initial.get(r, 0.0) for r in REGIMES], dtype=float)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if s <= 0:
        probs = np.full(len(REGIMES), 1.0 / len(REGIMES))
    else:
        probs = probs / s
    return rng.choice(len(REGIMES), size=n_paths, p=probs)


def _simulate_paths(
    initial_states: np.ndarray,
    transition: np.ndarray,
    n_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Returns: paths (n_paths, n_steps + 1) con stati interi 0..K-1."""
    K = transition.shape[0]
    n_paths = initial_states.shape[0]
    paths = np.zeros((n_paths, n_steps + 1), dtype=int)
    paths[:, 0] = initial_states

    # Pre-genera tutti i sample uniformi per vettorializzare le scelte
    cumprobs = np.cumsum(transition, axis=1)  # (K, K)
    for t in range(n_steps):
        u = rng.random(n_paths)
        # Per ogni path, lookup cumprobs[paths[:,t]]
        cur_states = paths[:, t]
        cum = cumprobs[cur_states]               # (n_paths, K)
        next_states = (cum >= u[:, None]).argmax(axis=1)
        paths[:, t + 1] = next_states
    return paths


def _compute_regime_bands(paths: np.ndarray) -> list[RegimeBand]:
    """Per ogni step e regime calcola la frazione di path nello stato + percentili
    bootstrap-like (qui: usiamo direttamente la frequenza)."""
    n_paths, n_periods = paths.shape
    K = len(REGIMES)

    # one_hot shape (n_paths, n_periods, K)
    one_hot = np.zeros((n_paths, n_periods, K))
    for k in range(K):
        one_hot[:, :, k] = (paths == k).astype(float)

    # Per ogni step e regime: distribuzione su path (sono 0/1) → useremo
    # bootstrap-style su fraction by sampling sub-batches
    bands = []
    for k, regime_name in enumerate(REGIMES):
        # frequency per step (mean across paths)
        freq = one_hot[:, :, k].mean(axis=0)
        # Per percentili usiamo bootstrap su sub-batch (200 batch da 100 path)
        n_bootstrap = 200
        batch_size = max(50, n_paths // 5)
        rng = np.random.default_rng(42)
        boots = np.empty((n_bootstrap, n_periods))
        for b in range(n_bootstrap):
            idx = rng.integers(0, n_paths, batch_size)
            boots[b, :] = one_hot[idx, :, k].mean(axis=0)
        bands.append(RegimeBand(
            regime=regime_name,
            median=list(map(float, np.percentile(boots, 50, axis=0))),
            p10=list(map(float, np.percentile(boots, 10, axis=0))),
            p25=list(map(float, np.percentile(boots, 25, axis=0))),
            p75=list(map(float, np.percentile(boots, 75, axis=0))),
            p90=list(map(float, np.percentile(boots, 90, axis=0))),
            mean=list(map(float, freq)),
        ))
    return bands


def _compute_asset_bands(paths: np.ndarray, assets: list[str]) -> list[AssetBand]:
    """Per ogni path e step, calcola asset_scores assumendo 1-hot sul regime corrente.
    Aggrega per asset i percentili sui path."""
    n_paths, n_periods = paths.shape
    n_assets = len(assets)

    # Pre-calcola i 4 score-vector (uno per regime puro)
    pure_scores = np.zeros((len(REGIMES), n_assets))
    for k, regime in enumerate(REGIMES):
        probs = {r: (1.0 if r == regime else 0.0) for r in REGIMES}
        scores = calculate_final_scores(probs)
        pure_scores[k, :] = [scores.get(a, 0.0) for a in assets]

    # asset_scores shape (n_paths, n_periods, n_assets)
    # path[i,t] = state idx → pure_scores[state_idx]
    asset_scores = pure_scores[paths]   # (n_paths, n_periods, n_assets)

    bands = []
    for j, asset in enumerate(assets):
        ts = asset_scores[:, :, j]  # (n_paths, n_periods)
        bands.append(AssetBand(
            asset=asset,
            median=list(map(float, np.percentile(ts, 50, axis=0))),
            p10=list(map(float, np.percentile(ts, 10, axis=0))),
            p25=list(map(float, np.percentile(ts, 25, axis=0))),
            p75=list(map(float, np.percentile(ts, 75, axis=0))),
            p90=list(map(float, np.percentile(ts, 90, axis=0))),
            mean=list(map(float, ts.mean(axis=0))),
        ))
    return bands


def run_monte_carlo(
    db: Session,
    n_paths: int = DEFAULT_N_PATHS,
    n_steps: int = DEFAULT_STEPS,
    horizon_days: int = DEFAULT_HORIZON,
    initial_distribution: dict[str, float] | None = None,
    assets: list[str] | None = None,
    seed: int = 42,
) -> MonteCarloResult:
    """Esegue Monte Carlo regime + asset scores.

    Args:
        db: session DB per pescare la transition matrix empirica
        n_paths: numero traiettorie simulate (default 500)
        n_steps: orizzonti futuri (default 12)
        horizon_days: giorni per step (default 30 = ~mese)
        initial_distribution: se None, usa l'ultima classification del DB
        assets: lista asset class da includere (default tutti)

    Returns: MonteCarloResult con bande regime + asset.
    """
    # Transition matrix empirica
    tm = compute_transition_matrix(db, horizon_days=horizon_days)
    if tm.total_observations < 10:
        raise ValueError(
            f"Monte Carlo: transition matrix ha solo {tm.total_observations} obs. "
            f"Esegui /regime/backfill/historical prima."
        )

    A = np.array([
        [tm.probabilities[r_from][r_to] for r_to in REGIMES]
        for r_from in REGIMES
    ])
    # Smoothing: aggiungi pseudo-conteggio Dirichlet per evitare zero stati
    A = A + 1e-3
    A = A / A.sum(axis=1, keepdims=True)

    # Distribuzione iniziale
    if initial_distribution is None:
        from app.models import RegimeClassification
        last = (
            db.query(RegimeClassification)
            .order_by(RegimeClassification.date.desc()).first()
        )
        if last is None:
            initial = {r: 0.25 for r in REGIMES}
        else:
            initial = {
                "reflation": last.probability_reflation,
                "stagflation": last.probability_stagflation,
                "deflation": last.probability_deflation,
                "goldilocks": last.probability_goldilocks,
            }
    else:
        initial = initial_distribution

    rng = np.random.default_rng(seed)
    initial_states = _sample_initial_states(initial, n_paths, rng)
    paths = _simulate_paths(initial_states, A, n_steps, rng)

    regime_bands = _compute_regime_bands(paths)
    assets_list = assets or list(ASSET_CLASSES)
    asset_bands = _compute_asset_bands(paths, assets_list)

    notes: list[str] = []
    if tm.total_observations < 100:
        notes.append(
            f"Transition matrix basata su solo {tm.total_observations} obs — proiezioni rumorose."
        )

    logger.info(
        f"Monte Carlo: {n_paths} paths × {n_steps} steps "
        f"(horizon {horizon_days}d, {tm.total_observations} obs in matrix)"
    )

    return MonteCarloResult(
        n_paths=n_paths,
        n_steps=n_steps,
        horizon_days=horizon_days,
        initial_distribution=initial,
        step_dates_offsets=[i * horizon_days for i in range(n_steps + 1)],
        regime_bands=regime_bands,
        asset_bands=asset_bands,
        transition_matrix_observations=tm.total_observations,
        notes=notes,
    )
