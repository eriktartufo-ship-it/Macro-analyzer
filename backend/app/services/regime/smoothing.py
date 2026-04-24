"""Smoothing temporale delle probabilita regime via forward-backward.

Il classifier rule-based emette una distribuzione P(regime | indicatori_t) per ogni
punto temporale, in modo indipendente tra punti. Questo ignora un fatto essenziale:
i regimi macro sono **persistenti**. Il 1996 era goldilocks, il 1997 con alta probabilita
sara ancora goldilocks — non e un coin flip.

Qui applichiamo un filtro HMM sovrapposto usando:
  - `P(regime | indicatori_t)` del rule-based come **emission likelihood** (soft label)
  - La **transition matrix empirica** calcolata dal backfill storico come prior di
    persistenza

Il risultato: probabilita temporalmente coerenti. Punti rumorosi vengono ammorbiditi
verso il regime adiacente; transizioni genuine emergono perche' supportate da
osservazioni consecutive, non da un singolo outlier.

Questo NON sostituisce il rule-based: lo raffina. Il rule-based fornisce il segnale
istantaneo, questo modulo aggiunge la memoria.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
from loguru import logger
from scipy.special import logsumexp
from sqlalchemy.orm import Session

from app.models import RegimeClassification
from app.services.regime.classifier import REGIMES
from app.services.regime.transition_matrix import compute_transition_matrix

_EPS = 1e-12


@dataclass
class SmoothedPoint:
    date: date
    raw: dict[str, float]       # posterior rule-based (invariato)
    smoothed: dict[str, float]  # posterior smoothed via forward-backward


@dataclass
class SmoothedHistoryResult:
    points: list[SmoothedPoint]
    transition_horizon_days: int
    total_observations: int


def _renormalize(p: np.ndarray, floor: float = 1e-4) -> np.ndarray:
    p = np.maximum(p, floor)
    return p / p.sum()


def smooth_probabilities_sequence(
    raw_probs: np.ndarray,  # (T, K)
    transition: np.ndarray,  # (K, K), righe = from, colonne = to
) -> np.ndarray:
    """Forward-backward con emission = raw_probs e transition data.

    In un HMM tradizionale l'emission e' una likelihood. Qui usiamo direttamente
    la posteriore rule-based P(r|x_t) come emission (approssimazione:
    P(x_t|r) ∝ P(r|x_t) assumendo prior uniforme su r).

    Returns: posterior smoothed, shape (T, K).
    """
    T, K = raw_probs.shape
    log_E = np.log(np.maximum(raw_probs, _EPS))
    log_A = np.log(np.maximum(transition, _EPS))
    log_pi = np.log(np.full(K, 1.0 / K))

    # Forward
    log_alpha = np.full((T, K), -np.inf)
    log_alpha[0] = log_pi + log_E[0]
    for t in range(1, T):
        log_alpha[t] = log_E[t] + logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0)

    # Backward
    log_beta = np.full((T, K), -np.inf)
    log_beta[T - 1] = 0.0
    for t in range(T - 2, -1, -1):
        log_beta[t] = logsumexp(
            log_A + (log_E[t + 1] + log_beta[t + 1])[None, :], axis=1
        )

    # Posterior
    log_gamma = log_alpha + log_beta
    log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
    gamma = np.exp(log_gamma)
    # Sicurezza numerica
    return np.array([_renormalize(gamma[t]) for t in range(T)])


def smooth_history(
    db: Session,
    days: int = 365 * 5,
    transition_horizon_days: int = 30,
) -> SmoothedHistoryResult:
    """Legge lo storico delle classificazioni, costruisce la transition matrix
    empirica, applica forward-backward smoothing.

    Args:
        days: finestra di storico (default 5 anni)
        transition_horizon_days: horizon usato per la transition matrix (default 30d)

    Returns:
        SmoothedHistoryResult con raw vs smoothed per ogni punto.
    """
    from datetime import date as _date, timedelta

    cutoff = _date.today() - timedelta(days=days)
    rows: list[RegimeClassification] = (
        db.query(RegimeClassification)
        .filter(RegimeClassification.date >= cutoff)
        .order_by(RegimeClassification.date.asc())
        .all()
    )

    if len(rows) < 5:
        return SmoothedHistoryResult(points=[], transition_horizon_days=transition_horizon_days, total_observations=0)

    # Costruisci matrice raw_probs (T, K) rispettando l'ordine REGIMES
    raw = np.array([
        [
            row.probability_reflation,
            row.probability_stagflation,
            row.probability_deflation,
            row.probability_goldilocks,
        ]
        for row in rows
    ])

    # Transition matrix empirica su TUTTO lo storico (non solo la finestra)
    tm = compute_transition_matrix(db, horizon_days=transition_horizon_days)
    if tm.total_observations == 0:
        # Fallback: matrice persistente uniforme
        K = len(REGIMES)
        A = np.full((K, K), (1.0 - 0.6) / (K - 1))
        np.fill_diagonal(A, 0.6)
    else:
        A = np.array([
            [tm.probabilities[r_from][r_to] for r_to in REGIMES]
            for r_from in REGIMES
        ])
        # Aggiungi piccolo prior Dirichlet per evitare righe con zero assoluti
        A = A + 1e-3
        A = A / A.sum(axis=1, keepdims=True)

    smoothed = smooth_probabilities_sequence(raw, A)

    points: list[SmoothedPoint] = []
    for i, row in enumerate(rows):
        points.append(SmoothedPoint(
            date=row.date,
            raw={r: float(raw[i, j]) for j, r in enumerate(REGIMES)},
            smoothed={r: float(smoothed[i, j]) for j, r in enumerate(REGIMES)},
        ))

    logger.info(
        f"Smoothed {len(points)} points using transition matrix "
        f"({tm.total_observations} transitions, h={transition_horizon_days}d)"
    )

    return SmoothedHistoryResult(
        points=points,
        transition_horizon_days=transition_horizon_days,
        total_observations=len(points),
    )


def smooth_current(db: Session, window_days: int = 180) -> Optional[dict[str, float]]:
    """Applica lo smoothing alla finestra piu' recente e restituisce il posterior
    dell'ultimo punto (corrente). Util per lo stato live."""
    result = smooth_history(db, days=window_days)
    if not result.points:
        return None
    return result.points[-1].smoothed
