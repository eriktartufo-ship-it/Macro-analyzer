"""HMM Market — features feature-disjoint dal rule-based classifier.

Riusa l'infrastruttura Baum-Welch da `hmm_classifier.py` ma con feature set
completamente diverso. Il mapping stati → regimi e' fatto via correlazione
con le posteriori rule-based DEL DB (non sui label hard, sui valori soft).
Questo riduce ulteriormente la tautologia: il modello non vede i label,
solo le distribuzioni che il rule-based produce.

Output: posterior P(regime | last market features), confrontabile col rule-based.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger
from scipy.special import logsumexp
from sqlalchemy.orm import Session

from app.models import RegimeClassification
from app.services.regime.classifier import REGIMES
from app.services.regime.hmm_classifier import (
    _baum_welch,
    _POSTERIOR_FLOOR,
    _POSTERIOR_TEMPERATURE,
)
from app.services.regime.market_features import compute_market_features

_EPS = 1e-12
MIN_TRAINING_OBSERVATIONS = 100


@dataclass
class HMMMarketResult:
    feature_names: list[str]
    probabilities: dict[str, float]
    current_state: int
    state_to_regime: dict[int, str]
    state_centroid_correlation: dict[int, dict[str, float]]
    n_training: int
    log_likelihood: float


def _standardize(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def _map_states_via_soft_correlation(
    gamma: np.ndarray,           # (T, n_states) HMM posteriors
    rule_probs: pd.DataFrame,    # (T, 4 regimes) rule-based posteriors aligned
) -> tuple[dict[int, str], dict[int, dict[str, float]]]:
    """Mappa ciascuno stato HMM al regime rule-based con cui correla di piu'.

    Per ogni stato s e regime r, calcola corr(gamma[:,s], rule_probs[r]).
    Stato → regime con corr massima, garantendo bigezione (greedy assignment
    sull'ordine di concentrazione massima).
    """
    n_states = gamma.shape[1]
    corr_matrix: dict[int, dict[str, float]] = {}
    for s in range(n_states):
        corr_matrix[s] = {}
        for r in REGIMES:
            x = gamma[:, s]
            y = rule_probs[r].values
            if x.std() == 0 or y.std() == 0:
                corr_matrix[s][r] = 0.0
            else:
                corr_matrix[s][r] = float(np.corrcoef(x, y)[0, 1])

    # Greedy assignment: stati con max-corr piu' alta scelgono per primi
    state_max = sorted(
        range(n_states),
        key=lambda s: max(corr_matrix[s].values()),
        reverse=True,
    )
    mapping: dict[int, str] = {}
    used: set[str] = set()
    for s in state_max:
        sorted_regs = sorted(corr_matrix[s].items(), key=lambda kv: kv[1], reverse=True)
        picked = next((r for r, _ in sorted_regs if r not in used), sorted_regs[0][0])
        mapping[s] = picked
        used.add(picked)
    return mapping, corr_matrix


def fit_and_predict_hmm_market(db: Session, n_states: int = 4) -> HMMMarketResult:
    """Addestra HMM-Market su features di mercato + mappa via soft correlation."""
    df = compute_market_features()
    df = df.dropna()
    if df.shape[0] < MIN_TRAINING_OBSERVATIONS:
        raise ValueError(
            f"HMM-Market: training data insufficiente ({df.shape[0]} < {MIN_TRAINING_OBSERVATIONS})"
        )

    feature_names = list(df.columns)
    X = df.values
    X_std = _standardize(X)

    try:
        pi, A, mu, var, gamma, ll = _baum_welch(X_std, n_states=n_states, max_iter=200)
    except Exception as e:
        raise ValueError(f"HMM-Market training failed: {e}") from e

    # Allinea posteriori rule-based dal DB ai mesi delle nostre features
    rows = (
        db.query(RegimeClassification)
        .order_by(RegimeClassification.date.asc())
        .all()
    )
    if not rows:
        raise ValueError("Nessuna classification in DB per il mapping stati→regimi.")

    rule_df = pd.DataFrame([
        {
            "date": pd.Timestamp(r.date),
            "reflation": r.probability_reflation,
            "stagflation": r.probability_stagflation,
            "deflation": r.probability_deflation,
            "goldilocks": r.probability_goldilocks,
        }
        for r in rows
    ]).set_index("date").sort_index().resample("ME").mean().dropna()

    common = df.index.intersection(rule_df.index)
    if len(common) < 12:
        raise ValueError(
            f"HMM-Market: overlap insufficiente con rule-based ({len(common)} mesi)"
        )

    gamma_aligned = pd.DataFrame(gamma, index=df.index).reindex(common)
    rule_aligned = rule_df.reindex(common)

    state_to_regime, state_corr = _map_states_via_soft_correlation(
        gamma_aligned.values, rule_aligned,
    )

    # Posterior corrente (ultima riga di gamma)
    last_post = gamma[-1]
    regime_probs = {r: 0.0 for r in REGIMES}
    for s, p in enumerate(last_post):
        regime_probs[state_to_regime.get(int(s), REGIMES[0])] += float(p)

    # Anti-saturazione: temperature + floor (riusa parametri di hmm_classifier)
    log_p = np.log(np.array([max(regime_probs[r], _EPS) for r in REGIMES]))
    log_p /= _POSTERIOR_TEMPERATURE
    smoothed = np.exp(log_p - logsumexp(log_p))
    regime_probs = {r: float(smoothed[i]) for i, r in enumerate(REGIMES)}
    regime_probs = {r: max(p, _POSTERIOR_FLOOR) for r, p in regime_probs.items()}
    total = sum(regime_probs.values())
    regime_probs = {r: v / total for r, v in regime_probs.items()}

    logger.info(
        f"HMM-Market trained: n={X_std.shape[0]} ll={ll:.2f} "
        f"state_map={state_to_regime}"
    )

    return HMMMarketResult(
        feature_names=feature_names,
        probabilities=regime_probs,
        current_state=int(np.argmax(last_post)),
        state_to_regime=state_to_regime,
        state_centroid_correlation=state_corr,
        n_training=int(X_std.shape[0]),
        log_likelihood=float(ll),
    )
