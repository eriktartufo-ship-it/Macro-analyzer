"""HMM Market — features feature-disjoint dal rule-based classifier.

Riusa l'infrastruttura Baum-Welch da `hmm_classifier.py` ma con feature set
completamente diverso. Il mapping stati → regimi e' fatto via correlazione
con le posteriori rule-based DEL DB (non sui label hard, sui valori soft).
Questo riduce ulteriormente la tautologia: il modello non vede i label,
solo le distribuzioni che il rule-based produce.

Output: posterior P(regime | last market features), confrontabile col rule-based.

Caching: il fit Baum-Welch e' deterministic + costoso (~3-5s). Cache in memoria
TTL 1h: stesso `n_states` riusa il fit per chiamate successive nella stessa
ora (ensemble + standalone HMM-Market non rifittano due volte).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

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
from app.services.regime.state_mapping import (
    compute_state_to_regime_distribution_corr,
)

_EPS = 1e-12
MIN_TRAINING_OBSERVATIONS = 100

# Cache TTL: il fit Baum-Welch su ~240 mesi market features e' deterministic;
# 1h e' sufficiente per uso interattivo (ensemble + standalone non rifittano).
_CACHE_TTL_SECONDS = 3600
_FIT_CACHE: dict[int, tuple[float, "HMMMarketResult"]] = {}


@dataclass
class HMMMarketResult:
    feature_names: list[str]
    probabilities: dict[str, float]
    current_state: int
    state_to_regime: dict[int, str]    # regime DOMINANTE per stato (argmax distribuzione)
    state_to_regime_distribution: dict[int, dict[str, float]]
    # ↑ distribuzione soft: per ogni stato, P(regime|stato) basata su correlazione
    # con i posterior rule-based (clipped a 0) + smoothing prior alpha=0.05.
    # Risolve il bug pre-fix dove `_map_states_via_soft_correlation` (bigezione
    # hard greedy) etichettava arbitrariamente stati con correlazioni deboli.
    state_centroid_correlation: dict[int, dict[str, float]]
    n_training: int
    log_likelihood: float
    # Regime probability history su ogni mese training (= Σ_s gamma[t,s] · P(regime|s)).
    # Esposto per il meta-classifier Brier-weighted (Tier 1.1 roadmap).
    # Lista di {"date": "YYYY-MM-DD", "regime_probs": {regime: prob}}.
    regime_history: list[dict] = field(default_factory=list)


def _standardize(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def _compute_state_correlation_matrix(
    gamma: np.ndarray,           # (T, n_states) HMM posteriors
    rule_probs: pd.DataFrame,    # (T, 4 regimes) rule-based posteriors aligned
) -> dict[int, dict[str, float]]:
    """Diagnostica: matrice di correlazione raw stato vs regime (no clipping,
    no normalizzazione). Esposta nel result come `state_centroid_correlation`
    per debug/UI. Il mapping vero e' fatto da
    `compute_state_to_regime_distribution_corr` (vedi state_mapping.py)."""
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
    return corr_matrix


def fit_and_predict_hmm_market(db: Session, n_states: int = 4) -> HMMMarketResult:
    """Addestra HMM-Market su features di mercato + mappa via soft correlation.

    Cache: TTL 1h sull'output keyed by n_states. Permette a ensemble +
    standalone di condividere il fit (ridotta latency da ~5s a sub-ms).
    Per forzare refresh: `_FIT_CACHE.clear()`.
    """
    now = time.time()
    cached = _FIT_CACHE.get(n_states)
    if cached and (now - cached[0]) < _CACHE_TTL_SECONDS:
        logger.info(f"HMM-Market cache hit (n_states={n_states}, age={now - cached[0]:.1f}s)")
        return cached[1]

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

    # Soft mapping corr-based: P(regime|stato) ∝ max(corr, 0) + smoothing alpha.
    # Sostituisce la bigezione hard greedy che etichettava arbitrariamente
    # stati con correlazione debole.
    state_to_regime_dist = compute_state_to_regime_distribution_corr(
        gamma_aligned, rule_aligned, alpha=0.05,
    )
    # Matrice raw di correlazione per debug/UI
    state_corr = _compute_state_correlation_matrix(gamma_aligned.values, rule_aligned)
    # Regime "dominante" per stato (argmax) per back-compat label/log
    state_to_regime = {
        s: max(state_to_regime_dist[s], key=state_to_regime_dist[s].get)
        for s in range(n_states)
    }

    # Posterior corrente: marginalizza P(regime) = Σ_s P(stato_s|obs) · P(regime|stato_s)
    last_post = gamma[-1]
    regime_probs = {r: 0.0 for r in REGIMES}
    for s in range(n_states):
        p_s = float(last_post[s])
        for r in REGIMES:
            regime_probs[r] += p_s * state_to_regime_dist[s][r]

    # Anti-saturazione: temperature + floor (mantenuto come post-correzione,
    # con soft mapping il suo effetto e' ridotto perche' la distribuzione
    # arriva gia' meno 1-hot)
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

    # Regime probability history: per ogni mese training,
    # P(regime_t) = Σ_s gamma[t,s] · P(regime|stato_s).
    history: list[dict] = []
    for t in range(gamma.shape[0]):
        regime_probs_t = {r: 0.0 for r in REGIMES}
        for s in range(n_states):
            p_s = float(gamma[t, s])
            for r in REGIMES:
                regime_probs_t[r] += p_s * state_to_regime_dist[s][r]
        history.append({
            "date": df.index[t].date().isoformat(),
            "regime_probs": regime_probs_t,
        })

    result = HMMMarketResult(
        feature_names=feature_names,
        probabilities=regime_probs,
        current_state=int(np.argmax(last_post)),
        state_to_regime=state_to_regime,
        state_to_regime_distribution=state_to_regime_dist,
        state_centroid_correlation=state_corr,
        n_training=int(X_std.shape[0]),
        log_likelihood=float(ll),
        regime_history=history,
    )
    _FIT_CACHE[n_states] = (now, result)
    return result
