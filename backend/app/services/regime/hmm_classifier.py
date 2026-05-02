"""HMM regime classifier (Gaussian, n stati, implementazione pure-numpy).

Approccio:
  1. Estrai feature vector da ogni RegimeClassification storica
     (gdp_roc, cpi_yoy, unrate, pmi, yield_curve_10y2y, lei_roc) e standardizza.
  2. Inizializza i parametri di emissione (mu, sigma, pi) con GaussianMixture
     di scikit-learn (robusto vs random init).
  3. Iterazioni Baum-Welch (EM) per stimare matrice di transizione A e
     raffinare emissioni. Forward/backward in log-space per stabilita numerica.
  4. Mappa ciascuno dei N stati latenti al regime del classifier rule-based
     piu rappresentato tra le sue osservazioni (majority vote).
  5. Per l'ultima osservazione restituisce le probabilita posteriori gamma_T
     aggregate per regime.

Motivazione della scelta implementativa: `hmmlearn` richiede compilatore C
non disponibile su Python 3.14 Windows. Il core EM fit in <200 loc e' sufficiente
per il nostro caso d'uso (sequenze <10^4, n_states <= 8).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from loguru import logger
from scipy.special import logsumexp
from sqlalchemy.orm import Session

from app.models import RegimeClassification
from app.services.regime.classifier import REGIMES
from app.services.regime.state_mapping import (
    compute_state_to_regime_distribution_freq,
)

# Feature core (sempre disponibili storicamente): devono essere presenti in tutte le righe
_CORE_FEATURES = (
    "gdp_roc",
    "cpi_yoy",
    "unrate",
    "unrate_roc",
    "yield_curve_10y2y",
    "lei_roc",
    "indpro_roc_12m",
    "payrolls_roc_12m",
    "core_pce_yoy",
    "baa_spread",
)

# Feature estese (alcune periodi storici non le hanno — es. VIX pre-1990, breakeven pre-2003)
# Se mancano vengono imputate con la media (= 0 dopo standardizzazione).
_EXTENDED_FEATURES = (
    "vix",
    "nfci",
    "breakeven_10y",
    "housing_starts_roc_12m",
)

_FEATURES = _CORE_FEATURES + _EXTENDED_FEATURES

MIN_TRAINING_OBSERVATIONS = 60
_EPS = 1e-12

# Parametri anti-saturazione
_DIRICHLET_ALPHA = 1.0  # pseudo-osservazioni uniformi su A, evita transizioni zero
_POSTERIOR_TEMPERATURE = 1.3  # smoothing finale: > 1 = distribuzione piu' piatta
_POSTERIOR_FLOOR = 0.02  # nessun regime puo' scendere sotto il 2%


@dataclass
class HMMResult:
    regimes: list[str]
    probabilities: dict[str, float]
    current_state: int
    state_to_regime: dict[int, str]    # regime DOMINANTE per stato (argmax distribuzione)
    state_to_regime_distribution: dict[int, dict[str, float]]
    # ↑ distribuzione soft: per ogni stato, P(regime|stato) basata su frequenza
    # empirica dei label rule-based + smoothing prior alpha=0.05.
    # Risolve il bug pre-fix dove `_map_states_to_regimes` (concentration-greedy
    # bigezione) etichettava state 3 (51% refl + 37% gold) come 'goldilocks'
    # solo perche' 'reflation' era stata gia' assegnata a un altro stato.
    n_training: int
    log_likelihood: float
    feature_means: dict[str, float]
    feature_stds: dict[str, float]


def _extract_feature_matrix(
    rows: list[RegimeClassification],
) -> tuple[np.ndarray, list[str], list[int]]:
    """Estrae matrice feature. Scarta la riga solo se mancano CORE features;
    le EXTENDED vengono imputate col valore medio della serie disponibile
    (calcolata al volo su tutte le righe che le contengono).

    Questo permette di usare l'intero storico 1970+ anche dove VIX (post-1990)
    o breakeven (post-2003) non sono disponibili.
    """
    extracted: list[dict] = []
    for row in rows:
        try:
            meta = json.loads(row.conditions_met) if row.conditions_met else {}
        except Exception:
            continue
        ind = meta.get("indicators", {}) or {}
        # Core: tutti presenti o scarta riga
        core_ok = all(ind.get(name) is not None for name in _CORE_FEATURES)
        if not core_ok:
            continue
        extracted.append({"ind": ind, "label": row.regime})

    if not extracted:
        return np.empty((0, len(_FEATURES))), [], []

    # Stima media per ogni extended feature sui dati disponibili
    extended_mean: dict[str, float] = {}
    for name in _EXTENDED_FEATURES:
        vals = [float(e["ind"][name]) for e in extracted if e["ind"].get(name) is not None]
        extended_mean[name] = float(sum(vals) / len(vals)) if vals else 0.0

    X: list[list[float]] = []
    labels: list[str] = []
    for e in extracted:
        vec: list[float] = []
        for name in _CORE_FEATURES:
            vec.append(float(e["ind"][name]))
        for name in _EXTENDED_FEATURES:
            v = e["ind"].get(name)
            vec.append(float(v) if v is not None else extended_mean[name])
        X.append(vec)
        labels.append(e["label"])

    return np.asarray(X, dtype=float), labels, list(range(len(X)))


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd, mu, sd


def _log_gaussian_diag(X: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    """log N(X | mean, diag(var)) per riga. var shape (D,).

    Diagonale anziche' full: gaussiane meno peaked → posterior meno saturato
    e molto piu' robusto a pochi campioni per stato (evita cov full ill-conditioned).
    """
    var = np.maximum(var, 1e-4)  # floor per evitare divisione per zero
    D = X.shape[1]
    diff = X - mean  # (T, D)
    mahal = np.sum((diff ** 2) / var, axis=1)
    log_det = np.sum(np.log(var))
    return -0.5 * (D * np.log(2 * np.pi) + log_det + mahal)


def _init_params(
    X: np.ndarray, n_states: int, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Inizializza pi, A, mu, var via GaussianMixture (diag)."""
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(
        n_components=n_states,
        covariance_type="diag",
        random_state=random_state,
        n_init=3,
        reg_covar=1e-2,
    )
    gmm.fit(X)
    mu = gmm.means_
    var = gmm.covariances_  # (K, D) con covariance_type='diag'
    pi = np.full(n_states, 1.0 / n_states)
    # Transizione uniforme con leggera persistenza
    A = np.full((n_states, n_states), 1.0 / (n_states + 2))
    np.fill_diagonal(A, 3.0 / (n_states + 2))
    A = A / A.sum(axis=1, keepdims=True)
    return pi, A, mu, var


def _forward_backward(
    log_B: np.ndarray, log_pi: np.ndarray, log_A: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Forward-backward in log-space. log_B shape (T, K)."""
    T, K = log_B.shape
    log_alpha = np.full((T, K), -np.inf)
    log_beta = np.full((T, K), -np.inf)

    log_alpha[0] = log_pi + log_B[0]
    for t in range(1, T):
        # log_alpha[t,j] = log_B[t,j] + logsumexp_i(log_alpha[t-1,i] + log_A[i,j])
        log_alpha[t] = log_B[t] + logsumexp(
            log_alpha[t - 1][:, None] + log_A, axis=0
        )

    log_beta[T - 1] = 0.0
    for t in range(T - 2, -1, -1):
        # log_beta[t,i] = logsumexp_j(log_A[i,j] + log_B[t+1,j] + log_beta[t+1,j])
        log_beta[t] = logsumexp(
            log_A + (log_B[t + 1] + log_beta[t + 1])[None, :], axis=1
        )

    ll = float(logsumexp(log_alpha[T - 1]))
    return log_alpha, log_beta, ll


def _baum_welch(
    X: np.ndarray, n_states: int, max_iter: int = 100, tol: float = 1e-3,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit Gaussian HMM via Baum-Welch (diag covariance + Dirichlet prior su A).

    Returns: pi, A, mu, var, gamma, ll.
    - var: shape (K, D), diagonale della covarianza per stato
    - Dirichlet prior con alpha=_DIRICHLET_ALPHA stabilizza A, evita transizioni
      saturate a 1.0 che amplificano il posterior fino a 100%.
    """
    T, D = X.shape
    pi, A, mu, var = _init_params(X, n_states, random_state=random_state)

    prev_ll = -np.inf
    for it in range(max_iter):
        # E-step
        log_B = np.column_stack([
            _log_gaussian_diag(X, mu[k], var[k]) for k in range(n_states)
        ])
        log_pi = np.log(pi + _EPS)
        log_A = np.log(A + _EPS)

        log_alpha, log_beta, ll = _forward_backward(log_B, log_pi, log_A)
        log_gamma = log_alpha + log_beta
        log_gamma = log_gamma - logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)  # (T, K)

        log_xi = (
            log_alpha[:-1, :, None]
            + log_A[None, :, :]
            + log_B[1:, None, :]
            + log_beta[1:, None, :]
        )
        log_xi = log_xi - logsumexp(
            log_xi.reshape(T - 1, -1), axis=1, keepdims=True
        ).reshape(T - 1, 1, 1)
        xi = np.exp(log_xi)  # (T-1, K, K)

        # M-step
        pi = gamma[0] + _EPS
        pi = pi / pi.sum()

        # Transition con Dirichlet prior: aggiunge alpha pseudo-count per cella
        A = xi.sum(axis=0) + _DIRICHLET_ALPHA
        A = A / A.sum(axis=1, keepdims=True)

        N_k = gamma.sum(axis=0) + _EPS
        mu = (gamma.T @ X) / N_k[:, None]  # (K, D)
        # Varianze diagonali ponderate + floor di regolarizzazione
        var = np.zeros((n_states, D))
        for k in range(n_states):
            diff = X - mu[k]
            var[k] = (gamma[:, k : k + 1] * diff ** 2).sum(axis=0) / N_k[k] + 1e-2

        if abs(ll - prev_ll) < tol:
            logger.info(f"Baum-Welch converged at iter {it}, ll={ll:.2f}")
            break
        prev_ll = ll

    return pi, A, mu, var, gamma, ll


def fit_and_predict_hmm(db: Session, n_states: int = 4) -> HMMResult:
    """Addestra il GaussianHMM sulle classificazioni storiche e restituisce
    la distribuzione posteriore per l'ultima osservazione."""
    rows: list[RegimeClassification] = (
        db.query(RegimeClassification)
        .order_by(RegimeClassification.date.asc())
        .all()
    )
    X_raw, labels, _ = _extract_feature_matrix(rows)
    if X_raw.shape[0] < MIN_TRAINING_OBSERVATIONS:
        raise ValueError(
            f"Dati insufficienti per training HMM: {X_raw.shape[0]} < {MIN_TRAINING_OBSERVATIONS}. "
            f"Esegui /regime/backfill/historical prima."
        )

    X_std, mu_raw, sd_raw = _standardize(X_raw)

    try:
        pi, A, mu, var, gamma, ll = _baum_welch(X_std, n_states=n_states)
    except Exception as e:
        raise ValueError(f"Training HMM fallito: {e}") from e

    states = gamma.argmax(axis=1)
    # Soft mapping freq-based: per ogni stato, P(regime|stato) = frequenza
    # empirica dei label rule-based + smoothing prior alpha=0.05.
    # Risolve il bug del concentration-greedy che etichettava uno stato
    # 51% reflation come 'goldilocks' per esclusione.
    state_to_regime_dist = compute_state_to_regime_distribution_freq(
        states, labels, n_states=n_states, alpha=0.05,
    )
    # Regime "dominante" per stato (argmax) per back-compat label/log
    state_to_regime = {
        s: max(state_to_regime_dist[s], key=state_to_regime_dist[s].get)
        for s in range(n_states)
    }

    # Posterior corrente: marginalizza P(regime) = Σ_s P(stato_s|obs) · P(regime|stato_s)
    last_post = gamma[-1]
    regime_probs: dict[str, float] = {r: 0.0 for r in REGIMES}
    for s in range(n_states):
        p_s = float(last_post[s])
        for r in REGIMES:
            regime_probs[r] += p_s * state_to_regime_dist[s][r]

    # Anti-saturazione: temperature smoothing (T > 1 = distribuzione piu' piatta).
    # Mantenuto come post-correzione anti-saturazione (parametro storico tunato);
    # con il soft mapping il suo effetto e' minore perche' la distribuzione
    # arriva gia' meno 1-hot.
    if _POSTERIOR_TEMPERATURE != 1.0:
        log_probs = np.log(np.array([max(regime_probs[r], _EPS) for r in REGIMES]))
        log_probs /= _POSTERIOR_TEMPERATURE
        smoothed_probs = np.exp(log_probs - logsumexp(log_probs))
        regime_probs = {r: float(smoothed_probs[i]) for i, r in enumerate(REGIMES)}

    # Floor su ogni probabilita (evita zero assoluto), poi renormalizza
    regime_probs = {r: max(p, _POSTERIOR_FLOOR) for r, p in regime_probs.items()}
    total = sum(regime_probs.values())
    regime_probs = {r: v / total for r, v in regime_probs.items()}

    logger.info(
        f"HMM trained: n_obs={X_raw.shape[0]} ll={ll:.2f} "
        f"state_map={state_to_regime}"
    )

    return HMMResult(
        regimes=list(REGIMES),
        probabilities=regime_probs,
        current_state=int(states[-1]),
        state_to_regime=state_to_regime,
        state_to_regime_distribution=state_to_regime_dist,
        n_training=int(X_raw.shape[0]),
        log_likelihood=float(ll),
        feature_means={name: float(mu_raw[i]) for i, name in enumerate(_FEATURES)},
        feature_stds={name: float(sd_raw[i]) for i, name in enumerate(_FEATURES)},
    )
