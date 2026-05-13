"""Tier 5.6 LOOP CLOSURE — Asset returns → regime Bayesian feedback.

Sperimentale. Il mercato ha già "votato" sul regime: performance recente
di gold/bonds/equities su finestra 6m implica P(regime|returns) via
likelihood gaussiana parametrizzata da ASSET_REGIME_DATA.

**Filosofia**:

- Gold +30% YoY + bonds long -15% → conferma stagflation_debasement.
- Equities growth +25% + gold -10% → conferma reflation/goldilocks.
- Tutti gli asset crollano → conferma deflation_crash.

**Math**:

Per ogni regime r e asset a con return osservato R_a (annualizzato):

- Likelihood: `N(R_a; mu=avg_return[r][a], sigma=vol[r][a])`
- log_lik[r] = Σ_a log_pdf(R_a; mu, sigma)
- posterior_unnorm[r] ∝ prior[r] · exp(log_lik[r])
- posterior = softmax/normalize

**Soft blend**: per limitare lookahead/causalità inversa:

`final = (1 - alpha) · prior + alpha · posterior`

con alpha=0.05 default → max ±5pp shift.

**Rischio lookahead**: asset returns possono essere conseguenza del regime,
non causa. Mitigato da window 6m + soft blend + opt-in flag + validation
NBER lead-time.
"""
from __future__ import annotations

import math

from app.services.config_flags import use_asset_feedback
from app.services.scoring.engine import ASSET_REGIME_DATA

_ALPHA_DEFAULT = 0.05
_MIN_VOL = 0.02  # floor su vol per evitare division by zero
_MAX_LOG_LIK = 50.0  # cap per evitare overflow numerico

# Asset class con segnale storicamente forte e meno rumoroso. Esclude
# silver/bitcoin/crypto (alta vol, segnali noisy), tips (mix nominal/real
# poco discriminante), real_estate (illiquido/lagging).
_FEEDBACK_ASSETS = (
    "us_equities_growth",
    "us_equities_value",
    "us_bonds_long",
    "gold",
    "energy",
    "broad_commodities",
)


def _gaussian_log_pdf(x: float, mu: float, sigma: float) -> float:
    """Log-pdf gaussiana, robusta a sigma piccolo."""
    sigma = max(sigma, _MIN_VOL)
    z = (x - mu) / sigma
    return -0.5 * z * z - math.log(sigma * math.sqrt(2 * math.pi))


def compute_likelihoods(
    asset_returns_annualized: dict[str, float],
) -> dict[str, float]:
    """Calcola log-likelihood per ogni regime data la dict di returns.

    Args:
        asset_returns_annualized: dict {asset: return_annualized}. Solo
            chiavi in `_FEEDBACK_ASSETS` vengono considerate (altre ignorate).

    Returns:
        dict {regime: log_likelihood}. Sum across assets per ogni regime.
    """
    regimes = ["reflation", "stagflation", "deflation", "goldilocks"]
    log_lik: dict[str, float] = {r: 0.0 for r in regimes}

    for asset, R in asset_returns_annualized.items():
        if asset not in _FEEDBACK_ASSETS:
            continue
        regime_data = ASSET_REGIME_DATA.get(asset)
        if not regime_data:
            continue
        for r in regimes:
            stats = regime_data.get(r)
            if not stats:
                continue
            mu = float(stats.get("avg_return", 0.0))
            sigma = float(stats.get("vol", _MIN_VOL))
            ll = _gaussian_log_pdf(R, mu, sigma)
            # Cap per stabilità numerica
            log_lik[r] += max(-_MAX_LOG_LIK, min(_MAX_LOG_LIK, ll))

    return log_lik


def _softmax_normalize(log_scores: dict[str, float]) -> dict[str, float]:
    """Stabile numericamente (subtract max prima di exp)."""
    if not log_scores:
        return {}
    max_log = max(log_scores.values())
    raw = {k: math.exp(v - max_log) for k, v in log_scores.items()}
    total = sum(raw.values())
    if total <= 0:
        return {k: 1.0 / len(log_scores) for k in log_scores}
    return {k: v / total for k, v in raw.items()}


def apply_asset_feedback(
    probs: dict[str, float],
    asset_returns_annualized: dict[str, float],
    alpha: float = _ALPHA_DEFAULT,
) -> dict[str, float]:
    """Bayesian update con soft blend.

    Args:
        probs: regime probabilities pre-update.
        asset_returns_annualized: dict {asset: return} con returns
            annualizzati (es. 2 × 6m return).
        alpha: blend factor [0, 1]. 0 = no update (return probs).
            Default 0.05 → max ±5pp shift.

    Returns:
        Posterior probs blendate. Σ=1.0. Identità se flag OFF o asset
        returns vuoti.
    """
    if not use_asset_feedback():
        return dict(probs)
    if not probs or not asset_returns_annualized:
        return dict(probs)
    if alpha <= 0:
        return dict(probs)

    # Log-prior + log-likelihood = log-posterior
    log_lik = compute_likelihoods(asset_returns_annualized)
    # Se nessun asset noto matchava, log_lik tutti 0 → posterior = uniform.
    # Skip update in quel caso.
    if all(v == 0.0 for v in log_lik.values()):
        return dict(probs)

    log_prior = {
        r: math.log(max(p, 1e-9))
        for r, p in probs.items()
    }
    log_post = {
        r: log_prior.get(r, math.log(1e-9)) + log_lik.get(r, 0.0)
        for r in probs.keys()
    }
    posterior = _softmax_normalize(log_post)

    # Soft blend: (1-α)·prior + α·posterior
    blended = {
        r: (1.0 - alpha) * probs.get(r, 0.0) + alpha * posterior.get(r, 0.0)
        for r in probs.keys()
    }
    # Renormalize per drift floating point
    total = sum(blended.values())
    if total <= 0:
        return dict(probs)
    return {r: v / total for r, v in blended.items()}
