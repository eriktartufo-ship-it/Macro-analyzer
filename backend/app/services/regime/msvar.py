"""Markov-Switching Regression univariata su S&P 500 returns (Hamilton 1989).

A differenza dell'HMM rule-based-tethered, qui usiamo statsmodels
`MarkovRegression` su una singola serie di mercato (S&P 500 monthly returns)
con 2 stati: `bull` (mu>0, low vol) vs `bear` (mu<0, high vol).

Nessun feature macro, solo prezzi storici. E' la baseline classica delle banche
per regime detection puro market-based, completamente indipendente sia dal
rule-based che dall'HMM-Market.

Output: posterior P(stato_t | observations) per ogni mese, rimappato sui 4
regimi tradizionali via correlazione con le posteriori rule-based:
  - bull state typically aligns con reflation/goldilocks
  - bear state typically aligns con stagflation/deflation
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy.orm import Session

from app.models import RegimeClassification
from app.services.prices.yahoo_fetcher import YahooFetcher
from app.services.regime.classifier import REGIMES


@dataclass
class MSVARResult:
    n_states: int
    state_means: dict[int, float]      # mean return per stato
    state_vols: dict[int, float]       # vol return per stato
    state_to_regime: dict[int, str]    # regime DOMINANTE per stato (argmax distribuzione)
    state_to_regime_distribution: dict[int, dict[str, float]]
    # ↑ distribuzione soft completa: per ogni stato, P(regime|stato) sui 4 regimi.
    # Risolve il bug pre-fix dove con n_states=2 due regimi su 4 erano ciechi.
    probabilities: dict[str, float]    # posterior corrente per regime
    current_state: int
    n_training: int
    log_likelihood: float


def _fit_markov(returns: pd.Series, n_states: int = 2):
    """Fit MarkovRegression con n_states e variance switching."""
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    # `switching_variance=True` → variance e' regime-dependent (high/low vol).
    model = MarkovRegression(
        returns,
        k_regimes=n_states,
        trend="c",  # constant
        switching_variance=True,
    )
    return model.fit(disp=False, maxiter=200)


def _compute_state_to_regime_distribution(
    state_probs: pd.DataFrame,  # (T, n_states)
    rule_df: pd.DataFrame,      # (T, 4 regimes)
    alpha: float = 0.05,
) -> dict[int, dict[str, float]]:
    """Per ogni stato MS-VAR ritorna P(regime | stato) come distribuzione completa
    sui 4 regimi, informata dalla correlazione storica con i posterior rule-based.

    Logica:
    - Per ogni (stato s, regime r) calcola corr(P(stato_s|t), P(regime_r|t)).
    - Clip a 0 le correlazioni negative (un regime anti-correlato non riceve massa
      dalla quota informata, solo lo smoothing prior).
    - Normalizza i pesi positivi e mescola con prior uniforme alpha/n_regimes.

    `alpha` (default 0.05) e' lo smoothing prior: garantisce che ogni regime
    riceva almeno alpha/n_regimes ≈ 1.25% di massa, evitando zero categorici e
    risolvendo il bug pre-fix (con n_states=2 due regimi su 4 erano ciechi).

    Fallback:
    - <12 osservazioni comuni: distribuzione uniforme (no fit affidabile).
    - Stato con tutte le correlazioni a 0 o NaN: distribuzione uniforme.
    """
    n_states = state_probs.shape[1]
    common = state_probs.index.intersection(rule_df.index)
    uniform = {r: 1.0 / len(REGIMES) for r in REGIMES}

    if len(common) < 12:
        return {s: dict(uniform) for s in range(n_states)}

    sp = state_probs.loc[common]
    rl = rule_df.loc[common]
    uniform_w = alpha / len(REGIMES)

    out: dict[int, dict[str, float]] = {}
    for s in range(n_states):
        weights: dict[str, float] = {}
        for r in REGIMES:
            x, y = sp.iloc[:, s].values, rl[r].values
            if x.std() == 0 or y.std() == 0:
                weights[r] = 0.0
            else:
                c = float(np.corrcoef(x, y)[0, 1])
                weights[r] = max(c, 0.0)  # clip negative correlations

        total = sum(weights.values())
        if total <= 0:
            out[s] = dict(uniform)
        else:
            out[s] = {
                r: (1.0 - alpha) * (w / total) + uniform_w
                for r, w in weights.items()
            }
    return out


def fit_and_predict_msvar(
    db: Session,
    n_states: int = 2,
    ticker: str = "SPY",
) -> MSVARResult:
    """Addestra MS Regression su S&P returns mensili, mappa stati via correlazione."""
    yahoo = YahooFetcher()
    px = yahoo.fetch(ticker)
    px_m = px.copy()
    px_m.index = pd.to_datetime(px_m.index)
    px_m = px_m.resample("ME").last().dropna()
    returns = px_m.pct_change().dropna() * 100  # percent monthly returns

    if len(returns) < 60:
        raise ValueError(f"MS-VAR: training data insufficiente ({len(returns)} mesi)")

    fit = _fit_markov(returns, n_states=n_states)

    # Posterior smoothed per ogni stato e mese
    smoothed = pd.DataFrame(
        fit.smoothed_marginal_probabilities.values,
        index=returns.index,
        columns=[f"state_{i}" for i in range(n_states)],
    )

    # Caratterizza stati (mean, vol)
    state_means = {i: float(fit.params[f"const[{i}]"]) for i in range(n_states)}
    state_vols = {
        i: float(np.sqrt(fit.params[f"sigma2[{i}]"])) for i in range(n_states)
    }

    # Carica rule-based posteriors per il mapping
    rows = (
        db.query(RegimeClassification)
        .order_by(RegimeClassification.date.asc())
        .all()
    )
    if rows:
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
        state_to_regime_dist = _compute_state_to_regime_distribution(smoothed, rule_df)
    else:
        # Nessun rule-based in DB: distribuzione uniforme per ogni stato
        state_to_regime_dist = {
            s: {r: 1.0 / len(REGIMES) for r in REGIMES} for s in range(n_states)
        }

    # Posterior corrente: marginalizza P(regime) = Σ_s P(stato_s|obs) · P(regime|stato_s)
    last = smoothed.iloc[-1]
    regime_probs = {r: 0.0 for r in REGIMES}
    for s in range(n_states):
        p_s = float(last.iloc[s])
        for r in REGIMES:
            regime_probs[r] += p_s * state_to_regime_dist[s][r]

    # Niente floor: la distribuzione e' gia' coperta dallo smoothing prior alpha
    total = sum(regime_probs.values())
    if total > 0:
        regime_probs = {r: v / total for r, v in regime_probs.items()}

    # Regime "dominante" per stato (argmax distribuzione) — back-compat per UI
    state_to_regime = {
        s: max(state_to_regime_dist[s], key=state_to_regime_dist[s].get)
        for s in range(n_states)
    }

    logger.info(
        f"MS-VAR trained on {ticker}: n={len(returns)} ll={fit.llf:.2f} "
        f"means={ {k: round(v, 2) for k, v in state_means.items()} } "
        f"vols={ {k: round(v, 2) for k, v in state_vols.items()} } "
        f"dominant_map={state_to_regime}"
    )

    return MSVARResult(
        n_states=n_states,
        state_means=state_means,
        state_vols=state_vols,
        state_to_regime=state_to_regime,
        state_to_regime_distribution=state_to_regime_dist,
        probabilities=regime_probs,
        current_state=int(last.values.argmax()),
        n_training=len(returns),
        log_likelihood=float(fit.llf),
    )
