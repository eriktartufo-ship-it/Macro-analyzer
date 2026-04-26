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
    state_to_regime: dict[int, str]    # mapping stati → regimi (via correlazione)
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


def _map_states_via_correlation(
    state_probs: pd.DataFrame,  # (T, n_states)
    rule_df: pd.DataFrame,      # (T, 4 regimes)
) -> dict[int, str]:
    """Mappa ogni stato MS al regime con cui correla di piu', greedy bigezione."""
    n_states = state_probs.shape[1]
    common = state_probs.index.intersection(rule_df.index)
    if len(common) < 12:
        # Fallback: mapping per posizione su 2 stati
        if n_states == 2:
            return {0: "reflation", 1: "deflation"}
        return {i: REGIMES[i % len(REGIMES)] for i in range(n_states)}

    sp = state_probs.loc[common]
    rl = rule_df.loc[common]

    corr: dict[int, dict[str, float]] = {}
    for s in range(n_states):
        corr[s] = {}
        for r in REGIMES:
            x, y = sp.iloc[:, s].values, rl[r].values
            if x.std() == 0 or y.std() == 0:
                corr[s][r] = 0.0
            else:
                corr[s][r] = float(np.corrcoef(x, y)[0, 1])

    # Greedy: stato con corr-max piu' alta sceglie per primo
    state_max = sorted(range(n_states), key=lambda s: max(corr[s].values()), reverse=True)
    mapping: dict[int, str] = {}
    used: set[str] = set()
    for s in state_max:
        for r, _ in sorted(corr[s].items(), key=lambda kv: kv[1], reverse=True):
            if r not in used:
                mapping[s] = r
                used.add(r)
                break
        if s not in mapping:
            mapping[s] = next(iter(REGIMES))
    return mapping


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
        state_to_regime = _map_states_via_correlation(smoothed, rule_df)
    else:
        state_to_regime = {i: REGIMES[i % len(REGIMES)] for i in range(n_states)}

    # Posterior corrente
    last = smoothed.iloc[-1]
    regime_probs = {r: 0.0 for r in REGIMES}
    for s in range(n_states):
        regime_probs[state_to_regime.get(s, REGIMES[0])] += float(last.iloc[s])

    # Floor minimo per evitare 0.0 assoluto sui regimi non assegnati
    floor = 0.05
    regime_probs = {r: max(p, floor) for r, p in regime_probs.items()}
    total = sum(regime_probs.values())
    regime_probs = {r: v / total for r, v in regime_probs.items()}

    logger.info(
        f"MS-VAR trained on {ticker}: n={len(returns)} ll={fit.llf:.2f} "
        f"means={ {k: round(v, 2) for k, v in state_means.items()} } "
        f"vols={ {k: round(v, 2) for k, v in state_vols.items()} } "
        f"map={state_to_regime}"
    )

    return MSVARResult(
        n_states=n_states,
        state_means=state_means,
        state_vols=state_vols,
        state_to_regime=state_to_regime,
        probabilities=regime_probs,
        current_state=int(last.values.argmax()),
        n_training=len(returns),
        log_likelihood=float(fit.llf),
    )
