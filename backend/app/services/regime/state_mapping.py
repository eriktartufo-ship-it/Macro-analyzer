"""Soft mapping HMM/MS-VAR state -> regime distribution.

Risolve il bug strutturale del mapping hard 1-a-1 (bigezione greedy):
con n_states diverso da n_regimes, alcuni regimi finivano "ciechi"
(prob solo da floor) o etichettati arbitrariamente con correlazioni deboli.

Due varianti, stessa filosofia (clip + smoothing prior alpha):

1. **corr-based**: per ogni stato calcola correlazione vs posterior
   rule-based di ogni regime, clippa negative a 0, normalizza.
   Usata da HMM-Market e MS-VAR.

2. **freq-based**: per ogni stato calcola frequenza empirica dei label
   rule-based osservati in quello stato. Usata da HMM macro.

`alpha` (default 0.05) garantisce che ogni regime riceva almeno
alpha/n_regimes ≈ 1.25% di massa, evitando zero categorici.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.regime.classifier import REGIMES


def compute_state_to_regime_distribution_corr(
    state_probs: pd.DataFrame,  # (T, n_states)
    rule_df: pd.DataFrame,      # (T, 4 regimes)
    alpha: float = 0.05,
) -> dict[int, dict[str, float]]:
    """Per ogni stato, distribuzione P(regime | stato) basata su correlazione
    storica con i posterior rule-based.

    Logica:
    - Per ogni (stato s, regime r) calcola corr(P(stato_s|t), P(regime_r|t)).
    - Clip a 0 le correlazioni negative (no peso negativo).
    - Normalizza i pesi positivi e mescola con prior uniforme alpha/n_regimes.

    Fallback:
    - <12 osservazioni comuni: uniforme (no fit affidabile).
    - Stato con tutte correlazioni a 0 o NaN: uniforme.
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
                weights[r] = max(c, 0.0)

        total = sum(weights.values())
        if total <= 0:
            out[s] = dict(uniform)
        else:
            out[s] = {
                r: (1.0 - alpha) * (w / total) + uniform_w
                for r, w in weights.items()
            }
    return out


def compute_state_to_regime_distribution_freq(
    states: np.ndarray,         # (T,) state assignment (gamma.argmax)
    labels: list[str],          # len T, rule-based labels per ogni obs
    n_states: int,
    alpha: float = 0.05,
) -> dict[int, dict[str, float]]:
    """Per ogni stato, distribuzione P(regime | stato) basata su frequenza
    empirica dei label rule-based osservati in quello stato.

    Logica:
    - Per ogni stato s, conta quante volte ogni label appare nelle obs
      assegnate a s.
    - Empirical frequency = count[r] / total.
    - Smoothing: out[s][r] = (1 - alpha) * empirical + alpha/n_regimes.

    Fallback: stato senza osservazioni -> distribuzione uniforme.

    Risolve il bug HMM macro pre-fix: `_map_states_to_regimes` usava
    concentration-greedy con bigezione hard, etichettando uno stato 51%
    reflation come 'goldilocks' solo perché 'reflation' era già stata
    presa da un altro stato. Col freq-based, lo stato mantiene la sua
    distribuzione vera 51% refl + 37% gold + ...
    """
    uniform = {r: 1.0 / len(REGIMES) for r in REGIMES}
    uniform_w = alpha / len(REGIMES)
    out: dict[int, dict[str, float]] = {}

    for s in range(n_states):
        idx = states == s
        if idx.sum() == 0:
            out[s] = dict(uniform)
            continue

        state_labels = [labels[i] for i in range(len(labels)) if idx[i]]
        total = len(state_labels)
        empirical = {r: state_labels.count(r) / total for r in REGIMES}
        out[s] = {
            r: (1.0 - alpha) * empirical[r] + uniform_w
            for r in REGIMES
        }
    return out
