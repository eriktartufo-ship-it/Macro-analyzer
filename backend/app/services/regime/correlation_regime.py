"""Tier 6.5 — Correlation regime detection.

Detect cross-asset correlation breakdown come confirming signal di crisis.

**Filosofia**:

Storicamente, in periodi di stress sistemico, le correlazioni cross-asset
collassano verso +1 ("flight to safety"). Equities, EM, credit, commodities
si muovono tutti negativamente insieme; bonds long e gold positivamente.
Pattern visto in:

- 2008-Q4 (GFC): SPX vs HY credit vs EM vs commodities tutti correlati +0.85.
- 2020-Q1 (COVID): correlazioni avg pairwise spike a ~0.75.
- 2022-Sep (UK gilt + crypto): correlazioni elevate ma più moderate.

In regime normale (diversification benefit), correlazioni pairwise medie
si aggirano 0.10-0.30.

**Math**:

Rolling 60d daily returns su N asset → matrix N×N correlazioni → average
upper-triangular (escluso diagonale) → 1 valore [0, 1].

**Thresholds**:

- avg_corr > 0.7 → `correlation_crisis` (panic flight to safety)
- 0.5 < avg_corr <= 0.7 → `correlation_elevated`
- 0.2 < avg_corr <= 0.5 → `correlation_moderate`
- avg_corr <= 0.2 → `correlation_diversified`

**Asset class target** (minimum 4 per matrix significativa):

- us_equities_growth (SPY proxy)
- us_bonds_long (TLT/IEF)
- gold (GLD)
- broad_commodities (DBC)

Espandibile a EM, credit, USD, ecc.

**Use cases**:

- Sub-regime informativo separato da regime principale 4-quadranti.
- Trigger `correlation_breakdown` in `crisis_indicator` quando regime ==
  "correlation_crisis", boost risk_score.
- Frontend tile dashboard.

NON modifica regime probs direttamente (info layer + crisis_indicator
modulator). Pattern come T5.7 vol_regime.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


# Asset whitelist per correlazione (richiesti almeno 3 in returns DataFrame).
_DEFAULT_ASSETS = (
    "us_equities_growth",
    "us_bonds_long",
    "gold",
    "broad_commodities",
)

# Threshold di classificazione avg pairwise correlation.
_CRISIS_THRESHOLD = 0.70
_ELEVATED_THRESHOLD = 0.50
_DIVERSIFIED_THRESHOLD = 0.20


@dataclass
class CorrelationRegimeAssessment:
    """Snapshot correlation regime + top correlazioni."""
    regime: str                                 # correlation_crisis/elevated/moderate/diversified
    avg_pairwise_correlation: float             # ∈ [-1, +1], tipicamente [0, 1] in stress
    n_assets: int
    n_observations: int                         # giorni di history
    matrix_excerpt: list[tuple[str, str, float]] = field(default_factory=list)
    # Top-3 correlazioni più alte (asset_a, asset_b, corr)
    description: str = ""
    is_breakdown: bool = False                  # True se regime == correlation_crisis


def _classify(avg_corr: float) -> str:
    """Classify avg correlation in 1 dei 4 regimi."""
    if avg_corr > _CRISIS_THRESHOLD:
        return "correlation_crisis"
    if avg_corr > _ELEVATED_THRESHOLD:
        return "correlation_elevated"
    if avg_corr <= _DIVERSIFIED_THRESHOLD:
        return "correlation_diversified"
    return "correlation_moderate"


def _describe(regime: str, avg_corr: float, n_assets: int) -> str:
    """Descrizione IT human-readable."""
    base = {
        "correlation_crisis":
            f"Correlazioni cross-asset breakdown (avg {avg_corr:.2f}): "
            f"flight to safety, diversification fallisce su {n_assets} asset",
        "correlation_elevated":
            f"Correlazioni elevate (avg {avg_corr:.2f}): "
            f"risk-off pattern emergente",
        "correlation_moderate":
            f"Correlazioni moderate (avg {avg_corr:.2f}): "
            f"diversificazione parziale",
        "correlation_diversified":
            f"Correlazioni basse (avg {avg_corr:.2f}): "
            f"diversificazione funzionante (regime normale)",
    }
    return base.get(regime, f"avg correlation {avg_corr:.2f}")


def compute_avg_pairwise_correlation(
    corr_matrix: pd.DataFrame,
) -> float:
    """Calcola media correlazioni upper-triangular (esclusa diagonale).

    Args:
        corr_matrix: N×N pandas DataFrame correlation matrix.

    Returns:
        avg pairwise correlation come float. 0.0 se matrix < 2×2.
    """
    if corr_matrix is None or corr_matrix.empty:
        return 0.0
    n = corr_matrix.shape[0]
    if n < 2 or corr_matrix.shape[1] != n:
        return 0.0
    arr = corr_matrix.values
    # Upper-triangular (k=1 esclude diagonale)
    mask = np.triu_indices_from(arr, k=1)
    pairs = arr[mask]
    pairs = pairs[~np.isnan(pairs)]
    if len(pairs) == 0:
        return 0.0
    return float(np.mean(pairs))


def _top_pairs(corr_matrix: pd.DataFrame, top_n: int = 3) -> list[tuple[str, str, float]]:
    """Top-N coppie con correlation più alta (upper-triangular)."""
    if corr_matrix.empty or corr_matrix.shape[0] < 2:
        return []
    pairs = []
    cols = list(corr_matrix.columns)
    n = len(cols)
    for i in range(n):
        for j in range(i + 1, n):
            v = corr_matrix.iat[i, j]
            if pd.isna(v):
                continue
            pairs.append((cols[i], cols[j], float(v)))
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    return pairs[:top_n]


def compute_correlation_regime(
    returns: pd.DataFrame,
    window_days: int = 60,
    assets: Iterable[str] | None = None,
    min_obs: int = 20,
) -> CorrelationRegimeAssessment:
    """Pipeline: returns DataFrame → correlation matrix → regime classification.

    Args:
        returns: DataFrame daily returns, indexed by date, columns by asset.
            Es. ['us_equities_growth', 'us_bonds_long', 'gold', ...].
        window_days: ampiezza finestra rolling per correlation (default 60).
        assets: subset di colonne da usare (default: tutte presenti).
        min_obs: minimum observations richieste per correlation valida.

    Returns:
        CorrelationRegimeAssessment con regime, avg_corr, descrizione.
    """
    if returns is None or returns.empty:
        return CorrelationRegimeAssessment(
            regime="correlation_moderate",
            avg_pairwise_correlation=0.0,
            n_assets=0,
            n_observations=0,
            description="Nessun dato disponibile",
            is_breakdown=False,
        )

    # Subset assets
    if assets is None:
        cols = list(returns.columns)
    else:
        cols = [a for a in assets if a in returns.columns]
    if len(cols) < 2:
        return CorrelationRegimeAssessment(
            regime="correlation_moderate",
            avg_pairwise_correlation=0.0,
            n_assets=len(cols),
            n_observations=len(returns),
            description=f"Asset insufficienti ({len(cols)}), serve >=2",
            is_breakdown=False,
        )

    # Tail window
    window = returns[cols].tail(window_days).dropna(how="all")
    n_obs = len(window)
    if n_obs < min_obs:
        return CorrelationRegimeAssessment(
            regime="correlation_moderate",
            avg_pairwise_correlation=0.0,
            n_assets=len(cols),
            n_observations=n_obs,
            description=f"Window troppo corta ({n_obs} obs, serve >={min_obs})",
            is_breakdown=False,
        )

    # Correlation matrix + avg pairwise + top pairs
    corr_mat = window.corr()
    avg_corr = compute_avg_pairwise_correlation(corr_mat)
    regime = _classify(avg_corr)
    is_breakdown = regime == "correlation_crisis"
    top = _top_pairs(corr_mat, top_n=3)
    description = _describe(regime, avg_corr, len(cols))

    return CorrelationRegimeAssessment(
        regime=regime,
        avg_pairwise_correlation=round(avg_corr, 4),
        n_assets=len(cols),
        n_observations=n_obs,
        matrix_excerpt=[(a, b, round(c, 3)) for a, b, c in top],
        description=description,
        is_breakdown=is_breakdown,
    )


def returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Helper: converte prezzi DataFrame in daily log-returns.

    Args:
        prices: DataFrame indexed by date, columns by asset.

    Returns:
        DataFrame stesso shape, log(price[t]/price[t-1]) dropna primo row.
    """
    if prices is None or prices.empty:
        return pd.DataFrame()
    return np.log(prices / prices.shift(1)).dropna(how="all")


def list_default_assets() -> list[str]:
    """Restituisce la whitelist default asset per correlation regime."""
    return list(_DEFAULT_ASSETS)
