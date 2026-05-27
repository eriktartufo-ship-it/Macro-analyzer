"""T9-AUDIT — Bootstrap confidence intervals 95% per metriche OOS.

Insight raydalio: "N=9 OOS. Wilson 95% CI su 7/9 = [0.45, 0.94]. Statisticamente
potresti essere al 45% di accuracy reale. Insignificante."

Soluzione: bootstrap resampling N=1000 per stimare CI 95% di ogni metrica.
Gate strategico: nessun claim "il modello fa X%" senza CI.

Metodologia:
1. Estrai i risultati per-episode (correct/wrong, top-5 overlap, ecc).
2. Resample N=1000 con replacement dal sample originale.
3. Per ogni resample: ricalcola la metrica.
4. CI 95% = percentile 2.5% e 97.5% della distribuzione bootstrap.

Output: dict {metric_name: {point_estimate, ci_lower, ci_upper, n_samples}}.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass
class BootstrapResult:
    """Singola metrica con CI 95%."""
    metric_name: str
    point_estimate: float
    ci_lower: float  # 2.5 percentile
    ci_upper: float  # 97.5 percentile
    n_samples: int
    n_bootstrap: int

    def is_significant(self, baseline: float = 0.0) -> bool:
        """Significativo se baseline è fuori dal CI 95%."""
        return baseline < self.ci_lower or baseline > self.ci_upper

    def summary(self) -> str:
        """Formato leggibile: 'accuracy 0.78 [0.45, 0.94]'."""
        return (
            f"{self.metric_name} = {self.point_estimate:.3f} "
            f"[{self.ci_lower:.3f}, {self.ci_upper:.3f}] "
            f"(N={self.n_samples}, n_bootstrap={self.n_bootstrap})"
        )


def bootstrap_metric(
    per_sample_values: Sequence[float],
    aggregator: Callable[[Sequence[float]], float] = np.mean,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
    metric_name: str = "metric",
) -> BootstrapResult:
    """Bootstrap CI per una metrica aggregata.

    Args:
        per_sample_values: lista di valori per-sample (es. [1,0,1,1,0,...] per
            correct/wrong oppure [2,3,3,2,1,...] per top-5 overlap).
        aggregator: funzione di aggregazione (default np.mean).
        n_bootstrap: N resample (default 1000, standard).
        confidence: livello CI (default 0.95).
        random_state: seed riproducibilità.
        metric_name: etichetta per output.

    Returns:
        BootstrapResult con point estimate + CI.
    """
    values = np.asarray(per_sample_values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return BootstrapResult(
            metric_name=metric_name, point_estimate=0.0,
            ci_lower=0.0, ci_upper=0.0,
            n_samples=0, n_bootstrap=n_bootstrap,
        )

    rng = np.random.default_rng(random_state)
    bootstrap_estimates = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        resample = values[sample_idx]
        bootstrap_estimates[i] = aggregator(resample)

    alpha = 1.0 - confidence
    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - alpha / 2.0) * 100.0
    ci_lower = float(np.percentile(bootstrap_estimates, lower_pct))
    ci_upper = float(np.percentile(bootstrap_estimates, upper_pct))
    point = float(aggregator(values))
    return BootstrapResult(
        metric_name=metric_name,
        point_estimate=point,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_samples=n,
        n_bootstrap=n_bootstrap,
    )


def block_bootstrap_metric(
    per_sample_values: Sequence[float],
    aggregator: Callable[[Sequence[float]], float] = np.mean,
    block_size: int = 12,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
    metric_name: str = "metric",
) -> BootstrapResult:
    """Q1.5 council: Block-bootstrap CI per time series con autocorrelazione.

    Standard bootstrap assume iid samples. Monthly time series violano questa
    assunzione (autocorrelazione persistente). Block-bootstrap resample BLOCCHI
    consecutivi di lunghezza `block_size`, preservando struttura within-block.

    Args:
        per_sample_values: serie temporale (es. monthly returns).
        aggregator: funzione metrica (default np.mean; per Sortino usare
            statistic come `lambda x: mean(x) / std(x[x<0]) * sqrt(12)`).
        block_size: lunghezza blocchi consecutivi (default 12 mesi = 1 anno).
        n_bootstrap: N resample (default 1000).
        confidence: livello CI (default 0.95).
        random_state: seed.
        metric_name: etichetta.

    Returns:
        BootstrapResult con point estimate + CI block-bootstrap.

    Reference:
        Künsch (1989), "The Jackknife and the Bootstrap for General Stationary
        Observations". Standard methodology per macroeconomic time series.
    """
    values = np.asarray(per_sample_values, dtype=np.float64)
    n = len(values)
    if n == 0 or n < block_size:
        return BootstrapResult(
            metric_name=metric_name, point_estimate=0.0,
            ci_lower=0.0, ci_upper=0.0,
            n_samples=n, n_bootstrap=n_bootstrap,
        )

    rng = np.random.default_rng(random_state)
    n_blocks_needed = (n + block_size - 1) // block_size  # ceil
    max_start = n - block_size
    bootstrap_estimates = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        block_starts = rng.integers(0, max_start + 1, size=n_blocks_needed)
        resample = np.concatenate([values[s:s + block_size] for s in block_starts])[:n]
        bootstrap_estimates[i] = aggregator(resample)

    alpha = 1.0 - confidence
    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - alpha / 2.0) * 100.0
    ci_lower = float(np.percentile(bootstrap_estimates, lower_pct))
    ci_upper = float(np.percentile(bootstrap_estimates, upper_pct))
    point = float(aggregator(values))
    return BootstrapResult(
        metric_name=metric_name,
        point_estimate=point,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_samples=n,
        n_bootstrap=n_bootstrap,
    )


def block_bootstrap_sortino_ci(
    monthly_returns: Sequence[float],
    block_size: int = 12,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> BootstrapResult:
    """Helper: Sortino annualizzato + CI 95% via block-bootstrap.

    Sortino = mean / downside_std * sqrt(12), dove downside_std considera solo
    returns < 0. Block-bootstrap su 12 mesi → CI robusto a autocorrelazione.
    """

    def sortino(arr: Sequence[float]) -> float:
        a = np.asarray(arr, dtype=np.float64)
        downside = a[a < 0]
        if len(downside) < 2:
            return 0.0
        dstd = downside.std(ddof=1)
        if dstd < 1e-9:
            return 0.0
        return float(a.mean() / dstd * np.sqrt(12))

    return block_bootstrap_metric(
        monthly_returns,
        aggregator=sortino,
        block_size=block_size,
        n_bootstrap=n_bootstrap,
        confidence=0.95,
        random_state=random_state,
        metric_name="sortino_annualized",
    )


def bootstrap_classification_metrics(
    correct_flags: Sequence[bool],
    top_n_overlaps: Sequence[int],
    top_n_max: int = 5,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> dict[str, BootstrapResult]:
    """Bootstrap suite metrics per regime classifier + asset prediction.

    Args:
        correct_flags: lista [True, False, True, ...] per regime correct OOS.
        top_n_overlaps: lista [3, 2, 4, ...] per top-N asset overlap.
        top_n_max: max possible overlap (default 5 per top-5).

    Returns:
        Dict con:
        - regime_accuracy
        - top_n_overlap_mean (normalized 0-1)
        - top_n_overlap_avg_count (absolute count out of top_n_max)
        - top_n_>=3 (frazione episodi con overlap >= 3)
    """
    correct_int = [1 if c else 0 for c in correct_flags]
    overlap_int = list(top_n_overlaps)

    out: dict[str, BootstrapResult] = {}
    out["regime_accuracy"] = bootstrap_metric(
        correct_int, aggregator=np.mean,
        n_bootstrap=n_bootstrap, random_state=random_state,
        metric_name="regime_accuracy",
    )
    out["top_n_overlap_avg"] = bootstrap_metric(
        overlap_int, aggregator=np.mean,
        n_bootstrap=n_bootstrap, random_state=random_state,
        metric_name="top_n_overlap_avg",
    )
    # Normalized 0-1
    overlap_norm = [v / top_n_max for v in overlap_int]
    out["top_n_overlap_pct"] = bootstrap_metric(
        overlap_norm, aggregator=np.mean,
        n_bootstrap=n_bootstrap, random_state=random_state,
        metric_name="top_n_overlap_pct",
    )
    # Frazione episodi con >=3/5
    high_overlap = [1 if v >= 3 else 0 for v in overlap_int]
    out["top_n_high_overlap_rate"] = bootstrap_metric(
        high_overlap, aggregator=np.mean,
        n_bootstrap=n_bootstrap, random_state=random_state,
        metric_name="top_n_high_overlap_rate",
    )
    return out
