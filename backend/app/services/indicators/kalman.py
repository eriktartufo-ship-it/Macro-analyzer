"""Kalman filter 1D scalare per smoothing indicatori macro rumorosi.

Modello state-space univariato:
    x_t = x_{t-1} + w_t        w_t ~ N(0, Q)   (random walk: il "vero" valore evolve lentamente)
    y_t = x_t + v_t            v_t ~ N(0, R)   (osservazione = stato + rumore)

Il filtro stima x_t (livello vero) dalle osservazioni rumorose y_t.

Parametro di tuning:
    lambda = R / Q  =  signal-to-noise ratio inverso
        - lambda alto (es. 30+) → fida poco delle osservazioni, smoothing aggressivo
        - lambda basso (es. 1-5) → segue le osservazioni quasi 1:1
        - lambda 10 default = bilanciato per indicatori macro mensili

Indicatori target per il smoothing:
    - unrate_roc, initial_claims_roc: spike spuri dovuti a stagionalita' / one-off
    - lei_roc: con CFNAI ora robusto, ma residui di rumore mese-su-mese
    - housing_starts_roc_12m: volatile per shock weather/permits

Output: per ogni indicatore, la serie filtered (stesso index della raw) + statistiche
sull'effetto smoothing (riduzione varianza, max delta).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

DEFAULT_LAMBDA = 10.0  # R/Q ratio: 10 = smoothing moderato adatto a macro mensili
DEFAULT_INITIAL_VARIANCE = 1e6  # alta = primo step si fida pienamente dell'osservazione


@dataclass
class KalmanResult:
    raw: pd.Series           # serie originale
    filtered: pd.Series      # forward filter (causal, real-time)
    smoothed: pd.Series      # RTS smoother (uses future data, retrospective)
    lambda_used: float
    variance_reduction: float    # 1 - var(filtered) / var(raw)


def kalman_filter_1d(
    series: pd.Series,
    lam: float = DEFAULT_LAMBDA,
    initial_variance: float = DEFAULT_INITIAL_VARIANCE,
) -> KalmanResult:
    """Applica un Kalman 1D scalare alla serie.

    Returns: KalmanResult con raw, filtered (real-time), smoothed (retrospective),
    e variance reduction stats.
    """
    s = series.dropna().sort_index()
    if len(s) < 5:
        raise ValueError(f"kalman: serie troppo corta ({len(s)} punti)")

    y = s.values.astype(float)
    n = len(y)

    # Q (process noise) e R (observation noise) — solo il loro ratio importa
    Q = 1.0
    R = lam * Q

    # FORWARD FILTER (causale, real-time)
    x_pred = np.zeros(n)        # prior x_{t|t-1}
    P_pred = np.zeros(n)        # prior var
    x_filt = np.zeros(n)        # posterior x_{t|t}
    P_filt = np.zeros(n)        # posterior var

    x_filt[0] = y[0]
    P_filt[0] = initial_variance

    for t in range(1, n):
        # Predict: x_pred = x_{t-1|t-1}, P_pred = P_{t-1|t-1} + Q
        x_pred[t] = x_filt[t - 1]
        P_pred[t] = P_filt[t - 1] + Q
        # Update: K = P_pred / (P_pred + R)
        K = P_pred[t] / (P_pred[t] + R)
        x_filt[t] = x_pred[t] + K * (y[t] - x_pred[t])
        P_filt[t] = (1 - K) * P_pred[t]

    # RTS SMOOTHER (retrospective, usa dati futuri)
    x_smooth = np.zeros(n)
    P_smooth = np.zeros(n)
    x_smooth[-1] = x_filt[-1]
    P_smooth[-1] = P_filt[-1]
    for t in range(n - 2, -1, -1):
        # Smoother gain: A_t = P_filt[t] / P_pred[t+1]
        if P_pred[t + 1] > 0:
            A = P_filt[t] / P_pred[t + 1]
        else:
            A = 0.0
        x_smooth[t] = x_filt[t] + A * (x_smooth[t + 1] - x_pred[t + 1])
        P_smooth[t] = P_filt[t] + A * A * (P_smooth[t + 1] - P_pred[t + 1])

    filtered = pd.Series(x_filt, index=s.index, name=s.name)
    smoothed = pd.Series(x_smooth, index=s.index, name=s.name)

    raw_var = float(np.var(y, ddof=1))
    filt_var = float(np.var(x_filt, ddof=1))
    var_red = 1.0 - filt_var / raw_var if raw_var > 0 else 0.0

    return KalmanResult(
        raw=s,
        filtered=filtered,
        smoothed=smoothed,
        lambda_used=lam,
        variance_reduction=var_red,
    )


# Indicatori notoriamente rumorosi che traggono beneficio dal Kalman
NOISY_INDICATORS = {
    "unrate": "Unemployment rate (livello, smussa step jumps mensili)",
    "initial_claims": "Initial jobless claims (volatile settimanale, smussa per analisi mensile)",
    "lei": "CFNAI 3MA (gia' smussato ma riduce ulteriormente outlier)",
    "housing_starts": "Housing starts (sensibile a meteo/permessi, bias one-off)",
    "consumer_sentiment": "UMich sentiment (rumoroso, indagine campionaria)",
    "ism_manufacturing": "PMI manufacturing (mensile, possibili spike)",
}


def smooth_macro_series(
    series_name: str,
    fred_fetcher=None,
    lam: float = DEFAULT_LAMBDA,
) -> KalmanResult:
    """Helper: pesca dal FRED fetcher e applica il filtro."""
    if fred_fetcher is None:
        from app.services.indicators.fetcher import FredFetcher
        fred_fetcher = FredFetcher()

    series = fred_fetcher.fetch_series(series_name)
    logger.info(f"Kalman smoothing {series_name}: n={len(series)}, lambda={lam}")
    return kalman_filter_1d(series, lam=lam)
