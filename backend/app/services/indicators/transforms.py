"""Trasformazioni matematiche per indicatori macro: ROC, Z-score, YoY."""

import pandas as pd
import numpy as np


def calculate_roc(series: pd.Series, periods: int = 1) -> pd.Series:
    """Calcola Rate of Change percentuale.

    ROC = (current - past) / past * 100

    Args:
        series: Serie temporale di valori
        periods: Numero di periodi per il calcolo

    Returns:
        Serie con ROC percentuale (primi N valori = NaN)
    """
    roc = series.pct_change(periods=periods) * 100
    return roc


def calculate_zscore(series: pd.Series, window: int = 12) -> pd.Series:
    """Calcola Z-score rolling.

    Z = (x - rolling_mean) / rolling_std

    Se rolling_std == 0 (serie costante), restituisce 0.

    Args:
        series: Serie temporale
        window: Finestra rolling in periodi

    Returns:
        Serie con z-score (primi window-1 valori = NaN)
    """
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()

    # Evita divisione per zero: se std=0, zscore=0
    zscore = pd.Series(
        np.where(rolling_std == 0, 0.0, (series - rolling_mean) / rolling_std),
        index=series.index,
    )

    # Mantieni NaN dove non c'e' abbastanza storia
    zscore.iloc[: window - 1] = np.nan

    return zscore


def calculate_yoy(series: pd.Series, periods: int = 12) -> pd.Series:
    """Calcola Year-over-Year percentuale.

    Identico a ROC con periods=12 per dati mensili.

    Args:
        series: Serie temporale (tipicamente mensile)
        periods: Periodi per YoY (12 per mensile, 4 per trimestrale)

    Returns:
        Serie con YoY percentuale
    """
    return calculate_roc(series, periods=periods)


def calculate_moving_average(series: pd.Series, window: int = 12) -> pd.Series:
    """Media mobile semplice.

    Args:
        series: Serie temporale
        window: Finestra

    Returns:
        Serie con media mobile
    """
    return series.rolling(window=window).mean()
