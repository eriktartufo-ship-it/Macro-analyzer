"""Trasformazioni matematiche per indicatori macro: ROC, Z-score, YoY."""

from typing import Optional

import pandas as pd
import numpy as np

# Quanto prima della data bersaglio puo' stare l'osservazione di confronto: copre weekend e
# festivi (giornaliere) e il passo delle settimanali, NON un mese mancante (30 giorni).
TOLLERANZA_GIORNI = 7


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


def calculate_roc_by_date(
    series: pd.Series, months: int, tolerance_days: int = TOLLERANZA_GIORNI
) -> pd.Series:
    """ROC % fra ogni osservazione e quella di `months` mesi prima, cercata per DATA.

    `calculate_roc(periods=12)` conta 12 RIGHE: vale 12 mesi solo se la serie e' mensile e
    senza buchi. Ottobre 2025 manca nelle serie BLS (shutdown) e `fetch_series` toglie i NaN,
    quindi da novembre 2025 a ottobre 2026 il confronto cadeva 13 mesi indietro; su una serie
    trimestrale 12 righe sono 3 anni, su una settimanale 63 righe sono 63 settimane.
    Qui il passato e' l'ultima osservazione entro `tolerance_days` prima della data bersaglio;
    se non c'e' (un mese mancante) il valore e' NaN, mai un altro mese al suo posto.
    """
    s = series.dropna().sort_index()
    if s.empty:
        return s.astype(float)
    bersagli = pd.DataFrame({"t": s.index - pd.DateOffset(months=months)})
    passato = pd.merge_asof(
        bersagli,
        pd.DataFrame({"t": s.index, "v": s.to_numpy(dtype=float)}),
        on="t",
        direction="backward",
        tolerance=pd.Timedelta(days=tolerance_days),
    )["v"].to_numpy()
    return pd.Series((s.to_numpy(dtype=float) / passato - 1) * 100, index=s.index, name=series.name)


def value_at_or_before(
    series: pd.Series, when: pd.Timestamp, tolerance_days: int = TOLLERANZA_GIORNI
) -> Optional[float]:
    """Ultima osservazione entro `tolerance_days` prima di `when` (inclusa), altrimenti None."""
    s = series.dropna()
    s = s[s.index <= when]
    if s.empty or (pd.Timestamp(when) - s.index[-1]) > pd.Timedelta(days=tolerance_days):
        return None
    return float(s.iloc[-1])


def roc_between(
    series: pd.Series, when: pd.Timestamp, months: int, tolerance_days: int = TOLLERANZA_GIORNI
) -> Optional[float]:
    """ROC % fra il valore a `when` e quello di `months` mesi prima, per DATA. None se manca uno dei due."""
    when = pd.Timestamp(when)
    ora = value_at_or_before(series, when, tolerance_days)
    prima = value_at_or_before(series, when - pd.DateOffset(months=months), tolerance_days)
    if ora is None or prima is None or prima == 0:
        return None
    return (ora / prima - 1) * 100


# Finestre degli z-score in OSSERVAZIONI, per frequenza dichiarata in fred_codes.
# ⚠️ Restano posizionali (su una settimanale 252 righe sono ~5 anni): nessuna logica li legge,
# finiscono solo nelle colonne zscore_* di macro_indicators.
_ZSCORE_WINDOWS = {
    "monthly": (12, 36),
    "quarterly": (4, 12),
    "annual": (3, 5),
}
_ZSCORE_WINDOWS_DEFAULT = (252, 756)  # daily, weekly


def standard_transforms(raw: pd.Series, frequency: str) -> dict[str, pd.Series]:
    """Le trasformazioni standard di una serie FRED: ROC a 3/6/12 mesi PER DATA + z-score.

    Unico scrittore: lo usano `FredFetcher.fetch_and_transform` (dal vivo) e
    `AsOfFetcher.fetch_and_transform` (ricostruzione storica), che prima ne avevano due copie.
    Le serie annuali hanno solo la variazione di un anno: va su tutti e tre gli orizzonti,
    come prima.
    """
    result: dict[str, pd.Series] = {"raw": raw}
    if frequency == "annual":
        roc_1y = calculate_roc_by_date(raw, 12)
        result["roc_3m"] = roc_1y
        result["roc_6m"] = roc_1y
        result["roc_12m"] = roc_1y
    else:
        result["roc_3m"] = calculate_roc_by_date(raw, 3)
        result["roc_6m"] = calculate_roc_by_date(raw, 6)
        result["roc_12m"] = calculate_roc_by_date(raw, 12)
    w12, w36 = _ZSCORE_WINDOWS.get(frequency, _ZSCORE_WINDOWS_DEFAULT)
    result["zscore_12m"] = calculate_zscore(raw, window=w12)
    result["zscore_36m"] = calculate_zscore(raw, window=w36)
    return result


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
