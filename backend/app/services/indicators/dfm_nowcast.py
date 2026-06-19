"""Dynamic Factor Model nowcast GDP (Tier 2.5 roadmap Bridgewater).

Stima un fattore latente "growth" da indicatori coincidenti FRED, mensili.
Il fattore correla con GDP YoY ma e' aggiornato in tempo reale (ogni nuovo
release di payrolls/retail/industrial production aggiorna il nowcast),
risolvendo il lag trimestrale di 1-2 mesi del classico `gdp_roc`.

Pipeline:
1. Fetch 5 indicatori coincidenti monthly: industrial_production,
   nonfarm_payrolls, retail_sales, consumer_sentiment, housing_starts.
2. Trasforma a YoY (% change vs 12 mesi fa) per stazionarieta'.
3. Standardize z-score.
4. Fit `statsmodels.tsa.statespace.dynamic_factor.DynamicFactor` con
   k_factors=1, factor_order=1.
5. Output: factor smoothed monthly + factor loadings + R² vs GDP YoY.

Approccio "minimal viable": 5 serie, 1 factor. Lo standard hedge fund
(Stock-Watson, Atlanta Fed) usa 30-50 serie e factor multipli — espandibile
in v2 senza rompere l'API.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

# Indicatori coincidenti FRED gia' disponibili nel sistema. Lista calibrata
# sui "Big Four" Conference Board (industrial production + payrolls + retail
# sales + personal income) sostituendo personal income (non in fred_codes)
# con consumer_sentiment + housing_starts come proxy di domanda.
COINCIDENT_INDICATORS = (
    "industrial_production",
    "nonfarm_payrolls",
    "retail_sales",
    "consumer_sentiment",
    "housing_starts",
)

MIN_OBSERVATIONS_FOR_FIT = 24  # 2 anni minimi per training DFM


@dataclass
class FactorModelResult:
    """Output del fit DFM."""
    factor: np.ndarray              # (T,) factor latente smoothed
    loadings: np.ndarray            # (K,) coefficienti per ogni serie
    series_names: list[str]
    n_observations: int
    log_likelihood: float
    converged: bool


@dataclass
class GDPNowcastResult:
    """Output orchestratore alto livello."""
    nowcast_history: list[dict]     # [{date, factor_value, gdp_yoy_implied}]
    latest_nowcast_date: str
    latest_factor_value: float
    latest_gdp_yoy_implied: float | None  # None se OLS mapping non disponibile
    ols_alpha: float | None         # mapping factor → GDP YoY: gdp = α + β·factor
    ols_beta: float | None
    ols_r_squared: float | None     # R² del mapping
    series_used: list[str]
    notes: list[str] = field(default_factory=list)


def _yoy_transform(s: pd.Series) -> pd.Series:
    """Trasforma a % change YoY (12 mesi fa). Primi 12 valori NaN.

    YoY = (val_t - val_{t-12}) / val_{t-12} × 100. Gestisce divisione per
    zero ritornando NaN (no inf).
    """
    if len(s) < 13:
        return pd.Series(dtype=float, index=s.index)
    prev = s.shift(12)
    # Sostituisci 0 con NaN per evitare div by zero
    prev_safe = prev.replace(0, np.nan)
    return ((s - prev_safe) / prev_safe.abs() * 100.0)


def _standardize_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score per colonna. Colonna costante → output 0 (no division)."""
    out = df.copy()
    for col in out.columns:
        mean = out[col].mean()
        std = out[col].std(ddof=0)
        if std == 0 or pd.isna(std):
            out[col] = 0.0
        else:
            out[col] = (out[col] - mean) / std
    return out


def fit_factor_model(
    panel: pd.DataFrame,
    n_factors: int = 1,
    max_iter: int = 50,
) -> FactorModelResult | None:
    """Fit DynamicFactor su un panel mensile standardizzato.

    Args:
        panel: DataFrame (T, K) standardizzato (z-score per colonna).
        n_factors: numero fattori latenti (default 1).
        max_iter: max iterazioni EM/ML.

    Returns:
        `FactorModelResult` con factor smoothed + loadings, oppure None se
        dati insufficienti (< MIN_OBSERVATIONS_FOR_FIT).

    Note: per dati con NaN, statsmodels gestisce automaticamente missing data
    via Kalman filter. Per minimizzare bias, ridurre missingness pre-fit.
    """
    if len(panel) < MIN_OBSERVATIONS_FOR_FIT:
        logger.warning(
            f"DFM fit skipped: only {len(panel)} obs (min {MIN_OBSERVATIONS_FOR_FIT})"
        )
        return None

    from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

    try:
        model = DynamicFactor(
            panel.values,
            k_factors=n_factors,
            factor_order=1,
            error_order=0,
        )
        result = model.fit(maxiter=max_iter, disp=False)
    except Exception as e:
        logger.warning(f"DFM fit failed: {e}")
        return None

    # Extract smoothed factors. statsmodels DynamicFactor stocca come
    # `factors.smoothed` con shape (k_factors, T), non (T, k_factors).
    # Per k_factors=1 prendiamo riga 0 → array (T,).
    smoothed = np.asarray(result.factors.smoothed)
    if smoothed.ndim == 1:
        factor = smoothed
    elif smoothed.shape[0] == n_factors:
        factor = smoothed[0]
    else:
        factor = smoothed[:, 0]

    # Loadings: coefficienti dell'observation equation per ciascuna serie sul factor.
    # Statsmodels stocca matrice di transizione observation come `result.params`
    # numpy array. I primi K valori (uno per serie) sono i loadings sul fattore 1.
    params = np.asarray(result.params)
    K = panel.shape[1]
    if len(params) >= K:
        loadings = params[:K].astype(float)
    else:
        loadings = np.zeros(K, dtype=float)
    loadings_list = loadings.tolist()

    return FactorModelResult(
        factor=factor,
        loadings=np.array(loadings_list),
        series_names=list(panel.columns),
        n_observations=len(panel),
        log_likelihood=float(result.llf),
        converged=bool(getattr(result, "mle_retvals", {}).get("converged", True)),
    )


# ============================================================================
# Orchestratore alto livello: compute_gdp_nowcast
# ============================================================================


def _fetch_coincident_panel(fetcher) -> pd.DataFrame | None:
    """Scarica le 5 serie coincidenti FRED, allinea su month-end, applica YoY."""
    series_dict: dict[str, pd.Series] = {}
    for name in COINCIDENT_INDICATORS:
        try:
            raw = fetcher.fetch_series(name)
        except Exception as e:
            logger.warning(f"DFM: skip {name} ({e})")
            continue
        if raw is None or raw.empty:
            continue
        s = raw.copy()
        s.index = pd.to_datetime(s.index)
        # Allinea su month-end (last available value of month)
        s = s.resample("ME").last().dropna()
        # YoY transform
        yoy = _yoy_transform(s).dropna()
        if not yoy.empty:
            series_dict[name] = yoy

    if not series_dict:
        return None

    panel = pd.DataFrame(series_dict).dropna()
    if panel.empty:
        return None
    return panel


def _ols_factor_to_gdp(
    factor: pd.Series,
    gdp_yoy: pd.Series,
) -> tuple[float, float, float] | None:
    """Linear regression: gdp_yoy ~ α + β·factor (allineate sulle date comuni).

    Returns (alpha, beta, r_squared) oppure None se < 12 obs comuni.
    """
    common = factor.index.intersection(gdp_yoy.index)
    if len(common) < 12:
        return None
    x = factor.loc[common].values
    y = gdp_yoy.loc[common].values
    if x.std() == 0 or y.std() == 0:
        return None
    # OLS chiuso
    n = len(x)
    x_mean, y_mean = x.mean(), y.mean()
    cov_xy = ((x - x_mean) * (y - y_mean)).sum() / n
    var_x = ((x - x_mean) ** 2).sum() / n
    if var_x == 0:
        return None
    beta = cov_xy / var_x
    alpha = y_mean - beta * x_mean
    y_pred = alpha + beta * x
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y_mean) ** 2).sum()
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return float(alpha), float(beta), float(r_squared)


def compute_gdp_nowcast(
    fetcher=None,
    n_factors: int = 1,
) -> GDPNowcastResult | None:
    """Pipeline completa GDP nowcast.

    Steps:
    1. Fetch 5 coincident series + YoY transform.
    2. Standardize panel.
    3. Fit DFM 1-factor.
    4. Fetch GDP storico, calcola YoY, allinea con factor mensile.
    5. OLS: gdp_yoy ~ α + β·factor → mapping.
    6. Output: nowcast history + latest value.
    """
    if fetcher is None:
        from app.services.indicators.fetcher import FredFetcher
        fetcher = FredFetcher()

    notes: list[str] = []

    panel = _fetch_coincident_panel(fetcher)
    if panel is None or len(panel) < MIN_OBSERVATIONS_FOR_FIT:
        return None

    std_panel = _standardize_panel(panel)
    fit = fit_factor_model(std_panel, n_factors=n_factors)
    if fit is None:
        return None

    factor_series = pd.Series(fit.factor, index=panel.index, name="factor")
    if not fit.converged:
        notes.append("DFM ML did not converge — output approssimato")

    # Fetch GDP YoY per OLS mapping
    ols_alpha, ols_beta, ols_r2 = None, None, None
    gdp_yoy_implied: float | None = None
    try:
        # AUDIT 2026-06-19 bughunter #2: era fetch_series("gdp") = PIL NOMINALE
        # (FRED GDP, $B SAAR) → YoY ~6% contaminato dall'inflazione. Il classifier
        # interpretava gdp_yoy_dfm come "crescita reale" → segnale sigmoid spurio
        # positivo verso Goldilocks in stagflation. Fix: usa real_gdp (GDPC1).
        gdp_raw = fetcher.fetch_series("real_gdp")
        gdp_raw.index = pd.to_datetime(gdp_raw.index)
        gdp_quarterly = gdp_raw.resample("ME").last().ffill()
        gdp_yoy = _yoy_transform(gdp_quarterly).dropna()
        ols = _ols_factor_to_gdp(factor_series, gdp_yoy)
        if ols is not None:
            ols_alpha, ols_beta, ols_r2 = ols
            gdp_yoy_implied_raw = ols_alpha + ols_beta * float(factor_series.iloc[-1])
            # T9-AUDIT-BUG-10 (rev2, bughunter HIGH-3): anchor a gdp_roc × 4
            # con discrepancy gate. Cap fisso [-8,+8] era troppo largo.
            # Nuova logica:
            # - Stima YoY proxy = gdp_roc (QoQ) × 4 (semplice annualization).
            # - Se |DFM raw - proxy| > 2.5pp → DFM è inaffidabile.
            #   In tal caso usa proxy (gdp_roc × 4) come fallback.
            # - Altrimenti cap a [proxy - 2, proxy + 2] per smoothing.
            if gdp_yoy_implied_raw is not None:
                # AUDIT 2026-06-13 bughunter: era `last_gdp_qoq = gdp_quarterly.iloc[-1]`
                # ma gdp_quarterly è PIL NOMINALE in $B (~29000), non QoQ %. Il calcolo
                # `proxy = last × 4 = 116000` produceva discrepancy enorme → cap a 5.0
                # fisso, semanticamente invertito. Fix: usa gdp_yoy.iloc[-1] (già YoY%
                # calcolato a riga 272), che è il proxy semanticamente corretto.
                try:
                    proxy_yoy = float(gdp_yoy.iloc[-1]) if not gdp_yoy.empty else None
                except Exception:
                    proxy_yoy = None
                if proxy_yoy is not None:
                    discrepancy = abs(gdp_yoy_implied_raw - proxy_yoy)
                    if discrepancy > 2.5:
                        gdp_yoy_implied = max(-5.0, min(5.0, proxy_yoy))
                        notes.append(
                            f"gdp_yoy_dfm raw={gdp_yoy_implied_raw:.2f} diverge "
                            f"{discrepancy:.1f}pp da proxy={proxy_yoy:.2f}, "
                            f"FALLBACK to proxy={gdp_yoy_implied:.2f}"
                        )
                    else:
                        gdp_yoy_implied = max(
                            proxy_yoy - 2.0,
                            min(proxy_yoy + 2.0, gdp_yoy_implied_raw),
                        )
                else:
                    # Fallback al vecchio cap range se gdp_yoy non disponibile
                    gdp_yoy_implied = max(-5.0, min(5.0, gdp_yoy_implied_raw))
            else:
                gdp_yoy_implied = None
        else:
            notes.append("OLS factor → GDP YoY skipped (overlap < 12 mesi)")
    except Exception as e:
        notes.append(f"GDP fetch fallito: {e}")
        logger.warning(f"DFM: GDP YoY fetch failed: {e}")

    # Output history
    history: list[dict] = []
    for ts, val in factor_series.items():
        implied = None
        if ols_alpha is not None and ols_beta is not None:
            implied = float(ols_alpha + ols_beta * float(val))
        history.append({
            "date": ts.date().isoformat(),
            "factor_value": float(val),
            "gdp_yoy_implied": implied,
        })

    last_date = factor_series.index[-1]
    last_factor = float(factor_series.iloc[-1])

    return GDPNowcastResult(
        nowcast_history=history,
        latest_nowcast_date=last_date.date().isoformat(),
        latest_factor_value=last_factor,
        latest_gdp_yoy_implied=gdp_yoy_implied,
        ols_alpha=ols_alpha,
        ols_beta=ols_beta,
        ols_r_squared=ols_r2,
        series_used=list(panel.columns),
        notes=notes,
    )
