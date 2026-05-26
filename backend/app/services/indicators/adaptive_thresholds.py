"""Tier 6.1 — Adaptive thresholds (z-score rolling 10y).

Sostituisce threshold hardcoded del classifier con quantili rolling 10y.
"VIX > 30" hardcoded diventa "VIX > quantile_95 ultimi 10y" → adattivo
al regime storico recente.

**Filosofia**:

- Quantile-based threshold = robusto a outlier (vs mean+std).
- Window 10y = cattura 1 ciclo macro completo + cushion.
- Solo indicatori "market-derived" (VIX, BAA, NFCI, yield_curve) e
  "rate-based" (CPI YoY). NON anchored a target Fed (fed_funds,
  breakeven, unrate restano hardcoded).

**Math** (per condizione type "high"):

```
center = quantile_75    (zona stress soft)
scale  = quantile_95 - quantile_75    (ampiezza transizione a panic)
sigmoid(x, center, scale) attiva a quantile_75, saturata a quantile_95
```

Per condizione type "low" (mirror):

```
center = quantile_25
scale  = quantile_50 - quantile_25
```

**Recipe registry**: dict che mappa condition_name → (indicator, type).
Helper `compute_adaptive_recipe(...)` ritorna (center, scale) per ogni
condizione target.

**Caching**: i quantili cambiano poco day-by-day → LRU cache in-memory
con key = (series_id, end_date_iso, n_quantiles). TTL 1 giorno via
date-based key.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd


# Recipe: indicator series_name + type (high/low/center)
# - "high": center=q75, scale=q95-q75 → sigmoid attivo sopra q75
# - "low":  center=q25, scale=q50-q25 → sigmoid_neg attivo sotto q25
# - "centered": center=q50 (median), scale=q75-q25 (IQR)
#
# Mapping condition_name → (indicator_key, threshold_type)
# indicator_key = chiave usata nel `indicators` dict del classifier
_CONDITION_RECIPES: dict[str, tuple[str, str]] = {
    # VIX conditions
    "vix_low":               ("vix", "low"),        # bottom 25% = risk-on
    "vix_calm":              ("vix", "low_extreme"),# bottom 10% = goldilocks
    "vix_elevated":          ("vix", "high"),       # top 25% = stress
    "vix_spike":             ("vix", "high_extreme"),# top 5% = panic
    # BAA credit spread
    "credit_spread_tight":   ("baa_spread", "low"),
    "credit_spread_wide":    ("baa_spread", "high"),
    # CPI YoY
    "inflation_rising":      ("cpi_yoy", "high"),
    "inflation_high":        ("cpi_yoy", "high_extreme"),
    "inflation_low":         ("cpi_yoy", "low"),
    # Yield curve 10y2y
    "yield_curve_steep":     ("yield_curve_10y2y", "high"),
    # NFCI Chicago Fed
    "financial_conditions_loose": ("nfci", "low"),
    "financial_conditions_easy":  ("nfci", "low_extreme"),
    "nfci_tight":                  ("nfci", "high"),
}


@dataclass
class AdaptiveRecipe:
    """Sigmoid (center, scale) calcolati da quantili rolling."""
    center: float
    scale: float
    quantile_used: str        # "q75", "q25", "q95", ecc.
    series_name: str
    n_observations: int = 0


def compute_quantiles(
    series: pd.Series,
    quantiles: Iterable[float] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
    end_date: date | None = None,
    window_years: int = 10,
) -> dict[float, float]:
    """Calcola quantili rolling su finestra `window_years` terminante a `end_date`.

    Args:
        series: pandas Series indexed by date.
        quantiles: iterable di percentili da calcolare (0-1).
        end_date: data finale finestra (default: ultima data della serie).
        window_years: ampiezza finestra in anni (default 10).

    Returns:
        dict {q: value} per ogni quantile richiesto. Empty dict se
        serie troppo corta (< window_years/2).
    """
    if series is None or series.empty:
        return {}
    s = series.dropna()
    if s.empty:
        return {}
    if end_date is None:
        end_ts = pd.Timestamp(s.index.max())
    else:
        end_ts = pd.Timestamp(end_date)
    start_ts = end_ts - pd.DateOffset(years=window_years)
    window = s.loc[(s.index >= start_ts) & (s.index <= end_ts)]
    if len(window) < 30:
        # Serie troppo corta per quantili stabili
        return {}
    arr = window.values
    return {float(q): float(np.quantile(arr, q)) for q in quantiles}


def quantiles_to_recipe(
    quantiles: dict[float, float],
    threshold_type: str,
    series_name: str,
    n_obs: int = 0,
) -> AdaptiveRecipe | None:
    """Converte dict quantili in AdaptiveRecipe (center, scale).

    Pattern per threshold_type:
    - "high":          center=q75, scale=q95-q75 (clamp ≥ small_eps)
    - "high_extreme":  center=q95, scale=q99-q95 oppure max(q95-q90, eps)
    - "low":           center=q25, scale=q50-q25 (positivo)
    - "low_extreme":   center=q05, scale=q10-q05
    - "centered":      center=q50, scale=q75-q25

    Per sigmoid "low" classifier usa `_sigmoid(-value, center=-X, scale=Y)`,
    quindi center qui è il valore DOMINIO (positivo), il classifier lo nega.

    Returns: None se quantili insufficienti per il tipo richiesto.
    """
    eps = 1e-3  # scale floor
    q = quantiles
    if not q:
        return None

    if threshold_type == "high":
        if 0.75 not in q or 0.95 not in q:
            return None
        center = q[0.75]
        scale = max(eps, q[0.95] - q[0.75])
        used = "q75 (scale=q95-q75)"
    elif threshold_type == "high_extreme":
        if 0.95 not in q:
            return None
        # Per high_extreme: center=q95. Scale stimato come q95-q90 se disponibile,
        # altrimenti q95-q75 / 4 (cushion arbitrario)
        if 0.90 in q:
            scale = max(eps, q[0.95] - q[0.90])
        elif 0.75 in q:
            scale = max(eps, (q[0.95] - q[0.75]) / 4.0)
        else:
            return None
        center = q[0.95]
        used = "q95 (scale=q95-q90)"
    elif threshold_type == "low":
        if 0.25 not in q or 0.50 not in q:
            return None
        center = q[0.25]
        scale = max(eps, q[0.50] - q[0.25])
        used = "q25 (scale=q50-q25)"
    elif threshold_type == "low_extreme":
        if 0.05 not in q or 0.10 not in q:
            return None
        center = q[0.05]
        scale = max(eps, q[0.10] - q[0.05])
        used = "q05 (scale=q10-q05)"
    elif threshold_type == "centered":
        if 0.50 not in q or 0.25 not in q or 0.75 not in q:
            return None
        center = q[0.50]
        scale = max(eps, q[0.75] - q[0.25])
        used = "q50 (scale=IQR)"
    else:
        return None

    return AdaptiveRecipe(
        center=center,
        scale=scale,
        quantile_used=used,
        series_name=series_name,
        n_observations=n_obs,
    )


def get_recipe_for_condition(
    condition_name: str,
    series: pd.Series | None,
    end_date: date | None = None,
    window_years: int = 10,
) -> AdaptiveRecipe | None:
    """Restituisce recipe (center, scale) per una condition_name dal mapping.

    Args:
        condition_name: chiave in `_CONDITION_RECIPES`.
        series: pandas Series storica dell'indicatore (indexed by date).
        end_date: data finale (default: oggi).

    Returns:
        AdaptiveRecipe o None se condition non mappata / serie insufficiente.
    """
    recipe_meta = _CONDITION_RECIPES.get(condition_name)
    if recipe_meta is None:
        return None
    series_name, threshold_type = recipe_meta
    if series is None or series.empty:
        return None
    quantiles = compute_quantiles(
        series,
        quantiles=(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
        end_date=end_date,
        window_years=window_years,
    )
    if not quantiles:
        return None
    recipe = quantiles_to_recipe(
        quantiles,
        threshold_type=threshold_type,
        series_name=series_name,
        n_obs=len(series.dropna()),
    )
    return recipe


def list_adaptive_conditions() -> list[str]:
    """Restituisce le condition_name supportate dal mapping adattivo."""
    return list(_CONDITION_RECIPES.keys())


# --- DB helpers per popolare le serie ---

_CACHE_TTL_KEY: dict[str, str] = {}  # series_id → last cache date_iso
_CACHE_RECIPES: dict[tuple[str, str], AdaptiveRecipe] = {}  # (condition, date_iso) → recipe


def load_series_from_db(db, series_id: str) -> pd.Series:
    """Carica serie pandas indexed by date da MacroIndicator table.

    Args:
        db: SQLAlchemy session.
        series_id: chiave serie (es. "vix", "baa_spread").

    Returns:
        pd.Series indicizzata per data (vuota se serie non trovata).
    """
    from app.models import MacroIndicator

    rows = (
        db.query(MacroIndicator)
        .filter(MacroIndicator.series_id == series_id)
        .order_by(MacroIndicator.date.asc())
        .all()
    )
    if not rows:
        return pd.Series(dtype=float)
    data = {pd.Timestamp(r.date): float(r.value) for r in rows if r.value is not None}
    return pd.Series(data).sort_index()


def load_series_from_classifications(db, indicator_key: str) -> pd.Series:
    """Carica serie storica di un indicator (es. "vix") dai records
    `RegimeClassification.conditions_met.indicators.{key}`.

    Pattern: nel progetto le serie storiche vivono dentro il JSON `conditions_met`
    dei record `RegimeClassification` (1027 records 1971-2026), non in
    `MacroIndicator` (vuota).

    Args:
        db: SQLAlchemy session.
        indicator_key: chiave dell'indicatore (es. "vix", "baa_spread", "cpi_yoy").

    Returns:
        pd.Series indicizzata per data (vuota se nessun record).
    """
    import json
    from app.models import RegimeClassification

    rows = (
        db.query(RegimeClassification)
        .order_by(RegimeClassification.date.asc())
        .all()
    )
    data: dict[pd.Timestamp, float] = {}
    for r in rows:
        if not r.conditions_met:
            continue
        try:
            meta = json.loads(r.conditions_met)
        except (ValueError, TypeError):
            continue
        indicators = meta.get("indicators", {}) or {}
        val = indicators.get(indicator_key)
        if val is None:
            continue
        try:
            data[pd.Timestamp(r.date)] = float(val)
        except (ValueError, TypeError):
            continue
    if not data:
        return pd.Series(dtype=float)
    return pd.Series(data).sort_index()


def load_cpi_yoy_from_db(db) -> pd.Series:
    """Calcola CPI YoY on-the-fly da CPI level (`series_id="cpi"`).

    CPI è level (es. 280.5), il classifier vuole cpi_yoy = (now/12m_ago - 1)*100.
    """
    cpi_level = load_series_from_db(db, "cpi")
    if cpi_level.empty:
        return pd.Series(dtype=float)
    # Resample monthly + pct_change 12 mesi
    monthly = cpi_level.resample("ME").last().dropna()
    yoy = monthly.pct_change(12) * 100.0
    return yoy.dropna()
