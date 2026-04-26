"""Sintesi total return per bond/cash da serie yield FRED.

Yahoo Finance fornisce TLT/SHY/BIL ETF solo dal 2002-2007. Pre-2002, i proxy
^TYX e ^IRX sono YIELD level — usarli come "prezzo" e' scientificamente sbagliato.

Soluzione: ricostruzione TR via duration approximation.

  monthly_return ≈ -modified_duration * yield_change + yield_carry

dove:
  - yield_change = y_t - y_{t-1}  (in decimali, es. 0.005 = 50bp)
  - price_return = -D * yield_change (capital gain/loss da variazione tassi)
  - yield_carry = y_{t-1} / 12 (interesse mensile maturato)
  - total_return = price_return + yield_carry

Approssimazione: duration costante per tutto il periodo. E' una semplificazione
ma sui livelli aggregati (12m forward) regge bene per validation di first-order.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd

from app.services.indicators.fetcher import FredFetcher


# Modified duration tipica per ogni proxy bond
# Fonti: iShares prospetti TLT/SHY/BIL e duration accademica per maturity buckets
_DURATION = {
    "us_bonds_long": 17.0,    # TLT ≈ 17y duration (20+y maturity)
    "us_bonds_short": 1.9,    # SHY ≈ 1.9y duration (1-3y maturity)
    "cash_money_market": 0.08,  # BIL ≈ 1 mese duration (~1/12 = 0.083)
}

# Mapping asset -> serie FRED yield rappresentativa
_YIELD_SOURCE = {
    "us_bonds_long": "treasury_10y",   # DGS10 — proxy 20+y, lievemente sotto-stima vol
    "us_bonds_short": "treasury_2y",   # DGS2
    "cash_money_market": "fed_funds",  # FEDFUNDS, monthly
}


def synthesize_bond_tr_index(
    asset: str,
    start_date: date = date(1962, 1, 1),
    fred: Optional[FredFetcher] = None,
) -> pd.Series:
    """Costruisce un indice total return ricostruito da yield.

    Returns:
        Series indexed by date, valori = livello indice (parte da 100).
    """
    if asset not in _DURATION:
        raise ValueError(f"Asset {asset} non supportato dal sintetizzatore TR")

    fred = fred or FredFetcher()
    series_name = _YIELD_SOURCE[asset]
    duration = _DURATION[asset]

    yield_pct = fred.fetch_series(series_name, start_date=start_date)
    # Resample a mensile (fine mese, ultimo valore)
    yield_pct = yield_pct.resample("ME").last().dropna()
    # Decimali (4.5% -> 0.045)
    yield_dec = yield_pct / 100.0

    # Variazione yield su base mensile
    yield_change = yield_dec.diff()
    # Capital return = -duration * delta_yield
    price_return = -duration * yield_change
    # Carry = yield iniziale del periodo / 12
    yield_carry = yield_dec.shift(1) / 12.0
    # Total return mensile
    monthly_tr = (price_return + yield_carry).fillna(0.0)
    # Cumulato per ottenere indice di livello
    tr_index = (1 + monthly_tr).cumprod() * 100.0
    tr_index.name = "tr_index"
    return tr_index
