"""NEW #4 council 2026-05-27 — Shiller dataset integration (1871+).

Robert Shiller pubblica monthly S&P + CPI dataset dal gennaio 1871 (yale.edu).
Per validare regime stagflation '70-'80 (oil shocks) servono real prices.
yfinance + FRED CPI partono ~1990, troppo corto.

Shiller dataset: http://www.econ.yale.edu/~shiller/data/ie_data.xls
Manual download richiesto, salvato in `backend/data/shiller_ie_data.csv`.

Columns: Date, P (S&P composite), D (dividend), E (earnings), CPI, GS10 (10y yield),
Real_Price, Real_Dividend, Real_Earnings, CAPE.

API:
- `load_shiller_dataset()`: legge CSV, ritorna pd.DataFrame
- `shiller_real_return_series(horizon_months=1)`: monthly real returns S&P
  da 1871. Compatibile con real_returns_matrix.

Use case: extend real_returns_matrix back to '70-'80 per asset us_equities_growth
(S&P composite come proxy). Altri asset (bonds, gold) richiedono dataset
addizionali (FRED GS10 + LBMA gold).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger


_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_SHILLER_CSV = _DATA_DIR / "shiller_ie_data.csv"


def shiller_csv_exists() -> bool:
    return _SHILLER_CSV.exists()


def load_shiller_dataset() -> pd.DataFrame | None:
    """Carica Shiller monthly dataset (S&P + CPI + Real_Price + ...).

    Returns:
        DataFrame indexed by date (month-end), o None se CSV missing.
        Columns expected: P, CPI, Real_Price, GS10 (almeno).
    """
    if not shiller_csv_exists():
        logger.warning(
            f"Shiller CSV missing at {_SHILLER_CSV}. "
            f"Download manuale da http://www.econ.yale.edu/~shiller/data/ie_data.xls "
            f"(save as CSV). NEW #4 council fallback."
        )
        return None

    try:
        df = pd.read_csv(_SHILLER_CSV)
        # Shiller date format è "YYYY.MM" come float (es. 1871.01)
        # Convert a month-end timestamps
        if "Date" in df.columns:
            year_month = df["Date"].astype(str).str.replace(".1$", ".10", regex=True)
            df["timestamp"] = pd.to_datetime(year_month, format="%Y.%m", errors="coerce")
            df = df.dropna(subset=["timestamp"]).set_index("timestamp")
            df.index = df.index + pd.offsets.MonthEnd(0)
        return df
    except Exception as e:
        logger.warning(f"Shiller CSV load failed: {e}")
        return None


def shiller_real_return_series(
    horizon_months: int = 1,
) -> pd.Series:
    """Real return forward `horizon_months` su S&P composite Shiller.

    real_return_t = real_price_{t+h} / real_price_t - 1
    (real_price = price / cpi, già adjusted in Shiller dataset)

    Args:
        horizon_months: forward horizon (default 1m).

    Returns:
        Series indexed by month-end con real returns. Vuoto se CSV missing.
    """
    df = load_shiller_dataset()
    if df is None:
        return pd.Series(dtype=float)

    if "Real_Price" not in df.columns:
        # Fallback: compute from P + CPI
        if "P" in df.columns and "CPI" in df.columns:
            df["Real_Price"] = df["P"] / df["CPI"]
        else:
            logger.warning("Shiller CSV missing Real_Price/P/CPI columns")
            return pd.Series(dtype=float)

    real_price = df["Real_Price"].dropna()
    if len(real_price) < horizon_months + 2:
        return pd.Series(dtype=float)

    forward = real_price.shift(-horizon_months)
    returns = forward / real_price - 1
    return returns.dropna()


def extend_real_returns_matrix_with_shiller(
    matrix: pd.DataFrame,
    cutoff_date: date | None = None,
) -> pd.DataFrame:
    """Extend real_returns_matrix backward usando Shiller per us_equities_growth.

    Combina:
    - Pre-cutoff (default 1990-01-01): Shiller Real_Price returns
    - Post-cutoff: matrix esistente (yfinance + FRED CPI)

    Per gli ALTRI asset (non us_equities_growth), Shiller non ha dati →
    rimangono NaN pre-1990.

    Args:
        matrix: pd.DataFrame esistente (output di build_real_returns_matrix).
        cutoff_date: data da cui matrix esistente prevale.

    Returns:
        Matrix esteso temporalmente per us_equities_growth.
    """
    shiller_ret = shiller_real_return_series(horizon_months=1)
    if shiller_ret.empty:
        return matrix

    if cutoff_date is None:
        cutoff_date = date(1990, 1, 1)
    cutoff_ts = pd.Timestamp(cutoff_date)

    pre = shiller_ret[shiller_ret.index < cutoff_ts]
    if pre.empty:
        return matrix

    # Build extended DataFrame
    extended_idx = pre.index.union(matrix.index)
    extended = matrix.reindex(extended_idx)
    if "us_equities_growth" in extended.columns:
        # Fill pre-cutoff con Shiller
        mask = extended.index < cutoff_ts
        extended.loc[mask, "us_equities_growth"] = pre.reindex(extended.index[mask])

    return extended.sort_index()
