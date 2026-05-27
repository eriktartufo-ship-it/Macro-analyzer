"""T9-TA Real returns matrix per random backtest.

Fix data leakage del random backtest che usava `ASSET_REGIME_DATA` sia come
return source che come scoring input (circular). Qui costruiamo una matrix
[(year, month), asset] -> real_return_1m da Yahoo + FRED CPI.

Real return formula:
    real_ret_t = (1 + nominal_ret_t) / (1 + cpi_inflation_t) - 1

dove nominal_ret_t = px_{t+1m} / px_t - 1 (forward 1-month) e
cpi_inflation_t = cpi_{t+1m} / cpi_t - 1.

Output: dict[(year, month), dict[asset, float]]. Mesi senza dato per un asset
sono assenti (consumer deve gestire missing).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from app.services.prices.returns import real_return_series

_CACHE_PATH = Path(__file__).resolve().parents[3] / ".cache" / "backtest" / "real_returns_1m.parquet"


def build_real_returns_matrix(
    assets: list[str],
    use_cache: bool = True,
) -> pd.DataFrame:
    """Costruisce matrix di real returns 1-month forward per gli asset richiesti.

    Args:
        assets: lista asset class (chiavi di ASSET_TICKERS).
        use_cache: se True usa il parquet su disco quando presente.

    Returns:
        DataFrame con index = month-end timestamps, columns = asset names,
        valori = real return forward 1-month. NaN dove dato mancante.
    """
    if use_cache and _CACHE_PATH.exists():
        try:
            df = pd.read_parquet(_CACHE_PATH)
            df.index = pd.to_datetime(df.index)
            missing = [a for a in assets if a not in df.columns]
            if not missing:
                return df[assets]
            logger.info(f"Cache miss su asset {missing}, rebuild parziale")
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")

    series_map: dict[str, pd.Series] = {}
    for asset in assets:
        try:
            s = real_return_series(asset, horizon_months=1)
            if not s.empty:
                series_map[asset] = s
                logger.info(f"  {asset}: {len(s)} months, "
                            f"{s.index.min().date()} to {s.index.max().date()}")
            else:
                logger.warning(f"  {asset}: empty series, skipping")
        except Exception as e:
            logger.warning(f"  {asset}: real_return_series failed: {e}")

    if not series_map:
        return pd.DataFrame()

    df = pd.DataFrame(series_map)
    df.index = pd.to_datetime(df.index)

    if use_cache:
        try:
            _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(_CACHE_PATH)
            logger.info(f"Saved real returns matrix: {df.shape} to {_CACHE_PATH}")
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

    return df


def get_real_return(
    matrix: pd.DataFrame,
    year: int,
    month: int,
    weights: dict[str, float],
) -> Optional[float]:
    """Real return aggregato del portfolio nel mese (year, month).

    Args:
        matrix: output di build_real_returns_matrix.
        year, month: mese target.
        weights: dict asset -> peso normalizzato (sum a 1.0).

    Returns:
        Real return del portfolio o None se nessun asset ha dato in quel mese.
    """
    # Trova il month-end di (year, month)
    target = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    if target not in matrix.index:
        return None

    row = matrix.loc[target]
    total = 0.0
    weight_sum = 0.0
    for asset, w in weights.items():
        if asset in row.index and not pd.isna(row[asset]):
            total += w * float(row[asset])
            weight_sum += w
    if weight_sum < 0.5:  # almeno metà del portfolio deve avere dati
        return None
    # Normalizza per i missing
    return total / weight_sum if weight_sum > 0 else None


def coverage_report(matrix: pd.DataFrame) -> dict:
    """Report di copertura per asset (n. mesi disponibili)."""
    return {
        "n_months_total": len(matrix),
        "date_min": matrix.index.min().date() if not matrix.empty else None,
        "date_max": matrix.index.max().date() if not matrix.empty else None,
        "coverage_by_asset": {
            asset: int(matrix[asset].notna().sum())
            for asset in matrix.columns
        },
    }


def cumulative_returns_provider(matrix: pd.DataFrame):
    """CRITICAL #1 council 2026-05-27: callable per evaluate_log().

    Ritorna funzione (start_date, end_date) -> {asset: cumulative_return}
    da usare in `prediction_logger.evaluate_log`. Cumulative compounded
    da real returns mensili: (1 + r1)(1 + r2)... - 1.

    Args:
        matrix: output di build_real_returns_matrix.

    Returns:
        Callable da passare a evaluate_log come asset_returns_provider.
    """
    def provider(start_date, end_date) -> dict[str, float]:
        from datetime import date as _date
        if not isinstance(start_date, _date):
            return {}
        start_ts = pd.Timestamp(start_date.year, start_date.month, 1) + pd.offsets.MonthEnd(0)
        end_ts = pd.Timestamp(end_date.year, end_date.month, 1) + pd.offsets.MonthEnd(0)

        mask = (matrix.index >= start_ts) & (matrix.index <= end_ts)
        sub = matrix.loc[mask]
        if sub.empty:
            return {}
        # Compound cumulative per ogni asset
        out: dict[str, float] = {}
        for asset in sub.columns:
            series = sub[asset].dropna()
            if len(series) == 0:
                continue
            cumulative = (1.0 + series).prod() - 1.0
            out[asset] = float(cumulative)
        return out

    return provider
