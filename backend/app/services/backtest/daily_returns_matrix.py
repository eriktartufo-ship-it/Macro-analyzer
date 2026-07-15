"""B2 (S41) — Daily returns matrix per il backtest daily-cadence.

Analogo daily di real_returns_matrix.py (mensile). Il backtest mensile quantizza
entrata/uscita a fine mese → ±7pp di rumore-fortuna (S39b) e può mancare le
V-recovery intra-mese. Qui: rendimenti NOMINALI giornalieri per ogni asset,
allineati al calendario di borsa, così l'harness daily può applicare i return
giorno-per-giorno (NAV/DD fedeli) con cadenza di decisione configurabile.

Nota real vs nominal: a granularità daily la deflazione CPI (mensile) non è
significativa e introdurrebbe artefatti di step → si usano rendimenti NOMINALI
(standard per un backtest daily). La matrice mensile resta real per le metriche
di regime; qui l'obiettivo è la fedeltà di timing/DD.

Output: DataFrame index = trading day (Timestamp), columns = asset, valori =
rendimento semplice giornaliero `px_t/px_{t-1} - 1`. NaN dove l'asset non ha
ancora storia (consumer deve gestire i missing come peso→cash).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from app.services.prices.asset_universe import ASSET_TICKERS
from app.services.prices.yahoo_fetcher import YahooFetcher

_CACHE_PATH = Path(__file__).resolve().parents[3] / ".cache" / "backtest" / "daily_returns_1d.parquet"

# Asset di riferimento per il calendario di borsa (storia lunga, liquido).
_CALENDAR_ASSET = "us_equities_growth"


def build_daily_returns_matrix(
    assets: list[str],
    use_cache: bool = True,
    yahoo: YahooFetcher | None = None,
) -> pd.DataFrame:
    """Costruisce la matrice di rendimenti semplici giornalieri per gli asset.

    Args:
        assets: lista asset class (chiavi di ASSET_TICKERS).
        use_cache: se True usa/scrive il parquet su disco.
        yahoo: fetcher opzionale (per test/iniezione).

    Returns:
        DataFrame [trading_day x asset] di rendimenti daily. NaN = no storia.
    """
    if use_cache and _CACHE_PATH.exists():
        try:
            df = pd.read_parquet(_CACHE_PATH)
            df.index = pd.to_datetime(df.index)
            missing = [a for a in assets if a not in df.columns]
            if not missing:
                return df[assets]
            logger.info(f"Daily matrix cache miss su {missing}, rebuild")
        except Exception as e:
            logger.warning(f"Daily matrix cache read failed: {e}")

    yahoo = yahoo or YahooFetcher()
    ret_map: dict[str, pd.Series] = {}
    for asset in assets:
        if asset not in ASSET_TICKERS:
            logger.warning(f"  {asset}: non in ASSET_TICKERS, skip")
            continue
        try:
            px = yahoo.fetch_asset(asset)
            if px is None or px.empty:
                logger.warning(f"  {asset}: prezzi vuoti, skip")
                continue
            px = px.copy()
            px.index = pd.to_datetime(px.index)
            px = px[~px.index.duplicated(keep="last")].sort_index()
            r = px.pct_change().dropna()
            if not r.empty:
                ret_map[asset] = r
                logger.info(f"  {asset}: {len(r)} giorni, "
                            f"{r.index.min().date()} -> {r.index.max().date()}")
        except Exception as e:
            logger.warning(f"  {asset}: fetch/pct_change failed: {e}")

    if not ret_map:
        return pd.DataFrame()

    df = pd.DataFrame(ret_map)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    if use_cache:
        try:
            _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(_CACHE_PATH)
            logger.info(f"Saved daily returns matrix: {df.shape} to {_CACHE_PATH}")
        except Exception as e:
            logger.warning(f"Daily matrix cache write failed: {e}")

    return df


def trading_days(matrix: pd.DataFrame, start, end) -> pd.DatetimeIndex:
    """Giorni di borsa nella finestra [start, end] presenti nella matrice."""
    idx = matrix.index
    return idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
