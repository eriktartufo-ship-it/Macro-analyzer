"""Portfolio simulator: monthly rebalance senza lookahead bias.

Disegno:
  - Allocation al mese t = funzione di info disponibili a t-1 (regime probs, scores)
  - Rendimento del mese t = media pesata dei rendimenti realizzati t-1 -> t
  - Cash residuo = mancanza segnale (allocation < 100%) -> 0% rendimento

Trading cost: bp configurabile, applicato sulla turnover del rebalance.

Output: Series di rendimenti mensili portfolio, allocations per mese, contributions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from loguru import logger

from app.services.prices.asset_universe import ASSET_TICKERS
from app.services.prices.yahoo_fetcher import YahooFetcher


@dataclass
class BacktestRun:
    monthly_returns: pd.Series       # net portfolio returns
    allocations: pd.DataFrame        # weights per asset per mese (rows=date, cols=asset)
    asset_returns: pd.DataFrame      # rendimenti realizzati per asset (per attribution)
    turnover: pd.Series              # turnover mensile
    cost_bps: float                  # bp di costo applicato per rebalance


def _to_monthly_close(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = pd.to_datetime(s.index)
    return s.resample("ME").last().dropna()


def fetch_asset_returns(
    asset_classes: list[str],
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Costruisce DataFrame mensile dei rendimenti realizzati per gli asset richiesti."""
    yahoo = YahooFetcher()
    cols = {}
    for asset in asset_classes:
        if asset not in ASSET_TICKERS:
            logger.warning(f"backtest: skip unknown asset {asset}")
            continue
        try:
            px = yahoo.fetch_asset(asset)
            px_m = _to_monthly_close(px)
            cols[asset] = px_m.pct_change()
        except Exception as e:
            logger.warning(f"backtest: skip {asset} ({e})")

    df = pd.DataFrame(cols).sort_index()
    if start:
        df = df.loc[df.index >= pd.Timestamp(start)]
    if end:
        df = df.loc[df.index <= pd.Timestamp(end)]
    return df


def run_backtest(
    target_weights: pd.DataFrame,    # index=date, cols=asset; valori 0..1, sum<=1
    asset_returns: pd.DataFrame,     # index=date, cols=asset
    cost_bps: float = 10.0,
) -> BacktestRun:
    """Simulazione monthly rebalance.

    target_weights[t] e' applicato al mese t con info noti a t (intended for use by
    chiamanti che calcolano i pesi shiftando t-1 internally).

    Returns: BacktestRun con metriche mensili.
    """
    # Allinea le date
    common = target_weights.index.intersection(asset_returns.index)
    if len(common) < 12:
        raise ValueError(f"backtest: overlap insufficiente ({len(common)} mesi)")
    weights = target_weights.loc[common].fillna(0.0)
    returns = asset_returns.loc[common].fillna(0.0)

    # Allinea colonne (asset comuni a entrambi)
    cols = weights.columns.intersection(returns.columns)
    weights = weights[cols]
    returns = returns[cols]

    # Per evitare lookahead: i pesi al mese t sono basati su info a t-1, ma
    # sono APPLICATI al rendimento del mese t. Quindi il rendimento del mese t
    # del portfolio = sum(w_t * r_t).
    portfolio_returns = (weights * returns).sum(axis=1)

    # Turnover = sum(|w_t - w_{t-1}|) / 2
    weight_diff = weights.diff().abs().sum(axis=1) / 2
    weight_diff.iloc[0] = float(weights.iloc[0].abs().sum()) / 2  # initial allocation cost
    turnover = weight_diff.fillna(0.0)

    # Trading cost: bp * turnover -> deduzione dal rendimento
    cost = (cost_bps / 10000.0) * turnover
    portfolio_net = portfolio_returns - cost

    return BacktestRun(
        monthly_returns=portfolio_net,
        allocations=weights,
        asset_returns=returns,
        turnover=turnover,
        cost_bps=cost_bps,
    )
