"""Calcolo rendimenti nominali e REALI (deflazionati CPI) per asset class.

Il regime classifier produce probabilita' P(regime) per ogni mese storico.
Per validare ASSET_REGIME_DATA (hardcoded in scoring/engine.py) servono
rendimenti misurati vs nominali sui prezzi reali.

real_return_t = (price_{t+h} / price_t) / (cpi_{t+h} / cpi_t) - 1

Sharpe: usiamo lo Sharpe REALE (real_return - risk_free_real) / vol(real)
con risk_free_real ~ 0% (T-bill ex-CPI ≈ 0 storicamente).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.services.indicators.fetcher import FredFetcher
from app.services.prices.asset_universe import ASSET_TICKERS
from app.services.prices.yahoo_fetcher import YahooFetcher


@dataclass
class RegimeAssetMetrics:
    asset: str
    regime: str
    n_observations: int
    hit_rate: float        # frazione di periodi con real_return > 0
    real_return: float     # media rendimento reale 12m forward
    volatility: float      # std rendimento reale 12m forward
    sharpe: float          # real_return / volatility (risk-free reale ~ 0)


def _align_monthly(s: pd.Series) -> pd.Series:
    """Riallinea una serie daily a fine mese, prendendo l'ultimo prezzo."""
    s = s.copy()
    s.index = pd.to_datetime(s.index)
    return s.resample("ME").last().dropna()


def real_return_series(
    asset: str,
    horizon_months: int = 12,
    yahoo: YahooFetcher | None = None,
    fred: FredFetcher | None = None,
) -> pd.Series:
    """Per ogni mese t restituisce real_return forward a `horizon_months` mesi.

    real_return_t = nominal_return_t / cpi_inflation_t

    Returns: Series indexed by month-end di tutti i punti per cui esiste
    sia il prezzo iniziale che quello a t+h e il CPI a entrambi.
    """
    yahoo = yahoo or YahooFetcher()
    fred = fred or FredFetcher()

    if asset not in ASSET_TICKERS:
        raise ValueError(f"Unknown asset: {asset}")

    px = yahoo.fetch_asset(asset)
    px_m = _align_monthly(px)
    cpi = fred.fetch_series("cpi")
    cpi_m = _align_monthly(cpi)

    # Allinea sui mesi comuni
    common = px_m.index.intersection(cpi_m.index)
    px_m = px_m.loc[common]
    cpi_m = cpi_m.loc[common]

    if len(px_m) < horizon_months + 2:
        return pd.Series(dtype=float)

    nominal_ret = px_m.shift(-horizon_months) / px_m - 1
    inflation = cpi_m.shift(-horizon_months) / cpi_m - 1
    real_ret = (1 + nominal_ret) / (1 + inflation) - 1
    return real_ret.dropna()


def metrics_by_regime(
    asset: str,
    regime_probs_monthly: pd.DataFrame,
    horizon_months: int = 12,
    threshold: float = 0.45,
    yahoo: YahooFetcher | None = None,
    fred: FredFetcher | None = None,
) -> list[RegimeAssetMetrics]:
    """Calcola hit_rate / real_return / vol / sharpe per ogni regime.

    Args:
        asset: nome asset class
        regime_probs_monthly: DataFrame con index = month-end, columns = REGIMES,
            valori = probabilita' marginale del regime in quel mese
        horizon_months: orizzonte forward per il return
        threshold: un mese e' "in regime r" se prob_r >= threshold

    Una osservazione t conta per il regime r se la sua prob_r >= threshold.
    Un mese puo' essere in 0 o piu' regimi (overlap, ma in pratica raro a 0.45).
    """
    real_ret = real_return_series(asset, horizon_months, yahoo, fred)
    if real_ret.empty:
        return []

    # Allinea probabilita' regime ai mesi del rendimento
    probs = regime_probs_monthly.copy()
    probs.index = pd.to_datetime(probs.index).to_period("M").to_timestamp("M")
    common = real_ret.index.intersection(probs.index)
    if len(common) < 12:
        return []

    real_ret = real_ret.loc[common]
    probs = probs.loc[common]

    out: list[RegimeAssetMetrics] = []
    for regime in probs.columns:
        mask = probs[regime] >= threshold
        rets = real_ret[mask]
        if len(rets) < 6:
            out.append(RegimeAssetMetrics(
                asset=asset, regime=regime, n_observations=len(rets),
                hit_rate=float("nan"), real_return=float("nan"),
                volatility=float("nan"), sharpe=float("nan"),
            ))
            continue
        hit_rate = float((rets > 0).mean())
        mean_ret = float(rets.mean())
        vol = float(rets.std(ddof=1))
        sharpe = float(mean_ret / vol) if vol > 0 else 0.0
        out.append(RegimeAssetMetrics(
            asset=asset, regime=regime, n_observations=len(rets),
            hit_rate=hit_rate, real_return=mean_ret,
            volatility=vol, sharpe=sharpe,
        ))
    return out


def regime_probs_dataframe(rows) -> pd.DataFrame:
    """Trasforma una lista di RegimeClassification in DataFrame mensile.

    Se ci sono piu' record per mese (per es. backfill daily + monthly), aggrega
    via media.
    """
    df = pd.DataFrame([
        {
            "date": pd.Timestamp(r.date),
            "reflation": r.probability_reflation,
            "stagflation": r.probability_stagflation,
            "deflation": r.probability_deflation,
            "goldilocks": r.probability_goldilocks,
        }
        for r in rows
    ])
    if df.empty:
        return df
    df = df.set_index("date").sort_index()
    monthly = df.resample("ME").mean().dropna()
    return monthly
