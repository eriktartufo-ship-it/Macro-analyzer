"""Market features feature-disjoint dal rule-based classifier.

Il rule-based usa: gdp/cpi/unrate/payrolls/indpro/lei/baa/sentiment/yield_curve/
fed_funds/vix/nfci/breakeven/housing. Per avere un secondo modello che vede cose
DIVERSE, costruiamo features puramente di mercato che il classifier non guarda:

  1. **Yield curve curvature** = (10y2y) - (10y3m): se 10y2y e' > 0 ma 10y3m < 0
     o viceversa, segnale di slope inusuale.
  2. **HY-IG credit risk premium** = hy_credit_spread - ig_credit_spread.
     Captura premio per default risk vs duration.
  3. **Copper/gold ratio** = copper_price / GLD price (proxy growth-vs-fear).
  4. **S&P 12m return** = momentum equity (lookback 252 giorni).
  5. **NDX/SPX ratio momentum** = QQQ/SPY 12m % change (growth-vs-broad).
  6. **Dollar 3m change** = DTWEXBGS rolling 3m % (cross-asset stress).
  7. **VIX percentile** = rank percentile su 5 anni (calma vs spike).

Questi 7 feature sono indipendenti dai macro indicators del rule-based ma molto
informativi su risk-on/risk-off. L'HMM trained su questi dovrebbe scoprire stati
con bassa correlazione coi label rule-based — quando concordano, e' convergenza
genuina di approcci diversi.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from loguru import logger

from app.services.indicators.fetcher import FredFetcher
from app.services.prices.yahoo_fetcher import YahooFetcher


def _to_monthly(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = pd.to_datetime(s.index)
    return s.resample("ME").last().dropna()


def compute_market_features(
    start: date = date(1990, 1, 1),
    end: date | None = None,
) -> pd.DataFrame:
    """Costruisce DataFrame mensile delle market features. Index = month-end."""
    fred = FredFetcher()
    yahoo = YahooFetcher()
    end = end or date.today()

    # --- FRED daily/monthly series ---
    yc_10y2y = _to_monthly(fred.fetch_series("yield_curve_10y2y", start_date=start, end_date=end))
    yc_10y3m = _to_monthly(fred.fetch_series("yield_curve_10y3m", start_date=start, end_date=end))
    baa = _to_monthly(fred.fetch_series("baa_spread", start_date=start, end_date=end))
    vix = _to_monthly(fred.fetch_series("vix", start_date=start, end_date=end))
    try:
        hy = _to_monthly(fred.fetch_series("hy_credit_spread", start_date=start, end_date=end))
    except Exception as e:
        logger.warning(f"hy_credit_spread fetch failed: {e}")
        hy = pd.Series(dtype=float)
    try:
        ig = _to_monthly(fred.fetch_series("ig_credit_spread", start_date=start, end_date=end))
    except Exception as e:
        logger.warning(f"ig_credit_spread fetch failed: {e}")
        ig = pd.Series(dtype=float)
    try:
        dxy = _to_monthly(fred.fetch_series("dxy_broad", start_date=start, end_date=end))
    except Exception as e:
        logger.warning(f"dxy_broad fetch failed: {e}")
        dxy = pd.Series(dtype=float)

    # --- Yahoo prices ---
    spy = _to_monthly(yahoo.fetch("SPY"))
    qqq = _to_monthly(yahoo.fetch("QQQ"))
    gld = _to_monthly(yahoo.fetch("GLD"))
    try:
        copper = _to_monthly(fred.fetch_series("copper_price", start_date=start, end_date=end))
    except Exception:
        copper = pd.Series(dtype=float)

    df = pd.DataFrame(index=pd.date_range(start, end, freq="ME"))

    # 1. Yield curve curvature
    df["yc_curvature"] = (yc_10y2y - yc_10y3m).reindex(df.index)

    # 2. HY-IG premium dove disponibile (post-2023), BAA spread come fallback storico.
    # Scaliamo BAA a HY-IG sull'overlap per consistenza dimensionale (HY-IG e' diff,
    # BAA e' spread vs Treasury — entrambi captano risk premium ma scale diverse).
    credit = baa.reindex(df.index).copy()
    if not hy.empty and not ig.empty:
        hy_ig = (hy - ig).reindex(df.index)
        overlap = hy_ig.dropna().index.intersection(credit.dropna().index)
        if len(overlap) >= 3:
            scale = float(credit.loc[overlap].mean()) / max(float(hy_ig.loc[overlap].mean()), 1e-6)
            hy_ig_scaled = hy_ig * scale
            credit = credit.where(hy_ig.isna(), hy_ig_scaled)
    df["credit_premium"] = credit

    # 3. Copper/gold ratio
    if not copper.empty and not gld.empty:
        common = copper.index.intersection(gld.index)
        if len(common) > 0:
            df["copper_gold"] = (copper.loc[common] / gld.loc[common]).reindex(df.index)
        else:
            df["copper_gold"] = np.nan
    else:
        df["copper_gold"] = np.nan

    # 4. S&P 12m return
    if not spy.empty:
        df["sp500_12m_return"] = spy.pct_change(12).reindex(df.index)

    # 5. NDX/SPX ratio momentum (12m delta)
    if not qqq.empty and not spy.empty:
        ratio = (qqq / spy).reindex(df.index).dropna()
        df["nasdaq_sp_ratio_12m"] = ratio.pct_change(12).reindex(df.index)

    # 6. Dollar 3m change
    if not dxy.empty:
        df["dollar_3m_pct"] = dxy.pct_change(3).reindex(df.index)

    # 7. VIX 5y percentile
    if not vix.empty:
        vix_aligned = vix.reindex(df.index)
        df["vix_5y_pct"] = vix_aligned.rolling(60, min_periods=24).rank(pct=True)

    df = df.dropna(how="all")
    return df


def latest_features(df: pd.DataFrame) -> dict[str, float]:
    """Estrae l'ultima riga in dict (per esposizione API)."""
    if df.empty:
        return {}
    last = df.iloc[-1].dropna()
    return {k: float(v) for k, v in last.items()}
