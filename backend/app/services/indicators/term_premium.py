"""Decomposizione yield curve via Adrian-Crump-Moench (ACM) term premium.

Il yield 10Y osservato e' decomposto in:
  - **expected path** = media attesa del Fed funds rate sui prossimi 10 anni
  - **term premium** = premio richiesto dagli investitori per detenere 10Y vs
    rolling short (compensa duration risk, inflation risk, scarcity)

  yield_10y_fitted ≈ expected_path + term_premium

Letture macro:
  - term_premium ALTO (>+1%) = mercato vuole compensation alta → risk-off,
    duration aversion (es. fine '70s, 2008-09 mild)
  - term_premium BASSO/NEGATIVO = mercato accetta low yields → flight to quality,
    QE-driven scarcity (es. 2014-21)
  - expected_path ALTA = mercato prevede Fed hawkish nel medio
  - expected_path BASSA = mercato prevede tagli/easing

NY Fed ACM model: pubblicato giornalmente, range 1990-oggi via FRED:
  - THREEFYTP10 = term premium 10Y
  - THREEFY10   = fitted yield 10Y
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
from loguru import logger
from sqlalchemy.orm import Session

from app.models import RegimeClassification
from app.services.indicators.fetcher import FredFetcher
from app.services.regime.classifier import REGIMES


@dataclass
class TermPremiumPoint:
    date: date
    fitted_yield: float       # ACM 10Y yield modellato
    term_premium: float       # ACM 10Y term premium
    expected_path: float      # fitted_yield - term_premium


@dataclass
class TermPremiumStat:
    regime: str
    n_observations: int
    mean_fitted: float
    mean_term_premium: float
    mean_expected_path: float
    pct_term_premium_positive: float


@dataclass
class TermPremiumReport:
    points: list[TermPremiumPoint]
    by_regime: list[TermPremiumStat]
    common_period: tuple[str, str]
    threshold: float
    n_observations: int


def fetch_term_premium_decomposition(
    start: date | None = None,
    fred: FredFetcher | None = None,
) -> pd.DataFrame:
    """Fetcha le serie ACM e calcola expected_path = fitted - term_premium.

    Returns: DataFrame con cols [fitted_yield, term_premium, expected_path] e
    index daily.
    """
    fred = fred or FredFetcher()
    fitted = fred.fetch_series("acm_fitted_yield_10y", start_date=start)
    tp = fred.fetch_series("acm_term_premium_10y", start_date=start)

    df = pd.DataFrame({"fitted_yield": fitted, "term_premium": tp}).dropna()
    df["expected_path"] = df["fitted_yield"] - df["term_premium"]
    df.index = pd.to_datetime(df.index)
    return df


def compute_term_premium_report(
    db: Session,
    threshold: float = 0.40,
    days: int = 365 * 30,
) -> TermPremiumReport:
    """Per ogni regime, calcola la media di term_premium / expected_path / fitted yield.

    Output: timeline + tabella per regime + metadati.
    """
    from datetime import timedelta as _td

    cutoff = date.today() - _td(days=days)
    df = fetch_term_premium_decomposition(start=cutoff)
    df_m = df.resample("ME").mean().dropna()

    # Carica regime probs
    rows = (
        db.query(RegimeClassification)
        .order_by(RegimeClassification.date.asc())
        .all()
    )
    if not rows:
        raise ValueError("Nessuna classificazione in DB")

    rp = pd.DataFrame([
        {
            "date": pd.Timestamp(r.date),
            "reflation": r.probability_reflation,
            "stagflation": r.probability_stagflation,
            "deflation": r.probability_deflation,
            "goldilocks": r.probability_goldilocks,
        }
        for r in rows
    ]).set_index("date").sort_index().resample("ME").mean().dropna()

    common = df_m.index.intersection(rp.index)
    if len(common) < 12:
        raise ValueError(f"Overlap term-premium/regime troppo corto ({len(common)} mesi)")

    df_aligned = df_m.loc[common]
    rp_aligned = rp.loc[common]

    by_regime: list[TermPremiumStat] = []
    for regime in REGIMES:
        mask = rp_aligned[regime] >= threshold
        sample = df_aligned.loc[mask]
        n = len(sample)
        if n < 6:
            by_regime.append(TermPremiumStat(
                regime=regime, n_observations=n,
                mean_fitted=float("nan"), mean_term_premium=float("nan"),
                mean_expected_path=float("nan"), pct_term_premium_positive=float("nan"),
            ))
            continue
        by_regime.append(TermPremiumStat(
            regime=regime, n_observations=n,
            mean_fitted=float(sample["fitted_yield"].mean()),
            mean_term_premium=float(sample["term_premium"].mean()),
            mean_expected_path=float(sample["expected_path"].mean()),
            pct_term_premium_positive=float((sample["term_premium"] > 0).mean()),
        ))

    points = [
        TermPremiumPoint(
            date=idx.date(),
            fitted_yield=float(df_aligned.loc[idx, "fitted_yield"]),
            term_premium=float(df_aligned.loc[idx, "term_premium"]),
            expected_path=float(df_aligned.loc[idx, "expected_path"]),
        )
        for idx in df_aligned.index
    ]

    logger.info(
        f"Term premium report: {len(common)} mesi, threshold {threshold}"
    )

    return TermPremiumReport(
        points=points,
        by_regime=by_regime,
        common_period=(str(common.min().date()), str(common.max().date())),
        threshold=threshold,
        n_observations=len(common),
    )
