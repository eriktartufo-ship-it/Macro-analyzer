"""Mapping Fama-French factor returns × regime macro.

Per ogni regime nel DB, calcola la performance media (annualizzata) di ogni
fattore quando quel regime e' dominante (prob >= threshold). Output: tabella
factor × regime con mean_annual, vol_annual, sharpe, n_observations.

Uso pratico: dentro il book equity, sapere quale stile (size/value/momentum)
funziona meglio in ciascun regime. Es. value funziona in stagflation, momentum
in reflation. Permette ranking sub-asset (growth vs value) regime-conditional.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.models import RegimeClassification
from app.services.factors.fama_french import factor_keys, fetch_fama_french
from app.services.regime.classifier import REGIMES


DEFAULT_THRESHOLD = 0.40


@dataclass
class FactorRegimeStat:
    factor: str
    regime: str
    n_observations: int
    mean_annual: float       # rendimento medio annualizzato (decimale, es. 0.08 = +8%)
    vol_annual: float        # volatilita annualizzata
    sharpe: float            # Sharpe annualizzato (rf gia' sottratto in Mkt-RF)
    win_rate: float          # frazione mesi con factor return > 0


@dataclass
class FactorRegimeReport:
    threshold: float
    n_months_analyzed: int
    factor_keys: list[str]
    regimes: list[str]
    stats: list[FactorRegimeStat]
    common_period: tuple[str, str]


def _regime_probs_monthly(db: Session) -> pd.DataFrame:
    rows = (
        db.query(RegimeClassification)
        .order_by(RegimeClassification.date.asc())
        .all()
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([
        {
            "date": pd.Timestamp(r.date),
            "reflation": r.probability_reflation,
            "stagflation": r.probability_stagflation,
            "deflation": r.probability_deflation,
            "goldilocks": r.probability_goldilocks,
        }
        for r in rows
    ]).set_index("date").sort_index()
    return df.resample("ME").mean().dropna()


def compute_factor_regime_report(
    db: Session,
    threshold: float = DEFAULT_THRESHOLD,
) -> FactorRegimeReport:
    """Per ogni (factor, regime) calcola statistiche su mesi dove prob_regime>=threshold."""
    fr = fetch_fama_french()
    factors = fr.monthly
    if factors.empty:
        raise ValueError("Fama-French data vuota")

    # Converti percentuale Kenneth French (es. 2.89) a decimale (0.0289)
    factor_cols = [c for c in factor_keys() if c in factors.columns]
    factors_dec = factors[factor_cols] / 100.0
    factors_dec.index = pd.to_datetime(factors_dec.index)

    rp = _regime_probs_monthly(db)
    if rp.empty:
        raise ValueError("Nessuna classification in DB")

    common = factors_dec.index.intersection(rp.index)
    if len(common) < 12:
        raise ValueError(f"Overlap factor/regime troppo corto ({len(common)} mesi)")

    factors_aligned = factors_dec.loc[common]
    rp_aligned = rp.loc[common]

    stats: list[FactorRegimeStat] = []
    for factor in factor_cols:
        for regime in REGIMES:
            mask = rp_aligned[regime] >= threshold
            sample = factors_aligned.loc[mask, factor].dropna()
            n = len(sample)
            if n < 6:
                stats.append(FactorRegimeStat(
                    factor=factor, regime=regime, n_observations=n,
                    mean_annual=float("nan"), vol_annual=float("nan"),
                    sharpe=float("nan"), win_rate=float("nan"),
                ))
                continue
            mean_m = float(sample.mean())
            std_m = float(sample.std(ddof=1))
            mean_ann = mean_m * 12
            vol_ann = std_m * np.sqrt(12)
            sharpe = mean_ann / vol_ann if vol_ann > 0 else 0.0
            win = float((sample > 0).mean())
            stats.append(FactorRegimeStat(
                factor=factor, regime=regime, n_observations=n,
                mean_annual=mean_ann, vol_annual=vol_ann,
                sharpe=sharpe, win_rate=win,
            ))

    return FactorRegimeReport(
        threshold=threshold,
        n_months_analyzed=len(common),
        factor_keys=factor_cols,
        regimes=list(REGIMES),
        stats=stats,
        common_period=(str(common.min().date()), str(common.max().date())),
    )
