"""Strategie di allocation per il backtest.

Generano `target_weights` (DataFrame mensile) DA shiftare di 1 mese prima dell'uso
(per evitare lookahead bias). Il chiamante (runner) si occupa dello shift.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy.orm import Session

from app.models import RegimeClassification
from app.services.scoring.engine import ASSET_CLASSES, calculate_final_scores


def regime_probs_monthly(db: Session) -> pd.DataFrame:
    """Carica posteriori regime mensili dal DB (aggregando se piu' record per mese)."""
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


def score_weighted_strategy(
    db: Session,
    top_n: int = 5,
    score_threshold: float = 30.0,
    asset_classes: list[str] | None = None,
    force_include_dedollar: bool | None = None,
) -> pd.DataFrame:
    """Per ogni mese: calcola asset scores dai regime probs, alloca proporzionalmente
    ai top N asset con score >= threshold. Resto = cash (0% peso).

    Returns: DataFrame index=month-end, cols=asset, valori in [0,1] con sum<=1.
    """
    assets = asset_classes or list(ASSET_CLASSES)
    rp = regime_probs_monthly(db)
    if rp.empty:
        return pd.DataFrame()

    rows = []
    for ts, probs in rp.iterrows():
        prob_dict = {k: float(v) for k, v in probs.items()}
        scores = calculate_final_scores(prob_dict, force_include_dedollar=force_include_dedollar)
        # Filtra asset richiesti
        scores = {a: s for a, s in scores.items() if a in assets}
        # Sopra threshold, top-N
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:top_n]
        top = [(a, s) for a, s in top if s >= score_threshold]
        if not top:
            # Nessun asset sufficiente -> tutto cash
            row = {a: 0.0 for a in assets}
        else:
            total = sum(s for _, s in top)
            row = {a: 0.0 for a in assets}
            for a, s in top:
                row[a] = s / total
        row["__date__"] = ts
        rows.append(row)

    df = pd.DataFrame(rows).set_index("__date__")
    df.index.name = None
    return df


def buy_and_hold_strategy(
    asset_classes: list[str], dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Equal-weight buy-and-hold (rebilanciato mensile per semplicita)."""
    n = len(asset_classes)
    if n == 0:
        return pd.DataFrame(index=dates)
    weight = 1.0 / n
    df = pd.DataFrame(weight, index=dates, columns=asset_classes)
    return df


def sixty_forty_strategy(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """60% us_equities_growth + 40% us_bonds_long (proxy classico 60/40)."""
    return pd.DataFrame({
        "us_equities_growth": 0.60,
        "us_bonds_long": 0.40,
    }, index=dates)


def spy_only_strategy(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Buy-and-hold SPY (us_equities_growth proxy)."""
    return pd.DataFrame({"us_equities_growth": 1.0}, index=dates)
