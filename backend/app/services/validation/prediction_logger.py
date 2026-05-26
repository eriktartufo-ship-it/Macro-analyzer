"""T9-AUDIT-TA — Service per logging predictions + evaluation realized returns.

Inspired da TradingAgents: traccia ogni prediction model + 1/3/6 mesi dopo
calcola P&L vero vs benchmark 60/40.

API:
- `log_prediction(db, date, classify_output, top5, allocation_mode)`:
  salva un nuovo record.
- `evaluate_pending(db, asset_returns)`:
  per ogni log non ancora valutato, se sono passati ≥1/3/6 mesi,
  computa returns realizzati.
- `summary_stats(db, horizon='3m')`:
  aggregate P&L del modello vs benchmark.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from loguru import logger
from sqlalchemy.orm import Session

from app.models import PredictionLog


# Benchmark 60/40 portfolio (simplified): 60% S&P, 40% 10y Treasury
BENCHMARK_60_40 = {"us_equities_growth": 0.60, "us_bonds_long": 0.40}


@dataclass
class PredictionLogged:
    id: int
    date_predicted: date
    regime: str
    top5_count: int


def _ensure_table_exists() -> None:
    """Lazy create table (pattern T6.6/T6.7)."""
    from app.database import engine
    try:
        PredictionLog.__table__.create(engine, checkfirst=True)
    except Exception as e:
        logger.warning(f"Lazy create prediction_log failed: {e}")


def log_prediction(
    db: Session,
    classify_output: dict,
    top5_assets: list[dict],
    date_predicted: date | None = None,
    allocation_mode: str = "regime_based",
) -> PredictionLogged:
    """Salva una nuova prediction con allocation suggerita.

    Args:
        db: SQLAlchemy session.
        classify_output: output di classify_regime() — deve avere
            'regime', 'probabilities', 'confidence'.
        top5_assets: lista [{'asset': str, 'score': float, 'weight': float}].
        date_predicted: data prediction (default today).
        allocation_mode: 'regime_based' | 'defensive_transition'.

    Returns:
        PredictionLogged con id.
    """
    _ensure_table_exists()
    if date_predicted is None:
        date_predicted = date.today()

    record = PredictionLog(
        date_predicted=date_predicted,
        regime=classify_output.get("regime", "unknown"),
        confidence=float(classify_output.get("confidence", 0.0)),
        probabilities_json=json.dumps(classify_output.get("probabilities", {})),
        top5_json=json.dumps(top5_assets),
        allocation_mode=allocation_mode,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return PredictionLogged(
        id=record.id, date_predicted=date_predicted,
        regime=record.regime, top5_count=len(top5_assets),
    )


def _compute_portfolio_return(
    weights: dict[str, float],
    asset_returns: dict[str, float],
) -> float:
    """Σ weight_i × return_i. Asset mancanti = 0 return assunto."""
    return sum(weights.get(a, 0) * asset_returns.get(a, 0) for a in weights)


def evaluate_log(
    db: Session,
    asset_returns_provider,
    today: date | None = None,
) -> int:
    """Evaluate logs pending (≥1m, ≥3m, ≥6m fa) con returns realizzati.

    Args:
        db: SQLAlchemy session.
        asset_returns_provider: callable (start_date, end_date) → dict[asset, cumulative_return].
        today: oggi (default today()). Utile per test/backtest.

    Returns:
        Numero di records aggiornati.
    """
    _ensure_table_exists()
    if today is None:
        today = date.today()

    pending = db.query(PredictionLog).filter(
        PredictionLog.evaluated_at.is_(None) | (PredictionLog.alpha_6m.is_(None))
    ).all()

    updated = 0
    for record in pending:
        dp = record.date_predicted
        if not isinstance(dp, date):
            continue

        top5 = json.loads(record.top5_json)
        # weights dict da top5 list
        if top5 and "weight" in top5[0]:
            weights = {item["asset"]: item["weight"] for item in top5}
        else:
            # fallback equal weight
            weights = {item["asset"]: 1.0 / len(top5) for item in top5}

        changed = False
        for horizon_months, attr_ret, attr_bench, attr_alpha in (
            (1, "realized_return_1m", "benchmark_return_1m", "alpha_1m"),
            (3, "realized_return_3m", "benchmark_return_3m", "alpha_3m"),
            (6, "realized_return_6m", "benchmark_return_6m", "alpha_6m"),
        ):
            if getattr(record, attr_ret) is not None:
                continue
            target_date = dp + timedelta(days=horizon_months * 30)
            if target_date > today:
                continue  # non ancora maturato

            try:
                returns = asset_returns_provider(dp, target_date)
            except Exception as e:
                logger.warning(f"asset_returns_provider error for {dp}-{target_date}: {e}")
                continue
            if not returns:
                continue

            r_pred = _compute_portfolio_return(weights, returns)
            r_bench = _compute_portfolio_return(BENCHMARK_60_40, returns)
            setattr(record, attr_ret, round(r_pred, 6))
            setattr(record, attr_bench, round(r_bench, 6))
            setattr(record, attr_alpha, round(r_pred - r_bench, 6))
            changed = True

        if changed:
            record.evaluated_at = datetime.utcnow()
            updated += 1

    if updated:
        db.commit()
    return updated


def summary_stats(db: Session, horizon: str = "3m") -> dict:
    """Aggregate P&L stats: mean alpha, hit rate, n_evaluated.

    Args:
        horizon: '1m' | '3m' | '6m'.

    Returns:
        dict con mean_alpha, hit_rate (% volte alpha > 0), n_evaluated.
    """
    _ensure_table_exists()
    attr_alpha = f"alpha_{horizon}"
    attr_ret = f"realized_return_{horizon}"

    rows = db.query(PredictionLog).filter(
        getattr(PredictionLog, attr_alpha).is_not(None)
    ).all()

    if not rows:
        return {
            "horizon": horizon, "n_evaluated": 0,
            "mean_alpha": None, "median_alpha": None,
            "hit_rate": None, "mean_realized_return": None,
        }

    alphas = [getattr(r, attr_alpha) for r in rows]
    rets = [getattr(r, attr_ret) for r in rows]
    n_positive = sum(1 for a in alphas if a > 0)

    return {
        "horizon": horizon,
        "n_evaluated": len(rows),
        "mean_alpha": round(sum(alphas) / len(alphas), 6),
        "median_alpha": round(sorted(alphas)[len(alphas) // 2], 6),
        "hit_rate": round(n_positive / len(rows), 4),
        "mean_realized_return": round(sum(rets) / len(rets), 6),
    }


def stats_by_regime(db: Session, horizon: str = "3m") -> dict[str, dict]:
    """Stats per regime: dove il modello fa bene/male?"""
    _ensure_table_exists()
    attr_alpha = f"alpha_{horizon}"
    rows = db.query(PredictionLog).filter(
        getattr(PredictionLog, attr_alpha).is_not(None)
    ).all()

    by_regime: dict[str, list[float]] = {}
    for r in rows:
        by_regime.setdefault(r.regime, []).append(getattr(r, attr_alpha))

    out: dict[str, dict] = {}
    for regime, alphas in by_regime.items():
        if not alphas:
            continue
        out[regime] = {
            "n": len(alphas),
            "mean_alpha": round(sum(alphas) / len(alphas), 6),
            "hit_rate": round(sum(1 for a in alphas if a > 0) / len(alphas), 4),
            "worst_alpha": round(min(alphas), 6),
            "best_alpha": round(max(alphas), 6),
        }
    return out
