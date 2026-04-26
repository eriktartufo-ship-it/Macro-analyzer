"""Calibrazione dei parametri ASSET_REGIME_DATA tramite shrinkage Bayesiano.

I valori hardcoded in `engine.py` sono prior expert (la mia stima qualitativa
basata su rierimenti storici). Le metriche misurate sono il dato empirico calcolato
dai prezzi reali (Yahoo) deflazionati col CPI sui regimi del backfill storico.

Strategia: shrinkage proporzionale alla numerosita' campionaria.
  - n < n_threshold (default 6): si tiene il prior intero (campione insufficiente)
  - n >= n_full (default 30): si prende il misurato al 100%
  - tra threshold e full: weighted average lineare

Output: JSON in `seed/calibrated_asset_regime.json` con metadata (data, n_total,
horizon, soglia regime). Il scoring engine legge questo file all'import; se
assente, usa l'hardcoded come fallback.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from app.models import RegimeClassification
from app.services.prices.asset_universe import ASSET_TICKERS
from app.services.prices.returns import metrics_by_regime, regime_probs_dataframe
from app.services.regime.classifier import REGIMES


# Path output: backend/seed/calibrated_asset_regime.json
_CALIB_PATH = Path(__file__).resolve().parents[3] / "seed" / "calibrated_asset_regime.json"


@dataclass
class CalibrationParams:
    horizon_months: int = 12
    regime_threshold: float = 0.40
    n_min: int = 6        # sotto questo: prior intero (no shrinkage)
    n_full: int = 30      # sopra questo: misurato intero
    # Bandi di plausibilita': se misurato esce fuori, scartato come outlier
    real_return_max_abs: float = 0.80   # nessun real return > 80% o < -80% e' plausibile
    sharpe_max_abs: float = 4.0


def _shrinkage_weight(n: int, n_min: int, n_full: int) -> float:
    """Peso 0..1 per il misurato. 0 = prior intero, 1 = misurato intero."""
    if n < n_min:
        return 0.0
    if n >= n_full:
        return 1.0
    return (n - n_min) / (n_full - n_min)


def _shrink(prior: float, measured: float | None, weight: float) -> float:
    if measured is None:
        return prior
    return weight * measured + (1.0 - weight) * prior


def _is_outlier(real_return: float | None, sharpe: float | None,
                p: CalibrationParams) -> bool:
    if real_return is not None and abs(real_return) > p.real_return_max_abs:
        return True
    if sharpe is not None and abs(sharpe) > p.sharpe_max_abs:
        return True
    return False


def calibrate(
    db_session,
    params: CalibrationParams | None = None,
) -> dict[str, Any]:
    """Esegue la calibrazione e ritorna dict pronto per JSON.

    Returns:
        {
            "version": int,
            "calibrated_on": iso_date,
            "params": {...},
            "n_classifications": int,
            "asset_regime_data": {asset: {regime: {hit_rate, avg_return, vol, sharpe,
                                                    n_observations, source}}},
            "diagnostics": [{asset, regime, prior, measured, calibrated, weight}],
        }
    """
    from app.services.scoring.engine import ASSET_REGIME_DATA  # local to break cycle

    p = params or CalibrationParams()

    rows = (
        db_session.query(RegimeClassification)
        .order_by(RegimeClassification.date.asc())
        .all()
    )
    probs_df = regime_probs_dataframe(rows)
    if probs_df.empty:
        raise ValueError("Nessuna classification in DB. Esegui /regime/backfill/historical.")

    out: dict[str, dict[str, dict[str, float | int | str]]] = {}
    diagnostics: list[dict[str, Any]] = []

    for asset in ASSET_TICKERS.keys():
        if asset not in ASSET_REGIME_DATA:
            continue
        try:
            metrics = metrics_by_regime(
                asset, probs_df,
                horizon_months=p.horizon_months, threshold=p.regime_threshold,
            )
        except Exception as e:
            logger.warning(f"Calibration {asset}: skip ({e})")
            continue

        m_by_regime = {m.regime: m for m in metrics}
        out[asset] = {}

        for regime in REGIMES:
            prior = ASSET_REGIME_DATA[asset].get(regime, {})
            if not prior:
                continue
            m = m_by_regime.get(regime)
            n_obs = m.n_observations if m else 0
            measured_present = bool(m and n_obs >= p.n_min)

            # Outlier filter
            if measured_present and _is_outlier(m.real_return, m.sharpe, p):
                logger.info(f"{asset}/{regime}: outlier scartato (n={n_obs}, ret={m.real_return:.2f}, sh={m.sharpe:.2f})")
                measured_present = False

            weight = _shrinkage_weight(n_obs, p.n_min, p.n_full) if measured_present else 0.0

            calibrated = {
                "hit_rate": _shrink(prior["hit_rate"], m.hit_rate if measured_present else None, weight),
                "avg_return": _shrink(prior["avg_return"], m.real_return if measured_present else None, weight),
                "vol": _shrink(prior["vol"], m.volatility if measured_present else None, weight),
                "sharpe": _shrink(prior["sharpe"], m.sharpe if measured_present else None, weight),
                "n_observations": n_obs,
                "source": "measured" if weight >= 1.0 else ("blended" if weight > 0 else "prior"),
            }
            out[asset][regime] = calibrated

            diagnostics.append({
                "asset": asset,
                "regime": regime,
                "n_observations": n_obs,
                "weight_measured": round(weight, 3),
                "prior": prior,
                "measured": {
                    "hit_rate": m.hit_rate if m else None,
                    "real_return": m.real_return if m else None,
                    "vol": m.volatility if m else None,
                    "sharpe": m.sharpe if m else None,
                } if m else None,
                "calibrated": {
                    "hit_rate": calibrated["hit_rate"],
                    "avg_return": calibrated["avg_return"],
                    "vol": calibrated["vol"],
                    "sharpe": calibrated["sharpe"],
                },
            })

    return {
        "version": 1,
        "calibrated_on": date.today().isoformat(),
        "params": asdict(p),
        "n_classifications": len(rows),
        "asset_regime_data": out,
        "diagnostics": diagnostics,
    }


def save_calibration(payload: dict[str, Any]) -> Path:
    """Persiste la calibrazione su seed/calibrated_asset_regime.json."""
    _CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CALIB_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info(f"Calibration salvata in {_CALIB_PATH}")
    return _CALIB_PATH


def load_calibration() -> dict[str, Any] | None:
    """Legge la calibrazione persistita. None se assente."""
    if not _CALIB_PATH.exists():
        return None
    try:
        return json.loads(_CALIB_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"Calibration load failed: {e}")
        return None
