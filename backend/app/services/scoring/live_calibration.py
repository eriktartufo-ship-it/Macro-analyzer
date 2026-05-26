"""Tier 6.6 — Live calibration `asset_regime_data` versionata in DB.

Sostituisce `seed/calibrated_asset_regime.json` (file statico) con DB
table `asset_regime_calibrations` (snapshot versionati + history audit).

**Pattern**:

1. Scheduler mensile chiama `calibrate_and_persist(db)`.
2. Riusa `services/scoring/calibration.calibrate(db)` esistente (Bayesian
   shrinkage contro rendimenti storici).
3. Persiste nuovo `AssetRegimeCalibration` record, mark `is_active=True`,
   deactivate snapshots precedenti.
4. `load_latest_from_db(db)` ritorna ultimo snapshot attivo.
5. `_calibrated_or_prior()` in scoring/engine.py: se `use_live_calibration()`
   ON, prevale lookup DB sopra JSON sopra hardcoded prior.

**Differenze vs JSON legacy (USE_CALIBRATED_SCORING)**:

- DB versionato: history snapshot mensili (audit trail + rollback).
- Auto-refresh: scheduler job mensile (no manual save_calibration).
- Query-able: SELECT history per A/B testing OOS.
- Avg Brier delta vs prior persistito (tracciamento miglioramento).

**Indipendente** da `USE_CALIBRATED_SCORING`: se entrambi ON, T6.6 prevale.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from loguru import logger
from sqlalchemy.orm import Session

from app.models import AssetRegimeCalibration


@dataclass
class LiveCalibrationResult:
    """Snapshot in-memory di una calibration appena calcolata."""
    calibrated_on: date
    version: str
    n_classifications: int
    n_assets: int
    payload: dict[str, Any]


def _ensure_table_exists(db: Session) -> None:
    """Garantisce che la table `asset_regime_calibrations` esista (lazy create).

    Pattern pragmatico vs Alembic migration (TODO follow-up).
    Idempotente: checkfirst=True evita errori se table già presente.
    """
    from app.database import engine
    try:
        AssetRegimeCalibration.__table__.create(engine, checkfirst=True)
    except Exception as e:
        logger.warning(f"Lazy create asset_regime_calibrations failed: {e}")


def calibrate_and_persist(db: Session) -> LiveCalibrationResult | None:
    """Calcola calibration via shrinkage Bayesiano + persiste in DB.

    Riusa `services/scoring/calibration.calibrate` (logica esistente).
    Marca questo snapshot `is_active=True`, deactivate gli altri.

    Returns:
        LiveCalibrationResult o None se calibration fallisce.
    """
    _ensure_table_exists(db)

    try:
        from app.services.scoring.calibration import calibrate
        payload = calibrate(db)
    except Exception as e:
        logger.warning(f"Calibration compute failed: {e}")
        return None

    if not payload or "asset_regime_data" not in payload:
        logger.warning("Calibration empty payload")
        return None

    asset_regime_data = payload.get("asset_regime_data", {})
    n_classifications = int(payload.get("n_classifications", 0))
    n_assets = len(asset_regime_data)
    today = date.today()
    version = datetime.utcnow().isoformat()

    # Deactivate snapshot precedenti
    try:
        db.query(AssetRegimeCalibration).filter(
            AssetRegimeCalibration.is_active == True  # noqa: E712
        ).update({"is_active": False})
    except Exception as e:
        logger.warning(f"Deactivate previous snapshots failed: {e}")

    # Insert nuovo snapshot
    record = AssetRegimeCalibration(
        calibrated_on=today,
        version=version,
        n_classifications=n_classifications,
        n_assets=n_assets,
        avg_brier_delta=None,  # TODO: compute Brier delta vs prior
        is_active=True,
        payload=json.dumps(payload, default=str),
    )
    try:
        db.add(record)
        db.commit()
        logger.info(
            f"Live calibration persisted: version={version} "
            f"n_classifications={n_classifications} n_assets={n_assets}"
        )
    except Exception as e:
        db.rollback()
        logger.warning(f"Calibration persist failed: {e}")
        return None

    return LiveCalibrationResult(
        calibrated_on=today,
        version=version,
        n_classifications=n_classifications,
        n_assets=n_assets,
        payload=payload,
    )


def load_latest_from_db(db: Session) -> dict[str, Any] | None:
    """Restituisce ultimo snapshot calibration attivo da DB.

    Returns:
        payload dict (asset_regime_data + diagnostics + params) o None
        se nessun snapshot disponibile.
    """
    _ensure_table_exists(db)

    try:
        latest = (
            db.query(AssetRegimeCalibration)
            .filter(AssetRegimeCalibration.is_active == True)  # noqa: E712
            .order_by(AssetRegimeCalibration.created_at.desc())
            .first()
        )
    except Exception as e:
        logger.warning(f"Load latest calibration failed: {e}")
        return None

    if latest is None or not latest.payload:
        return None

    try:
        return json.loads(latest.payload)
    except (ValueError, TypeError) as e:
        logger.warning(f"Calibration payload JSON parse failed: {e}")
        return None


def list_snapshots(db: Session, limit: int = 24) -> list[dict[str, Any]]:
    """Lista snapshot history per audit + UI tile (ultimi `limit` records)."""
    _ensure_table_exists(db)
    try:
        rows = (
            db.query(AssetRegimeCalibration)
            .order_by(AssetRegimeCalibration.created_at.desc())
            .limit(limit)
            .all()
        )
    except Exception:
        return []
    return [
        {
            "id": r.id,
            "calibrated_on": str(r.calibrated_on),
            "version": r.version,
            "n_classifications": r.n_classifications,
            "n_assets": r.n_assets,
            "avg_brier_delta": r.avg_brier_delta,
            "is_active": r.is_active,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]
