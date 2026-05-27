"""CRITICAL #2 council 2026-05-27: model snapshot lock service.

Lock filosofia: per validare true OOS, ogni prediction deve essere attribuibile
a uno STATO ESATTO del modello. Snapshot freezato in DB con:
- version: stringa human-readable (es. "v1.0-sealed-2026-05-27")
- flag state: dict di tutti i 27 flag opt-in + 3 default-ON al momento
- classifier_hash: md5 di REGIME_CONDITIONS dict (cattura tweak ai pesi/threshold)
- locked_at + description

API:
- `lock_current_model(version, description)`: salva snapshot CURRENT state.
- `get_active_version()`: ritorna ultimo snapshot lockato, fallback "unsealed".
- `compute_classifier_hash()`: md5 dei REGIME_CONDITIONS (detect drift).

Usato da `prediction_logger.log_prediction()` per attribuire model_version FK.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from loguru import logger
from sqlalchemy.orm import Session


def _ensure_snapshot_table_exists() -> None:
    """Lazy create table (pattern T11+)."""
    from app.database import engine
    from app.models import ModelSnapshot
    try:
        ModelSnapshot.__table__.create(engine, checkfirst=True)
    except Exception as e:
        logger.warning(f"Lazy create model_snapshots failed: {e}")


def compute_classifier_hash() -> str:
    """md5(REGIME_CONDITIONS dict) — cattura tutti i pesi/threshold del classifier.

    Se i pesi cambiano (es. T11 rebalance), hash cambia → snapshot diverso.
    """
    from app.services.regime.classifier import REGIME_CONDITIONS

    # Stable serialization (sort keys)
    serialized = json.dumps(REGIME_CONDITIONS, sort_keys=True, default=str)
    return hashlib.md5(serialized.encode()).hexdigest()


def lock_current_model(
    version: str,
    description: str | None = None,
    db: Session | None = None,
) -> dict:
    """Lock state corrente del modello come snapshot freezato.

    Args:
        version: identifier human-readable (es. "v1.0-sealed-2026-05-27").
        description: note manuali optional.
        db: SQLAlchemy session optional (creata internamente se None).

    Returns:
        dict con version, locked_at, classifier_hash, flags_count.
    """
    _ensure_snapshot_table_exists()

    from app.database import SessionLocal
    from app.models import ModelSnapshot
    from app.services.config_flags import get_all_flags_state

    flags = get_all_flags_state()
    flags_json = json.dumps(flags, default=str)
    chash = compute_classifier_hash()

    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True

    try:
        existing = db.query(ModelSnapshot).filter_by(version=version).first()
        if existing is not None:
            # Idempotent: refresh i campi se chiamato di nuovo con stessa versione
            existing.flags_state_json = flags_json
            existing.classifier_hash = chash
            if description:
                existing.description = description
        else:
            snapshot = ModelSnapshot(
                version=version,
                locked_at=datetime.now(timezone.utc),
                flags_state_json=flags_json,
                classifier_hash=chash,
                description=description,
            )
            db.add(snapshot)
        db.commit()
        logger.info(f"Model snapshot LOCKED: {version} (classifier_hash={chash[:8]}...)")
        return {
            "version": version,
            "locked_at": datetime.now(timezone.utc).isoformat(),
            "classifier_hash": chash,
            "flags_count": len(flags),
            "n_default_on": sum(1 for v in flags.values() if v.get("value")),
        }
    finally:
        if close_db:
            db.close()


def get_active_version(db: Session | None = None) -> str:
    """Ritorna versione del modello ATTIVO (ultimo snapshot lockato).

    Se nessuno snapshot esiste, ritorna "unsealed" (predictions logged senza
    sealing, non OOS-valide ma utili per shadow mode iniziale).
    """
    _ensure_snapshot_table_exists()

    from app.database import SessionLocal
    from app.models import ModelSnapshot

    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True

    try:
        latest = db.query(ModelSnapshot).order_by(ModelSnapshot.locked_at.desc()).first()
        return latest.version if latest else "unsealed"
    finally:
        if close_db:
            db.close()


def list_snapshots(db: Session | None = None) -> list[dict]:
    """Lista tutti gli snapshot lockati (per audit/debug)."""
    _ensure_snapshot_table_exists()

    from app.database import SessionLocal
    from app.models import ModelSnapshot

    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True

    try:
        rows = db.query(ModelSnapshot).order_by(ModelSnapshot.locked_at.desc()).all()
        return [
            {
                "version": r.version,
                "locked_at": r.locked_at.isoformat() if r.locked_at else None,
                "classifier_hash": r.classifier_hash,
                "description": r.description,
            }
            for r in rows
        ]
    finally:
        if close_db:
            db.close()
