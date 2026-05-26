"""Tier 6.7 — Discretionary override engine.

Pattern Bridgewater: modello mechanical + judgment layer audit-trail.
Override discrezionali stored in DB con `expires_at` per auto-rollback.

Due tipi:
- `target_type="asset"`: delta additivo a `final_score(asset)` (range tipico ±30).
- `target_type="regime"`: shift additivo su `probs[regime]`, normalize Σ=1.

Solo override con `expires_at > now` sono applicati. Lazy table create
pattern (come T6.6) — no Alembic migration per ora.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from loguru import logger
from sqlalchemy.orm import Session

from app.models import DiscretionaryOverride


@dataclass
class OverrideApplied:
    """Record di un override effettivamente applicato (audit)."""
    id: int
    target_type: str
    target_name: str
    delta: float
    reason: str
    author: str
    expires_at: datetime


def _ensure_table_exists() -> None:
    """Lazy create tabella `discretionary_overrides`. Idempotente."""
    from app.database import engine
    try:
        DiscretionaryOverride.__table__.create(engine, checkfirst=True)
    except Exception as e:
        logger.warning(f"Lazy create discretionary_overrides failed: {e}")


def load_active_overrides(
    db: Session, now: datetime | None = None
) -> list[DiscretionaryOverride]:
    """Restituisce gli override con `expires_at > now` (default now=utcnow)."""
    _ensure_table_exists()
    current = now or datetime.utcnow()
    try:
        return (
            db.query(DiscretionaryOverride)
            .filter(DiscretionaryOverride.expires_at > current)
            .order_by(DiscretionaryOverride.created_at.desc())
            .all()
        )
    except Exception as e:
        logger.warning(f"Load active overrides failed: {e}")
        return []


def apply_asset_overrides(
    scores: dict[str, float],
    overrides: Iterable[DiscretionaryOverride],
) -> tuple[dict[str, float], list[OverrideApplied]]:
    """Applica override `target_type='asset'` ai score finali asset.

    - Score clamp [0, 100] dopo delta.
    - Asset non in `scores` ignorati silently.
    - Returns (new_scores, applied_list).
    """
    new_scores = dict(scores)
    applied: list[OverrideApplied] = []
    for ov in overrides:
        if ov.target_type != "asset":
            continue
        if ov.target_name not in new_scores:
            continue
        new_scores[ov.target_name] = max(
            0.0, min(100.0, new_scores[ov.target_name] + ov.delta)
        )
        applied.append(
            OverrideApplied(
                id=ov.id,
                target_type=ov.target_type,
                target_name=ov.target_name,
                delta=ov.delta,
                reason=ov.reason,
                author=ov.author,
                expires_at=ov.expires_at,
            )
        )
    return new_scores, applied


def apply_regime_overrides(
    probs: dict[str, float],
    overrides: Iterable[DiscretionaryOverride],
) -> tuple[dict[str, float], list[OverrideApplied]]:
    """Applica override `target_type='regime'` alle probs regime.

    Semantica (T7.4): la massa rimossa/aggiunta dal target viene ridistribuita
    SOLO sui regimi non-target proporzionalmente alle loro prob originali.
    In questo modo un override su un regime NON gonfia silentemente gli altri
    che il PM non voleva toccare.

    - Floor 0.0001 sul target post-delta.
    - Massa rimossa = old[target] - new[target] (può essere negativa = injection).
    - Ridistribuzione: new[other] = old[other] * (1 - mass_removed/sum_others).
      Se sum_others = 0 (edge case) ridistribuzione uniforme tra non-target.
    """
    new_probs = dict(probs)
    applied: list[OverrideApplied] = []
    targets_touched: list[str] = []
    for ov in overrides:
        if ov.target_type != "regime":
            continue
        if ov.target_name not in new_probs:
            continue
        target = ov.target_name
        old_target = new_probs[target]
        new_target = max(0.0001, old_target + ov.delta)
        mass_removed = old_target - new_target  # positivo se delta < 0
        new_probs[target] = new_target
        # Ridistribuisci la massa SOLO sui non-target
        non_targets = [k for k in new_probs if k != target]
        sum_others = sum(new_probs[k] for k in non_targets)
        if sum_others > 0:
            for k in non_targets:
                new_probs[k] += mass_removed * (new_probs[k] / sum_others)
        elif non_targets:
            equal_share = mass_removed / len(non_targets)
            for k in non_targets:
                new_probs[k] += equal_share
        targets_touched.append(target)
        applied.append(
            OverrideApplied(
                id=ov.id,
                target_type=ov.target_type,
                target_name=target,
                delta=ov.delta,
                reason=ov.reason,
                author=ov.author,
                expires_at=ov.expires_at,
            )
        )
    # Floor di sicurezza + renormalize finale per gestire effetti compositi
    # di multiple overrides + clamp negativi creati da grandi mass_removed.
    for k in new_probs:
        new_probs[k] = max(0.0001, new_probs[k])
    total = sum(new_probs.values())
    if total > 0:
        new_probs = {k: v / total for k, v in new_probs.items()}
    return new_probs, applied


def create_override(
    db: Session,
    target_type: str,
    target_name: str,
    delta: float,
    expires_at: datetime,
    reason: str,
    author: str = "anonymous",
) -> DiscretionaryOverride:
    """Crea nuovo override + persiste in DB."""
    if target_type not in ("asset", "regime"):
        raise ValueError(f"target_type must be 'asset' or 'regime', got '{target_type}'")
    if not reason or not reason.strip():
        raise ValueError("reason is required (audit trail)")
    _ensure_table_exists()
    record = DiscretionaryOverride(
        target_type=target_type,
        target_name=target_name,
        delta=float(delta),
        expires_at=expires_at,
        reason=reason.strip(),
        author=author.strip() or "anonymous",
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def delete_override(db: Session, override_id: int) -> bool:
    """Cancella override per id. Returns True se trovato e rimosso."""
    _ensure_table_exists()
    record = db.query(DiscretionaryOverride).filter(
        DiscretionaryOverride.id == override_id
    ).first()
    if record is None:
        return False
    db.delete(record)
    db.commit()
    return True
