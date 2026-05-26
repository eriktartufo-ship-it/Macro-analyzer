"""Tier 6.7 — Discretionary override layer.

Audit-trail di override discrezionali applicati al scoring engine. Pattern
Bridgewater: il modello è 100% mechanical, ma occasionalmente un PM ha
un'opinione (es. "in questo regime preferisco -10 score su bonds_long").

Schema:
- `target_type`: "asset" (modifica score finale) | "regime" (shift prob)
- `target_name`: ticker asset (`us_equities_growth`) o regime (`reflation`).
- `delta`: float, additivo. Range tipico [-30, +30] per asset, [-0.20, +0.20] per regime.
- `expires_at`: timestamp UTC. Dopo questa data l'override è inattivo.
- `reason`: text obbligatorio (per audit).
- `author`: text (default "anonymous" per CLI/test).

Pattern lazy create table come T6.6 (no Alembic migration formale).
"""
from datetime import datetime

from sqlalchemy import DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class DiscretionaryOverride(Base):
    """Override discrezionale applicabile a scoring asset o probs regime."""

    __tablename__ = "discretionary_overrides"

    id: Mapped[int] = mapped_column(primary_key=True)
    target_type: Mapped[str] = mapped_column(
        String(20), nullable=False, comment="'asset' | 'regime'"
    )
    target_name: Mapped[str] = mapped_column(
        String(64), nullable=False, comment="asset class o regime name"
    )
    delta: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Additivo (score points o prob delta)"
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, comment="UTC. Override attivo se now < expires_at"
    )
    reason: Mapped[str] = mapped_column(
        Text, nullable=False, comment="Audit trail: motivazione discrezionale"
    )
    author: Mapped[str] = mapped_column(
        String(64), nullable=False, default="anonymous"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )

    __table_args__ = (
        Index("ix_discretionary_overrides_active", "expires_at"),
        Index("ix_discretionary_overrides_target", "target_type", "target_name"),
    )
