"""CRITICAL #2 council 2026-05-27: lock model snapshots per true OOS validation.

Ogni prediction in PredictionLog ha `model_version` FK qui. Questo snapshot
documenta lo state ESATTO del modello al momento della prediction:
- versione (stringa human-readable, es. "v1.0-sealed-2026-05-27")
- flag state (JSON di get_all_flags_state())
- classifier_hash (md5 di REGIME_CONDITIONS dict)
- locked_at timestamp
- description (note manuali)

Filosofia: per validare true OOS, devi sapere QUALE modello ha fatto QUALE
prediction. Senza snapshot, refactor classifier invalida tutto lo storico
predictions (perché non si può più ricostruire cosa ha visto quel giorno).
"""
from datetime import datetime

from sqlalchemy import DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class ModelSnapshot(Base):
    """Snapshot freezato di un modello specifico, usato come reference dalle
    predictions logged via PredictionLog.model_version.
    """
    __tablename__ = "model_snapshots"

    version: Mapped[str] = mapped_column(String(64), primary_key=True)
    locked_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    flags_state_json: Mapped[str] = mapped_column(Text, nullable=False)
    classifier_hash: Mapped[str] = mapped_column(String(32), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<ModelSnapshot({self.version}, locked_at={self.locked_at})>"
