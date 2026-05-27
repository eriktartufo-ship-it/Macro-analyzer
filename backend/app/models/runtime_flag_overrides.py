"""T11+ Runtime flag persistence — survives docker compose restart.

Salva runtime overrides settati via POST /api/v1/config/flags in DB. Al
startup del lifespan, `app.main` carica overrides via `load_runtime_flags_from_db()`
e li applica via `set_runtime_flag()`. Quindi i flag attivati da UI rimangono
attivi tra restart container.

Senza questo, il container Docker perdeva tutte le runtime activations a ogni
deploy → l'utente doveva ri-attivare manualmente (es. USE_MOMENTUM_PILLARS).
"""
from datetime import datetime

from sqlalchemy import Boolean, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class RuntimeFlagOverride(Base):
    """Override persistente per un singolo flag config.

    PK: flag_name. Upsert per ogni POST /config/flags. updated_at audit trail.
    """
    __tablename__ = "runtime_flag_overrides"

    flag_name: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[bool] = mapped_column(Boolean, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return f"<RuntimeFlagOverride({self.flag_name}={self.value})>"
