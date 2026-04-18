from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class SecularTrend(Base):
    """Indicatori dedollarizzazione e multipolarita (Fase 2)."""

    __tablename__ = "secular_trends"

    id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, comment="Data osservazione")
    trend_name: Mapped[str] = mapped_column(
        String(50), nullable=False,
        comment="Nome trend (dedollarization, multipolar_shift, energy_transition)"
    )
    score: Mapped[float] = mapped_column(
        Float, nullable=False,
        comment="Score trend 0-1 (intensita)"
    )
    components: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="JSON breakdown componenti del trend"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )

    __table_args__ = (
        Index("ix_secular_trends_date_name", "date", "trend_name", unique=True),
    )
