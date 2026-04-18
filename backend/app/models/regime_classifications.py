from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class RegimeClassification(Base):
    """Classificazione regime macro giornaliera/settimanale."""

    __tablename__ = "regime_classifications"

    id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, unique=True, comment="Data classificazione")
    regime: Mapped[str] = mapped_column(
        String(30), nullable=False,
        comment="Regime: reflation, stagflation, deflation, goldilocks"
    )
    probability_reflation: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0, comment="Probabilita regime reflation"
    )
    probability_stagflation: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0, comment="Probabilita regime stagflation"
    )
    probability_deflation: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0, comment="Probabilita regime deflation"
    )
    probability_goldilocks: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0, comment="Probabilita regime goldilocks"
    )
    confidence: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Confidence score 0-1 (concordanza condizioni)"
    )
    conditions_met: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="JSON con dettaglio condizioni soddisfatte"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, comment="Ultimo aggiornamento"
    )

    __table_args__ = (
        Index("ix_regime_classifications_date", "date"),
        Index("ix_regime_classifications_regime", "regime"),
    )
