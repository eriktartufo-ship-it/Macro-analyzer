from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class DailySignal(Base):
    """Output finale del modello: score giornaliero per asset class."""

    __tablename__ = "daily_signals"

    id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, comment="Data segnale")
    asset_class: Mapped[str] = mapped_column(
        String(50), nullable=False, comment="Asset class"
    )
    final_score: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Score finale 0-100"
    )
    regime_component: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Componente regime-based"
    )
    secular_trend_bonus: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0,
        comment="Bonus trend secolare (dedollarizzazione, fase 2)"
    )
    news_signal: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0,
        comment="Segnale news con decay esponenziale 7gg (fase 2)"
    )
    momentum_penalty: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0,
        comment="Penalita se asset sopravvalutato"
    )
    breakdown: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="JSON breakdown componenti per debug"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )

    __table_args__ = (
        Index("ix_daily_signals_date_asset", "date", "asset_class", unique=True),
        Index("ix_daily_signals_date", "date"),
    )
