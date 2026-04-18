from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class NewsSignal(Base):
    """Segnali estratti da notizie via Claude API (Fase 2)."""

    __tablename__ = "news_signals"

    id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, comment="Data notizia")
    source: Mapped[str] = mapped_column(
        String(100), nullable=False, comment="Fonte (es. reuters, bloomberg, youtube)"
    )
    title: Mapped[str] = mapped_column(
        String(500), nullable=False, comment="Titolo o descrizione"
    )
    content_summary: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="Riassunto contenuto (Claude API)"
    )
    sentiment_score: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0,
        comment="Score sentiment -1 (bearish) a +1 (bullish)"
    )
    relevance_score: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0,
        comment="Rilevanza per macro analysis 0-1"
    )
    affected_assets: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="JSON lista asset class impattati"
    )
    decay_weight: Mapped[float] = mapped_column(
        Float, nullable=False, default=1.0,
        comment="Peso con decay esponenziale (dimezza ogni 7gg)"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )

    __table_args__ = (
        Index("ix_news_signals_date", "date"),
        Index("ix_news_signals_source_date", "source", "date"),
    )
