from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class MacroIndicator(Base):
    """Time series di indicatori macro raw e derivati."""

    __tablename__ = "macro_indicators"

    id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, comment="Data osservazione")
    series_id: Mapped[str] = mapped_column(
        String(50), nullable=False, comment="Codice FRED (es. GDP, CPIAUCSL)"
    )
    value: Mapped[float] = mapped_column(Float, nullable=False, comment="Valore raw")
    roc_3m: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Rate of change 3 mesi"
    )
    roc_6m: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Rate of change 6 mesi"
    )
    roc_12m: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Rate of change 12 mesi"
    )
    zscore_12m: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Z-score rolling 12 mesi"
    )
    zscore_36m: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Z-score rolling 36 mesi"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, comment="Ultimo aggiornamento"
    )

    __table_args__ = (
        Index("ix_macro_indicators_date_series", "date", "series_id", unique=True),
        Index("ix_macro_indicators_series_date", "series_id", "date"),
    )
