from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class BacktestResult(Base):
    """Storico validazione walk-forward del modello."""

    __tablename__ = "backtest_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_date: Mapped[date] = mapped_column(
        Date, nullable=False, comment="Data esecuzione backtest"
    )
    period_start: Mapped[date] = mapped_column(
        Date, nullable=False, comment="Inizio periodo test"
    )
    period_end: Mapped[date] = mapped_column(
        Date, nullable=False, comment="Fine periodo test"
    )
    asset_class: Mapped[str] = mapped_column(
        String(50), nullable=False, comment="Asset class testato"
    )
    predicted_regime: Mapped[str] = mapped_column(
        String(30), nullable=False, comment="Regime predetto dal modello"
    )
    actual_return: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Rendimento effettivo nel periodo"
    )
    predicted_score: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Score assegnato dal modello"
    )
    hit: Mapped[int] = mapped_column(
        Integer, nullable=False,
        comment="1 se predizione corretta (outperformance), 0 altrimenti"
    )
    details: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="JSON dettagli backtest"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )

    __table_args__ = (
        Index("ix_backtest_results_run_date", "run_date"),
        Index("ix_backtest_results_asset_period", "asset_class", "period_start", "period_end"),
    )
