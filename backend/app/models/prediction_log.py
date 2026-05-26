"""T9-AUDIT-TA — Prediction log con realized returns.

Inspired da TradingAgents decision log: ogni allocation suggerita dal modello
viene loggata + 1/3/6 mesi dopo si confronta con returns realizzati.

Filosofia: "Se ascolti il modello, quanto guadagni REALE?" — la sola accuracy
non misura il P&L vero.

Schema:
- date_predicted: data classification + allocation suggerita
- regime: regime predicted at that date
- probabilities JSON: probs regime
- top5_assets JSON: lista [asset, weight] per top-5 allocation
- realized_return_1m/3m/6m: P&L cumulativo se avesse seguito allocation
- benchmark_return_1m/3m/6m: P&L di 60/40 portfolio nello stesso periodo
- alpha_1m/3m/6m: predicted - benchmark
- evaluated_at: timestamp ultimo evaluation
"""
from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class PredictionLog(Base):
    """Log allocation predictions + realized returns post-hoc."""

    __tablename__ = "prediction_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    date_predicted: Mapped[date] = mapped_column(Date, nullable=False, index=True)

    regime: Mapped[str] = mapped_column(String(20), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    probabilities_json: Mapped[str] = mapped_column(Text, nullable=False)

    # Top-5 asset suggested (JSON list of {asset, score, weight})
    top5_json: Mapped[str] = mapped_column(Text, nullable=False)

    # Allocation mode: "regime_based" | "defensive_transition"
    allocation_mode: Mapped[str] = mapped_column(String(30), nullable=False, default="regime_based")

    # Realized returns post-hoc (None se non ancora evaluato)
    realized_return_1m: Mapped[float | None] = mapped_column(Float, nullable=True)
    realized_return_3m: Mapped[float | None] = mapped_column(Float, nullable=True)
    realized_return_6m: Mapped[float | None] = mapped_column(Float, nullable=True)

    benchmark_return_1m: Mapped[float | None] = mapped_column(Float, nullable=True)
    benchmark_return_3m: Mapped[float | None] = mapped_column(Float, nullable=True)
    benchmark_return_6m: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Alpha vs benchmark
    alpha_1m: Mapped[float | None] = mapped_column(Float, nullable=True)
    alpha_3m: Mapped[float | None] = mapped_column(Float, nullable=True)
    alpha_6m: Mapped[float | None] = mapped_column(Float, nullable=True)

    evaluated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_prediction_log_date", "date_predicted"),
        Index("ix_prediction_log_regime", "regime"),
    )
