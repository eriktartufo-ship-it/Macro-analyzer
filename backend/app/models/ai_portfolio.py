"""AI Portfolio — paper trading autonomo daily con tranche DCA.

User request 2026-06-10: portafoglio con tutte le asset class, libertà
ingressi/uscite, position sizing via vol + indicatori, posizioni incrementali
(tranche), tab dedicata con tracking AI.

3 tabelle:
- ai_portfolio_positions: posizioni live (1 riga per asset attivo)
- ai_portfolio_decisions: log immutabile decisioni daily
- ai_portfolio_performance: snapshot NAV daily

Capital base: $100,000 paper. Tutte le metriche % del NAV corrente.
"""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class AiPortfolioPosition(Base):
    """Posizione corrente per asset (unique per asset)."""

    __tablename__ = "ai_portfolio_positions"

    id: Mapped[int] = mapped_column(primary_key=True)
    asset_class: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)

    # Sizing
    target_weight_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    current_weight_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    tranches_filled: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tranches_total: Mapped[int] = mapped_column(Integer, nullable=False, default=3)

    # Entry metadata
    opened_at: Mapped[date] = mapped_column(Date, nullable=False)
    entry_regime: Mapped[str] = mapped_column(String(20), nullable=False)
    avg_entry_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Real P&L tracking (Yahoo daily prices Fase 2)
    avg_entry_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    estimated_pnl_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    took_partial_tp: Mapped[bool] = mapped_column(default=False, nullable=False)

    last_action: Mapped[str] = mapped_column(String(24), nullable=False, default="OPEN_TRANCHE")
    last_action_date: Mapped[date] = mapped_column(Date, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow,
    )


class AiPortfolioDecision(Base):
    """Log immutabile decisioni AI daily (1 riga per asset per giorno se != HOLD)."""

    __tablename__ = "ai_portfolio_decisions"

    id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    asset_class: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # action enum: OPEN_TRANCHE | CLOSE_TRANCHE | FULL_CLOSE | HOLD | SKIP_NO_SIGNAL | STOP_LOSS | TAKE_PROFIT
    action: Mapped[str] = mapped_column(String(24), nullable=False)
    size_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    tranche_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tranches_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Context at decision time
    score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    regime: Mapped[str] = mapped_column(String(20), nullable=False)
    momentum_signal: Mapped[str] = mapped_column(String(16), nullable=False, default="NEUTRAL")
    vol_60d: Mapped[float | None] = mapped_column(Float, nullable=True)
    estimated_pnl_pct: Mapped[float | None] = mapped_column(Float, nullable=True)

    reason_short: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_ai_decisions_date_asset", "date", "asset_class"),
    )


class AiPortfolioPerformance(Base):
    """Snapshot NAV + metriche portafoglio (1 riga per data, unique)."""

    __tablename__ = "ai_portfolio_performance"

    id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, unique=True, index=True)

    nav: Mapped[float] = mapped_column(Float, nullable=False, default=100000.0)
    cumulative_return_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    drawdown_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    peak_nav: Mapped[float] = mapped_column(Float, nullable=False, default=100000.0)

    n_positions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    deployed_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    top_position_asset: Mapped[str | None] = mapped_column(String(64), nullable=True)
    top_position_weight_pct: Mapped[float | None] = mapped_column(Float, nullable=True)

    portfolio_vol_30d: Mapped[float | None] = mapped_column(Float, nullable=True)
    benchmark_60_40_return_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    benchmark_sp500_return_pct: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AiPortfolioLearning(Base):
    """Post-mortem learnings: pattern di decisioni con outcome aggregato.

    Update incrementale: ogni volta che una posizione viene chiusa con loss
    (STOP_LOSS o FULL_CLOSE in perdita), il learning service identifica il
    pattern (regime + asset_type + momentum) e aggiorna stats.

    Used by strategist per adattare entry thresholds dinamicamente.
    """

    __tablename__ = "ai_portfolio_learnings"

    id: Mapped[int] = mapped_column(primary_key=True)
    pattern_key: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)

    # Aggregated stats
    n_wins: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_losses: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_pnl_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    avg_pnl_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    win_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)

    # Recommendation: shift threshold (+/-) for entries matching this pattern
    entry_threshold_shift: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    last_pattern_seen_at: Mapped[date] = mapped_column(Date, nullable=False)

    # Human-readable insight (LLM generated, optional)
    insight_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow,
    )


class AiPortfolioReasoning(Base):
    """LLM reasoning cache per giorno: explanation + regime challenge.

    1 riga per data. Cache invalidata se data_hash cambia (logica simile
    a llm/macro_analyzer.py).
    """

    __tablename__ = "ai_portfolio_reasoning"

    id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, unique=True, index=True)
    data_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    # LLM output structured
    daily_summary: Mapped[str] = mapped_column(Text, nullable=False)
    decisions_reasoning_json: Mapped[str] = mapped_column(Text, nullable=False)  # {asset: reasoning_short}

    # Regime challenge
    regime_classifier_says: Mapped[str] = mapped_column(String(20), nullable=False)
    ai_agrees: Mapped[str] = mapped_column(String(16), nullable=False, default="high")  # high|medium|low
    ai_alternative_regime: Mapped[str | None] = mapped_column(String(20), nullable=True)
    regime_challenge_reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Metadata
    provider: Mapped[str] = mapped_column(String(48), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

