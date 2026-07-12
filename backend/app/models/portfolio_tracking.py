"""Portfolio Tracking (pt_*) — tracking manuale portafogli reali a prezzi di mercato.

User request 2026-07-12: sostituire l'Excel del fratello (oro/argento/BTC) con
tracking manuale + prezzi live + consigli macro buy-timing (NO trading), e vetrina
del portafoglio di Erik pilotata da Claude Code.

Modulo ADDITIVO dietro flag ENABLE_PORTFOLIO_TRACKING (default OFF): nessuna
interazione col modello macro / AI Portfolio. Namespace `pt_` per evitare
collisioni con ai_portfolio_* e /portfolio/allocation.

4 tabelle:
- pt_transactions: acquisti/vendite manuali (source of truth, posizioni derivate)
- pt_price_cache: cache spot per ticker (TTL breve, serve stale su errore fetch)
- pt_advice: output advisor Gemini settimanale/manuale (semaforo oro/argento/BTC)
- pt_strategy: JSON vetrina per-utente (Erik: push da Claude Code via admin token)
"""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class PtTransaction(Base):
    """Movimento manuale (buy/sell). Le posizioni si calcolano da qui (average cost)."""

    __tablename__ = "pt_transactions"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(32), nullable=False, index=True)

    # asset_key: "gold" | "silver" | "bitcoin" (preset) oppure ticker Yahoo uppercase
    asset_key: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    asset_class: Mapped[str] = mapped_column(String(16), nullable=False)  # metal|crypto|ticker
    label: Mapped[str] = mapped_column(String(64), nullable=False)
    ticker: Mapped[str] = mapped_column(String(24), nullable=False)  # simbolo Yahoo per pricing

    side: Mapped[str] = mapped_column(String(4), nullable=False, default="buy")  # buy|sell
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    # unit della quantity inserita: oz | g | btc | share
    unit: Mapped[str] = mapped_column(String(8), nullable=False, default="unit")
    # prezzo unitario pagato/incassato in EUR, nella stessa unit della quantity
    unit_price_eur: Mapped[float] = mapped_column(Float, nullable=False)
    fee_eur: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    trade_date: Mapped[date] = mapped_column(Date, nullable=False)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_pt_tx_user_date", "username", "trade_date"),
    )


class PtPriceCache(Base):
    """Ultimo spot noto per ticker. TTL breve in lettura; stale-if-error."""

    __tablename__ = "pt_price_cache"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(String(24), nullable=False, unique=True)
    price: Mapped[float] = mapped_column(Float, nullable=False)  # nella valuta nativa
    currency: Mapped[str] = mapped_column(String(8), nullable=False, default="USD")
    price_eur: Mapped[float] = mapped_column(Float, nullable=False)  # convertito EUR
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class PtAdvice(Base):
    """Snapshot consigli macro buy-timing (semaforo per asset). 1 riga per run."""

    __tablename__ = "pt_advice"

    id: Mapped[int] = mapped_column(primary_key=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    trigger: Mapped[str] = mapped_column(String(16), nullable=False, default="scheduled")  # scheduled|manual
    provider: Mapped[str] = mapped_column(String(24), nullable=False, default="gemini")
    content_json: Mapped[str] = mapped_column(Text, nullable=False)


class PtStrategy(Base):
    """Vetrina strategia per-utente (JSON libero). Per Erik la scrive Claude Code."""

    __tablename__ = "pt_strategy"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(32), nullable=False, unique=True)
    content_json: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow,
    )
