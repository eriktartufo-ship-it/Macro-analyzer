"""Libro dei bot — quote e cassa in dollari (2026-11-15, Sessione 57, opzione A scelta da Erik).

Prima i bot avevano DUE contabilità che non si parlavano: la curva era `somma(peso × rendimento)`
con pesi che non seguivano il prezzo, i movimenti confrontavano il prezzo con un prezzo d'ingresso
grezzo. Una diceva +6%, l'altra perdeva. Qui c'è un libro solo, e curva e movimenti escono da lì.

Una sola fonte di verità: `bot_movements`. Posizioni, cassa e utile si RICAVANO rigiocando i
movimenti eseguiti (come `rb_movements` del libro reale). Le altre tabelle sono contorno:
- bot_books: quando è partito ogni libro, con che capitale, e il riassunto della curva archiviata.
- bot_intent_snapshot: l'ultimo stato del bot già tradotto in ordini (per vedere cosa è cambiato).
- bot_nav: il valore del libro a ogni chiusura di Wall Street, con la quadratura.
"""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Boolean, Date, DateTime, Float, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class BotBook(Base):
    __tablename__ = "bot_books"

    id: Mapped[int] = mapped_column(primary_key=True)
    strategy_type: Mapped[str] = mapped_column(String(16), nullable=False, unique=True)
    started_on: Mapped[date] = mapped_column(Date, nullable=False)
    initial_capital_usd: Mapped[float] = mapped_column(Float, nullable=False)
    # chiusura di partenza: benchmark e rendimenti si misurano da qui (prima chiusura vista)
    start_close: Mapped[date | None] = mapped_column(Date, nullable=True)
    # riassunto della curva archiviata alla ripartenza (JSON), per dirlo sulla pagina
    archive_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class BotMovement(Base):
    """Movimento del libro: acquisto, vendita, cedola o interesse sulla liquidità.

    Ciclo di un ordine: `pending` (deciso dopo la chiusura, in attesa dell'apertura) → `filled`
    (eseguito all'apertura successiva) | `cancelled` (il bot ha cambiato idea o non c'era niente da
    vendere) | `skipped` (niente liquidità). Cedole e interessi nascono già `filled`.
    """

    __tablename__ = "bot_movements"

    id: Mapped[int] = mapped_column(primary_key=True)
    strategy_type: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    asset_class: Mapped[str] = mapped_column(String(64), nullable=False)
    ticker: Mapped[str] = mapped_column(String(16), nullable=False)

    side: Mapped[str] = mapped_column(String(10), nullable=False)  # buy | sell | dividend | interest
    # tranche | floor | close_tranche | full_close | stop_loss | take_profit | cap_trim | switch |
    # riallineamento | dividend | interest
    kind: Mapped[str] = mapped_column(String(24), nullable=False)
    status: Mapped[str] = mapped_column(String(12), nullable=False, default="pending")

    # Intento (in pesi del bot): quanto doveva cambiare, e per le vendite che frazione della posizione
    weight_delta_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    sell_fraction: Mapped[float | None] = mapped_column(Float, nullable=True)
    amount_usd: Mapped[float | None] = mapped_column(Float, nullable=True)  # acquisti: dollari da spendere

    decided_on: Mapped[date] = mapped_column(Date, nullable=False)
    decided_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    fill_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)

    # Esecuzione — valori VERI del giorno dell'esecuzione (non si riscrivono dopo un frazionamento)
    quantity: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    open_usd: Mapped[float | None] = mapped_column(Float, nullable=True)        # apertura di quel giorno
    fill_price_usd: Mapped[float | None] = mapped_column(Float, nullable=True)  # apertura ± metà spread
    gross_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    fee_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    spread_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    cost_basis_sold_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    realized_pnl_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    cash_after_usd: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Perché
    decision_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    entry_regime: Mapped[str | None] = mapped_column(String(20), nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_bot_mov_strategy_status", "strategy_type", "status"),
    )


class BotIntentSnapshot(Base):
    """Ultimo stato del bot (pesi di intento) già tradotto in ordini, per (strategia, asset)."""

    __tablename__ = "bot_intent_snapshot"

    id: Mapped[int] = mapped_column(primary_key=True)
    strategy_type: Mapped[str] = mapped_column(String(16), nullable=False)
    asset_class: Mapped[str] = mapped_column(String(64), nullable=False)
    weight_pct: Mapped[float] = mapped_column(Float, nullable=False)
    tranches_filled: Mapped[int] = mapped_column(Integer, nullable=False)
    tranches_total: Mapped[int] = mapped_column(Integer, nullable=False)
    opened_at: Mapped[date] = mapped_column(Date, nullable=False)
    last_action: Mapped[str] = mapped_column(String(24), nullable=False)
    last_action_date: Mapped[date] = mapped_column(Date, nullable=False)
    # regime all'APERTURA della posizione: l'apprendimento lo usa quando la posizione è già chiusa
    entry_regime: Mapped[str | None] = mapped_column(String(20), nullable=True)

    __table_args__ = (
        UniqueConstraint("strategy_type", "asset_class", name="uq_bot_intent_strategy_asset"),
    )


class BotNav(Base):
    """Valore del libro alla chiusura di Wall Street `date`, con la quadratura dei conti."""

    __tablename__ = "bot_nav"

    id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    strategy_type: Mapped[str] = mapped_column(String(16), nullable=False)
    nav_usd: Mapped[float] = mapped_column(Float, nullable=False)
    cash_usd: Mapped[float] = mapped_column(Float, nullable=False)
    invested_usd: Mapped[float] = mapped_column(Float, nullable=False)
    n_positions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cumulative_return_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    peak_nav_usd: Mapped[float] = mapped_column(Float, nullable=False)
    drawdown_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # scomposizione: nav − capitale = realizzato + latente + cedole + interessi
    realized_cum_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    unrealized_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    dividends_cum_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    interest_cum_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    fees_cum_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    spread_cum_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    reconciled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    reconciliation_gap_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # perché i conti non sono verificati (scarto, movimento incoerente, cedola doppia, scala, continuità)
    reconciliation_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    bench_sp500_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    bench_acwi_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    bench_6040_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    positions_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("date", "strategy_type", name="uq_bot_nav_date_strategy"),
    )
