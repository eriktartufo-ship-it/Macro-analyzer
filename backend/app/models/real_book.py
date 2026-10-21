"""Libro reale — paper trading con strumenti VERI (richiesta Erik 2026-10-21).

I 3 bot dell'horserace decidono in % del NAV su ETF americani (GLD, XLE...), senza
orari, quote, commissioni né cambio. Il libro reale li SEGUE con un portafoglio che
un investitore italiano potrebbe tenere davvero: gemelli UCITS/ETC quotati in euro,
quote intere, ordine al mercato per l'apertura, commissione Scalable FREE, spread
stimato, cedole accreditate, cassa in euro. I 3 bot e il loro storico NON si toccano.

Una sola fonte di verità: `rb_movements`. Posizioni e cassa si RICAVANO da lì (nessun
secondo scrittore che possa divergere). Le altre tabelle sono contorno:
- rb_books: quando è partito ogni libro, con che capitale, da che punto si misura.
- rb_bot_snapshot: lo stato del bot che il libro ha già copiato (per vedere cosa è cambiato).
- rb_nav: il valore del libro a ogni chiusura di borsa.
"""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class RbBook(Base):
    """Un libro per strategia: data di partenza e punti di misura iniziali."""

    __tablename__ = "rb_books"

    id: Mapped[int] = mapped_column(primary_key=True)
    strategy_type: Mapped[str] = mapped_column(String(16), nullable=False, unique=True)
    started_on: Mapped[date] = mapped_column(Date, nullable=False)
    initial_capital_eur: Mapped[float] = mapped_column(Float, nullable=False)
    # NAV del bot "di carta" il giorno di partenza: serve a dire quanto costa la realtà
    bot_nav_start: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Benchmark in euro all'apertura del primo giorno eseguito: S&P 500 (VUAA), MSCI ACWI (IUSQ),
    # e la gamba obbligazionaria del 60/40 (IDTL)
    benchmark_start_price_eur: Mapped[float | None] = mapped_column(Float, nullable=True)
    acwi_start_price_eur: Mapped[float | None] = mapped_column(Float, nullable=True)
    bond_start_price_eur: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class RbMovement(Base):
    """Movimento del libro: acquisto, vendita o cedola. Log immutabile salvo lo stato.

    Ciclo di un ordine: `pending` (deciso, in attesa dell'apertura) → `filled`
    (eseguito al prezzo d'apertura) | `cancelled` (la strategia ha cambiato idea prima
    dell'apertura) | `skipped` (quantità zero, niente liquidità: si dice perché).
    """

    __tablename__ = "rb_movements"

    id: Mapped[int] = mapped_column(primary_key=True)
    strategy_type: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    asset_class: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    instrument_key: Mapped[str] = mapped_column(String(24), nullable=False)

    side: Mapped[str] = mapped_column(String(8), nullable=False)  # buy | sell | dividend
    # tranche | presa_in_carico | floor | close_tranche | full_close | stop_loss |
    # take_profit | riallineamento | dividend
    kind: Mapped[str] = mapped_column(String(24), nullable=False)
    status: Mapped[str] = mapped_column(String(12), nullable=False, default="pending")

    tranche_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tranches_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    weight_delta_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    sell_fraction: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Quando
    decided_on: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    decided_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    fill_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    fill_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)  # UTC

    # Quanto
    quantity: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    est_price_eur: Mapped[float | None] = mapped_column(Float, nullable=True)
    fill_price_eur: Mapped[float | None] = mapped_column(Float, nullable=True)
    gross_eur: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    fee_eur: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    spread_eur: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    realized_pnl_eur: Mapped[float | None] = mapped_column(Float, nullable=True)
    # tasse trattenute dal broker (regime amministrato) e movimenti dello zainetto
    tax_eur: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    zainetto_used_eur: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    zainetto_added_eur: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    cash_after_eur: Mapped[float | None] = mapped_column(Float, nullable=True)
    weight_after_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    position_qty_after: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_price_after_eur: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Perché (il "come" e il "quando" si compongono dai campi sopra, non dal testo)
    decision_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    facts_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    narrative: Mapped[str | None] = mapped_column(Text, nullable=True)
    narrative_source: Mapped[str | None] = mapped_column(String(96), nullable=True)
    exit_plan: Mapped[str | None] = mapped_column(Text, nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)  # skip/cancel/riduzioni

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_rb_mov_strategy_status", "strategy_type", "status"),
    )


class RbBotSnapshot(Base):
    """Ultimo stato del bot già copiato nel libro, per (strategia, asset)."""

    __tablename__ = "rb_bot_snapshot"

    id: Mapped[int] = mapped_column(primary_key=True)
    strategy_type: Mapped[str] = mapped_column(String(16), nullable=False)
    asset_class: Mapped[str] = mapped_column(String(64), nullable=False)
    current_weight_pct: Mapped[float] = mapped_column(Float, nullable=False)
    tranches_filled: Mapped[int] = mapped_column(Integer, nullable=False)
    tranches_total: Mapped[int] = mapped_column(Integer, nullable=False)
    opened_at: Mapped[date] = mapped_column(Date, nullable=False)
    last_action: Mapped[str] = mapped_column(String(24), nullable=False)
    last_action_date: Mapped[date] = mapped_column(Date, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("strategy_type", "asset_class", name="uq_rb_snap_strategy_asset"),
    )


class RbNav(Base):
    """Valore del libro alla chiusura di borsa (solo giorni con mercato aperto)."""

    __tablename__ = "rb_nav"

    id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    strategy_type: Mapped[str] = mapped_column(String(16), nullable=False)
    nav_eur: Mapped[float] = mapped_column(Float, nullable=False)
    cash_eur: Mapped[float] = mapped_column(Float, nullable=False)
    invested_eur: Mapped[float] = mapped_column(Float, nullable=False)
    n_positions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # lordo = come se le tasse non esistessero (tasse pagate rimesse in cassa, nessuna latente)
    # netto = valore se si vendesse tutto oggi (tasse pagate + tasse latenti sulle plusvalenze)
    return_gross_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    return_net_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    taxes_cum_eur: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    latent_tax_eur: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    zainetto_eur: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    bot_return_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    bench_sp_gross_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    bench_sp_net_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    bench_acwi_gross_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    bench_acwi_net_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    bench_6040_gross_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    bench_6040_net_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    fees_cum_eur: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    spread_cum_eur: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    dividends_cum_eur: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # posizioni valorizzate alla chiusura: la pagina le legge senza chiamare Yahoo
    positions_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("date", "strategy_type", name="uq_rb_nav_date_strategy"),
    )
