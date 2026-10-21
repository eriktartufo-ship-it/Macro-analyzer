"""Contabilità del libro reale — funzioni PURE (niente DB, niente rete).

Regole, scritte una volta sola:
- quote intere (un ETF non si compra a frazioni su Xetra);
- prezzo eseguito = apertura ± metà spread (si paga la lettera, si incassa il denaro);
- commissione fissa per ordine (Scalable FREE, scelta Erik 2026-10-21);
- costo medio di carico COMPRENSIVO di commissioni (come lo calcola il fisco italiano);
- niente leva: un acquisto che non sta nella cassa si riduce, non va in negativo
  ([[feedback_no_leverage_margin]]).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

FEE_PER_ORDER_EUR = 0.99  # Scalable Capital, piano FREE (scelta Erik 2026-10-21)
INITIAL_CAPITAL_EUR = 100_000.0


@dataclass
class Lot:
    """Posizione di UNA asset class di una strategia (due asset possono condividere lo strumento)."""

    instrument_key: str
    qty: float = 0.0
    cost_eur: float = 0.0  # costo di carico residuo, commissioni incluse

    @property
    def avg_price(self) -> float | None:
        return self.cost_eur / self.qty if self.qty > 0 else None


@dataclass
class Fill:
    qty: int
    price: float      # prezzo eseguito (spread incluso)
    gross: float      # qty × price
    spread: float     # quanto dello `gross` è spread (informativo)
    fee: float
    note: str | None = None


@dataclass
class LedgerState:
    cash: float
    lots: dict[str, Lot] = field(default_factory=dict)
    fees: float = 0.0
    spread: float = 0.0
    dividends: float = 0.0


def fill_price(open_price: float, half_spread: float, side: str) -> float:
    return open_price * (1 + half_spread) if side == "buy" else open_price * (1 - half_spread)


def size_buy(amount_eur: float, est_price: float, half_spread: float, available_cash: float,
             fee: float = FEE_PER_ORDER_EUR) -> tuple[int, str | None]:
    """Quote da comprare per spendere ~`amount_eur`. Arrotonda al più vicino (non per
    difetto: per difetto il libro resterebbe SEMPRE sotto-investito di mezza quota),
    poi taglia se non sta nella cassa. Ritorna (quote, nota)."""
    if est_price <= 0 or amount_eur <= 0:
        return 0, "importo o prezzo non validi"
    unit = est_price * (1 + half_spread)
    qty = int(round(amount_eur / unit))
    note = None
    max_affordable = int(math.floor((available_cash - fee) / unit)) if available_cash > fee else 0
    if qty > max_affordable:
        qty = max(0, max_affordable)
        note = "ridotta: la liquidità disponibile non bastava"
    if qty <= 0:
        return 0, note or f"una quota costa {unit:.2f} €, più della tranche ({amount_eur:.2f} €)"
    return qty, note


def execute_buy(qty: int, open_price: float, half_spread: float, cash: float,
                fee: float = FEE_PER_ORDER_EUR) -> Fill | None:
    """Esegue all'apertura. Se l'apertura è salita oltre la cassa, riduce le quote."""
    price = fill_price(open_price, half_spread, "buy")
    note = None
    if qty * price + fee > cash:
        qty = int(math.floor((cash - fee) / price)) if cash > fee else 0
        note = "ridotta all'apertura: il prezzo era salito oltre la liquidità"
    if qty <= 0:
        return None
    return Fill(qty=qty, price=price, gross=qty * price,
                spread=qty * open_price * half_spread, fee=fee, note=note)


def execute_sell(qty: float, open_price: float, half_spread: float,
                 fee: float = FEE_PER_ORDER_EUR) -> Fill | None:
    if qty <= 0:
        return None
    price = fill_price(open_price, half_spread, "sell")
    return Fill(qty=int(qty), price=price, gross=int(qty) * price,
                spread=int(qty) * open_price * half_spread, fee=fee)


def sell_qty(lot_qty: float, fraction: float | None) -> int:
    """Quote da vendere: tutto (fraction None/≥1) o una frazione arrotondata, mai zero se c'è
    qualcosa da vendere e la frazione è > 0."""
    if lot_qty <= 0:
        return 0
    if fraction is None or fraction >= 0.999:
        return int(lot_qty)
    q = int(round(lot_qty * fraction))
    return min(int(lot_qty), max(1, q)) if fraction > 0 else 0


def apply_buy(state: LedgerState, asset: str, instrument_key: str, f: Fill) -> Lot:
    lot = state.lots.setdefault(asset, Lot(instrument_key=instrument_key))
    lot.qty += f.qty
    lot.cost_eur += f.gross + f.fee
    state.cash -= f.gross + f.fee
    state.fees += f.fee
    state.spread += f.spread
    return lot


def apply_sell(state: LedgerState, asset: str, f: Fill) -> tuple[Lot, float]:
    """Ritorna (lotto dopo, utile realizzato netto commissioni di vendita)."""
    lot = state.lots[asset]
    avg = lot.avg_price or 0.0
    qty = min(f.qty, int(lot.qty))
    proceeds = qty * f.price - f.fee
    realized = proceeds - avg * qty
    lot.qty -= qty
    lot.cost_eur -= avg * qty
    if lot.qty <= 0:
        lot.qty, lot.cost_eur = 0.0, 0.0
    state.cash += proceeds
    state.fees += f.fee
    state.spread += f.spread
    return lot, realized


def apply_dividend(state: LedgerState, amount: float) -> None:
    state.cash += amount
    state.dividends += amount


def nav(state: LedgerState, prices: dict[str, float]) -> tuple[float, float]:
    """(NAV, investito) con `prices` per instrument_key. Uno strumento senza prezzo vale
    il suo costo di carico: meglio un valore vecchio dichiarato che zero."""
    invested = 0.0
    for lot in state.lots.values():
        if lot.qty <= 0:
            continue
        p = prices.get(lot.instrument_key)
        invested += lot.qty * p if p else lot.cost_eur
    return state.cash + invested, invested
