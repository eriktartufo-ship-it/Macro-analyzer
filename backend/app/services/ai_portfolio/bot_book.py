"""Libro dei bot in quote e cassa (2026-11-15, Sessione 57 — opzione A scelta da Erik).

IL DIFETTO CHE CHIUDE. Fino al 15/11 i bot avevano due contabilità che non si parlavano:
- la curva = somma(peso × rendimento), con pesi che non seguivano il prezzo, calcolata DOPO le
  decisioni del giorno: un acquisto incassava la seduta prima di essere comprato, una vendita
  saltava quella in cui era ancora in mano (bot LLM: ~+2,2 pp di curva non guadagnati in 10 settimane);
- i movimenti = prezzo contro un prezzo d'ingresso grezzo: il frazionamento 4:1 di QUAL (05/11/2026)
  è diventato un -75% e uno STOP_LOSS su una posizione sana;
- il «minimo investito» aggiungeva peso senza lasciare un movimento.

COME FUNZIONA ORA (un solo libro; curva e movimenti escono da qui):
1. Le strategie restano com'erano: decidono in PESI di intento su `AiPortfolioPosition`.
2. `create_orders` confronta l'intento con l'ultimo stato già ordinato (`bot_intent_snapshot`) e
   scrive un ORDINE per ogni differenza (anche floor e tetto: nessun peso cambia senza movimento).
   Gli acquisti non superano la cassa che ci sarà (cassa + vendite dello stesso giro); la parte di
   un acquisto saltata o ridotta all'apertura torna da ordinare.
3. `settle`, al giro dopo, esegue TUTTI gli ordini di una decisione nella stessa seduta di Wall
   Street, la prima dopo la decisione: ETF all'apertura, cripto alla chiusura della barra di quel
   giorno (che apre a mezzanotte UTC, prima della decisione). Nella stessa seduta: cedole, vendite,
   acquisti, interessi — lo STESSO ordine con cui `replay` ricostruisce lo stato.
4. Valore = cassa + quote × chiusura. Le quote si riportano alla scala di prezzo di oggi con il
   rapporto fra l'apertura salvata il giorno dell'esecuzione e la stessa apertura come la dà Yahoo
   adesso: dopo un frazionamento 4:1 quel rapporto vale 4. Se quell'apertura manca, la scala NON si
   indovina: i conti si dichiarano non verificabili e il bot non decide.
5. Quadratura a ogni giro, entro 1 $: valore − capitale = utile realizzato (salvato sui movimenti)
   + latente + cedole + interessi. In più quattro controlli che la quadratura da sola non vede:
   ogni movimento coerente (importo = quantità × prezzo), nessuna cedola doppia, scala verificabile,
   e il valore di ogni posizione rimasta ferma uguale a quello della chiusura prima × il rendimento
   del suo prezzo. Se uno fallisce, quel bot NON decide (il giro si ferma e lo dice).

Semplificazioni dichiarate: quote frazionarie (conto in dollari, niente arrotondamenti), commissione
fissa 1 $ per ordine, metà spread 2 bp sugli ETF e 10 bp sulle cripto, interessi solo sulla
liquidità rimasta ferma fra due chiusure, al rendimento di BIL (flag
`USE_AI_PORTFOLIO_CASH_OVERNIGHT`), niente tasse (le simula il libro reale in euro).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models.ai_portfolio import AiPortfolioDecision, AiPortfolioPosition
from app.models.bot_book import BotBook, BotIntentSnapshot, BotMovement, BotNav
from app.services.real_book.market import close_on, first_bar_on_or_after, last_close_on_or_before

logger = logging.getLogger(__name__)

STRATEGIES = ("model_driven", "llm_driven", "risk_managed_sp")
INITIAL_CAPITAL_USD = 100_000.0
FEE_PER_ORDER_USD = 1.0
MIN_ORDER_USD = 50.0           # sotto questa cifra un acquisto non si ordina (costerebbe solo commissioni)
DEFAULT_HALF_SPREAD = 0.0002   # 2 bp: ETF USA liquidi
CRYPTO_TICKERS = {"BTC-USD", "ETH-USD"}
HALF_SPREAD = {"BTC-USD": 0.0010, "ETH-USD": 0.0010}
CASH_TICKER = "BIL"
CALENDAR_TICKER = "SPY"        # le sedute di Wall Street le dà l'S&P
RECONCILE_TOLERANCE_USD = 1.0
CONTINUITY_TOLERANCE = 0.005   # valore di una posizione ferma vs rendimento del suo prezzo
# Un ETF largo non si muove del 40% in una seduta (2020: S&P −12% nel giorno peggiore). Un salto così
# è quasi sempre un frazionamento che Yahoo mostra sulla barra nuova prima di riscalare lo storico:
# lo si dichiara e quel giorno il bot non decide, invece di far scattare uno stop loss su un dato falso.
MAX_ETF_SESSION_MOVE = 0.40
SPLIT_FACTORS = (1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 20.0, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 10)
EPS_W = 0.0005                 # differenze di peso più piccole si ACCUMULANO, non diventano ordini
# stesso ordine in esecuzione e nel rigioco: cedole (a chi aveva le quote la sera prima), vendite
# (liberano cassa), acquisti, interessi (sulla liquidità rimasta ferma)
EVENT_RANK = {"dividend": 0, "sell": 1, "buy": 2, "interest": 3}
# vendite che chiudono una posizione: alimentano l'apprendimento con l'utile REALIZZATO
FULL_EXIT_KINDS = ("full_close", "stop_loss", "take_profit")
LEARNING_STRATEGIES = ("model_driven", "llm_driven")
SELL_KIND = {"FULL_CLOSE": "full_close", "STOP_LOSS": "stop_loss", "TAKE_PROFIT": "take_profit",
             "CLOSE_TRANCHE": "close_tranche", "CAP_TRIM": "cap_trim"}
KIND_REASON = {
    "floor": "minimo investito: il bot schiera la quota che manca al 30%",
    "riallineamento": "riallineamento del libro all'intento del bot",
}


def half_spread(ticker: str) -> float:
    return HALF_SPREAD.get(ticker, DEFAULT_HALF_SPREAD)


def ticker_for(asset: str) -> str | None:
    from app.services.ai_portfolio.momentum import asset_to_ticker

    return asset_to_ticker(asset)


# ---------------------------------------------------------------------------
# Orari e sedute
# ---------------------------------------------------------------------------

def us_summer_time(d: date) -> bool:
    """Ora legale USA: dalla seconda domenica di marzo alla prima di novembre."""
    mar1 = date(d.year, 3, 1)
    start = mar1 + timedelta(days=(6 - mar1.weekday()) % 7 + 7)
    nov1 = date(d.year, 11, 1)
    end = nov1 + timedelta(days=(6 - nov1.weekday()) % 7)
    return start <= d < end


def us_open_utc(d: date) -> datetime:
    """Apertura di Wall Street (09:30 New York) in UTC naive."""
    return datetime(d.year, d.month, d.day, 13 if us_summer_time(d) else 14, 30)


def fill_start(decided_on: date, decided_at: datetime) -> date:
    """Primo giorno in cui l'ordine può essere eseguito: il giro delle 05:30 UTC precede l'apertura
    di Wall Street; un giro manuale fatto dopo l'apertura passa al giorno dopo."""
    if decided_at > us_open_utc(decided_on):
        return decided_on + timedelta(days=1)
    return decided_on


def last_session_close(market, today: date) -> date | None:
    """Ultima seduta di Wall Street CONCLUSA prima di oggi (UTC)."""
    df = market.bars(CALENDAR_TICKER, today - timedelta(days=14), today - timedelta(days=1))
    if df is None or df.empty:
        return None
    return max(df.index)


def session_on_or_after(market, start: date, until: date) -> date | None:
    df = market.bars(CALENDAR_TICKER, start, until)
    if df is None or df.empty:
        return None
    return min(df.index)


def execution_price(market, ticker: str, session: date, until: date) -> tuple[date, float] | None:
    """(data, prezzo di riferimento) dell'esecuzione nella seduta `session`.
    ETF: apertura della prima barra dalla seduta in poi. Cripto: chiusura della barra di quel
    giorno (la sua apertura, a mezzanotte UTC, viene prima della decisione delle 05:30)."""
    if ticker in CRYPTO_TICKERS:
        df = market.bars(ticker, session, session)
        if df is None or df.empty:
            return None
        return session, float(df.iloc[-1]["Close"])
    bar = first_bar_on_or_after(market, ticker, session, until)
    return (bar[0], bar[1]) if bar is not None else None


# ---------------------------------------------------------------------------
# Stato del libro — funzioni PURE (il prezzo entra solo nella valutazione)
# ---------------------------------------------------------------------------

@dataclass
class BotLot:
    fill_date: date
    qty: float           # quote alla scala del giorno dell'esecuzione
    open_usd: float      # prezzo di riferimento di quel giorno, stessa scala
    cost_usd: float      # costo residuo (lordo + commissione), si riduce con le vendite
    remaining: float = 1.0


@dataclass
class BotPosition:
    ticker: str
    lots: list[BotLot] = field(default_factory=list)

    @property
    def basis(self) -> float:
        return sum(lot.cost_usd for lot in self.lots)


@dataclass
class BookState:
    cash: float
    positions: dict[str, BotPosition] = field(default_factory=dict)
    realized: float = 0.0
    fees: float = 0.0
    spread: float = 0.0
    dividends: float = 0.0
    interest: float = 0.0


def apply_buy(state: BookState, asset: str, ticker: str, fill_date: date, qty: float, open_usd: float,
              gross: float, fee: float, spread: float) -> None:
    pos = state.positions.setdefault(asset, BotPosition(ticker=ticker))
    pos.lots.append(BotLot(fill_date=fill_date, qty=qty, open_usd=open_usd, cost_usd=gross + fee))
    state.cash -= gross + fee
    state.fees += fee
    state.spread += spread


def apply_sell(state: BookState, asset: str, fraction: float, gross: float, fee: float,
               spread: float) -> tuple[float, float]:
    """Vende `fraction` della posizione (pro-quota su ogni lotto = costo medio).
    Ritorna (costo di carico venduto, utile realizzato al netto della commissione)."""
    pos = state.positions[asset]
    f = min(max(fraction, 0.0), 1.0)
    basis_sold = pos.basis * f
    for lot in pos.lots:
        lot.cost_usd *= 1 - f
        lot.remaining *= 1 - f
    if f >= 0.999999:
        del state.positions[asset]
    realized = (gross - fee) - basis_sold
    state.cash += gross - fee
    state.fees += fee
    state.spread += spread
    state.realized += realized
    return basis_sold, realized


def apply_income(state: BookState, side: str, amount: float) -> None:
    state.cash += amount
    if side == "dividend":
        state.dividends += amount
    else:
        state.interest += amount


def _phase(side: str, ticker: str | None) -> int:
    """Momento dentro la seduta: cedole prima dell'apertura, ETF all'apertura, cripto alla chiusura
    della loro barra (dopo), interessi a fine giornata. Così una vendita di cripto non paga un
    acquisto di ETF fatto ore prima."""
    if side == "dividend":
        return 0
    if side == "interest":
        return 3
    return 2 if ticker in CRYPTO_TICKERS else 1


def _event_key(mv: BotMovement) -> tuple:
    return (mv.fill_date or date.min, _phase(mv.side, mv.ticker), EVENT_RANK.get(mv.side, 9), mv.id or 0)


def replay(movements: list[BotMovement], initial: float) -> BookState:
    """Stato del libro rigiocando i movimenti eseguiti nell'ordine di `settle` (`EVENT_RANK`)."""
    state = BookState(cash=initial)
    for mv in sorted((m for m in movements if m.status == "filled"), key=_event_key):
        if mv.side == "buy" and mv.fill_date is not None:
            apply_buy(state, mv.asset_class, mv.ticker, mv.fill_date, mv.quantity, mv.open_usd or 0.0,
                      mv.gross_usd, mv.fee_usd, mv.spread_usd)
        elif mv.side == "sell" and mv.asset_class in state.positions:
            apply_sell(state, mv.asset_class, mv.sell_fraction if mv.sell_fraction is not None else 1.0,
                       mv.gross_usd, mv.fee_usd, mv.spread_usd)
        elif mv.side in ("dividend", "interest"):
            apply_income(state, mv.side, mv.gross_usd)
    return state


# ---------------------------------------------------------------------------
# Prezzi
# ---------------------------------------------------------------------------

def _open_on(market, ticker: str, d: date) -> float | None:
    df = market.bars(ticker, d, d)
    if df is None or df.empty:
        return None
    return float(df.iloc[-1]["Open"])


def _close_asof(market, ticker: str, d: date) -> float | None:
    return close_on(market, ticker, d) or last_close_on_or_before(market, ticker, d)


def _adj_close_asof(market, ticker: str, d: date) -> float | None:
    df = market.bars(ticker, d - timedelta(days=14), d)
    if df is None or df.empty:
        return None
    col = "AdjClose" if "AdjClose" in df.columns else "Close"
    v = df.loc[max(df.index)][col]
    return float(v) if v == v else None  # NaN → None


def scale_factor(market, ticker: str, lot: BotLot) -> float | None:
    """Quante quote di OGGI vale una quota del giorno dell'esecuzione (4 dopo un frazionamento 4:1).
    None se l'apertura di quel giorno non c'è più: la scala non si indovina (era il caso di QUAL)."""
    if ticker in CRYPTO_TICKERS:
        return 1.0  # nessun frazionamento; il riferimento è la chiusura, non l'apertura
    if not lot.open_usd:
        return None
    now_open = _open_on(market, ticker, lot.fill_date)
    if now_open:
        return lot.open_usd / now_open
    # La barra di quel giorno manca (dato scartato da Yahoo): si guarda la più vicina entro 5 giorni
    # e il rapporto si accetta solo se è un fattore di frazionamento plausibile entro il 6%.
    df = market.bars(ticker, lot.fill_date - timedelta(days=5), lot.fill_date + timedelta(days=5))
    if df is None or df.empty:
        return None
    near = min(df.index, key=lambda d: abs((d - lot.fill_date).days))
    o = float(df.loc[near]["Open"])
    if o <= 0:
        return None
    r = lot.open_usd / o
    for k in SPLIT_FACTORS:
        if abs(r / k - 1) < 0.06:
            return k
    return None


def qty_now(market, pos: BotPosition) -> tuple[float, bool]:
    """(quote alla scala di oggi, scala verificata su tutti i lotti)."""
    q, ok = 0.0, True
    for lot in pos.lots:
        k = scale_factor(market, pos.ticker, lot)
        if k is None:
            ok, k = False, 1.0
        q += lot.qty * lot.remaining * k
    return q, ok


def valuate(state: BookState, market, on: date) -> tuple[float, float, list[dict]]:
    """(valore, investito, posizioni) alla chiusura `on`. Senza prezzo una posizione vale il suo
    costo di carico (`price_missing`); senza scala verificabile lo dice `scale_unverified`."""
    invested, rows = 0.0, []
    for asset, pos in sorted(state.positions.items()):
        q, scale_ok = qty_now(market, pos)
        px = _close_asof(market, pos.ticker, on)
        value = q * px if px else pos.basis
        invested += value
        rows.append({"asset_class": asset, "ticker": pos.ticker, "qty": q, "price_usd": px,
                     "value_usd": value, "cost_usd": pos.basis,
                     "avg_cost_usd": (pos.basis / q) if q > 0 else None,
                     "pnl_usd": value - pos.basis,
                     "pnl_pct": (value / pos.basis - 1) if pos.basis > 0 else 0.0,
                     "price_missing": not px, "scale_unverified": not scale_ok})
    nav = state.cash + invested
    for r in rows:
        r["weight_pct"] = r["value_usd"] / nav if nav > 0 else None
    return nav, invested, rows


def benchmarks(market, c0: date | None, c: date) -> dict[str, float | None]:
    """S&P 500, MSCI ACWI e 60/40 (S&P + Treasury lunghi, ribilanciato ogni seduta) dalla chiusura
    di partenza, sui prezzi rettificati per i dividendi: stessa base del libro, che le cedole le
    incassa."""
    out: dict[str, float | None] = {"sp500": None, "acwi": None, "6040": None}
    if c0 is None or c <= c0:
        return {"sp500": 0.0, "acwi": 0.0, "6040": 0.0}
    for key, t in (("sp500", "SPY"), ("acwi", "ACWI")):
        a, b = _adj_close_asof(market, t, c0), _adj_close_asof(market, t, c)
        out[key] = (b / a - 1) if (a and b) else None
    spy = market.bars("SPY", c0, c)
    if spy is not None and not spy.empty:
        days = sorted(spy.index)
        v, ok = 1.0, True
        for d0, d1 in zip(days, days[1:]):
            rs = _ratio(market, "SPY", d0, d1)
            rb = _ratio(market, "TLT", d0, d1)
            if rs is None or rb is None:
                ok = False
                break
            v *= 1 + 0.6 * rs + 0.4 * rb
        out["6040"] = (v - 1) if ok else None
    return out


def _ratio(market, ticker: str, a: date, b: date) -> float | None:
    p0, p1 = _adj_close_asof(market, ticker, a), _adj_close_asof(market, ticker, b)
    return (p1 / p0 - 1) if (p0 and p1) else None


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

def get_book(db: Session, strategy: str) -> BotBook | None:
    return db.query(BotBook).filter(BotBook.strategy_type == strategy).first()


def movements_of(db: Session, strategy: str) -> list[BotMovement]:
    return db.query(BotMovement).filter(BotMovement.strategy_type == strategy).all()


def latest_nav(db: Session, strategy: str, on: date | None = None) -> BotNav | None:
    q = db.query(BotNav).filter(BotNav.strategy_type == strategy)
    if on is not None:
        q = q.filter(BotNav.date <= on)
    return q.order_by(BotNav.date.desc()).first()


@dataclass
class SettleResult:
    new_close: bool
    why: str
    close: date | None = None
    nav: float | None = None
    state: BookState | None = None
    reconciled: bool = True
    gap: float = 0.0
    filled: int = 0
    note: str | None = None


def _give_back_to_intent(db: Session, strategy: str, mv: BotMovement, filled_amount: float) -> None:
    """Un acquisto saltato o ridotto all'apertura: la parte non eseguita torna «da ordinare»
    (il peso già ordinato scende), così il giro dopo la riprova se c'è cassa."""
    req = mv.amount_usd or 0.0
    if req <= 0 or mv.weight_delta_pct <= 0:
        return
    missing_w = mv.weight_delta_pct * max(0.0, 1 - filled_amount / req)
    snap = db.query(BotIntentSnapshot).filter(BotIntentSnapshot.strategy_type == strategy,
                                              BotIntentSnapshot.asset_class == mv.asset_class).first()
    if snap is not None and missing_w > 0:
        snap.weight_pct = max(0.0, snap.weight_pct - missing_w)


def _checks(db: Session, strategy: str, rows: list[dict], last: BotNav | None, market, c: date,
            filled_assets: set[str]) -> list[str]:
    """I controlli che la quadratura da sola non vede (è un'identità sui movimenti)."""
    problems: list[str] = []
    movs = movements_of(db, strategy)
    for m in movs:
        if m.status == "filled" and m.side in ("buy", "sell") and m.fill_price_usd is not None:
            if abs(m.gross_usd - m.quantity * m.fill_price_usd) > 0.01 + 1e-9 * abs(m.gross_usd):
                problems.append(f"movimento {m.id} ({m.asset_class}): importo diverso da quantità × prezzo")
    seen: dict[tuple, int] = {}
    for m in movs:
        if m.side == "dividend" and m.status == "filled":
            seen[(m.asset_class, m.fill_date)] = seen.get((m.asset_class, m.fill_date), 0) + 1
    problems += [f"cedola doppia: {a} {d}" for (a, d), n in seen.items() if n > 1]
    problems += [f"scala non verificabile per {r['asset_class']}: manca l'apertura del giorno d'acquisto"
                 for r in rows if r.get("scale_unverified")]
    problems += [f"prezzo di chiusura mancante per {r['asset_class']}: valutata al costo"
                 for r in rows if r.get("price_missing")]
    if last is not None:
        before = {r["asset_class"]: r for r in json.loads(last.positions_json or "[]")}
        for r in rows:
            p0, p1 = _close_asof(market, r["ticker"], last.date), _close_asof(market, r["ticker"], c)
            if not p0 or not p1:
                continue
            move = p1 / p0 - 1
            if r["ticker"] not in CRYPTO_TICKERS and abs(move) > MAX_ETF_SESSION_MOVE:
                problems.append(f"{r['asset_class']}: prezzo {move * 100:+.1f}% fra il {last.date} e il {c}, "
                                f"probabile frazionamento non ancora riscalato da Yahoo")
                continue
            b = before.get(r["asset_class"])
            # la chiusura prima non verificata non fa da metro: il giorno dopo un blocco si riparte
            if b is None or r["asset_class"] in filled_assets or not b.get("value_usd") or not last.reconciled:
                continue
            expected = b["value_usd"] * p1 / p0
            if abs(r["value_usd"] - expected) > max(1.0, CONTINUITY_TOLERANCE * expected):
                problems.append(f"{r['asset_class']}: vale {r['value_usd']:.2f} $, il suo prezzo dice "
                                f"{expected:.2f} $ (dalla chiusura del {last.date})")
    return problems


def settle(db: Session, strategy: str, market, now: datetime) -> SettleResult:
    """Esegue gli ordini in attesa, accredita cedole e interessi, scrive il valore all'ultima
    chiusura conclusa. Idempotente: con la chiusura già contata non fa niente."""
    book = get_book(db, strategy)
    if book is None:
        return SettleResult(False, "libro non avviato")
    today = now.date()
    c = last_session_close(market, today)
    if c is None:
        return SettleResult(False, "chiusure di Wall Street non disponibili")
    last = latest_nav(db, strategy)
    if last is not None and last.date >= c:
        return SettleResult(False, f"nessuna chiusura nuova dopo il {last.date}", close=last.date,
                            nav=last.nav_usd, reconciled=last.reconciled)
    prev_close = last.date if last else None

    movs = movements_of(db, strategy)
    state = replay(movs, book.initial_capital_usd)
    cash_at_prev_close = state.cash

    # Ordini: tutti quelli di una decisione nella STESSA seduta (la prima dopo la decisione)
    events: list[tuple] = []
    for mv in sorted((m for m in movs if m.status == "pending"), key=lambda m: m.id):
        session = session_on_or_after(market, fill_start(mv.decided_on, mv.decided_at), c)
        ex = execution_price(market, mv.ticker, session, c) if session else None
        if ex is not None:
            events.append((ex[0], _phase(mv.side, mv.ticker), EVENT_RANK[mv.side], mv.id, "order", mv, ex[1]))
    held = {a: p.ticker for a, p in state.positions.items()}
    held.update({e[5].asset_class: e[5].ticker for e in events if e[5].side == "buy"})
    paid = {(m.asset_class, m.fill_date) for m in movs if m.side == "dividend"}
    if prev_close is not None:
        for asset, t in held.items():
            divs = market.dividends(t, prev_close + timedelta(days=1), c)
            for ex_d, per_share in (divs.items() if divs is not None else []):
                if (asset, ex_d) not in paid:
                    paid.add((asset, ex_d))  # la stessa data due volte nel feed = una cedola sola
                    events.append((ex_d, _phase("dividend", t), EVENT_RANK["dividend"], 0, "dividend",
                                   (asset, t), float(per_share)))
    events.sort(key=lambda e: (e[0], e[1], e[2], e[3]))

    filled, filled_assets = 0, set()
    for d, _, _, _, kind, obj, px in events:
        if kind == "dividend":
            asset, t = obj
            pos = state.positions.get(asset)
            q = qty_now(market, pos)[0] if pos else 0.0
            if q <= 0 or px <= 0:
                continue
            amount = q * px
            apply_income(state, "dividend", amount)
            db.add(BotMovement(strategy_type=strategy, asset_class=asset, ticker=t, side="dividend",
                               kind="dividend", status="filled", decided_on=d, decided_at=now, fill_date=d,
                               quantity=q, fill_price_usd=px, gross_usd=amount, cash_after_usd=state.cash,
                               note=f"cedola {px:.4f} $ per quota"))
            db.flush()
            continue
        mv, ref_px = obj, px
        h = half_spread(mv.ticker)
        if mv.side == "sell":
            pos = state.positions.get(mv.asset_class)
            if pos is None:
                mv.status, mv.note = "cancelled", "niente da vendere all'esecuzione"
                continue
            f = 1.0 if (mv.sell_fraction is None or mv.sell_fraction >= 0.999) else max(mv.sell_fraction, 0.0)
            q = qty_now(market, pos)[0] * f
            price = ref_px * (1 - h)
            gross = q * price
            fee = min(FEE_PER_ORDER_USD, gross)
            spread = q * ref_px * h
            basis_sold, realized = apply_sell(state, mv.asset_class, f, gross, fee, spread)
            mv.sell_fraction = f
            mv.quantity, mv.cost_basis_sold_usd, mv.realized_pnl_usd = q, basis_sold, realized
            if mv.kind in FULL_EXIT_KINDS and strategy in LEARNING_STRATEGIES and basis_sold > 0:
                _learn(db, strategy, mv, realized / basis_sold, d)
        else:
            requested = mv.amount_usd or 0.0
            amount, note = requested, None
            if amount > state.cash:
                amount, note = max(state.cash, 0.0), "ridotto all'esecuzione: la liquidità non bastava"
            if amount < MIN_ORDER_USD:
                mv.status = "skipped"
                mv.note = note or "importo troppo piccolo"
                _give_back_to_intent(db, strategy, mv, 0.0)
                continue
            fee = FEE_PER_ORDER_USD
            price = ref_px * (1 + h)
            q = (amount - fee) / price
            gross, spread = q * price, q * ref_px * h
            apply_buy(state, mv.asset_class, mv.ticker, d, q, ref_px, gross, fee, spread)
            mv.quantity = q
            if note:
                mv.note = note
                _give_back_to_intent(db, strategy, mv, amount)
        mv.status, mv.fill_date = "filled", d
        mv.open_usd, mv.fill_price_usd, mv.gross_usd = ref_px, price, gross
        mv.fee_usd, mv.spread_usd, mv.cash_after_usd = fee, spread, state.cash
        filled_assets.add(mv.asset_class)
        db.flush()
        filled += 1

    # Interessi: solo la liquidità rimasta FERMA fra le due chiusure (quella incassata vendendo
    # all'apertura era ancora investita; quella spesa all'apertura non c'è più)
    from app.services import config_flags

    idle = min(cash_at_prev_close, state.cash)
    if prev_close is not None and config_flags.use_ai_portfolio_cash_overnight() and idle > 0:
        r = _ratio(market, CASH_TICKER, prev_close, c)
        if r is not None and abs(idle * r) >= 0.005:
            amount = idle * r
            apply_income(state, "interest", amount)
            db.add(BotMovement(strategy_type=strategy, asset_class="cash", ticker=CASH_TICKER, side="interest",
                               kind="interest", status="filled", decided_on=c, decided_at=now, fill_date=c,
                               gross_usd=amount, cash_after_usd=state.cash,
                               note=f"{idle:.2f} $ ferme al rendimento di BIL {prev_close} → {c}: {r * 100:+.4f}%"))
            db.flush()

    if book.start_close is None:
        book.start_close = c
    nav, invested, rows = valuate(state, market, c)
    realized_saved = sum((m.realized_pnl_usd or 0.0) for m in db.query(BotMovement).filter(
        BotMovement.strategy_type == strategy, BotMovement.side == "sell", BotMovement.status == "filled").all())
    unrealized = invested - sum(p.basis for p in state.positions.values())
    explained = realized_saved + unrealized + state.dividends + state.interest
    gap = (nav - book.initial_capital_usd) - explained
    problems = _checks(db, strategy, rows, last, market, c, filled_assets)
    if abs(gap) > RECONCILE_TOLERANCE_USD:
        problems.insert(0, f"valore − capitale e scomposizione differiscono di {gap:.2f} $")
    if state.cash < -0.01:
        problems.append(f"liquidità negativa: {state.cash:.2f} $")
    if nav <= 0 or invested > nav * 1.001:
        problems.append("investito oltre il valore del libro (leva)")
    reconciled = not problems
    note = "; ".join(problems) if problems else None
    if not reconciled:
        logger.error(f"bot_book {strategy}: conti NON verificati al {c}: {note}")

    peak = max(last.peak_nav_usd if last else book.initial_capital_usd, nav)
    bm = benchmarks(market, book.start_close, c)
    row = BotNav(date=c, strategy_type=strategy, nav_usd=nav, cash_usd=state.cash, invested_usd=invested,
                 n_positions=len(rows), cumulative_return_pct=nav / book.initial_capital_usd - 1,
                 peak_nav_usd=peak, drawdown_pct=nav / peak - 1, realized_cum_usd=realized_saved,
                 unrealized_usd=unrealized, dividends_cum_usd=state.dividends,
                 interest_cum_usd=state.interest, fees_cum_usd=state.fees, spread_cum_usd=state.spread,
                 reconciled=reconciled, reconciliation_gap_usd=gap, reconciliation_note=note,
                 bench_sp500_pct=bm["sp500"], bench_acwi_pct=bm["acwi"], bench_6040_pct=bm["6040"],
                 positions_json=json.dumps(rows))
    db.add(row)
    db.flush()
    return SettleResult(True, f"chiusura nuova: {c}", close=c, nav=nav, state=state,
                        reconciled=reconciled, gap=gap, filled=filled, note=note)


def _learn(db: Session, strategy: str, mv: BotMovement, pnl_pct: float, on: date) -> None:
    from app.services.ai_portfolio import learning

    try:
        learning.record_outcome(db, strategy_type=strategy, regime=mv.entry_regime or "unknown",
                                asset_class=mv.asset_class, momentum="NEUTRAL", pnl=pnl_pct, seen_on=on)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"bot_book: apprendimento non aggiornato per {mv.asset_class}: {e}")


def sync_positions(db: Session, strategy: str, state: BookState, market, on: date) -> None:
    """Il guadagno che le strategie leggono (stop, prese di profitto, prompt dell'LLM) viene dal
    libro: valore delle quote contro il loro costo, frazionamenti compresi."""
    _, _, rows = valuate(state, market, on)
    by_asset = {r["asset_class"]: r for r in rows}
    for p in db.query(AiPortfolioPosition).filter(AiPortfolioPosition.strategy_type == strategy).all():
        r = by_asset.get(p.asset_class)
        if r is None:
            p.estimated_pnl_pct = 0.0  # ordine non ancora eseguito
            continue
        p.estimated_pnl_pct = r["pnl_pct"]
        p.current_price = r["price_usd"]
        p.avg_entry_price = r["avg_cost_usd"]
    db.flush()


# ---------------------------------------------------------------------------
# Intento del bot → ordini
# ---------------------------------------------------------------------------

def _decision_for(db: Session, strategy: str, asset: str, actions: tuple[str, ...],
                  today: date) -> AiPortfolioDecision | None:
    return (db.query(AiPortfolioDecision)
            .filter(AiPortfolioDecision.strategy_type == strategy, AiPortfolioDecision.asset_class == asset,
                    AiPortfolioDecision.action.in_(actions),
                    AiPortfolioDecision.date >= today - timedelta(days=3))
            .order_by(AiPortfolioDecision.id.desc()).first())


def create_orders(db: Session, strategy: str, now: datetime, nav_usd: float,
                  state: BookState) -> list[BotMovement]:
    """Ogni differenza fra l'intento del bot e l'ultimo stato già ordinato diventa un ordine.
    Prima le vendite, poi gli acquisti, che non superano la cassa che ci sarà all'esecuzione."""
    today = now.date()
    intent = {p.asset_class: p for p in db.query(AiPortfolioPosition)
              .filter(AiPortfolioPosition.strategy_type == strategy).all()}
    snaps = {s.asset_class: s for s in db.query(BotIntentSnapshot)
             .filter(BotIntentSnapshot.strategy_type == strategy).all()}
    pending = db.query(BotMovement).filter(BotMovement.strategy_type == strategy,
                                           BotMovement.status == "pending").all()
    last = latest_nav(db, strategy)
    value_at_close = {r["asset_class"]: r.get("value_usd") or 0.0
                      for r in json.loads(last.positions_json or "[]")} if last else {}
    created: list[BotMovement] = []
    sells: list[tuple] = []   # (asset, kind, fraction, dw, dec, regime)
    buys: list[tuple] = []    # (asset, kind, dw, dec, regime, base_w)
    ordered_w: dict[str, float] = {}

    for asset in sorted(set(intent) | set(snaps)):
        pos, snap = intent.get(asset), snaps.get(asset)
        if snap is not None and (pos is None or pos.opened_at != snap.opened_at):
            dec = _decision_for(db, strategy, asset, tuple(SELL_KIND), today)
            kind = SELL_KIND.get(dec.action, "full_close") if dec else "riallineamento"
            regime = snap.entry_regime or (dec.regime if dec else None)
            sells.append((asset, kind, None, -snap.weight_pct, dec, regime))
            snap = None
        if pos is None:
            continue
        base = snap.weight_pct if snap is not None else 0.0
        dw = pos.current_weight_pct - base
        ordered_w[asset] = base  # finché un ordine non lo sposta, resta quello già ordinato
        if dw > EPS_W:
            if snap is None and pos.last_action == "FLOOR_DEPLOY" and pos.tranches_total <= 1:
                buys.append((asset, "floor", dw, None, pos.entry_regime, base))
            elif snap is None or pos.tranches_filled > snap.tranches_filled:
                buys.append((asset, "tranche", dw, _decision_for(db, strategy, asset, ("OPEN_TRANCHE",), today),
                             pos.entry_regime, base))
            else:
                buys.append((asset, "floor" if pos.last_action == "FLOOR_DEPLOY" else "riallineamento", dw,
                             None, pos.entry_regime, base))
        elif dw < -EPS_W:
            fraction = 1 - pos.current_weight_pct / base if base > 0 else None
            if pos.last_action == "CAP_TRIM":
                kind, dec = "cap_trim", _decision_for(db, strategy, asset, ("CAP_TRIM",), today)
            elif snap is not None and pos.tranches_filled < snap.tranches_filled:
                kind, dec = "close_tranche", _decision_for(db, strategy, asset, ("CLOSE_TRANCHE",), today)
            else:
                kind, dec = "riallineamento", None
            sells.append((asset, kind, fraction, dw, dec, (snap.entry_regime if snap else None) or pos.entry_regime))
            ordered_w[asset] = pos.current_weight_pct

    # Una vendita TOTALE già in attesa (es. decisa sabato, eseguita lunedì) non si ripete nei giri
    # senza chiusura nuova: sennò domenica e lunedì ne nascerebbe un'altra, e i suoi incassi
    # verrebbero contati due volte come cassa disponibile.
    exiting = {m.asset_class for m in pending
               if m.side == "sell" and (m.sell_fraction is None or m.sell_fraction >= 0.999)}
    sells = [s for s in sells if s[0] not in exiting]

    # Rete di sicurezza: quote di un asset che il bot non ha né aveva
    for asset in list(state.positions):
        if asset not in intent and asset not in snaps and asset not in exiting:
            sells.append((asset, "riallineamento", None, 0.0, None, None))

    def new(asset: str, side: str, kind: str, dw: float, dec, regime: str | None, **kw) -> BotMovement | None:
        t = ticker_for(asset) or (state.positions[asset].ticker if asset in state.positions else None)
        if not t:
            logger.warning(f"bot_book: nessun ticker per {asset}")
            return None
        mv = BotMovement(strategy_type=strategy, asset_class=asset, ticker=t, side=side, kind=kind,
                         status="pending", weight_delta_pct=dw, decided_on=today, decided_at=now,
                         decision_id=dec.id if dec else None,
                         reason=(dec.reason_short if dec else KIND_REASON.get(kind)),
                         entry_regime=regime, **kw)
        db.add(mv)
        created.append(mv)
        return mv

    # cassa che ci sarà all'esecuzione: quella di oggi, meno gli acquisti ancora in attesa,
    # più le vendite (stimate al valore di chiusura)
    avail = state.cash - sum((m.amount_usd or 0.0) for m in pending if m.side == "buy")
    avail += sum(value_at_close.get(m.asset_class, 0.0) * (m.sell_fraction if m.sell_fraction is not None else 1.0)
                 for m in pending if m.side == "sell")

    for asset, kind, fraction, dw, dec, regime in sells:
        bought_pending = [m for m in pending if m.side == "buy" and m.asset_class == asset and m.status == "pending"]
        if fraction is None:
            for m in bought_pending:  # uscita totale: l'acquisto in attesa non deve più partire
                m.status, m.note = "cancelled", "annullato prima dell'esecuzione: il bot è uscito"
                avail += m.amount_usd or 0.0
        else:
            # taglio parziale: vale anche per la parte non ancora eseguita (sennò, con una tranche
            # in attesa, il libro finirebbe sopra l'intento)
            for m in bought_pending:
                cut = (m.amount_usd or 0.0) * fraction
                m.amount_usd = (m.amount_usd or 0.0) - cut
                m.weight_delta_pct *= 1 - fraction
                m.note = f"ridotto del {fraction * 100:.0f}% prima dell'esecuzione: il bot ha tagliato"
                avail += cut
        if asset not in state.positions:
            continue
        new(asset, "sell", kind, dw, dec, regime, sell_fraction=fraction)
        avail += value_at_close.get(asset, 0.0) * (fraction if fraction is not None else 1.0)

    for asset, kind, dw, dec, regime, base in buys:
        amount = min(dw * nav_usd, max(avail, 0.0))
        if amount < MIN_ORDER_USD:
            logger.info(f"bot_book {strategy}: {asset} +{dw * 100:.2f}% non ordinato, cassa disponibile "
                        f"{avail:.2f} $")
            continue  # resta «da ordinare»: il peso già ordinato non si muove
        ordered_dw = amount / nav_usd if nav_usd > 0 else 0.0
        note = "ridotto alla cassa disponibile" if amount < dw * nav_usd - 0.01 else None
        mv = new(asset, "buy", kind, ordered_dw, dec, regime, amount_usd=amount)
        if mv is not None:
            mv.note = note
            avail -= amount
            ordered_w[asset] = base + ordered_dw

    db.query(BotIntentSnapshot).filter(BotIntentSnapshot.strategy_type == strategy).delete()
    for p in intent.values():
        prev = snaps.get(p.asset_class)
        reopened = prev is not None and prev.opened_at != p.opened_at
        db.add(BotIntentSnapshot(strategy_type=strategy, asset_class=p.asset_class,
                                 weight_pct=ordered_w.get(p.asset_class, p.current_weight_pct),
                                 tranches_filled=p.tranches_filled, tranches_total=p.tranches_total,
                                 opened_at=p.opened_at, last_action=p.last_action,
                                 last_action_date=p.last_action_date,
                                 entry_regime=(prev.entry_regime if (prev and not reopened and prev.entry_regime)
                                               else p.entry_regime)))
    db.flush()
    return created


# ---------------------------------------------------------------------------
# Giro completo
# ---------------------------------------------------------------------------

def _strategy_module(strategy: str):
    from app.services.ai_portfolio import llm_strategist, risk_managed_sp, strategist

    return {"model_driven": strategist, "llm_driven": llm_strategist, "risk_managed_sp": risk_managed_sp}[strategy]


def run_giro(db: Session, now: datetime | None = None, market=None, strategies=STRATEGIES,
             reasoning: bool = True) -> dict:
    """Per ogni bot: chiude i conti della seduta nuova, poi decide, poi scrive gli ordini.
    Senza chiusura nuova non decide (la decisione sarebbe sugli stessi prezzi); con i conti che
    non quadrano non decide (si deciderebbe su un libro sbagliato)."""
    from app.services.real_book.market import YahooMarket

    now = now or datetime.utcnow()
    market = market or YahooMarket()
    out: dict = {}
    for s in strategies:
        try:
            res = settle(db, s, market, now)
            db.commit()
            if res.why == "libro non avviato":
                out[s] = {"skipped": "libro non avviato: serve la ripartenza"}
                continue
            if not res.new_close:
                out[s] = {"skipped": res.why, "nav": res.nav}
                # Recupero: un giro caduto fra il salvataggio dell'intento e la scrittura degli ordini
                # lascia differenze. Senza far ridecidere la strategia, le si trasforma in ordini ora.
                last = latest_nav(db, s)
                if last is not None and last.reconciled:
                    book = get_book(db, s)
                    st = replay(movements_of(db, s), book.initial_capital_usd if book else INITIAL_CAPITAL_USD)
                    late = create_orders(db, s, now, last.nav_usd, st)
                    db.commit()
                    if late:
                        out[s]["ordini_recuperati"] = len(late)
                continue
            if not res.reconciled:
                out[s] = {"bloccato": f"conti non verificati: {res.note}: nessuna decisione", "nav": res.nav}
                continue
            assert res.state is not None and res.close is not None and res.nav is not None
            sync_positions(db, s, res.state, market, res.close)
            db.commit()
            summary = _strategy_module(s).daily_scan(db)
            orders = create_orders(db, s, now, res.nav, res.state)
            db.commit()
            out[s] = {"chiusura": str(res.close), "nav": round(res.nav, 2), "eseguiti": res.filled,
                      "nuovi_ordini": len(orders), "scan": summary}
            if reasoning:
                try:
                    from app.services.ai_portfolio import llm_reasoner

                    llm_reasoner.generate_reasoning(db, strategy_type=s)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"bot_book: analisi LLM {s} non generata: {e}")
        except Exception as e:  # noqa: BLE001
            db.rollback()
            logger.exception(f"bot_book: giro {s} fallito: {e}")
            out[s] = {"errore": str(e)}
    return out


# ---------------------------------------------------------------------------
# Ripartenza da zero (Erik 2026-11-15: «riparti da zero i vari portafogli»)
# ---------------------------------------------------------------------------

# Tutto ciò che descrive i portafogli: la contabilità a pesi, il libro reale che la copiava e
# (se c'è) un libro dei bot precedente. Si COPIA in tabelle d'archivio, poi si svuota.
RESTART_TABLES = (
    "ai_portfolio_positions", "ai_portfolio_decisions", "ai_portfolio_performance",
    "ai_portfolio_learnings", "ai_portfolio_reasoning",
    "rb_books", "rb_movements", "rb_bot_snapshot", "rb_nav",
    "bot_books", "bot_movements", "bot_intent_snapshot", "bot_nav",
)


def _archive_summary(db: Session) -> dict:
    """Ultimo valore della curva vecchia come la mostrava il sito (corretta alla lettura)."""
    out: dict = {}
    try:
        from app.services.ai_portfolio.history_view import corrected_history_for

        for s in STRATEGIES:
            rows, _ = corrected_history_for(db, s)
            if rows:
                out[s] = {"date": rows[-1]["date"], "nav": rows[-1]["nav"],
                          "cumulative_return_pct": rows[-1]["cumulative_return_pct"],
                          "since": rows[0]["date"]}
    except Exception as e:  # noqa: BLE001
        logger.warning(f"bot_book: riassunto della curva archiviata non disponibile: {e}")
    return out


def restart(db: Session, today: date, label: str | None = None, summary: dict | None = None) -> dict:
    """Archivia (copia integrale) e svuota i portafogli, poi apre un libro nuovo per ogni bot.
    Rifiuta se l'archivio con quell'etichetta esiste già: non si archivia due volte sopra."""
    label = label or today.strftime("%Y%m%d")
    if summary is None:
        summary = _archive_summary(db)
    bind = db.get_bind()
    from sqlalchemy import inspect

    existing = set(inspect(bind).get_table_names())
    for t in RESTART_TABLES:
        if f"arch_{label}_{t}" in existing:
            raise ValueError(f"archivio arch_{label}_* già presente: ripartenza già fatta con questa etichetta")
    counts = {}
    for t in RESTART_TABLES:
        if t not in existing:
            counts[t] = None
            continue
        db.execute(text(f"CREATE TABLE arch_{label}_{t} AS SELECT * FROM {t}"))
        n = db.execute(text(f"SELECT COUNT(*) FROM arch_{label}_{t}")).scalar()
        src = db.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
        if n != src:
            raise RuntimeError(f"archivio di {t} incompleto ({n} righe su {src}): niente è stato cancellato")
        counts[t] = n
    for t in RESTART_TABLES:
        if counts.get(t) is not None:
            db.execute(text(f"DELETE FROM {t}"))
    for s in STRATEGIES:
        db.add(BotBook(strategy_type=s, started_on=today, initial_capital_usd=INITIAL_CAPITAL_USD,
                       archive_json=json.dumps({"label": label, "old_curve": summary.get(s)})))
    db.commit()
    return {"label": label, "archived_rows": counts, "old_curve": summary, "started_on": str(today)}


# ---------------------------------------------------------------------------
# Viste per l'API (pagina «Bot»)
# ---------------------------------------------------------------------------

KIND_LABEL = {
    "tranche": "Acquisto (tranche)", "floor": "Acquisto (minimo investito)",
    "riallineamento": "Riallineamento", "close_tranche": "Vendita parziale",
    "full_close": "Vendita totale", "stop_loss": "Stop loss", "take_profit": "Presa di profitto",
    "cap_trim": "Taglio al tetto", "dividend": "Cedola", "interest": "Interessi sulla liquidità",
}
STATUS_LABEL = {"pending": "in attesa dell'apertura", "filled": "eseguito", "cancelled": "annullato",
                "skipped": "non eseguito"}


def _book_info(book: BotBook | None) -> dict | None:
    if book is None:
        return None
    try:
        archive = json.loads(book.archive_json or "{}")
    except ValueError:
        archive = {}
    return {"started_on": str(book.started_on), "start_close": str(book.start_close) if book.start_close else None,
            "initial_capital_usd": book.initial_capital_usd, "archive": archive}


def _nav_row(r: BotNav) -> dict:
    return {
        "date": str(r.date), "price_date": str(r.date), "nav": r.nav_usd,
        "cumulative_return_pct": r.cumulative_return_pct, "drawdown_pct": r.drawdown_pct,
        "peak_nav": r.peak_nav_usd, "benchmark_sp500_return_pct": r.bench_sp500_pct,
        "benchmark_60_40_return_pct": r.bench_6040_pct, "benchmark_acwi_return_pct": r.bench_acwi_pct,
    }


def performance_view(db: Session, strategy: str, days: int = 180) -> dict:
    book = get_book(db, strategy)
    rows = db.query(BotNav).filter(BotNav.strategy_type == strategy).order_by(BotNav.date).all()
    cutoff = date.today() - timedelta(days=days)
    current = None
    if rows:
        last = rows[-1]
        positions = json.loads(last.positions_json or "[]")
        top = max(positions, key=lambda p: p.get("value_usd") or 0) if positions else None
        initial = book.initial_capital_usd if book else INITIAL_CAPITAL_USD
        current = {
            **_nav_row(last),
            "n_positions": last.n_positions,
            "deployed_pct": last.invested_usd / last.nav_usd if last.nav_usd else 0.0,
            "top_position_asset": top["asset_class"] if top else None,
            "top_position_weight_pct": top.get("weight_pct") if top else None,
            "cash_usd": last.cash_usd, "invested_usd": last.invested_usd, "initial_capital_usd": initial,
            "realized_pnl_usd": last.realized_cum_usd, "unrealized_pnl_usd": last.unrealized_usd,
            "dividends_usd": last.dividends_cum_usd, "interest_usd": last.interest_cum_usd,
            "fees_usd": last.fees_cum_usd, "spread_usd": last.spread_cum_usd,
            "reconciled": last.reconciled, "reconciliation_gap_usd": last.reconciliation_gap_usd,
            "reconciliation_note": last.reconciliation_note,
        }
    return {"current": current, "history": [_nav_row(r) for r in rows if r.date >= cutoff],
            "book": _book_info(book)}


def positions_view(db: Session, strategy: str) -> list[dict]:
    """Ciò che il libro POSSIEDE (ultima chiusura) accanto a ciò che il bot VUOLE (intento)."""
    last = latest_nav(db, strategy)
    held = {p["asset_class"]: p for p in json.loads(last.positions_json or "[]")} if last else {}
    intent = {p.asset_class: p for p in db.query(AiPortfolioPosition)
              .filter(AiPortfolioPosition.strategy_type == strategy).all()}
    out = []
    for asset in sorted(set(held) | set(intent), key=lambda a: -((held.get(a) or {}).get("value_usd") or 0)):
        h, p = held.get(asset), intent.get(asset)
        status = "in portafoglio" if (h and p) else ("ordine in attesa" if p else "in uscita")
        out.append({
            "asset_class": asset, "strategy_type": strategy, "status": status,
            "current_weight_pct": (h or {}).get("weight_pct") or 0.0,
            "target_weight_pct": p.current_weight_pct if p else 0.0,
            "qty": (h or {}).get("qty"), "price_usd": (h or {}).get("price_usd"),
            "value_usd": (h or {}).get("value_usd"), "cost_usd": (h or {}).get("cost_usd"),
            "avg_cost_usd": (h or {}).get("avg_cost_usd"), "pnl_usd": (h or {}).get("pnl_usd"),
            "estimated_pnl_pct": (h or {}).get("pnl_pct") or 0.0,
            "tranches_filled": p.tranches_filled if p else 0, "tranches_total": p.tranches_total if p else 0,
            "entry_regime": p.entry_regime if p else "",
            "avg_entry_score": p.avg_entry_score if p else 0.0,
            "took_partial_tp": p.took_partial_tp if p else False,
            "opened_at": str(p.opened_at) if p else None,
            "last_action": p.last_action if p else "", "last_action_date": str(p.last_action_date) if p else None,
            "days_held": (date.today() - p.opened_at).days if p else 0,
        })
    return out


def movements_view(db: Session, strategy: str, limit: int = 200, include_interest: bool = False) -> list[dict]:
    q = db.query(BotMovement).filter(BotMovement.strategy_type == strategy)
    if not include_interest:
        q = q.filter(BotMovement.side != "interest")
    rows = q.order_by(BotMovement.decided_on.desc(), BotMovement.id.desc()).limit(limit).all()
    return [{
        "id": m.id, "asset_class": m.asset_class, "side": m.side, "kind": m.kind,
        "kind_label": KIND_LABEL.get(m.kind, m.kind), "status": m.status,
        "status_label": STATUS_LABEL.get(m.status, m.status),
        "decided_on": str(m.decided_on), "fill_date": str(m.fill_date) if m.fill_date else None,
        "weight_delta_pct": m.weight_delta_pct, "sell_fraction": m.sell_fraction, "amount_usd": m.amount_usd,
        "quantity": m.quantity, "fill_price_usd": m.fill_price_usd, "gross_usd": m.gross_usd,
        "fee_usd": m.fee_usd, "spread_usd": m.spread_usd, "realized_pnl_usd": m.realized_pnl_usd,
        "realized_pnl_pct": (m.realized_pnl_usd / m.cost_basis_sold_usd)
        if (m.realized_pnl_usd is not None and m.cost_basis_sold_usd) else None,
        "cash_after_usd": m.cash_after_usd, "reason": m.reason, "note": m.note,
    } for m in rows]


def leaderboard_rows(db: Session) -> dict:
    out: dict = {}
    for s in STRATEGIES:
        last = latest_nav(db, s)
        if last is None:
            out[s] = None
            continue

        def alpha(b: float | None, r: BotNav = last) -> float | None:
            return None if b is None else r.cumulative_return_pct - b

        out[s] = {
            "date": str(last.date), "nav": last.nav_usd, "cumulative_return_pct": last.cumulative_return_pct,
            "drawdown_pct": last.drawdown_pct, "peak_nav": last.peak_nav_usd, "n_positions": last.n_positions,
            "deployed_pct": last.invested_usd / last.nav_usd if last.nav_usd else 0.0,
            "benchmark_60_40_return_pct": last.bench_6040_pct, "benchmark_sp500_return_pct": last.bench_sp500_pct,
            "benchmark_acwi_return_pct": last.bench_acwi_pct,
            "alpha_vs_60_40": alpha(last.bench_6040_pct), "alpha_vs_sp500": alpha(last.bench_sp500_pct),
            "alpha_vs_acwi": alpha(last.bench_acwi_pct), "reconciled": last.reconciled,
        }
    return out
