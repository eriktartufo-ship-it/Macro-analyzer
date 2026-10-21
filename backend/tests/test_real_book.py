"""Libro reale (2026-10-21): i 3 bot copiati su ETF UCITS in euro.

Ogni test nomina il difetto che impedisce. Quelli che contano di più:
- il pavimento di investimento cambia le posizioni SENZA decisione ⇒ il libro deve vederlo;
- una decisione di sabato si esegue all'apertura di lunedì, non al prezzo di venerdì;
- chi compra il giorno dello stacco non incassa la cedola;
- il racconto non può contenere un numero che la decisione non aveva;
- il diario pubblico non nomina strumenti né ISIN.
"""
from __future__ import annotations

import json
from datetime import date, datetime

import pandas as pd
import pytest

from app.services.real_book import instruments as I
from app.services.real_book import ledger as L
from app.services.real_book import narrator as N
from app.services.real_book.market import eu_summer_time, open_time_utc


# ---------------------------------------------------------------------------
# Registro strumenti
# ---------------------------------------------------------------------------

def _isin_ok(isin: str) -> bool:
    """Cifra di controllo ISIN (Luhn sulle lettere convertite in numeri)."""
    if len(isin) != 12 or not isin[:2].isalpha():
        return False
    digits = "".join(str(int(c, 36)) for c in isin[:-1])
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            d = d - 9 if d > 9 else d
        total += d
    return (10 - total % 10) % 10 == int(isin[-1])


def test_ogni_asset_del_modello_ha_uno_strumento_vero():
    # censimento: un asset nuovo nello scoring senza gemello UCITS fa fallire qui, non in prod
    from app.services.scoring.engine import ASSET_CLASSES

    mancanti = [a for a in ASSET_CLASSES if I.instrument_for(a) is None]
    assert mancanti == []
    assert len(ASSET_CLASSES) >= 20  # controllo positivo: il censimento vede davvero la lista


def test_isin_hanno_la_cifra_di_controllo_giusta():
    sbagliati = [i.key for i in I.INSTRUMENTS.values() if not _isin_ok(i.isin)]
    assert sbagliati == []
    # controllo positivo: la funzione boccia un ISIN con una cifra cambiata
    assert _isin_ok("IE00B579F325") and not _isin_ok("IE00B579F326")


def test_tutti_gli_strumenti_quotano_in_euro_su_borse_raggiungibili():
    for i in I.INSTRUMENTS.values():
        assert i.yahoo.endswith((".DE", ".MI")), i.key
        assert 0 < i.half_spread <= 0.005, i.key


def test_concentrazione_nascosta_energy_dichiarata():
    shared = I.shared_instruments()
    assert sorted(shared["SXLE"]) == ["energy", "sector_energy"]


# ---------------------------------------------------------------------------
# Orario d'apertura (ora legale fatta a mano)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d,hour", [
    (date(2026, 10, 23), 7),   # venerdì prima del cambio: ora legale → 09:00 = 07:00 UTC
    (date(2026, 10, 26), 8),   # lunedì dopo l'ultima domenica di ottobre: ora solare
    (date(2026, 3, 27), 8),    # venerdì prima dell'ultima domenica di marzo
    (date(2026, 3, 30), 7),    # lunedì dopo
])
def test_apertura_alle_nove_locali(d, hour):
    assert open_time_utc(d) == datetime(d.year, d.month, d.day, hour, 0)


def test_cambio_ora_cade_sull_ultima_domenica():
    assert eu_summer_time(date(2026, 3, 29)) and not eu_summer_time(date(2026, 3, 28))
    assert not eu_summer_time(date(2026, 10, 25)) and eu_summer_time(date(2026, 10, 24))


# ---------------------------------------------------------------------------
# Contabilità pura
# ---------------------------------------------------------------------------

def test_quote_arrotondate_al_piu_vicino_non_per_difetto():
    # 3000 / 646 = 4,64 → 5 quote (per difetto sarebbero 4 = sempre sotto-investiti)
    qty, note = L.size_buy(3000, 646.0, 0.0, available_cash=100_000)
    assert qty == 5 and note is None


def test_niente_leva_si_taglia_alla_cassa():
    qty, note = L.size_buy(10_000, 100.0, 0.0, available_cash=2_500)
    assert qty == 24  # (2500 - 0,99) / 100 = 24,99 → 24
    assert "liquidità" in note


def test_quota_piu_cara_della_tranche_non_si_compra_e_si_dice_perche():
    qty, note = L.size_buy(500, 1520.0, 0.0, available_cash=100_000)
    assert qty == 0 and "costa" in note


def test_apertura_salita_oltre_la_cassa_riduce_le_quote():
    f = L.execute_buy(10, open_price=110.0, half_spread=0.0, cash=1_000)
    assert f is not None and f.qty == 9 and f.note


def test_costo_medio_comprende_le_commissioni_e_utile_realizzato():
    st = L.LedgerState(cash=10_000)
    L.apply_buy(st, "gold", "SGLD", L.Fill(10, 100.0, 1000.0, 0.0, 0.99))
    L.apply_buy(st, "gold", "SGLD", L.Fill(10, 120.0, 1200.0, 0.0, 0.99))
    lot = st.lots["gold"]
    assert lot.avg_price == pytest.approx((2200 + 1.98) / 20)
    _, realized = L.apply_sell(st, "gold", L.Fill(20, 130.0, 2600.0, 0.0, 0.99))
    assert realized == pytest.approx(2600 - 0.99 - 2201.98)
    assert st.cash == pytest.approx(10_000 - 2201.98 + 2599.01)
    assert st.fees == pytest.approx(2.97)


def test_prezzo_eseguito_paga_lo_spread_nei_due_versi():
    assert L.fill_price(100, 0.001, "buy") == pytest.approx(100.1)
    assert L.fill_price(100, 0.001, "sell") == pytest.approx(99.9)


@pytest.mark.parametrize("lot,frac,exp", [(9, None, 9), (9, 1 / 3, 3), (2, 0.1, 1), (0, 0.5, 0), (5, 0.0, 0)])
def test_quote_da_vendere(lot, frac, exp):
    assert L.sell_qty(lot, frac) == exp


def test_strumento_senza_prezzo_vale_il_costo_non_zero():
    st = L.LedgerState(cash=0)
    L.apply_buy(st, "gold", "SGLD", L.Fill(1, 100.0, 100.0, 0.0, 0.0))
    st.cash = 0
    assert L.nav(st, {})[0] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Narratore e guardia
# ---------------------------------------------------------------------------

_FATTI = {"asset_nome": "Oro fisico", "punteggio": 58.0, "prob_regime_pct": 42.1, "motivo_del_bot": "rel_3m +3.0%"}


def test_guardia_boccia_un_numero_inventato():
    ok, why = N.check_narrative("Compro oro: punteggio 58 e inflazione al 4,2%.", json.dumps(_FATTI), [])
    assert not ok and "4,20" in why


def test_guardia_accetta_numeri_arrotondati_dai_fatti():
    ok, why = N.check_narrative("Compro oro: punteggio 58, regime al 42%, forza relativa +3% in 3 mesi.",
                                json.dumps(_FATTI), [])
    assert ok, why


def test_guardia_boccia_il_nome_dello_strumento_e_dell_emittente():
    forb = N.forbidden_tokens("SGLD")
    assert not N.check_narrative("Compro SGLD perché l'oro tiene, punteggio 58.", json.dumps(_FATTI), forb)[0]
    assert not N.check_narrative("Compro l'ETC di Invesco perché l'oro tiene.", json.dumps(_FATTI), forb)[0]
    assert N.check_narrative("Compro oro perché tiene, punteggio 58.", json.dumps(_FATTI), forb)[0]


@pytest.mark.parametrize("strategy", ["model_driven", "llm_driven", "risk_managed_sp"])
@pytest.mark.parametrize("kind,side", [
    ("tranche", "buy"), ("presa_in_carico", "buy"), ("floor", "buy"), ("riallineamento", "buy"),
    ("stop_loss", "sell"), ("take_profit", "sell"), ("close_tranche", "sell"), ("full_close", "sell"),
])
def test_il_perche_deterministico_non_nomina_mai_lo_strumento(strategy, kind, side):
    for asset in I.ASSET_MAP:
        inst = I.instrument_for(asset)
        f = {"strategy": strategy, "asset": asset, "asset_nome": I.label_for(asset), "evento": kind,
             "lato": "acquisto" if side == "buy" else "vendita", "tranche": 1, "tranche_totali": 2,
             "regime_it": "stagflazione", "prob_regime_pct": 42.1, "fiducia_pct": 54,
             "punteggio": 58.0, "pnl_pct": -10.2, "motivo_del_bot": "score 40 < exit threshold",
             "soglia_ingresso": 55, "soglia_uscita": 45, "stop_pct": -10, "tp_parziale_pct": 15,
             "tp_pct": 30, "floor_pct": 30, "aperta_il": "2026-07-12", "regime_di_ingresso_it": "stagflazione"}
        testo = N.why_template(f) + " " + (N.exit_plan(f) or "")
        for tok in N.forbidden_tokens(inst.key):
            assert tok.lower() not in testo.lower().replace("l'asset", ""), (asset, kind, tok, testo)


def test_il_racconto_non_aggiunge_cause_che_il_bot_non_ha_detto():
    # caso vero della fase di progetto: «RSI alto, per questo non aggiungo altro» era inventato
    f = {"strategy": "llm_driven", "asset": "sector_staples", "asset_nome": "Beni di prima necessità USA",
         "evento": "tranche", "lato": "acquisto", "tranche": 3, "tranche_totali": 3,
         "motivo_del_bot": "consumer_sentiment depresso (51.7); RSI 77 e breakout +4.1% su MA50",
         "regime_it": "stagflazione", "prob_regime_pct": 42.1}
    t = N.why_template(f)
    assert "fiducia dei consumatori" in t and "51.7" in t
    assert "non aggiungo" not in t


# ---------------------------------------------------------------------------
# Orchestrazione col DB (sqlite in memoria, mercato finto: nessuna rete)
# ---------------------------------------------------------------------------

class FakeMarket:
    """Barre per data; ogni simbolo ha gli stessi prezzi salvo override."""

    def __init__(self, days: dict[date, tuple[float, float]], per_symbol=None, divs=None):
        self.days = days
        self.per_symbol = per_symbol or {}
        self.divs = divs or {}

    def bars(self, symbol, start, end):
        src = self.per_symbol.get(symbol, self.days)
        rows = {d: v for d, v in src.items() if start <= d <= end}
        return pd.DataFrame({"Open": [v[0] for v in rows.values()], "Close": [v[1] for v in rows.values()]},
                            index=list(rows.keys()))

    def dividends(self, symbol, start, end):
        s = self.divs.get(symbol, {})
        return pd.Series({d: v for d, v in s.items() if start <= d <= end}, dtype=float)


# venerdì 16/10, lunedì 19, martedì 20 (sabato/domenica senza barre)
DAYS = {date(2026, 10, 15): (100.0, 100.0), date(2026, 10, 16): (100.0, 100.0),
        date(2026, 10, 19): (110.0, 111.0), date(2026, 10, 20): (112.0, 113.0)}


@pytest.fixture
def db():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.database import Base
    from app.models.ai_portfolio import AiPortfolioDecision, AiPortfolioPerformance, AiPortfolioPosition
    from app.models.real_book import RbBook, RbBotSnapshot, RbMovement, RbNav
    from app.models.regime_classifications import RegimeClassification

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine, tables=[
        AiPortfolioPosition.__table__, AiPortfolioDecision.__table__, AiPortfolioPerformance.__table__,
        RegimeClassification.__table__, RbBook.__table__, RbMovement.__table__,
        RbBotSnapshot.__table__, RbNav.__table__,
    ])
    s = sessionmaker(bind=engine)()
    s.add(RegimeClassification(
        date=date(2026, 10, 16), regime="stagflation", probability_reflation=0.385,
        probability_stagflation=0.421, probability_deflation=0.087, probability_goldilocks=0.107,
        confidence=0.543,
        conditions_met=json.dumps({"conditions": {"stagflation": {
            "gdp_weak": {"raw_score": 1, "weight": 0.09, "weighted_score": 0.09},
            # sfumata: conta al 63% anche se il CPI è 3,717 (caso vero del 21/10/2026)
            "inflation_high": {"raw_score": 0.627, "weight": 0.08, "weighted_score": 0.05}}},
            "indicators": {"cpi_yoy": 3.717}}),
    ))
    s.add(AiPortfolioPerformance(date=date(2026, 10, 16), strategy_type="model_driven", nav=112_000.0))
    s.commit()
    yield s
    s.close()


def _pos(db, asset, w, filled, total, opened=date(2026, 7, 12), last="OPEN_TRANCHE", last_date=None,
         strategy="model_driven"):
    from app.models.ai_portfolio import AiPortfolioPosition

    p = AiPortfolioPosition(asset_class=asset, strategy_type=strategy, target_weight_pct=w,
                            current_weight_pct=w, tranches_filled=filled, tranches_total=total,
                            opened_at=opened, entry_regime="stagflation", avg_entry_score=60.0,
                            estimated_pnl_pct=0.0, took_partial_tp=False, last_action=last,
                            last_action_date=last_date or opened)
    db.add(p)
    db.commit()
    return p


def _dec(db, asset, action, d, reason, size=0.0, k=0, n=0, pnl=None, strategy="model_driven"):
    from app.models.ai_portfolio import AiPortfolioDecision

    db.add(AiPortfolioDecision(date=d, asset_class=asset, strategy_type=strategy, action=action,
                               size_pct=size, tranche_index=k, tranches_total=n, score=58.0,
                               confidence=0.54, regime="stagflation", momentum_signal="BULLISH",
                               vol_60d=0.18, estimated_pnl_pct=pnl, reason_short=reason))
    db.commit()


def _mov(db, **f):
    from app.models.real_book import RbMovement

    q = db.query(RbMovement).filter_by(strategy_type="model_driven")
    for k, v in f.items():
        q = q.filter(getattr(RbMovement, k) == v)
    return q.order_by(RbMovement.id).all()


SAT = datetime(2026, 10, 17, 0, 30)
SUN = datetime(2026, 10, 18, 0, 30)
MON = datetime(2026, 10, 19, 0, 30)


def test_presa_in_carico_e_esecuzione_all_apertura_di_lunedi(db):
    from app.services.real_book import book

    _pos(db, "gold", 0.20, 3, 3)
    _dec(db, "gold", "OPEN_TRANCHE", date(2026, 7, 15), "score 64 strong, add tranche 3/3", 0.07, 3, 3)
    m = FakeMarket(DAYS)
    book.mirror(db, "model_driven", m, SAT)
    db.commit()
    [mv] = _mov(db, side="buy")
    assert mv.kind == "presa_in_carico" and mv.status == "pending"
    assert mv.quantity == 200  # 20% di 100.000 € / 100 € (chiusura di venerdì)
    assert "presa" not in (mv.narrative or "").lower() or "allineo" in mv.narrative

    # sabato sera: nessuna barra ⇒ resta in attesa (non si esegue al prezzo di venerdì)
    assert book.fill_pending(db, "model_driven", m, date(2026, 10, 18)) == 0
    assert book.fill_pending(db, "model_driven", m, date(2026, 10, 19)) == 1
    db.refresh(mv)
    assert mv.status == "filled" and mv.fill_date == date(2026, 10, 19)
    assert mv.fill_at == datetime(2026, 10, 19, 7, 0)  # 09:00 a Francoforte, ora legale
    sgld = I.INSTRUMENTS["SGLD"]
    assert mv.fill_price_eur == pytest.approx(110.0 * (1 + sgld.half_spread))
    assert mv.fee_eur == pytest.approx(0.99)


def test_tranche_di_sabato_e_domenica_partono_insieme_lunedi(db):
    from app.services.real_book import book

    m = FakeMarket(DAYS)
    book.mirror(db, "model_driven", m, datetime(2026, 10, 16, 0, 30))  # libro aperto, bot vuoto
    p = _pos(db, "energy", 0.045, 1, 2, opened=date(2026, 10, 17))
    _dec(db, "energy", "OPEN_TRANCHE", date(2026, 10, 17), "entry signal score 58.1", 0.045, 1, 2)
    book.mirror(db, "model_driven", m, SAT)
    p.current_weight_pct, p.tranches_filled, p.last_action_date = 0.09, 2, date(2026, 10, 18)
    db.commit()
    _dec(db, "energy", "OPEN_TRANCHE", date(2026, 10, 18), "score 58.0 strong, add tranche 2/2", 0.045, 2, 2)
    book.mirror(db, "model_driven", m, SUN)
    db.commit()
    book.fill_pending(db, "model_driven", m, date(2026, 10, 19))
    buys = _mov(db, side="buy", status="filled")
    assert [b.tranche_index for b in buys] == [1, 2]
    assert {b.fill_date for b in buys} == {date(2026, 10, 19)}
    assert buys[0].fill_price_eur == buys[1].fill_price_eur  # stesso prezzo d'apertura


def test_il_perche_macro_cita_i_valori_non_le_soglie_delle_condizioni(db):
    # il difetto: la condizione "CPI YoY > 4%" conta al 63% col CPI a 3,72 ⇒ scrivere la soglia
    # come causa sarebbe falso, e la guardia sui numeri non lo fermerebbe (il 4 è nei fatti)
    from app.services.real_book import book

    m = FakeMarket(DAYS)
    book.mirror(db, "model_driven", m, datetime(2026, 10, 16, 0, 30))  # libro aperto, bot vuoto
    _pos(db, "gold", 0.05, 1, 2, opened=date(2026, 10, 17))
    _dec(db, "gold", "OPEN_TRANCHE", date(2026, 10, 17), "entry signal score 58.1 momentum BULLISH", 0.05, 1, 2)
    book.mirror(db, "model_driven", m, SAT)
    db.commit()
    [mv] = _mov(db, side="buy")
    assert "inflazione USA 3,72%" in mv.narrative and "stagflazione" in mv.narrative
    import re
    assert re.search(r"(?<![\d,])4\s?%", mv.narrative) is None, mv.narrative
    assert "> 4" not in mv.narrative
    assert "CPI YoY" not in mv.narrative


def test_il_pavimento_senza_decisione_non_sparisce(db):
    # il difetto: deploy_boost.apply_deployment_floor cambia la posizione SENZA scrivere una
    # decisione. Un libro che seguisse le decisioni non comprerebbe niente qui.
    from app.services.real_book import book

    m = FakeMarket(DAYS)
    book.mirror(db, "model_driven", m, datetime(2026, 10, 16, 0, 30))
    _pos(db, "gold", 0.06, 1, 1, opened=date(2026, 10, 17), last="FLOOR_DEPLOY")
    book.mirror(db, "model_driven", m, SAT)
    db.commit()
    buys = _mov(db, side="buy")
    assert len(buys) == 1, "il pavimento di investimento è sparito dal libro"
    mv = buys[0]
    assert mv.kind == "floor" and mv.quantity == 60
    assert "investimento minimo" in mv.narrative and "non guarda l'andamento del prezzo" in mv.narrative


def test_chiusura_prima_dell_apertura_annulla_l_acquisto_in_attesa(db):
    from app.services.real_book import book

    m = FakeMarket(DAYS)
    book.mirror(db, "model_driven", m, datetime(2026, 10, 16, 0, 30))
    p = _pos(db, "gold", 0.05, 1, 2, opened=date(2026, 10, 17))
    book.mirror(db, "model_driven", m, SAT)
    db.delete(p)
    db.commit()
    _dec(db, "gold", "FULL_CLOSE", date(2026, 10, 18), "regime shift adverse (out of top-12)", pnl=0.0)
    book.mirror(db, "model_driven", m, SUN)
    db.commit()
    [buy] = _mov(db, side="buy")
    assert buy.status == "cancelled" and "cambiato idea" in buy.note
    assert _mov(db, side="sell") == []  # niente da vendere: non era mai stato comprato


def test_presa_di_profitto_parziale_vende_una_frazione(db):
    from app.services.real_book import book

    m = FakeMarket(DAYS)
    p = _pos(db, "gold", 0.15, 3, 3)
    book.mirror(db, "model_driven", m, datetime(2026, 10, 15, 0, 30))
    book.fill_pending(db, "model_driven", m, date(2026, 10, 15))
    p.current_weight_pct, p.tranches_filled, p.last_action, p.last_action_date = 0.10, 2, "CLOSE_TRANCHE", date(2026, 10, 17)
    db.commit()
    _dec(db, "gold", "CLOSE_TRANCHE", date(2026, 10, 17), "partial TP PnL 16.8%", pnl=0.168)
    book.mirror(db, "model_driven", m, SAT)
    book.fill_pending(db, "model_driven", m, date(2026, 10, 19))
    db.commit()
    [s] = _mov(db, side="sell")
    assert s.kind == "close_tranche" and s.status == "filled"
    assert s.quantity == 50  # 1/3 di 150 quote
    assert s.realized_pnl_eur > 0
    assert "+16,8%" in s.narrative


def test_cedola_solo_a_chi_aveva_le_quote_prima_dello_stacco(db):
    from app.services.real_book import book

    ibts = I.INSTRUMENTS["IBTS"].yahoo
    m = FakeMarket(DAYS, divs={ibts: {date(2026, 10, 19): 2.0}})
    _pos(db, "us_bonds_short", 0.10, 1, 1)
    book.mirror(db, "model_driven", m, datetime(2026, 10, 15, 0, 30))
    book.fill_pending(db, "model_driven", m, date(2026, 10, 15))   # 100 quote il 15/10
    db.commit()
    assert book.credit_dividends(db, "model_driven", m, date(2026, 10, 20), MON) == 1
    [d] = _mov(db, side="dividend")
    assert d.gross_eur == pytest.approx(200.0) and d.fill_date == date(2026, 10, 19)
    # idempotente: il giro successivo non la accredita due volte
    assert book.credit_dividends(db, "model_driven", m, date(2026, 10, 20), MON) == 0


def test_comprato_il_giorno_dello_stacco_niente_cedola(db):
    from app.services.real_book import book

    ibts = I.INSTRUMENTS["IBTS"].yahoo
    m = FakeMarket(DAYS, divs={ibts: {date(2026, 10, 19): 2.0}})
    _pos(db, "us_bonds_short", 0.10, 1, 1)
    book.mirror(db, "model_driven", m, SAT)
    book.fill_pending(db, "model_driven", m, date(2026, 10, 19))   # eseguito il 19 = giorno di stacco
    assert book.credit_dividends(db, "model_driven", m, date(2026, 10, 20), MON) == 0


def test_valore_del_libro_solo_nei_giorni_di_borsa_e_benchmark_in_euro(db):
    from app.services.real_book import book

    _pos(db, "gold", 0.20, 3, 3)
    m = FakeMarket(DAYS)
    book.mirror(db, "model_driven", m, SAT)
    book.fill_pending(db, "model_driven", m, date(2026, 10, 19))
    assert book.snapshot_nav(db, "model_driven", m, date(2026, 10, 18)) is None  # domenica
    row = book.snapshot_nav(db, "model_driven", m, date(2026, 10, 20))
    assert row is not None
    # 200 quote: valore a 113, cassa = 100.000 - 200 × prezzo eseguito - 0,99
    px = 110.0 * (1 + I.INSTRUMENTS["SGLD"].half_spread)
    assert row.nav_eur == pytest.approx(100_000 - 200 * px - 0.99 + 200 * 113.0)
    # benchmark: S&P in euro dall'apertura del primo giorno (110) alla chiusura (113)
    assert row.benchmark_return_pct == pytest.approx(113 / 110 - 1)
    assert json.loads(row.positions_json)[0]["asset_class"] == "gold"


def test_diario_pubblico_senza_strumenti_admin_con(db):
    from app.services.real_book import book, views

    _pos(db, "gold", 0.20, 3, 3)
    m = FakeMarket(DAYS)
    book.mirror(db, "model_driven", m, SAT)
    book.fill_pending(db, "model_driven", m, date(2026, 10, 19))
    db.commit()
    pub = views.journal(db, "model_driven", 10, admin=False)[0]
    txt = json.dumps(pub).lower()
    for tok in ("ie00b579f325", "sgld", "8psg", "invesco", "fill_price_eur", "quantity"):
        assert tok not in txt, tok
    assert pub["asset_label"] == "Oro fisico" and pub["gross_eur"] > 0
    adm = views.journal(db, "model_driven", 10, admin=True)[0]
    assert adm["instrument"]["isin"] == "IE00B579F325" and adm["quantity"] == 200


def test_concentrazione_segnalata_quando_due_asset_finiscono_sullo_stesso_etf(db):
    from app.services.real_book import book, views

    _pos(db, "energy", 0.09, 2, 2)
    _pos(db, "sector_energy", 0.21, 3, 3)
    m = FakeMarket(DAYS)
    book.mirror(db, "model_driven", m, SAT)
    book.fill_pending(db, "model_driven", m, date(2026, 10, 19))
    book.snapshot_nav(db, "model_driven", m, date(2026, 10, 19))
    db.commit()
    ov = views.overview(db, admin=False)["strategies"]["model_driven"]
    [c] = ov["concentration"]
    assert sorted(c["assets"]) == ["Energia USA", "Settore energia USA"]
    # quote fissate sulla chiusura di venerdì (100 €): 90 + 210 = 300. Il mercato apre a 110 e
    # chiude a 111 ⇒ il peso vero è più alto del 30% deciso: è l'ordine "al mercato in apertura".
    px = 110.0 * (1 + I.INSTRUMENTS["SXLE"].half_spread)
    nav = 100_000 - 300 * px - 2 * 0.99 + 300 * 111.0
    assert c["weight_pct"] == pytest.approx(300 * 111.0 / nav)
    assert c["weight_pct"] > 0.30
    assert "instrument_key" not in c
