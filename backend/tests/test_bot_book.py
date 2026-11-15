"""Libro dei bot in quote e cassa (2026-11-15, Sessione 57 — opzione A di Erik).

Ogni test nomina il difetto che impedisce. I tre che hanno fatto nascere il libro:
- un acquisto incassava la seduta PRIMA di essere comprato (curva +5% su un utile 0);
- un frazionamento 4:1 diventava -75% e faceva scattare lo stop loss;
- il «minimo investito» cambiava i pesi senza lasciare un movimento.
E quello che li tiene chiusi: la QUADRATURA (valore − capitale = realizzato + latente + cedole +
interessi), provata in tutti e due i versi: torna sul caso sano, scatta su un movimento alterato.
"""
from __future__ import annotations

import json
from datetime import date, datetime

import pandas as pd
import pytest

from app.services.ai_portfolio import bot_book as BB

MON, TUE, WED, THU, FRI = (date(2026, 11, d) for d in (9, 10, 11, 12, 13))
PREV_FRI = date(2026, 11, 6)


def run_at(d: date, hour: int = 5, minute: int = 30) -> datetime:
    return datetime(d.year, d.month, d.day, hour, minute)


class FakeMarket:
    """Barre giornaliere {ticker: {data: (apertura, chiusura)}} e cedole {ticker: {data: importo}}."""

    def __init__(self, bars: dict, divs: dict | None = None):
        self._bars, self._divs = bars, divs or {}

    def bars(self, symbol, start, end):
        rows = {d: v for d, v in self._bars.get(symbol, {}).items() if start <= d <= end}
        if not rows:
            return pd.DataFrame(columns=["Open", "Close"])
        idx = sorted(rows)
        return pd.DataFrame({"Open": [rows[d][0] for d in idx], "Close": [rows[d][1] for d in idx]}, index=idx)

    def dividends(self, symbol, start, end):
        d = {k: v for k, v in self._divs.get(symbol, {}).items() if start <= k <= end}
        return pd.Series(d, dtype=float)


def spy(days=(PREV_FRI, MON, TUE, WED, THU, FRI)) -> dict:
    return {d: (100.0, 100.0) for d in days}


@pytest.fixture
def db(monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.database import Base
    from app.models import ai_portfolio as AP
    from app.models import bot_book as MB
    from app.models import real_book as RB
    from app.models.regime_classifications import RegimeClassification
    from app.services import config_flags

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine, tables=[
        AP.AiPortfolioPosition.__table__, AP.AiPortfolioDecision.__table__, AP.AiPortfolioPerformance.__table__,
        AP.AiPortfolioLearning.__table__, AP.AiPortfolioReasoning.__table__, RegimeClassification.__table__,
        MB.BotBook.__table__, MB.BotMovement.__table__, MB.BotIntentSnapshot.__table__, MB.BotNav.__table__,
        RB.RbBook.__table__, RB.RbMovement.__table__, RB.RbBotSnapshot.__table__, RB.RbNav.__table__,
    ])
    monkeypatch.setattr(config_flags, "use_ai_portfolio_cash_overnight", lambda: True)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()


def _start(db, strategy="model_driven", started=MON):
    from app.models.bot_book import BotBook

    db.add(BotBook(strategy_type=strategy, started_on=started, initial_capital_usd=100_000.0))
    db.commit()


def _intent(db, asset, w, strategy="model_driven", tranches=(1, 2), last_action="OPEN_TRANCHE",
            opened=TUE):
    from app.models.ai_portfolio import AiPortfolioPosition

    p = AiPortfolioPosition(asset_class=asset, strategy_type=strategy, target_weight_pct=w, current_weight_pct=w,
                            tranches_filled=tranches[0], tranches_total=tranches[1], opened_at=opened,
                            entry_regime="stagflation", avg_entry_score=60.0, estimated_pnl_pct=0.0,
                            took_partial_tp=False, last_action=last_action, last_action_date=opened)
    db.add(p)
    db.commit()
    return p


def _settle(db, market, d, strategy="model_driven"):
    res = BB.settle(db, strategy, market, run_at(d))
    db.commit()
    return res


def _orders(db, res, d, strategy="model_driven"):
    out = BB.create_orders(db, strategy, run_at(d), res.nav, res.state)
    db.commit()
    return out


def _movs(db, **flt):
    from app.models.bot_book import BotMovement

    q = db.query(BotMovement)
    for k, v in flt.items():
        q = q.filter(getattr(BotMovement, k) == v)
    return q.order_by(BotMovement.id).all()


# ---------------------------------------------------------------------------
# 1. Il look-ahead è impossibile per costruzione
# ---------------------------------------------------------------------------

def test_un_acquisto_non_incassa_la_seduta_prima_di_essere_comprato(db):
    """XLV sale del 10% lunedì. Il giro di martedì mattina decide di comprarlo: lo compra
    all'apertura di martedì e da lì guadagna. Il +10% di lunedì non è suo."""
    m = FakeMarket({"SPY": spy(), "XLV": {PREV_FRI: (100, 100), MON: (100, 110), TUE: (110, 110)}})
    _start(db)
    r1 = _settle(db, m, TUE)                       # chiusura di lunedì: solo cassa
    assert r1.new_close and r1.close == MON and r1.nav == pytest.approx(100_000)
    _intent(db, "sector_healthcare", 0.5)
    (buy,) = _orders(db, r1, TUE)
    assert buy.side == "buy" and buy.amount_usd == pytest.approx(50_000)

    r2 = _settle(db, m, WED)                       # chiusura di martedì
    filled = _movs(db, side="buy", status="filled")
    assert len(filled) == 1 and filled[0].fill_date == TUE and filled[0].open_usd == 110
    # piatto da quando è stato comprato ⇒ il valore scende solo dei costi (1 $ + 2 bp di spread)
    costs = 1.0 + (50_000 - 1.0) / (110 * 1.0002) * 110 * 0.0002
    assert r2.nav == pytest.approx(100_000 - costs, abs=0.01)
    assert r2.nav < 100_000  # col vecchio calcolo sarebbe stato 105.000


def test_la_decisione_presa_dopo_l_apertura_va_alla_seduta_dopo():
    assert BB.fill_start(TUE, run_at(TUE, 5, 30)) == TUE
    assert BB.fill_start(TUE, run_at(TUE, 15, 0)) == WED   # dopo le 14:30 UTC di novembre


def test_la_cripto_si_esegue_nella_stessa_seduta_degli_etf_dopo_la_decisione():
    """La barra cripto del giorno apre a mezzanotte UTC, PRIMA della decisione delle 05:30: si usa la
    sua chiusura. E la seduta è quella di Wall Street, così le vendite di ETF pagano gli acquisti."""
    m = FakeMarket({"SPY": spy(), "BTC-USD": {TUE: (50_000.0, 51_000.0), WED: (51_000.0, 52_000.0)},
                    "XLV": {d: (100.0, 100.0) for d in spy()}})
    assert BB.execution_price(m, "BTC-USD", TUE, WED) == (TUE, 51_000.0)
    assert BB.execution_price(m, "XLV", TUE, WED) == (TUE, 100.0)
    sab = date(2026, 11, 14)
    assert BB.session_on_or_after(m, BB.fill_start(sab, run_at(sab)), date(2026, 11, 20)) is None  # niente lunedì nel finto


@pytest.mark.parametrize("d,summer", [
    (date(2026, 3, 7), False), (date(2026, 3, 8), True),       # seconda domenica di marzo
    (date(2026, 10, 31), True), (date(2026, 11, 1), False),    # prima domenica di novembre
])
def test_ora_legale_usa(d, summer):
    assert BB.us_summer_time(d) is summer


def test_senza_chiusura_nuova_il_giro_non_rifa_i_conti(db):
    m = FakeMarket({"SPY": spy()})
    _start(db)
    assert _settle(db, m, TUE).new_close
    again = _settle(db, m, TUE)
    assert not again.new_close and "nessuna chiusura nuova" in again.why


# ---------------------------------------------------------------------------
# 2. Un frazionamento non è un guadagno né una perdita
# ---------------------------------------------------------------------------

def test_un_frazionamento_non_e_una_perdita_e_non_fa_scattare_lo_stop(db):
    """QUAL comprata a 220 il 03/11; il 05/11 frazionamento 4:1 e Yahoo riscala TUTTA la storia:
    adesso dice che il 03/11 apriva a 55. Le quote del libro vanno lette ×4."""
    from app.models.bot_book import BotMovement
    from app.services.ai_portfolio import strategist

    _start(db)
    q = (40_000 - 1) / (220 * 1.0002)
    db.add(BotMovement(strategy_type="model_driven", asset_class="factor_quality", ticker="QUAL", side="buy",
                       kind="tranche", status="filled", decided_on=date(2026, 11, 3), decided_at=run_at(date(2026, 11, 3)),
                       fill_date=date(2026, 11, 3), quantity=q, open_usd=220.0, fill_price_usd=220 * 1.0002,
                       gross_usd=q * 220 * 1.0002, fee_usd=1.0, spread_usd=q * 220 * 0.0002, amount_usd=40_000))
    db.commit()
    post_split = {date(2026, 11, 3): (55.0, 55.0), PREV_FRI: (55.0, 55.0), MON: (55.0, 55.0)}
    m = FakeMarket({"SPY": spy(), "QUAL": post_split})
    res = _settle(db, m, TUE)
    pos_value = res.nav - res.state.cash
    assert pos_value == pytest.approx(q * 220, rel=1e-9)     # 4× le quote a un quarto del prezzo

    p = _intent(db, "factor_quality", 0.4, opened=date(2026, 11, 3))
    BB.sync_positions(db, "model_driven", res.state, m, res.close)
    assert p.estimated_pnl_pct == pytest.approx(-(1 + q * 220 * 0.0002) / 40_000, abs=1e-6)  # solo i costi
    action, reason, _ = strategist._decide("factor_quality", p, score=58.0, target=0.4, confidence=0.5,
                                           momentum_sig="NEUTRAL", in_favor=True, n_tranches_default=2, vol=0.15)
    assert action != "STOP_LOSS", reason


# ---------------------------------------------------------------------------
# 3. Nessun peso cambia senza un movimento
# ---------------------------------------------------------------------------

def test_il_minimo_investito_diventa_un_ordine(db, monkeypatch):
    from app.services.ai_portfolio import deploy_boost, pnl_tracker

    m = FakeMarket({"SPY": spy()})
    _start(db)
    r = _settle(db, m, TUE)
    monkeypatch.setattr(deploy_boost, "enabled", lambda: True)
    monkeypatch.setattr(pnl_tracker, "get_latest_price", lambda a, today=None: 100.0)
    deploy_boost.apply_deployment_floor(db, "model_driven", {"sector_healthcare": 70.0, "gold": 65.0},
                                        {"sector_healthcare", "gold"}, TUE, "stagflation")
    db.commit()
    orders = _orders(db, r, TUE)
    assert {o.kind for o in orders} == {"floor"}
    assert sum(o.weight_delta_pct for o in orders) == pytest.approx(0.30)
    assert sum(o.amount_usd for o in orders) == pytest.approx(30_000)
    assert all(o.reason for o in orders)  # il perché c'è anche senza una decisione della strategia


def test_tranche_chiusura_parziale_e_uscita_diventano_ordini_coerenti(db):
    from app.models.ai_portfolio import AiPortfolioPosition

    m = FakeMarket({"SPY": spy(), "XLV": {d: (100.0, 100.0) for d in spy()}})
    _start(db)
    r = _settle(db, m, TUE)
    _intent(db, "sector_healthcare", 0.2, tranches=(1, 2))
    _orders(db, r, TUE)
    r = _settle(db, m, WED)
    p = db.query(AiPortfolioPosition).one()
    p.current_weight_pct, p.tranches_filled = 0.4, 2                    # seconda tranche
    db.commit()
    (o2,) = _orders(db, r, WED)
    assert (o2.side, o2.kind) == ("buy", "tranche") and o2.weight_delta_pct == pytest.approx(0.2)
    r = _settle(db, m, THU)
    p.current_weight_pct, p.tranches_filled, p.last_action = 0.2, 1, "CLOSE_TRANCHE"   # presa parziale
    db.commit()
    (o3,) = _orders(db, r, THU)
    assert (o3.side, o3.kind) == ("sell", "close_tranche") and o3.sell_fraction == pytest.approx(0.5)
    r = _settle(db, m, FRI)
    db.delete(p)
    db.commit()
    (o4,) = _orders(db, r, FRI)
    assert o4.side == "sell" and o4.sell_fraction is None


# ---------------------------------------------------------------------------
# 4. Quadratura: torna sul caso sano, scatta su un movimento alterato
# ---------------------------------------------------------------------------

def _scenario(db):
    """Compra, il prezzo sale, vende metà, stacca una cedola, interessi sulla liquidità, vende tutto."""
    from app.models.ai_portfolio import AiPortfolioPosition

    days = [PREV_FRI, MON, TUE, WED, THU, FRI, date(2026, 11, 16), date(2026, 11, 17)]
    xlv = dict(zip(days, [(100, 100), (100, 100), (100, 104), (105, 108), (107, 110), (111, 112),
                          (112, 113), (113, 113)]))
    bil = {d: (91.5, 91.5 * (1 + 0.0002 * i)) for i, d in enumerate(days)}
    m = FakeMarket({"SPY": spy(days), "XLV": xlv, "BIL": bil}, divs={"XLV": {THU: 0.5}})
    _start(db)
    r = _settle(db, m, TUE)
    _intent(db, "sector_healthcare", 0.6, tranches=(2, 2))
    _orders(db, r, TUE)
    r = _settle(db, m, WED)
    p = db.query(AiPortfolioPosition).one()
    p.current_weight_pct, p.tranches_filled, p.last_action = 0.3, 1, "CLOSE_TRANCHE"
    db.commit()
    _orders(db, r, WED)
    rows = [_settle(db, m, THU)]
    db.delete(p)
    db.commit()
    _orders(db, rows[-1], THU)
    rows.append(_settle(db, m, FRI))
    rows.append(_settle(db, m, date(2026, 11, 17)))
    return m, rows


def test_i_conti_quadrano_su_una_sequenza_completa(db):
    from app.models.bot_book import BotNav

    _m, _ = _scenario(db)
    navs = db.query(BotNav).order_by(BotNav.date).all()
    assert len(navs) >= 4
    assert all(n.reconciled for n in navs), [(n.date, n.reconciliation_gap_usd) for n in navs]
    last = navs[-1]
    explained = last.realized_cum_usd + last.unrealized_usd + last.dividends_cum_usd + last.interest_cum_usd
    assert last.nav_usd - 100_000 == pytest.approx(explained, abs=0.01)
    # controllo positivo: la sequenza ha davvero prodotto ognuna delle voci
    assert last.realized_cum_usd != 0 and last.dividends_cum_usd > 0 and last.interest_cum_usd > 0
    assert last.n_positions == 0 and last.cash_usd == pytest.approx(last.nav_usd)
    sells = _movs(db, side="sell", status="filled")
    assert len(sells) == 2 and all(s.realized_pnl_usd is not None for s in sells)


def test_la_quadratura_scatta_se_un_movimento_e_alterato_e_il_bot_non_decide(db, monkeypatch):
    from app.models.bot_book import BotNav

    m, _ = _scenario(db)
    sell = _movs(db, side="sell", status="filled")[0]
    sell.realized_pnl_usd += 50.0          # un utile scritto male sul movimento
    db.commit()
    db.query(BotNav).filter(BotNav.date == date(2026, 11, 16)).delete()
    db.commit()
    called = []
    monkeypatch.setattr(BB, "_strategy_module", lambda s: called.append(s))
    out = BB.run_giro(db, now=run_at(date(2026, 11, 17)), market=m, strategies=("model_driven",), reasoning=False)
    assert "bloccato" in out["model_driven"] and called == []
    last = db.query(BotNav).order_by(BotNav.date.desc()).first()
    assert last.reconciled is False and last.reconciliation_gap_usd == pytest.approx(-50.0, abs=0.01)


def test_niente_leva_un_acquisto_oltre_la_cassa_si_riduce(db):
    m = FakeMarket({"SPY": spy(), "XLV": {d: (100.0, 100.0) for d in spy()}})
    _start(db)
    r = _settle(db, m, TUE)
    _intent(db, "sector_healthcare", 1.2)       # intento oltre il capitale
    (buy,) = _orders(db, r, TUE)
    assert buy.amount_usd == pytest.approx(100_000) and "cassa" in (buy.note or "")   # ridotto all'ordine
    r2 = _settle(db, m, WED)
    assert buy.status == "filled" and r2.state.cash == pytest.approx(0.0, abs=1e-6) and r2.reconciled
    # il 20% che manca resta «da ordinare», ma senza cassa non nasce un ordine ogni giorno
    assert _orders(db, r2, WED) == []


# ---------------------------------------------------------------------------
# 5. Apprendimento dall'utile REALIZZATO
# ---------------------------------------------------------------------------

def test_l_apprendimento_registra_l_utile_realizzato_all_esecuzione(db):
    from app.models.ai_portfolio import AiPortfolioDecision, AiPortfolioLearning

    xlv = {PREV_FRI: (100, 100), MON: (100, 100), TUE: (100, 100), WED: (90, 90), THU: (90, 90)}
    m = FakeMarket({"SPY": spy(), "XLV": xlv})
    _start(db)
    r = _settle(db, m, TUE)
    p = _intent(db, "sector_healthcare", 0.5)
    _orders(db, r, TUE)
    r = _settle(db, m, WED)
    db.add(AiPortfolioDecision(date=WED, asset_class="sector_healthcare", strategy_type="model_driven",
                               action="STOP_LOSS", regime="stagflation", reason_short="stop loss PnL -10%"))
    db.delete(p)
    db.commit()
    (o,) = _orders(db, r, WED)
    assert o.kind == "stop_loss" and o.reason == "stop loss PnL -10%"
    assert db.query(AiPortfolioLearning).count() == 0          # alla decisione non si impara niente
    _settle(db, m, THU)
    (learn,) = db.query(AiPortfolioLearning).all()
    sell = _movs(db, side="sell", status="filled")[0]
    assert learn.n_losses == 1
    assert learn.avg_pnl_pct == pytest.approx(sell.realized_pnl_usd / sell.cost_basis_sold_usd)
    assert -0.11 < learn.avg_pnl_pct < -0.09


# ---------------------------------------------------------------------------
# 6. Ripartenza da zero: si archivia, poi si svuota
# ---------------------------------------------------------------------------

def test_la_ripartenza_archivia_tutto_prima_di_svuotare(db):
    from sqlalchemy import inspect, text

    from app.models.ai_portfolio import AiPortfolioPerformance
    from app.models.bot_book import BotBook

    for i in range(3):
        db.add(AiPortfolioPerformance(date=date(2026, 11, 2 + i), strategy_type="model_driven", nav=100_000 + i,
                                      peak_nav=100_000 + i))
    db.commit()
    out = BB.restart(db, today=date(2026, 11, 15), summary={"model_driven": {"cumulative_return_pct": 0.063}})
    assert out["archived_rows"]["ai_portfolio_performance"] == 3
    tables = set(inspect(db.get_bind()).get_table_names())
    assert "arch_20261115_ai_portfolio_performance" in tables
    assert db.execute(text("SELECT COUNT(*) FROM arch_20261115_ai_portfolio_performance")).scalar() == 3
    assert db.query(AiPortfolioPerformance).count() == 0
    books = db.query(BotBook).all()
    assert sorted(b.strategy_type for b in books) == sorted(BB.STRATEGIES)
    assert json.loads(books[0].archive_json)["label"] == "20261115"
    with pytest.raises(ValueError):
        BB.restart(db, today=date(2026, 11, 15), summary={})


def test_senza_libro_avviato_i_bot_non_girano(db):
    m = FakeMarket({"SPY": spy()})
    out = BB.run_giro(db, now=run_at(TUE), market=m, strategies=("model_driven",), reasoning=False)
    assert "libro non avviato" in out["model_driven"]["skipped"]


# ---------------------------------------------------------------------------
# 7. Giro completo con una strategia finta
# ---------------------------------------------------------------------------

def test_giro_completo_decide_ordina_esegue_e_quadra(db, monkeypatch):
    from app.models.ai_portfolio import AiPortfolioPosition

    class FakeStrategy:
        @staticmethod
        def daily_scan(db_):
            if not db_.query(AiPortfolioPosition).count():
                db_.add(AiPortfolioPosition(asset_class="gold", strategy_type="model_driven", target_weight_pct=0.25,
                                            current_weight_pct=0.25, tranches_filled=1, tranches_total=1,
                                            opened_at=TUE, entry_regime="stagflation", avg_entry_score=60.0,
                                            estimated_pnl_pct=0.0, took_partial_tp=False,
                                            last_action="OPEN_TRANCHE", last_action_date=TUE))
                db_.commit()
            return {"ok": True}

    monkeypatch.setattr(BB, "_strategy_module", lambda s: FakeStrategy)
    gld = {d: (200.0, 200.0 + i) for i, d in enumerate(spy())}
    m = FakeMarket({"SPY": spy(), "GLD": gld})
    _start(db)
    out1 = BB.run_giro(db, now=run_at(TUE), market=m, strategies=("model_driven",), reasoning=False)
    assert out1["model_driven"]["nuovi_ordini"] == 1
    out2 = BB.run_giro(db, now=run_at(WED), market=m, strategies=("model_driven",), reasoning=False)
    assert out2["model_driven"]["eseguiti"] == 1 and out2["model_driven"]["nuovi_ordini"] == 0
    view = BB.performance_view(db, "model_driven")
    assert view["current"]["reconciled"] is True and view["current"]["n_positions"] == 1
    pos = BB.positions_view(db, "model_driven")
    assert pos[0]["status"] == "in portafoglio" and pos[0]["value_usd"] > 0
    movs = BB.movements_view(db, "model_driven")
    assert movs[0]["status"] == "filled" and movs[0]["quantity"] > 0 and movs[0]["fill_price_usd"] > 200


def test_il_libro_reale_legge_il_valore_del_bot_dal_libro_in_quote(db):
    """Il «costo della realtà» confronta il libro in euro col bot: il bot ora è il suo libro."""
    from app.models.bot_book import BotNav
    from app.services.real_book import book as RB
    from app.services.real_book import views as RV

    for d, nav in ((MON, 100_000.0), (TUE, 101_000.0), (WED, 102_010.0)):
        db.add(BotNav(date=d, strategy_type="model_driven", nav_usd=nav, cash_usd=nav, invested_usd=0.0,
                      peak_nav_usd=nav))
    db.commit()
    assert RB._latest_bot_nav(db, "model_driven", TUE) == pytest.approx(101_000.0)
    assert RB._latest_bot_nav(db, "model_driven") == pytest.approx(102_010.0)
    at = RV._bot_return_at(db, "model_driven", TUE)
    assert at(WED) == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# 8. Regressioni dalla revisione indipendente del 15/11 (scenari riprodotti, poi corretti)
# ---------------------------------------------------------------------------

MON2, TUE2 = date(2026, 11, 16), date(2026, 11, 17)
SESSIONS = (PREV_FRI, MON, TUE, WED, THU, FRI, MON2, TUE2)


def _flat(*tickers, days=SESSIONS):
    return {t: {d: (100.0, 100.0) for d in days} for t in tickers}


def test_rev_quantita_alterata_non_passa_la_verifica(db):
    m = FakeMarket({"SPY": spy(SESSIONS), **_flat("XLV")})
    _start(db)
    r = _settle(db, m, TUE)
    _intent(db, "sector_healthcare", 0.5)
    _orders(db, r, TUE)
    _settle(db, m, WED)
    (buy,) = _movs(db, side="buy", status="filled")
    buy.quantity *= 4
    db.commit()
    r = _settle(db, m, THU)
    assert not r.reconciled and "quantità" in r.note


def test_rev_frazionamento_senza_apertura_del_giorno_d_acquisto_blocca_invece_di_indovinare(db):
    from app.models.bot_book import BotMovement

    _start(db)
    q = (40_000 - 1) / (220 * 1.0002)
    db.add(BotMovement(strategy_type="model_driven", asset_class="factor_quality", ticker="QUAL", side="buy",
                       kind="tranche", status="filled", decided_on=date(2026, 11, 3),
                       decided_at=run_at(date(2026, 11, 3)), fill_date=date(2026, 11, 3), quantity=q,
                       open_usd=220.0, fill_price_usd=220 * 1.0002, gross_usd=q * 220 * 1.0002, fee_usd=1.0,
                       spread_usd=q * 220 * 0.0002, amount_usd=40_000))
    db.commit()
    # nessuna barra entro 5 giorni dal 03/11: la scala non si indovina, si blocca e lo si dice
    m = FakeMarket({"SPY": spy(), "QUAL": {date(2026, 11, 9): (55.0, 55.0)}})
    r = _settle(db, m, TUE)
    assert not r.reconciled and "scala non verificabile" in r.note


def test_rev_barra_mancante_ma_vicina_ricava_il_fattore_di_frazionamento(db):
    """Il 03/11 manca, il 06/11 c'è (55 contro 220 pagati): 4 è un fattore plausibile entro il 6%,
    quindi le quote si leggono ×4 e la posizione vale quanto è costata, non −75%."""
    from app.models.bot_book import BotMovement

    _start(db)
    q = (40_000 - 1) / (220 * 1.0002)
    db.add(BotMovement(strategy_type="model_driven", asset_class="factor_quality", ticker="QUAL", side="buy",
                       kind="tranche", status="filled", decided_on=date(2026, 11, 3),
                       decided_at=run_at(date(2026, 11, 3)), fill_date=date(2026, 11, 3), quantity=q,
                       open_usd=220.0, fill_price_usd=220 * 1.0002, gross_usd=q * 220 * 1.0002, fee_usd=1.0,
                       spread_usd=q * 220 * 0.0002, amount_usd=40_000))
    db.commit()
    m = FakeMarket({"SPY": spy(), "QUAL": {PREV_FRI: (55.0, 55.0), MON: (55.0, 55.0)}})
    r = _settle(db, m, TUE)
    assert r.reconciled and (r.nav - r.state.cash) == pytest.approx(q * 220, rel=1e-9)


def test_rev_salto_da_frazionamento_non_ancora_riscalato_blocca_lo_stop_loss(db):
    """Yahoo mostra 55 sulla barra nuova ma lo storico è ancora a 220: −75% in una seduta non è un
    prezzo vero per un ETF largo. Il bot non decide quel giorno (lo stop loss non scatta)."""
    days = (PREV_FRI, MON, TUE, WED)
    m = FakeMarket({"SPY": spy(days), "QUAL": {PREV_FRI: (220.0, 220.0), MON: (220.0, 220.0),
                                              TUE: (220.0, 220.0), WED: (55.0, 55.0)}})
    _start(db)
    r = _settle(db, m, TUE)
    _intent(db, "factor_quality", 0.4)
    _orders(db, r, TUE)
    assert _settle(db, m, WED).reconciled
    r = _settle(db, m, THU)
    assert not r.reconciled and "frazionamento non ancora riscalato" in r.note


def test_rev_cedola_doppia_non_passa_e_il_giro_non_la_ripete(db):
    from app.models.bot_book import BotMovement

    m = FakeMarket({"SPY": spy(SESSIONS), **_flat("XLV")}, divs={"XLV": {THU: 0.5}})
    _start(db)
    r = _settle(db, m, TUE)
    _intent(db, "sector_healthcare", 0.5)
    _orders(db, r, TUE)
    _settle(db, m, WED)
    _settle(db, m, FRI)
    assert len(_movs(db, side="dividend")) == 1
    (dv,) = _movs(db, side="dividend")
    db.add(BotMovement(strategy_type="model_driven", asset_class=dv.asset_class, ticker=dv.ticker, side="dividend",
                       kind="dividend", status="filled", decided_on=dv.decided_on, decided_at=dv.decided_at,
                       fill_date=dv.fill_date, quantity=dv.quantity, gross_usd=dv.gross_usd))
    db.commit()
    r = _settle(db, m, MON2)
    assert not r.reconciled and "cedola doppia" in r.note


def test_rev_esecuzione_e_rigioco_seguono_lo_stesso_ordine(db):
    """Giro manuale dopo l'apertura (acquisto rimandato a giovedì) + giro del mattino che vende
    metà (anche lui giovedì): l'acquisto ha l'id più vecchio. Esecuzione e rigioco devono fare
    entrambi vendita-poi-acquisto, sennò il libro perde ~10.000 $ e il bot resta bloccato."""
    from app.models.ai_portfolio import AiPortfolioPosition

    xlv = {d: (100.0, 100.0) for d in SESSIONS}
    xlv[THU] = xlv[FRI] = xlv[MON2] = (120.0, 120.0)
    m = FakeMarket({"SPY": spy(SESSIONS), "XLV": xlv})
    _start(db)
    r = _settle(db, m, TUE)
    _intent(db, "sector_healthcare", 0.2, tranches=(1, 2))
    _orders(db, r, TUE)
    r = BB.settle(db, "model_driven", m, run_at(WED, 15, 0))
    db.commit()
    p = db.query(AiPortfolioPosition).one()
    p.current_weight_pct, p.tranches_filled = 0.4, 2
    db.commit()
    BB.create_orders(db, "model_driven", run_at(WED, 15, 0), r.nav, r.state)
    db.commit()
    r = _settle(db, m, THU)
    p.current_weight_pct, p.tranches_filled, p.last_action = 0.2, 1, "CLOSE_TRANCHE"
    db.commit()
    _orders(db, r, THU)
    r_fri = _settle(db, m, FRI)
    r_mon = _settle(db, m, TUE2)
    assert r_fri.reconciled and r_mon.reconciled
    assert r_mon.nav == pytest.approx(r_fri.nav, abs=0.01)   # prezzi fermi: il valore non si muove


def test_rev_l_apprendimento_usa_il_regime_d_ingresso(db):
    from app.models.ai_portfolio import AiPortfolioDecision, AiPortfolioLearning

    xlv = {PREV_FRI: (100, 100), MON: (100, 100), TUE: (100, 100), WED: (90, 90), THU: (90, 90)}
    m = FakeMarket({"SPY": spy(), "XLV": xlv})
    _start(db)
    r = _settle(db, m, TUE)
    p = _intent(db, "sector_healthcare", 0.5)          # entrata in stagflazione
    _orders(db, r, TUE)
    r = _settle(db, m, WED)
    db.add(AiPortfolioDecision(date=WED, asset_class="sector_healthcare", strategy_type="model_driven",
                               action="STOP_LOSS", regime="goldilocks", reason_short="stop"))
    db.delete(p)
    db.commit()
    _orders(db, r, WED)
    _settle(db, m, THU)
    (learn,) = db.query(AiPortfolioLearning).all()
    assert "regime=stagflation" in learn.pattern_key


def test_rev_un_acquisto_rimasto_indietro_si_riprende_quando_arriva_la_cassa(db):
    from app.models.ai_portfolio import AiPortfolioPosition

    m = FakeMarket({"SPY": spy(SESSIONS), **_flat("XLV", "GLD")})
    _start(db)
    r = _settle(db, m, TUE)
    _intent(db, "gold", 0.15, tranches=(1, 1))
    _intent(db, "sector_healthcare", 0.95, tranches=(1, 1))   # 110%: la sanità resta corta del 10%
    first = {o.asset_class: o for o in _orders(db, r, TUE)}
    assert first["gold"].amount_usd == pytest.approx(15_000)
    assert first["sector_healthcare"].amount_usd == pytest.approx(85_000)
    r = _settle(db, m, WED)
    assert _orders(db, r, WED) == []                   # niente cassa: nessun ordine ripetuto
    gold = db.query(AiPortfolioPosition).filter_by(asset_class="gold").one()
    db.delete(gold)                                    # il bot esce dall'oro: arriva cassa
    db.commit()
    orders = {(o.side, o.asset_class): o for o in _orders(db, r, THU)}
    assert ("sell", "gold") in orders
    retry = orders[("buy", "sector_healthcare")]       # il 10% rimasto indietro riparte
    assert retry.weight_delta_pct == pytest.approx(0.10, abs=1e-6)
    assert retry.amount_usd == pytest.approx(r.nav * 0.10, rel=1e-6)


def test_rev_interessi_solo_sulla_liquidita_rimasta_ferma(db):
    days = SESSIONS
    bil = {d: (91.5, 91.5 * (1 + 0.001 * i)) for i, d in enumerate(days)}
    m = FakeMarket({"SPY": spy(days), **_flat("XLV"), "BIL": bil})
    _start(db)
    r = _settle(db, m, TUE)
    p = _intent(db, "sector_healthcare", 1.0, tranches=(1, 1))
    _orders(db, r, TUE)
    r = _settle(db, m, WED)                            # tutto investito all'apertura di martedì
    db.delete(p)
    db.commit()
    _orders(db, r, WED)                                # vende all'apertura di mercoledì
    _settle(db, m, THU)
    assert [mv for mv in _movs(db, side="interest") if mv.fill_date == WED] == []  # investita fino all'apertura
    _settle(db, m, FRI)
    (it,) = [mv for mv in _movs(db, side="interest") if mv.fill_date == THU]
    assert it.gross_usd > 90                           # dal giorno dopo, sì


def test_rev_scambio_etf_cripto_di_sabato_si_paga_nella_stessa_seduta(db):
    days = SESSIONS
    sab = date(2026, 11, 14)
    m = FakeMarket({"SPY": spy(days), **_flat("XLV"),
                    "BTC-USD": {d: (50_000.0, 50_000.0) for d in (sab, date(2026, 11, 15), MON2, TUE2)}})
    _start(db)
    r = _settle(db, m, THU)
    p = _intent(db, "sector_healthcare", 1.0, tranches=(1, 1), opened=THU)
    _orders(db, r, THU)
    _settle(db, m, FRI)
    r = _settle(db, m, sab)
    db.delete(p)
    db.commit()
    _intent(db, "bitcoin", 1.0, tranches=(1, 1), opened=sab)
    _orders(db, r, sab)
    r = _settle(db, m, TUE2)
    fills = {(mv.asset_class, mv.side): mv for mv in _movs(db, status="filled") if mv.decided_on == sab}
    assert fills[("sector_healthcare", "sell")].fill_date == MON2 == fills[("bitcoin", "buy")].fill_date
    assert "bitcoin" in r.state.positions and r.reconciled


def test_rev_dopo_un_crash_fra_intento_e_ordini_il_giro_li_recupera(db, monkeypatch):
    from app.models.ai_portfolio import AiPortfolioPosition

    class FakeStrategy:
        @staticmethod
        def daily_scan(db_):
            if not db_.query(AiPortfolioPosition).count():
                _intent(db_, "gold", 0.25, tranches=(1, 1))
            return {"ok": True}

    monkeypatch.setattr(BB, "_strategy_module", lambda s: FakeStrategy)
    m = FakeMarket({"SPY": spy(SESSIONS), **_flat("GLD")})
    _start(db)
    real = BB.create_orders
    monkeypatch.setattr(BB, "create_orders", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    out1 = BB.run_giro(db, now=run_at(TUE), market=m, strategies=("model_driven",), reasoning=False)
    monkeypatch.setattr(BB, "create_orders", real)
    out2 = BB.run_giro(db, now=run_at(TUE, 6, 0), market=m, strategies=("model_driven",), reasoning=False)
    assert "errore" in out1["model_driven"] and out2["model_driven"]["ordini_recuperati"] == 1
    assert [(mv.asset_class, mv.side) for mv in _movs(db)] == [("gold", "buy")]


def test_rev_le_piccole_differenze_si_accumulano_invece_di_sparire(db):
    from app.models.ai_portfolio import AiPortfolioPosition

    m = FakeMarket({"SPY": spy(SESSIONS), **_flat("XLV")})
    _start(db)
    r = _settle(db, m, TUE)
    _intent(db, "sector_healthcare", 0.30, tranches=(1, 1))
    _orders(db, r, TUE)
    r = _settle(db, m, WED)
    p = db.query(AiPortfolioPosition).one()
    p.current_weight_pct, p.last_action = 0.3003, "FLOOR_DEPLOY"     # +0,03%: sotto la soglia
    db.commit()
    assert _orders(db, r, WED) == []
    p.current_weight_pct = 0.3007                                      # in tutto +0,07%: ora sì
    db.commit()
    (o,) = _orders(db, r, WED)
    assert o.weight_delta_pct == pytest.approx(0.0007, abs=1e-9)


def test_rev_il_valore_di_una_posizione_ferma_deve_seguire_il_suo_prezzo(db):
    """Riga coerente (importo = quantità × prezzo) ma riferimento di scala alterato: la quadratura
    non lo vede (è un'identità), il controllo di continuità sì."""
    m = FakeMarket({"SPY": spy(SESSIONS), **_flat("XLV")})
    _start(db)
    r = _settle(db, m, TUE)
    _intent(db, "sector_healthcare", 0.5)
    _orders(db, r, TUE)
    assert _settle(db, m, WED).reconciled
    (buy,) = _movs(db, side="buy", status="filled")
    buy.open_usd *= 2          # la posizione ora «vale» il doppio senza che il prezzo si sia mosso
    db.commit()
    r = _settle(db, m, THU)
    assert not r.reconciled and "il suo prezzo dice" in r.note


# ---------------------------------------------------------------------------
# 9. Regressioni dal secondo giro di revisione (15/11): difetti introdotti dalle correzioni
# ---------------------------------------------------------------------------

def test_rev2_una_vendita_totale_in_attesa_non_si_ripete_nel_weekend(db, monkeypatch):
    from app.models.ai_portfolio import AiPortfolioPosition

    sab, dom = date(2026, 11, 14), date(2026, 11, 15)
    phase = {"n": 0}

    class S:
        @staticmethod
        def daily_scan(db_):
            if phase["n"] == 0 and not db_.query(AiPortfolioPosition).count():
                _intent(db_, "sector_healthcare", 1.0, tranches=(1, 1), opened=THU)
            elif phase["n"] == 1:
                for p in db_.query(AiPortfolioPosition).all():
                    db_.delete(p)
                db_.commit()
                _intent(db_, "gold", 0.6, tranches=(1, 1), opened=sab)
                _intent(db_, "silver", 0.6, tranches=(1, 1), opened=sab)
            return {}

    monkeypatch.setattr(BB, "_strategy_module", lambda s: S)
    m = FakeMarket({"SPY": spy(SESSIONS), **_flat("XLV", "GLD", "SLV")})
    _start(db)

    def giro(d):
        return BB.run_giro(db, now=run_at(d), market=m, strategies=("model_driven",), reasoning=False)

    giro(THU)
    giro(FRI)
    phase["n"] = 1
    giro(sab)                                          # sabato: esce dalla sanità, compra oro e argento
    phase["n"] = 2
    giro(dom)
    giro(MON2)                                         # domenica e lunedì: nessuna chiusura nuova
    pend = [(mv.asset_class, mv.side) for mv in _movs(db, status="pending")]
    assert pend.count(("sector_healthcare", "sell")) == 1
    assert sum(mv.amount_usd or 0 for mv in _movs(db, status="pending", side="buy")) <= 100_000  # niente cassa contata due volte
    giro(TUE2)
    assert [mv for mv in _movs(db) if mv.status == "cancelled"] == []


def test_rev2_taglio_parziale_vale_anche_per_la_tranche_in_attesa(db):
    from app.models.ai_portfolio import AiPortfolioPosition

    m = FakeMarket({"SPY": spy(SESSIONS), **_flat("XLV")})
    _start(db)
    r = _settle(db, m, TUE)
    _intent(db, "sector_healthcare", 0.2, tranches=(1, 2))
    _orders(db, r, TUE)
    r = BB.settle(db, "model_driven", m, run_at(WED, 15, 0))     # giro manuale dopo l'apertura
    db.commit()
    p = db.query(AiPortfolioPosition).one()
    p.current_weight_pct, p.tranches_filled = 0.4, 2
    db.commit()
    BB.create_orders(db, "model_driven", run_at(WED, 15, 0), r.nav, r.state)   # +20% in attesa fino a giovedì
    db.commit()
    r = _settle(db, m, THU)
    p.current_weight_pct, p.tranches_filled, p.last_action = 0.2, 1, "CLOSE_TRANCHE"
    db.commit()
    _orders(db, r, THU)
    r = _settle(db, m, FRI)
    assert (r.nav - r.state.cash) / r.nav == pytest.approx(0.2, abs=0.002)     # come l'intento, non 0,3


class _DupDivMarket(FakeMarket):
    def dividends(self, symbol, start, end):
        if symbol == "XLV" and start <= THU <= end:
            return pd.Series([0.25, 0.25], index=[THU, THU], dtype=float)
        return pd.Series(dtype=float)


def test_rev2_la_stessa_cedola_due_volte_nel_feed_si_accredita_una_volta(db):
    m = _DupDivMarket({"SPY": spy(SESSIONS), **_flat("XLV")})
    _start(db)
    r = _settle(db, m, TUE)
    _intent(db, "sector_healthcare", 0.5)
    _orders(db, r, TUE)
    _settle(db, m, WED)
    r1 = _settle(db, m, FRI)
    r2 = _settle(db, m, MON2)
    assert len(_movs(db, side="dividend")) == 1 and r1.reconciled and r2.reconciled
