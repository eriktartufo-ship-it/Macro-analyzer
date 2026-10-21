"""Correzioni ai bot dell'AI Portfolio (2026-10-21), ognuna con il difetto che impedisce.

1. Il valore contava il venerdì tre volte (sabato, domenica e lunedì notte non arriva una
   chiusura nuova): S&P sul sito +5,58% contro un vero +2,56% dal 9/07.
2. Rilanciare lo scan nello stesso giorno applicava il rendimento due volte.
3. Nel weekend i bot decidevano due volte sugli stessi prezzi.
4. energy + sector_energy = stesso ETF (XLE): 29,8% su uno strumento col tetto al 25%.
5. Il bot «S&P a rischio gestito» teneva il Nasdaq-100.
6. Tasse italiane nel libro reale (Erik: «di base conta che ci sia»).
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from app.services.ai_portfolio import market_calendar as mc


def _series(points: dict[date, float]) -> pd.Series:
    return pd.Series(list(points.values()), index=pd.to_datetime(list(points.keys())))


# giovedì 15/10, venerdì 16, lunedì 19, martedì 20 (2026)
ETF = _series({date(2026, 10, 15): 100.0, date(2026, 10, 16): 102.0,
               date(2026, 10, 19): 101.0, date(2026, 10, 20): 103.0})


# ---------------------------------------------------------------------------
# 1-3. Calendario: una chiusura conta una volta sola
# ---------------------------------------------------------------------------

def test_il_venerdi_conta_una_volta_sola():
    giri = [date(2026, 10, 16), date(2026, 10, 17), date(2026, 10, 18), date(2026, 10, 19), date(2026, 10, 20),
            date(2026, 10, 21)]
    cum = 1.0
    for prev, run in zip(giri, giri[1:]):
        cum *= 1 + mc.return_between(ETF, prev, run)
    # dal giro di venerdì (vede giovedì 100) a quello di mercoledì (vede martedì 103)
    assert cum == pytest.approx(103 / 100)
    # e i giri di domenica e lunedì notte non aggiungono niente
    assert mc.return_between(ETF, date(2026, 10, 17), date(2026, 10, 18)) == 0
    assert mc.return_between(ETF, date(2026, 10, 18), date(2026, 10, 19)) == 0


def test_controllo_positivo_il_vecchio_metodo_lo_contava_tre_volte():
    # la vecchia regola: a ogni giro ultima ÷ penultima chiusura disponibile
    dates = sorted(ETF.index.date)
    def last_prev(run):
        c = [d for d in dates if d < run]
        return ETF[pd.Timestamp(c[-1])] / ETF[pd.Timestamp(c[-2])]
    vecchio = 1.0
    for run in [date(2026, 10, 17), date(2026, 10, 18), date(2026, 10, 19)]:
        vecchio *= last_prev(run)
    assert vecchio == pytest.approx(1.02 ** 3)  # +2% del venerdì contato tre volte


def test_il_bitcoin_prende_il_movimento_vero_del_weekend():
    btc = _series({date(2026, 10, 16): 100.0, date(2026, 10, 17): 110.0, date(2026, 10, 18): 99.0})
    tot = (1 + mc.return_between(btc, date(2026, 10, 17), date(2026, 10, 18))) * \
          (1 + mc.return_between(btc, date(2026, 10, 18), date(2026, 10, 19)))
    assert tot == pytest.approx(99 / 100)


@pytest.mark.parametrize("today,prev,run", [
    (date(2026, 10, 18), date(2026, 10, 17), False),  # domenica dopo il giro di sabato
    (date(2026, 10, 19), date(2026, 10, 17), False),  # lunedì notte: ancora venerdì
    (date(2026, 10, 20), date(2026, 10, 17), True),   # martedì notte: c'è lunedì
    (date(2026, 10, 17), date(2026, 10, 16), True),   # sabato notte: c'è venerdì
    (date(2026, 10, 17), None, True),                 # primo giro in assoluto
])
def test_i_bot_decidono_solo_con_una_chiusura_nuova(today, prev, run):
    assert mc.has_new_close(sorted(ETF.index.date), today, prev) is run


@pytest.fixture
def db():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.database import Base
    from app.models.ai_portfolio import AiPortfolioDecision, AiPortfolioPerformance, AiPortfolioPosition
    from app.models.regime_classifications import RegimeClassification

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine, tables=[
        AiPortfolioPosition.__table__, AiPortfolioDecision.__table__, AiPortfolioPerformance.__table__,
        RegimeClassification.__table__,
    ])
    s = sessionmaker(bind=engine)()
    s.add(RegimeClassification(date=date(2026, 10, 16), regime="stagflation", probability_reflation=0.38,
                               probability_stagflation=0.42, probability_deflation=0.09,
                               probability_goldilocks=0.11, confidence=0.54, conditions_met="{}"))
    s.commit()
    yield s
    s.close()


def test_rilanciare_lo_scan_nello_stesso_giorno_non_raddoppia(db, monkeypatch):
    from app.models.ai_portfolio import AiPortfolioPerformance
    from app.services.ai_portfolio import performance as P

    db.add(AiPortfolioPerformance(date=date(2026, 10, 19), strategy_type="model_driven", nav=100_000.0,
                                  peak_nav=100_000.0))
    db.commit()
    monkeypatch.setattr(P, "_portfolio_return_between", lambda *a: 0.01)
    monkeypatch.setattr(P, "_ticker_return_between", lambda *a: 0.0)
    first = P.snapshot(db, target_date=date(2026, 10, 20), strategy_type="model_driven").nav
    again = P.snapshot(db, target_date=date(2026, 10, 20), strategy_type="model_driven").nav
    assert first == pytest.approx(101_000) and again == pytest.approx(101_000)


def test_cancello_del_giro_notturno(db, monkeypatch):
    from app.models.ai_portfolio import AiPortfolioPerformance
    from app.services.ai_portfolio import pnl_tracker

    class F:
        def fetch(self, *a, **k):
            return ETF

    monkeypatch.setattr(pnl_tracker, "_fetcher", lambda: F())
    db.add(AiPortfolioPerformance(date=date(2026, 10, 17), strategy_type="model_driven", nav=100_000.0))
    db.commit()
    run, why = mc.bots_should_run(db, today=date(2026, 10, 18))
    assert run is False and "nessuna chiusura nuova" in why
    assert mc.bots_should_run(db, today=date(2026, 10, 20))[0] is True


# ---------------------------------------------------------------------------
# 4. Tetto per strumento
# ---------------------------------------------------------------------------

def _pos(db, asset, w, strategy="model_driven"):
    from app.models.ai_portfolio import AiPortfolioPosition

    db.add(AiPortfolioPosition(asset_class=asset, strategy_type=strategy, target_weight_pct=w,
                               current_weight_pct=w, tranches_filled=2, tranches_total=3,
                               opened_at=date(2026, 7, 13), entry_regime="stagflation", avg_entry_score=60.0,
                               estimated_pnl_pct=0.0, took_partial_tp=False, last_action="OPEN_TRANCHE",
                               last_action_date=date(2026, 7, 13)))
    db.commit()


def test_energy_e_sector_energy_sono_lo_stesso_strumento():
    from app.services.ai_portfolio.instrument_cap import instrument_groups

    assert instrument_groups()["XLE"] == ["energy", "sector_energy"]


def test_tetto_per_strumento_riduce_in_proporzione_e_lo_dice(db):
    from app.models.ai_portfolio import AiPortfolioDecision, AiPortfolioPosition
    from app.services.ai_portfolio.instrument_cap import enforce_instrument_cap

    _pos(db, "energy", 0.0905)
    _pos(db, "sector_energy", 0.2079)
    _pos(db, "gold", 0.30)  # un asset da solo oltre il 25% NON è affare di questa guardia
    n = enforce_instrument_cap(db, "model_driven", date(2026, 10, 20), "stagflation")
    db.commit()
    w = {p.asset_class: p.current_weight_pct for p in db.query(AiPortfolioPosition).all()}
    assert n == 2
    assert w["energy"] + w["sector_energy"] == pytest.approx(0.25)
    assert w["energy"] / w["sector_energy"] == pytest.approx(0.0905 / 0.2079)
    assert w["gold"] == pytest.approx(0.30)
    decs = db.query(AiPortfolioDecision).filter_by(action="CAP_TRIM").all()
    assert len(decs) == 2 and all("stesso ETF (XLE)" in d.reason_short for d in decs)


def test_tetto_non_tocca_un_gruppo_sotto_soglia(db):
    from app.services.ai_portfolio.instrument_cap import enforce_instrument_cap

    _pos(db, "energy", 0.10)
    _pos(db, "sector_energy", 0.10)
    assert enforce_instrument_cap(db, "model_driven", date(2026, 10, 20), "stagflation") == 0


# ---------------------------------------------------------------------------
# 5. Il bot S&P tiene l'S&P
# ---------------------------------------------------------------------------

def test_il_bot_sp_tiene_l_sp500_vero_fuori_dallo_scoring():
    from app.services.ai_portfolio import momentum, risk_managed_sp
    from app.services.prices.asset_universe import ASSET_TICKERS
    from app.services.real_book.instruments import instrument_for

    assert momentum.asset_to_ticker(risk_managed_sp.SP_ASSET) == "SPY"
    assert risk_managed_sp.SP_ASSET not in ASSET_TICKERS  # calibrazione e flussi non cambiano
    for a in (risk_managed_sp.SP_ASSET, *risk_managed_sp.REFUGE_CANDIDATES):
        assert instrument_for(a) is not None, a
    assert instrument_for(risk_managed_sp.SP_ASSET).key == "VUAA"


# ---------------------------------------------------------------------------
# Storico corretto alla lettura
# ---------------------------------------------------------------------------

class _Row:
    def __init__(self, d, nav):
        self.date, self.nav = d, nav
        self.cumulative_return_pct = nav / 100_000 - 1
        self.drawdown_pct = 0.0
        self.peak_nav = nav
        self.benchmark_sp500_return_pct = None
        self.benchmark_60_40_return_pct = None


def test_storico_corretto_neutralizza_i_duplicati_e_ricalcola_i_benchmark():
    from app.services.ai_portfolio.history_view import corrected_history

    # giri: sab (vede ven, +2%), dom e lun (duplicati che il vecchio metodo gonfiava), mar (vede lun)
    rows = [_Row(date(2026, 10, 17), 102_000), _Row(date(2026, 10, 18), 104_040),
            _Row(date(2026, 10, 19), 106_120.8), _Row(date(2026, 10, 20), 105_080.4)]
    out, info = corrected_history(rows, fetch=lambda t, s, e: ETF)
    assert info == {"corrected": True, "duplicates": 2}
    # solo sabato (+2%) e martedì (105.080,4/106.120,8 = −0,98%) contano
    assert out[-1]["nav"] == pytest.approx(102_000 * (105_080.4 / 106_120.8))
    # benchmark dai prezzi: base = chiusura prima di venerdì (giovedì 100), fino a lunedì (101)
    assert out[-1]["benchmark_sp500_return_pct"] == pytest.approx(101 / 100 - 1)
    assert out[-1]["benchmark_acwi_return_pct"] == pytest.approx(101 / 100 - 1)


def test_storico_senza_prezzi_resta_grezzo_e_lo_dichiara():
    from app.services.ai_portfolio.history_view import corrected_history

    def boom(*a):
        raise RuntimeError("yahoo giù")

    out, info = corrected_history([_Row(date(2026, 10, 17), 101_000)], fetch=boom)
    assert info["corrected"] is False and out[0]["nav"] == 101_000


# ---------------------------------------------------------------------------
# 6. Tasse italiane
# ---------------------------------------------------------------------------

def test_tasse_etf_26_senza_compensazione_e_minus_nello_zainetto():
    from app.services.real_book import ledger as L

    st = L.LedgerState(cash=0)
    tax, used, added = L.tax_on_realized(st, -1000, "oicr", 0.26, 2026)
    assert (tax, used, added) == (0, 0, 1000) and st.zainetto == [[1000, 2030]]
    tax, used, _ = L.tax_on_realized(st, 500, "oicr", 0.26, 2026)  # ETF: lo zainetto NON si usa
    assert tax == pytest.approx(130) and used == 0 and st.cash == pytest.approx(-130)


def test_etc_usa_lo_zainetto_prima_di_pagare():
    from app.services.real_book import ledger as L

    st = L.LedgerState(cash=0, zainetto=[[300.0, 2030]])
    tax, used, _ = L.tax_on_realized(st, 500, "diversi", 0.26, 2026)
    assert used == pytest.approx(300) and tax == pytest.approx(200 * 0.26) and st.zainetto == []


def test_zainetto_scaduto_non_si_usa():
    from app.services.real_book import ledger as L

    st = L.LedgerState(cash=0, zainetto=[[300.0, 2025]])
    tax, used, _ = L.tax_on_realized(st, 500, "diversi", 0.26, 2026)
    assert used == 0 and tax == pytest.approx(130)


def test_titoli_di_stato_12_5_e_minus_al_48_per_cento():
    from app.services.real_book import ledger as L
    from app.services.real_book.instruments import INSTRUMENTS

    assert INSTRUMENTS["IDTL"].tax_rate == 0.125 and INSTRUMENTS["SGLD"].tax_class == "diversi"
    assert INSTRUMENTS["EQQQ"].tax_rate == 0.26 and INSTRUMENTS["EQQQ"].tax_class == "oicr"
    st = L.LedgerState(cash=0)
    assert L.tax_on_realized(st, 1000, "oicr", 0.125, 2026)[0] == pytest.approx(125)
    assert L.tax_on_realized(st, -1000, "oicr", 0.125, 2026)[2] == pytest.approx(1000 * 0.125 / 0.26)


def test_tasse_latenti_per_il_valore_netto():
    from app.services.real_book import ledger as L

    st = L.LedgerState(cash=0, zainetto=[[100.0, 2030]])
    st.lots["a"] = L.Lot("ETFX", qty=10, cost_eur=1000)   # +500 latente su ETF
    st.lots["b"] = L.Lot("ETCX", qty=10, cost_eur=1000)   # +300 latente su ETC, 100 di zainetto
    st.lots["c"] = L.Lot("ETFY", qty=10, cost_eur=1000)   # −200: nessun credito in contanti
    tax = L.latent_tax(st, {"ETFX": 150, "ETCX": 130, "ETFY": 80},
                       {"ETFX": ("oicr", 0.26), "ETCX": ("diversi", 0.26), "ETFY": ("oicr", 0.26)}, 2026)
    assert tax == pytest.approx(500 * 0.26 + 200 * 0.26)
