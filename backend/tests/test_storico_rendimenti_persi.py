"""Storico dei bot: i rendimenti persi per la cache Yahoo si ricalcolano ALLA LETTURA (2026-10-31).

Caso vero: dal 23/10 i NAV di Modello e LLM restavano fermi su chiusure nuove (rendimento 0) in 6
giri su 8. Scelta di Erik (opzione A): non si riscrive il DB, la pagina mostra i valori ricalcolati
dai prezzi veri coi pesi che il bot aveva quel giorno, ricostruiti dal libro reale.

Ogni test nomina il difetto che impedisce:
- un giro perso resta a 0 nell'equity curve, nella classifica e nel «costo della realtà»;
- pesi ricostruiti male usati lo stesso (la quadratura deve fermarli, e dirlo);
- un bot tutto in liquidità o un giro senza chiusura nuova scambiati per giri persi.
"""
from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from app.services.ai_portfolio import history_view as hv
from app.services.ai_portfolio import pnl_tracker

D = date


def _serie(daily: dict[date, float] | None = None, base: float = 100.0, step: float = 0.0) -> pd.Series:
    """Chiusure feriali 01/10 → 30/10: `base` costante (+`step` al giorno) o valori espliciti."""
    idx = pd.bdate_range("2026-10-01", "2026-10-30")
    vals, v = [], base
    for i in idx:
        d = i.date()
        if daily and d in daily:
            v = daily[d]
        elif step:
            v = v * (1 + step)
        vals.append(v)
    return pd.Series(vals, index=idx, name="close")


# XLE: +1% al giorno dal 22/10 · QUAL piatto · XLV +2% il 27/10 · BIL +0,01% al giorno
_XLE = {D(2026, 10, d): 100.0 for d in (1, 2, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 19, 20, 21)}
for _k, _d in enumerate((22, 23, 26, 27, 28, 29, 30), start=1):
    _XLE[D(2026, 10, _d)] = 100.0 * 1.01 ** _k
PRICES = {
    "SPY": _serie(step=0.002), "TLT": _serie(), "ACWI": _serie(step=0.001),
    "XLE": _serie(_XLE), "QUAL": _serie(base=200.0),
    "XLV": _serie({D(2026, 10, 27): 51.0}, base=50.0), "BIL": _serie(step=0.0001),
}


def _fetch(t, s, e):
    return PRICES[t][PRICES[t].index <= pd.Timestamp(e)]


def r_(t, a, b):
    s = PRICES[t]
    return float(s[pd.Timestamp(b)] / s[pd.Timestamp(a)] - 1)


def _row(d, nav, price_date, n_pos=2):
    return SimpleNamespace(date=d, nav=nav, price_date=price_date, n_positions=n_pos,
                           cumulative_return_pct=0.0, drawdown_pct=0.0, peak_nav=nav,
                           benchmark_sp500_return_pct=None, benchmark_60_40_return_pct=None)


# come in PROD: 22/10 giro buono, 23 e 24 persi, 27 buono (+1%), 28 perso
ROWS = [
    _row(D(2026, 10, 22), 100_000.0, D(2026, 10, 21)),
    _row(D(2026, 10, 23), 100_000.0, D(2026, 10, 22)),
    _row(D(2026, 10, 24), 100_000.0, D(2026, 10, 23)),
    _row(D(2026, 10, 27), 101_000.0, D(2026, 10, 26)),
    _row(D(2026, 10, 28), 101_000.0, D(2026, 10, 27)),
]
W22 = {"energy": 0.5, "factor_quality": 0.3}
W27 = {"energy": 0.5, "sector_healthcare": 0.2}
WEIGHTS = {D(2026, 10, 22): W22, D(2026, 10, 27): W27}


@pytest.fixture(autouse=True)
def _memo_pulito():
    hv._LOST_MEMO.clear()


def test_giri_persi_ricalcolati_coi_pesi_di_quel_giorno():
    out, info = hv.corrected_history(ROWS, fetch=_fetch, weights=WEIGHTS)
    assert info["lost_returns"] == 3 and info["recomputed"] == 3
    assert [r["return_recomputed"] for r in out] == [False, True, True, False, True]

    r23 = 0.5 * r_("XLE", D(2026, 10, 21), D(2026, 10, 22)) + 0.2 * r_("BIL", D(2026, 10, 21), D(2026, 10, 22))
    r24 = 0.5 * r_("XLE", D(2026, 10, 22), D(2026, 10, 23)) + 0.2 * r_("BIL", D(2026, 10, 22), D(2026, 10, 23))
    r28 = (0.5 * r_("XLE", D(2026, 10, 26), D(2026, 10, 27)) + 0.2 * r_("XLV", D(2026, 10, 26), D(2026, 10, 27))
           + 0.3 * r_("BIL", D(2026, 10, 26), D(2026, 10, 27)))
    assert r23 > 0.004 and r28 > 0.008  # controllo positivo: il ricalcolo muove davvero il NAV
    expected = 100_000.0 * (1 + r23) * (1 + r24) * 1.01 * (1 + r28)
    assert out[-1]["nav"] == pytest.approx(expected)
    # il giro buono del 27 resta il rendimento salvato (+1%), non viene toccato
    assert out[3]["nav"] / out[2]["nav"] - 1 == pytest.approx(0.01)


def test_senza_pesi_i_giri_persi_si_contano_ma_restano_a_zero():
    out, info = hv.corrected_history(ROWS, fetch=_fetch)
    assert info["lost_returns"] == 3 and info["recomputed"] == 0
    assert out[-1]["nav"] == pytest.approx(101_000.0)


def test_bot_tutto_in_liquidita_e_giro_senza_chiusura_nuova_non_sono_persi():
    rows = [
        _row(D(2026, 10, 22), 100_000.0, D(2026, 10, 21), n_pos=0),
        _row(D(2026, 10, 23), 100_000.0, D(2026, 10, 22), n_pos=0),   # cassa piatta: vero zero
        _row(D(2026, 10, 24), 100_000.0, D(2026, 10, 22), n_pos=2),   # stessa chiusura: duplicato
    ]
    out, info = hv.corrected_history(rows, fetch=_fetch, weights=WEIGHTS)
    assert info["lost_returns"] == 0 and info["recomputed"] == 0 and info["duplicates"] == 1


# ---------------------------------------------------------------------------
# Pesi dal libro reale + lettori (classifica, libro reale) sul DB
# ---------------------------------------------------------------------------

@pytest.fixture
def db(monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.database import Base
    from app.models.ai_portfolio import AiPortfolioPerformance, AiPortfolioPosition
    from app.models.real_book import RbBook, RbBotSnapshot, RbMovement, RbNav

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine, tables=[
        AiPortfolioPosition.__table__, AiPortfolioPerformance.__table__,
        RbBook.__table__, RbMovement.__table__, RbNav.__table__, RbBotSnapshot.__table__,
    ])
    s = sessionmaker(bind=engine)()
    for r in ROWS:
        s.add(AiPortfolioPerformance(date=r.date, strategy_type="model_driven", nav=r.nav,
                                     price_date=r.price_date, n_positions=r.n_positions))

    def mov(d, asset, dw, side="buy", kind="tranche"):
        s.add(RbMovement(strategy_type="model_driven", asset_class=asset, instrument_key="X", side=side,
                         kind=kind, weight_delta_pct=dw, decided_on=d,
                         decided_at=datetime(d.year, d.month, d.day, 5, 30)))

    mov(D(2026, 10, 22), "energy", 0.5, kind="presa_in_carico")
    mov(D(2026, 10, 22), "factor_quality", 0.3, kind="presa_in_carico")
    mov(D(2026, 10, 27), "factor_quality", -0.3, side="sell", kind="full_close")
    mov(D(2026, 10, 27), "sector_healthcare", 0.2)
    s.add(RbMovement(strategy_type="model_driven", asset_class="energy", instrument_key="X",
                     side="dividend", kind="dividend", weight_delta_pct=0.0, decided_on=D(2026, 10, 28),
                     decided_at=datetime(2026, 10, 28, 17, 15)))  # la cedola non è un peso
    s.add(RbBook(strategy_type="model_driven", started_on=D(2026, 10, 22),
                 initial_capital_eur=100_000.0, bot_nav_start=100_000.0))
    s.add(RbNav(strategy_type="model_driven", date=D(2026, 10, 28), nav_eur=101_500.0, cash_eur=0.0,
                invested_eur=101_500.0, return_gross_pct=0.015, return_net_pct=0.012,
                bot_return_pct=0.01))  # salvato col NAV grezzo del bot
    s.commit()
    monkeypatch.setattr(pnl_tracker, "_fetcher_singleton",
                        SimpleNamespace(fetch=lambda t, start_date=None, end_date=None, **k: _fetch(t, start_date, end_date)))
    yield s
    s.close()


def _positions(db, weights):
    from app.models.ai_portfolio import AiPortfolioPosition

    for a, w in weights.items():
        db.add(AiPortfolioPosition(asset_class=a, strategy_type="model_driven", target_weight_pct=w,
                                   current_weight_pct=w, tranches_filled=3, tranches_total=3,
                                   opened_at=D(2026, 10, 1), entry_regime="stagflation", avg_entry_score=60.0,
                                   estimated_pnl_pct=0.0, took_partial_tp=False, last_action="OPEN_TRANCHE",
                                   last_action_date=D(2026, 10, 1)))
    db.commit()


def test_pesi_ricostruiti_dal_libro_reale_quadrano(db):
    _positions(db, W27)
    w, why = hv.bot_weights_by_day(db, "model_driven")
    assert why == ""
    assert w[D(2026, 10, 22)] == pytest.approx(W22)
    assert w[D(2026, 10, 27)] == pytest.approx(W27)


def test_pesi_che_non_quadrano_non_si_usano_e_lo_dice(db):
    _positions(db, {"energy": 0.6, "sector_healthcare": 0.2})  # scarto 0,1 su energy
    out, info = hv.corrected_history_for(db, "model_driven")
    assert info["recomputed"] == 0 and info["lost_returns"] == 3
    assert "non quadrano" in info["recompute_skipped"]


def test_classifica_e_costo_della_realta_leggono_lo_storico_ricalcolato(db):
    from app.services.real_book import views

    _positions(db, W27)
    out, info = hv.corrected_history_for(db, "model_driven")
    assert info["recomputed"] == 3
    ov = views.overview(db, admin=False)["strategies"]["model_driven"]["nav"]
    atteso = out[-1]["nav"] / out[0]["nav"] - 1
    assert ov["bot_return_pct"] == pytest.approx(atteso)
    assert ov["bot_return_pct"] > 0.02  # non più l'1% salvato col NAV fermo
    assert ov["gross"]["reality_gap_pct"] == pytest.approx(0.015 - atteso)
    [nav] = views.nav_series(db, "model_driven", days=3650)
    assert nav["bot_return_pct"] == pytest.approx(atteso)
