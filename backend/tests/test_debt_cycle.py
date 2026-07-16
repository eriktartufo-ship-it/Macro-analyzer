"""Test debt cycle thermometer. Spec: S46."""
from __future__ import annotations

from app.services.backtest.debt_cycle import compute_debt_thermometer


def test_low_leverage_is_early():
    # bassa leva + term premium alto (mercato sano) -> inizio ciclo
    out = compute_debt_thermometer({"debt_gdp": 55.0, "fedbs_gdp": 6.0, "term_premium": 1.0})
    assert out["composite"] < 0.1
    assert out["stage"] == "early"


def test_high_leverage_is_late():
    # alta leva + Fed gonfio + term premium negativo (repressione) -> tardo ciclo
    out = compute_debt_thermometer({"debt_gdp": 130.0, "fedbs_gdp": 35.0, "term_premium": -1.0})
    assert out["composite"] > 0.9
    assert out["stage"] == "late"


def test_term_premium_axis_inverted():
    # a parita' di leva, term premium PIU' BASSO -> composito PIU' ALTO (repressione)
    hi_tp = compute_debt_thermometer({"debt_gdp": 100.0, "fedbs_gdp": 20.0, "term_premium": 1.0})
    lo_tp = compute_debt_thermometer({"debt_gdp": 100.0, "fedbs_gdp": 20.0, "term_premium": -1.0})
    assert lo_tp["composite"] > hi_tp["composite"]


def test_composite_bounded_and_monotonic_in_debt():
    a = compute_debt_thermometer({"debt_gdp": 70.0, "fedbs_gdp": 15.0, "term_premium": 0.0})
    b = compute_debt_thermometer({"debt_gdp": 120.0, "fedbs_gdp": 15.0, "term_premium": 0.0})
    assert 0.0 <= a["composite"] <= 1.0 and 0.0 <= b["composite"] <= 1.0
    assert b["composite"] > a["composite"]  # piu' debito -> piu' tardo


def test_missing_component_renormalized_not_silent():
    out = compute_debt_thermometer({"debt_gdp": 130.0, "fedbs_gdp": None, "term_premium": -1.0})
    assert out["missing"] == ["fedbs_gdp"]
    assert out["used"] == ["debt_gdp", "term_premium"]
    assert out["composite"] is not None  # rinormalizzato sulle presenti
    # componente mancante segnalata nei rows
    fedbs = next(r for r in out["components"] if r["key"] == "fedbs_gdp")
    assert fedbs["available"] is False and fedbs["value"] is None


def test_all_missing_is_na():
    out = compute_debt_thermometer({"debt_gdp": None, "fedbs_gdp": None, "term_premium": None})
    assert out["composite"] is None and out["stage"] == "na"


def test_out_of_range_clamped():
    # valori assurdi -> sub-score clampati in [0,1], composito valido
    out = compute_debt_thermometer({"debt_gdp": 500.0, "fedbs_gdp": 200.0, "term_premium": -10.0})
    assert out["composite"] == 1.0 and out["stage"] == "late"
