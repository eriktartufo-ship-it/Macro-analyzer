"""Test WeeklyRegimeGrid — lookup ffill + no-future-leak (no rete: classify_fn fake)."""
from __future__ import annotations

from datetime import date

import pandas as pd

from app.services.backtest.daily_regime_grid import (
    WeeklyRegimeGrid,
    build_weekly_grid_from_fn,
    _grid_fridays,
)


def _cm(regime, conf, probs):
    return {"regime": regime, "confidence": conf, "probs": probs}


def _grid_two_points():
    # due venerdì: 2020-03-06 (goldilocks) e 2020-03-13 (deflation)
    def fn(d):
        if d <= date(2020, 3, 6):
            return _cm("goldilocks", 0.6, {"reflation": 0.1, "stagflation": 0.1,
                                           "deflation": 0.2, "goldilocks": 0.6})
        return _cm("deflation", 0.7, {"reflation": 0.05, "stagflation": 0.05,
                                      "deflation": 0.7, "goldilocks": 0.2})
    dates = pd.to_datetime(["2020-03-06", "2020-03-13"])
    return WeeklyRegimeGrid(build_weekly_grid_from_fn(dates, fn))


def test_ffill_uses_last_grid_on_or_before():
    g = _grid_two_points()
    # martedì 2020-03-10 → deve usare il venerdì 2020-03-06 (goldilocks), NON il 03-13
    r = g.get(date(2020, 3, 10))
    assert r["regime"] == "goldilocks"
    assert r["grid_date"] == pd.Timestamp("2020-03-06")


def test_no_future_leak_before_transition():
    g = _grid_two_points()
    # giovedì 2020-03-12: la transizione a deflation è del 03-13 → NON deve vederla
    r = g.get(date(2020, 3, 12))
    assert r["regime"] == "goldilocks"
    # il 03-13 stesso (venerdì) vede deflation
    assert g.get(date(2020, 3, 13))["regime"] == "deflation"
    # dopo (lunedì) resta deflation
    assert g.get(date(2020, 3, 16))["regime"] == "deflation"


def test_before_first_grid_returns_none():
    g = _grid_two_points()
    assert g.get(date(2020, 2, 1)) is None


def test_probs_roundtrip():
    g = _grid_two_points()
    r = g.get(date(2020, 3, 20))
    assert r["probs"]["deflation"] == 0.7
    assert abs(sum(r["probs"].values()) - 1.0) < 1e-9


def test_none_classifications_skipped():
    def fn(d):
        return None if d == date(2020, 3, 6) else _cm(
            "reflation", 0.5, {"reflation": 0.5, "stagflation": 0.2,
                               "deflation": 0.1, "goldilocks": 0.2})
    dates = pd.to_datetime(["2020-03-06", "2020-03-13"])
    g = WeeklyRegimeGrid(build_weekly_grid_from_fn(dates, fn))
    assert len(g) == 1
    assert g.get(date(2020, 3, 15))["regime"] == "reflation"


def test_grid_fridays_are_fridays():
    fr = _grid_fridays(date(2020, 1, 1), date(2020, 2, 1))
    assert all(d.weekday() == 4 for d in fr)
