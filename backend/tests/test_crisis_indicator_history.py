"""Test backtest storico Crisis Risk Indicator.

Coverage:
- aggregate_crisis_stats: distribuzione livelli/tipi + episodi notevoli (run consecutivi).
- compute_crisis_history: pipeline E2E con DB mock (records sintetici).
- Verifica che scenari noti (2008-style, 1973-style, bubble) producano il
  crisis_type atteso nel backtest.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.services.regime.crisis_indicator_history import (
    CrisisHistoryPoint,
    aggregate_crisis_stats,
    compute_crisis_history,
    serialize_points,
)


def _pt(d: str, level: str, crisis_type: str, score: float = 0.5) -> CrisisHistoryPoint:
    return CrisisHistoryPoint(
        date=d,
        risk_level=level,
        risk_score=score,
        crisis_type=crisis_type,
        crisis_type_confidence=0.7,
        historical_analog=None,
        triggers_fired=3,
        triggers_total=15,
        summary="",
    )


class TestAggregateCrisisStats:
    def test_empty_input_returns_zero(self):
        stats = aggregate_crisis_stats([])
        assert stats["n_points"] == 0
        assert stats["notable_episodes"] == []

    def test_level_distribution_correct(self):
        pts = [
            _pt("2020-01-01", "low", "no_crisis"),
            _pt("2020-01-02", "low", "no_crisis"),
            _pt("2020-01-03", "elevated", "deflation_crash"),
            _pt("2020-01-04", "high", "deflation_crash"),
        ]
        stats = aggregate_crisis_stats(pts)
        assert stats["n_points"] == 4
        assert stats["level_distribution"]["low"]["count"] == 2
        assert stats["level_distribution"]["low"]["pct"] == 0.5
        assert stats["level_distribution"]["high"]["count"] == 1

    def test_type_distribution_correct(self):
        pts = [
            _pt("2020-01-01", "low", "no_crisis"),
            _pt("2020-01-02", "elevated", "deflation_crash"),
            _pt("2020-01-03", "high", "deflation_crash"),
            _pt("2020-01-04", "extreme", "deflation_crash"),
        ]
        stats = aggregate_crisis_stats(pts)
        assert stats["type_distribution"]["deflation_crash"]["count"] == 3
        assert stats["type_distribution"]["no_crisis"]["count"] == 1

    def test_notable_episode_detected(self):
        """Run di 4 punti high/extreme consecutivi = episode."""
        pts = [
            _pt("2020-01-01", "low", "no_crisis"),
            _pt("2020-01-02", "high", "deflation_crash", score=0.6),
            _pt("2020-01-03", "extreme", "deflation_crash", score=0.85),
            _pt("2020-01-04", "extreme", "deflation_crash", score=0.92),
            _pt("2020-01-05", "high", "deflation_crash", score=0.7),
            _pt("2020-01-06", "low", "no_crisis"),
        ]
        stats = aggregate_crisis_stats(pts)
        eps = stats["notable_episodes"]
        assert len(eps) == 1
        assert eps[0]["start"] == "2020-01-02"
        assert eps[0]["duration_points"] == 4
        assert eps[0]["crisis_type"] == "deflation_crash"
        assert eps[0]["peak_risk_score"] == 0.92

    def test_short_run_not_an_episode(self):
        """Run di solo 2 punti high non basta (min 3)."""
        pts = [
            _pt("2020-01-01", "low", "no_crisis"),
            _pt("2020-01-02", "high", "deflation_crash"),
            _pt("2020-01-03", "high", "deflation_crash"),
            _pt("2020-01-04", "low", "no_crisis"),
        ]
        stats = aggregate_crisis_stats(pts)
        assert stats["notable_episodes"] == []

    def test_trailing_episode_captured(self):
        """Run terminante alla fine della serie deve essere capturato."""
        pts = [
            _pt("2020-01-01", "low", "no_crisis"),
            _pt("2020-01-02", "high", "deflation_crash"),
            _pt("2020-01-03", "high", "deflation_crash"),
            _pt("2020-01-04", "extreme", "deflation_crash"),
        ]
        stats = aggregate_crisis_stats(pts)
        assert len(stats["notable_episodes"]) == 1
        assert stats["notable_episodes"][0]["duration_points"] == 3


def _make_mock_record(d: date, indicators: dict) -> SimpleNamespace:
    return SimpleNamespace(
        date=d,
        conditions_met=json.dumps({"indicators": indicators}),
    )


def _make_mock_db(records: list, dedollar_records: list | None = None):
    """Crea un mock db con due query: RegimeClassification e SecularTrend."""
    db = MagicMock()

    def query_side_effect(model):
        if "RegimeClassification" in model.__name__:
            return _make_query_chain(records)
        if "SecularTrend" in model.__name__:
            return _make_query_chain(dedollar_records or [])
        return _make_query_chain([])

    db.query.side_effect = query_side_effect
    return db


def _make_query_chain(records):
    """Mock chainable: query.filter.filter.order_by.all() o .first() o .limit().all()."""
    chain = MagicMock()
    chain.filter.return_value = chain
    chain.order_by.return_value = chain
    chain.limit.return_value = chain
    chain.all.return_value = list(records)
    chain.first.return_value = records[0] if records else None
    return chain


class TestComputeCrisisHistory:
    def test_empty_db_returns_empty(self):
        db = _make_mock_db([])
        out = compute_crisis_history(db, days=30)
        assert out == []

    def test_skips_records_without_indicators(self):
        records = [
            SimpleNamespace(date=date(2026, 1, 1), conditions_met=None),
            SimpleNamespace(date=date(2026, 1, 2), conditions_met=""),
            SimpleNamespace(date=date(2026, 1, 3), conditions_met='{"foo":"bar"}'),
        ]
        db = _make_mock_db(records)
        out = compute_crisis_history(db, days=365)
        assert out == []

    def test_normal_scenario_no_crisis(self):
        normal_ind = {
            "gdp_roc": 2.0, "cpi_yoy": 2.2, "unrate": 4.0, "vix": 17.0,
            "baa_spread": 1.7, "yield_curve_10y2y": 0.8,
            "breakeven_10y": 2.0, "nfci": -0.3, "unrate_roc": 0.0,
        }
        records = [_make_mock_record(date(2026, 1, 1), normal_ind)]
        db = _make_mock_db(records)
        out = compute_crisis_history(db, days=365)
        assert len(out) == 1
        assert out[0].crisis_type == "no_crisis"
        assert out[0].risk_level == "low"

    def test_2008_style_scenario_produces_deflation_crash(self):
        """Replica scenario 2008 GFC nel backtest."""
        crash_ind = {
            "gdp_roc": -3.0, "cpi_yoy": 1.0, "unrate": 8.0, "unrate_roc": 80.0,
            "vix": 45.0, "baa_spread": 5.5, "yield_curve_10y2y": -0.5,
            "breakeven_10y": 1.2, "nfci": 1.5,
        }
        records = [_make_mock_record(date(2008, 11, 1), crash_ind)]
        db = _make_mock_db(records)
        out = compute_crisis_history(db, days=365 * 30)
        assert len(out) == 1
        assert out[0].crisis_type == "deflation_crash"
        assert out[0].risk_level in {"elevated", "high", "extreme"}

    def test_dedollar_lookup_applied_to_stagflation_signal(self):
        """High dedollar combined deve boostare stagflation_debasement detection."""
        stag_ind = {
            "gdp_roc": -1.5, "cpi_yoy": 8.0, "unrate": 5.5, "unrate_roc": 30.0,
            "vix": 24.0, "baa_spread": 2.8, "yield_curve_10y2y": 0.3,
            "breakeven_10y": 3.4, "nfci": 0.4,
        }
        records = [_make_mock_record(date(2022, 6, 1), stag_ind)]
        dedollar_records = [
            SimpleNamespace(
                date=date(2022, 6, 1),
                score=0.75,
                components=json.dumps({"combined_score": 0.75}),
            ),
        ]
        db = _make_mock_db(records, dedollar_records)
        out = compute_crisis_history(db, days=365 * 30)
        assert len(out) == 1
        assert out[0].crisis_type == "stagflation_debasement"

    def test_serialize_points_returns_dicts(self):
        pts = [_pt("2020-01-01", "low", "no_crisis")]
        out = serialize_points(pts)
        assert isinstance(out, list)
        assert isinstance(out[0], dict)
        assert out[0]["date"] == "2020-01-01"
        assert out[0]["risk_level"] == "low"
