"""Test prediction logger."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.services.validation.prediction_logger import (
    BENCHMARK_60_40,
    _compute_portfolio_return,
    log_prediction,
    summary_stats,
    stats_by_regime,
)


class TestComputePortfolioReturn:
    def test_basic(self):
        weights = {"us_equities_growth": 0.6, "us_bonds_long": 0.4}
        returns = {"us_equities_growth": 0.10, "us_bonds_long": 0.02}
        # 0.6*0.10 + 0.4*0.02 = 0.068
        assert _compute_portfolio_return(weights, returns) == pytest.approx(0.068)

    def test_missing_asset(self):
        weights = {"gold": 0.5, "cash": 0.5}
        returns = {"gold": 0.05}  # cash missing
        assert _compute_portfolio_return(weights, returns) == pytest.approx(0.025)

    def test_empty(self):
        assert _compute_portfolio_return({}, {"x": 0.05}) == 0


class TestLogPrediction:
    def test_log_creates_record(self):
        db = MagicMock()
        with patch("app.services.validation.prediction_logger._ensure_table_exists"):
            out = log_prediction(
                db,
                classify_output={
                    "regime": "reflation",
                    "confidence": 0.75,
                    "probabilities": {"reflation": 0.6, "stagflation": 0.2,
                                      "deflation": 0.1, "goldilocks": 0.1},
                },
                top5_assets=[
                    {"asset": "us_equities_growth", "score": 75, "weight": 0.3},
                    {"asset": "us_equities_value", "score": 70, "weight": 0.25},
                ],
            )
        # add + commit + refresh chiamati
        db.add.assert_called_once()
        db.commit.assert_called_once()
        assert out.regime == "reflation"

    def test_default_date_today(self):
        db = MagicMock()
        with patch("app.services.validation.prediction_logger._ensure_table_exists"):
            log_prediction(
                db,
                classify_output={"regime": "deflation", "probabilities": {}},
                top5_assets=[],
            )
        # Verify date was set
        record_arg = db.add.call_args[0][0]
        assert record_arg.date_predicted == date.today()


class TestSummaryStats:
    def test_empty_returns_none(self):
        db = MagicMock()
        chain = MagicMock()
        chain.filter.return_value = chain
        chain.all.return_value = []
        db.query.return_value = chain
        with patch("app.services.validation.prediction_logger._ensure_table_exists"):
            out = summary_stats(db, horizon="3m")
        assert out["n_evaluated"] == 0
        assert out["mean_alpha"] is None

    def test_positive_alpha_stats(self):
        # Mock 3 records con alpha [0.02, 0.05, -0.01]
        records = [
            SimpleNamespace(alpha_3m=0.02, realized_return_3m=0.05),
            SimpleNamespace(alpha_3m=0.05, realized_return_3m=0.08),
            SimpleNamespace(alpha_3m=-0.01, realized_return_3m=0.01),
        ]
        db = MagicMock()
        chain = MagicMock()
        chain.filter.return_value = chain
        chain.all.return_value = records
        db.query.return_value = chain
        with patch("app.services.validation.prediction_logger._ensure_table_exists"):
            out = summary_stats(db, horizon="3m")
        assert out["n_evaluated"] == 3
        assert out["mean_alpha"] == pytest.approx(0.02, abs=0.001)
        assert out["hit_rate"] == pytest.approx(2/3, abs=0.01)


class TestStatsByRegime:
    def test_groups_by_regime(self):
        records = [
            SimpleNamespace(regime="reflation", alpha_3m=0.03),
            SimpleNamespace(regime="reflation", alpha_3m=0.01),
            SimpleNamespace(regime="deflation", alpha_3m=-0.02),
        ]
        db = MagicMock()
        chain = MagicMock()
        chain.filter.return_value = chain
        chain.all.return_value = records
        db.query.return_value = chain
        with patch("app.services.validation.prediction_logger._ensure_table_exists"):
            out = stats_by_regime(db, horizon="3m")
        assert "reflation" in out
        assert out["reflation"]["n"] == 2
        assert out["reflation"]["mean_alpha"] == pytest.approx(0.02)
        assert out["deflation"]["n"] == 1
