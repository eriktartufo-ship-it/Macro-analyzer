"""Test Tier 6.6 — Live calibration `asset_regime_data` versionata DB.

Coverage:
- load_latest_from_db: DB vuoto → None; con snapshot attivo → payload.
- calibrate_and_persist: success path + failure graceful.
- _calibrated_or_prior priority: DB (T6.6) > JSON (T1) > hardcoded prior.
- list_snapshots history.
- Flag-gating: USE_LIVE_CALIBRATION OFF → ignora DB.
"""
from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.services.scoring.live_calibration import (
    LiveCalibrationResult,
    list_snapshots,
    load_latest_from_db,
)


def _make_mock_db(records: list | None = None):
    """Mock SQLAlchemy session con query chainable."""
    db = MagicMock()
    chain = MagicMock()
    chain.filter.return_value = chain
    chain.order_by.return_value = chain
    chain.limit.return_value = chain
    chain.first.return_value = records[0] if records else None
    chain.all.return_value = list(records) if records else []
    chain.update.return_value = None
    db.query.return_value = chain
    return db


class TestLoadLatestFromDb:
    def test_empty_db_returns_none(self):
        db = _make_mock_db([])
        with patch("app.services.scoring.live_calibration._ensure_table_exists"):
            out = load_latest_from_db(db)
        assert out is None

    def test_active_snapshot_returns_payload(self):
        payload_dict = {
            "asset_regime_data": {
                "us_equities_growth": {
                    "reflation": {"hit_rate": 0.8, "avg_return": 0.15, "vol": 0.17, "sharpe": 0.88}
                }
            },
            "n_classifications": 800,
        }
        record = SimpleNamespace(
            id=1,
            payload=json.dumps(payload_dict),
            is_active=True,
        )
        db = _make_mock_db([record])
        with patch("app.services.scoring.live_calibration._ensure_table_exists"):
            out = load_latest_from_db(db)
        assert out is not None
        assert "asset_regime_data" in out
        assert out["asset_regime_data"]["us_equities_growth"]["reflation"]["hit_rate"] == 0.8

    def test_invalid_json_payload_returns_none(self):
        record = SimpleNamespace(id=1, payload="not valid json {", is_active=True)
        db = _make_mock_db([record])
        with patch("app.services.scoring.live_calibration._ensure_table_exists"):
            out = load_latest_from_db(db)
        assert out is None

    def test_empty_payload_returns_none(self):
        record = SimpleNamespace(id=1, payload="", is_active=True)
        db = _make_mock_db([record])
        with patch("app.services.scoring.live_calibration._ensure_table_exists"):
            out = load_latest_from_db(db)
        assert out is None


class TestListSnapshots:
    def test_empty_returns_empty_list(self):
        db = _make_mock_db([])
        with patch("app.services.scoring.live_calibration._ensure_table_exists"):
            out = list_snapshots(db)
        assert out == []

    def test_returns_serialized_records(self):
        from datetime import datetime
        records = [
            SimpleNamespace(
                id=2, calibrated_on=date(2026, 5, 1), version="v2",
                n_classifications=850, n_assets=15, avg_brier_delta=-0.005,
                is_active=True, created_at=datetime(2026, 5, 1, 4, 0),
            ),
            SimpleNamespace(
                id=1, calibrated_on=date(2026, 4, 1), version="v1",
                n_classifications=820, n_assets=15, avg_brier_delta=-0.003,
                is_active=False, created_at=datetime(2026, 4, 1, 4, 0),
            ),
        ]
        db = _make_mock_db(records)
        with patch("app.services.scoring.live_calibration._ensure_table_exists"):
            out = list_snapshots(db, limit=10)
        assert len(out) == 2
        assert out[0]["id"] == 2
        assert out[0]["is_active"] is True
        assert out[0]["version"] == "v2"
        assert out[1]["is_active"] is False


class TestCalibratedOrPriorPriority:
    """Verifica lookup priority: DB (T6.6) > JSON (T1) > hardcoded prior."""

    def test_default_flags_off_returns_hardcoded(self, monkeypatch):
        """T10b: USE_CALIBRATED_SCORING promosso default-ON. Disable esplicito."""
        monkeypatch.setenv("USE_LIVE_CALIBRATION", "0")
        monkeypatch.setenv("USE_CALIBRATED_SCORING", "0")
        from app.services.scoring.engine import ASSET_REGIME_DATA, _calibrated_or_prior

        out = _calibrated_or_prior()
        # Hardcoded prior usato → stessi object reference
        assert out is ASSET_REGIME_DATA

    def test_live_calibration_on_db_empty_falls_back(self, monkeypatch):
        """USE_LIVE_CALIBRATION ON ma DB empty → fallback hardcoded.

        T10b: disable CALIBRATED_SCORING per evitare JSON path.
        """
        monkeypatch.setenv("USE_LIVE_CALIBRATION", "1")
        monkeypatch.setenv("USE_CALIBRATED_SCORING", "0")
        from app.services.scoring.engine import ASSET_REGIME_DATA, _calibrated_or_prior

        # Mock load_latest_from_db returning None
        with patch(
            "app.services.scoring.live_calibration.load_latest_from_db",
            return_value=None,
        ):
            out = _calibrated_or_prior()
        assert out is ASSET_REGIME_DATA

    def test_live_calibration_on_db_present(self, monkeypatch):
        """USE_LIVE_CALIBRATION ON + DB ha snapshot → usa DB payload."""
        monkeypatch.setenv("USE_LIVE_CALIBRATION", "1")
        from app.services.scoring.engine import _calibrated_or_prior

        fake_payload = {
            "asset_regime_data": {
                "us_equities_growth": {
                    "reflation": {"hit_rate": 0.99, "avg_return": 0.50, "vol": 0.10, "sharpe": 5.0}
                }
            }
        }
        with patch(
            "app.services.scoring.live_calibration.load_latest_from_db",
            return_value=fake_payload,
        ):
            out = _calibrated_or_prior()
        # Calibrato → growth.reflation deve avere hit_rate 0.99 (vs ~0.78 hardcoded)
        assert out["us_equities_growth"]["reflation"]["hit_rate"] == 0.99

    def test_live_calibration_priority_over_json(self, monkeypatch):
        """Se ENTRAMBI flag ON, DB prevale su JSON."""
        monkeypatch.setenv("USE_LIVE_CALIBRATION", "1")
        monkeypatch.setenv("USE_CALIBRATED_SCORING", "1")
        from app.services.scoring.engine import _calibrated_or_prior

        db_payload = {
            "asset_regime_data": {
                "us_equities_growth": {
                    "reflation": {"hit_rate": 0.95, "avg_return": 0.20, "vol": 0.15, "sharpe": 1.33}
                }
            }
        }
        with patch(
            "app.services.scoring.live_calibration.load_latest_from_db",
            return_value=db_payload,
        ), patch(
            "app.services.scoring.calibration.load_calibration",
            return_value={"asset_regime_data": {"us_equities_growth": {
                "reflation": {"hit_rate": 0.50, "avg_return": 0.0, "vol": 1.0, "sharpe": 0.0}
            }}},
        ):
            out = _calibrated_or_prior()
        # DB prevale: hit_rate 0.95 (non 0.50 da JSON)
        assert out["us_equities_growth"]["reflation"]["hit_rate"] == 0.95

    def test_json_fallback_when_db_off(self, monkeypatch):
        """LIVE_CALIBRATION OFF + CALIBRATED_SCORING ON → usa JSON."""
        monkeypatch.delenv("USE_LIVE_CALIBRATION", raising=False)
        monkeypatch.setenv("USE_CALIBRATED_SCORING", "1")
        from app.services.scoring.engine import _calibrated_or_prior

        json_payload = {
            "asset_regime_data": {
                "us_equities_growth": {
                    "reflation": {"hit_rate": 0.75, "avg_return": 0.12, "vol": 0.18, "sharpe": 0.67}
                }
            }
        }
        with patch(
            "app.services.scoring.calibration.load_calibration",
            return_value=json_payload,
        ):
            out = _calibrated_or_prior()
        assert out["us_equities_growth"]["reflation"]["hit_rate"] == 0.75


class TestCalibrateAndPersist:
    def test_compute_failure_returns_none(self):
        from app.services.scoring.live_calibration import calibrate_and_persist

        db = _make_mock_db([])
        with patch(
            "app.services.scoring.calibration.calibrate",
            side_effect=Exception("compute failed"),
        ), patch(
            "app.services.scoring.live_calibration._ensure_table_exists",
        ):
            out = calibrate_and_persist(db)
        assert out is None

    def test_empty_payload_returns_none(self):
        from app.services.scoring.live_calibration import calibrate_and_persist

        db = _make_mock_db([])
        with patch(
            "app.services.scoring.calibration.calibrate",
            return_value={},  # no asset_regime_data key
        ), patch(
            "app.services.scoring.live_calibration._ensure_table_exists",
        ):
            out = calibrate_and_persist(db)
        assert out is None

    def test_success_path_returns_result(self):
        from app.services.scoring.live_calibration import calibrate_and_persist

        db = _make_mock_db([])
        db.add = MagicMock()
        db.commit = MagicMock()
        fake_payload = {
            "asset_regime_data": {"us_equities_growth": {"reflation": {"hit_rate": 0.8}}},
            "n_classifications": 800,
        }
        with patch(
            "app.services.scoring.calibration.calibrate",
            return_value=fake_payload,
        ), patch(
            "app.services.scoring.live_calibration._ensure_table_exists",
        ):
            out = calibrate_and_persist(db)
        assert out is not None
        assert isinstance(out, LiveCalibrationResult)
        assert out.n_classifications == 800
        assert out.n_assets == 1
        assert "asset_regime_data" in out.payload
        db.add.assert_called_once()
        db.commit.assert_called_once()
