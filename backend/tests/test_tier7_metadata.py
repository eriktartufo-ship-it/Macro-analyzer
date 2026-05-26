"""Test Tier 7.6 — applied_overrides metadata in calculate_final_scores response.

Bridgewater "radical transparency": ogni override applicato deve essere
visibile a valle.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.services.scoring.engine import (
    calculate_final_scores,
    calculate_final_scores_with_metadata,
    reload_calibration,
)


@pytest.fixture(autouse=True)
def _isolate_flags(monkeypatch):
    for flag in (
        "USE_CALIBRATED_SCORING",
        "USE_LIVE_CALIBRATION",
        "USE_DEDOLLAR_BONUS",
        "USE_DFM_ASSET_BONUS",
        "USE_DISCRETIONARY_OVERRIDES",
        "USE_RANK_PERCENTILE_SCORING",
        "USE_DOWNSIDE_PROTECTION_BONUS",
    ):
        monkeypatch.delenv(flag, raising=False)
    reload_calibration()
    yield
    reload_calibration()


def _make_override(target_type, target_name, delta, expires_in_days=1, id_=1):
    return SimpleNamespace(
        id=id_,
        target_type=target_type,
        target_name=target_name,
        delta=delta,
        expires_at=datetime.utcnow() + timedelta(days=expires_in_days),
        reason="test reason",
        author="tester",
        created_at=datetime.utcnow(),
    )


class TestMetadataShape:
    def test_no_overrides_returns_empty_lists(self):
        probs = {"reflation": 1.0, "stagflation": 0.0, "deflation": 0.0, "goldilocks": 0.0}
        out = calculate_final_scores_with_metadata(probs)
        assert isinstance(out, dict)
        assert "scores" in out
        assert "applied_regime_overrides" in out
        assert "applied_asset_overrides" in out
        assert out["applied_regime_overrides"] == []
        assert out["applied_asset_overrides"] == []

    def test_flags_state_included(self, monkeypatch):
        probs = {"stagflation": 0.7, "deflation": 0.3, "reflation": 0.0, "goldilocks": 0.0}
        monkeypatch.setenv("USE_DOWNSIDE_PROTECTION_BONUS", "1")
        monkeypatch.setenv("USE_RANK_PERCENTILE_SCORING", "1")
        out = calculate_final_scores_with_metadata(probs)
        assert out["downside_protection_active"] is True
        assert out["rank_percentile_active"] is True

    def test_downside_inactive_in_reflation(self, monkeypatch):
        probs = {"reflation": 1.0, "stagflation": 0.0, "deflation": 0.0, "goldilocks": 0.0}
        monkeypatch.setenv("USE_DOWNSIDE_PROTECTION_BONUS", "1")
        out = calculate_final_scores_with_metadata(probs)
        assert out["downside_protection_active"] is False


class TestOverridesInMetadata:
    def test_asset_override_visible_in_metadata(self, monkeypatch):
        monkeypatch.setenv("USE_DISCRETIONARY_OVERRIDES", "1")
        ov = _make_override("asset", "gold", -10.0, id_=42)

        # Mock load_active_overrides to return our test override
        with patch(
            "app.services.scoring.discretionary.load_active_overrides",
            return_value=[ov],
        ):
            with patch("app.database.SessionLocal") as mock_session:
                mock_session.return_value.__enter__.return_value = MagicMock()
                probs = {"reflation": 1.0, "stagflation": 0.0, "deflation": 0.0, "goldilocks": 0.0}
                out = calculate_final_scores_with_metadata(probs)

        applied = out["applied_asset_overrides"]
        assert len(applied) == 1
        assert applied[0]["target_name"] == "gold"
        assert applied[0]["delta"] == -10.0
        assert applied[0]["id"] == 42
        assert applied[0]["reason"] == "test reason"

    def test_regime_override_visible_in_metadata(self, monkeypatch):
        monkeypatch.setenv("USE_DISCRETIONARY_OVERRIDES", "1")
        ov = _make_override("regime", "deflation", +0.10, id_=7)

        with patch(
            "app.services.scoring.discretionary.load_active_overrides",
            return_value=[ov],
        ):
            with patch("app.database.SessionLocal") as mock_session:
                mock_session.return_value.__enter__.return_value = MagicMock()
                probs = {
                    "reflation": 0.40, "stagflation": 0.30,
                    "deflation": 0.20, "goldilocks": 0.10,
                }
                out = calculate_final_scores_with_metadata(probs)

        applied_regime = out["applied_regime_overrides"]
        assert len(applied_regime) == 1
        assert applied_regime[0]["target_name"] == "deflation"
        assert applied_regime[0]["delta"] == 0.10


class TestBackCompat:
    def test_calculate_final_scores_still_returns_dict(self):
        """T7.6 NON deve rompere il return type di calculate_final_scores."""
        probs = {"reflation": 1.0, "stagflation": 0.0, "deflation": 0.0, "goldilocks": 0.0}
        scores = calculate_final_scores(probs)
        assert isinstance(scores, dict)
        # Tutti i valori numerici 0-100
        for asset, score in scores.items():
            assert 0.0 <= score <= 100.0

    def test_metadata_scores_match_legacy(self):
        """`with_metadata['scores']` deve essere identico a `calculate_final_scores`."""
        probs = {"reflation": 0.5, "stagflation": 0.3, "deflation": 0.1, "goldilocks": 0.1}
        legacy = calculate_final_scores(probs)
        wrapped = calculate_final_scores_with_metadata(probs)
        assert wrapped["scores"] == legacy
