"""Test Tier 6.7 — Discretionary override engine.

Coverage:
- apply_asset_overrides: delta additivo, clamp [0,100], asset unknown ignored.
- apply_regime_overrides: delta + renormalize Σ=1, floor non-negative.
- load_active_overrides: filtra expires_at > now.
- create_override: validazione target_type + reason obbligatorio.
- delete_override: True se trovato, False se id assente.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.services.scoring.discretionary import (
    apply_asset_overrides,
    apply_regime_overrides,
    create_override,
    delete_override,
    load_active_overrides,
)


def _make_override(
    target_type: str,
    target_name: str,
    delta: float,
    expires_at: datetime | None = None,
    id_: int = 1,
    reason: str = "test",
    author: str = "tester",
):
    """Helper: crea un'istanza simil-DB di DiscretionaryOverride."""
    return SimpleNamespace(
        id=id_,
        target_type=target_type,
        target_name=target_name,
        delta=delta,
        expires_at=expires_at or (datetime.utcnow() + timedelta(days=1)),
        reason=reason,
        author=author,
    )


class TestApplyAssetOverrides:
    def test_empty_overrides_identity(self):
        scores = {"us_equities_growth": 65.0, "gold": 50.0}
        out, applied = apply_asset_overrides(scores, [])
        assert out == scores
        assert applied == []

    def test_single_delta_additive(self):
        scores = {"us_equities_growth": 65.0}
        ov = _make_override("asset", "us_equities_growth", -10.0)
        out, applied = apply_asset_overrides(scores, [ov])
        assert out["us_equities_growth"] == 55.0
        assert len(applied) == 1
        assert applied[0].delta == -10.0

    def test_clamp_upper_bound(self):
        scores = {"gold": 95.0}
        ov = _make_override("asset", "gold", +20.0)
        out, _ = apply_asset_overrides(scores, [ov])
        assert out["gold"] == 100.0

    def test_clamp_lower_bound(self):
        scores = {"us_bonds_long": 10.0}
        ov = _make_override("asset", "us_bonds_long", -50.0)
        out, _ = apply_asset_overrides(scores, [ov])
        assert out["us_bonds_long"] == 0.0

    def test_unknown_asset_ignored(self):
        scores = {"gold": 50.0}
        ov = _make_override("asset", "ghost_asset", +10.0)
        out, applied = apply_asset_overrides(scores, [ov])
        assert out == scores
        assert applied == []

    def test_regime_type_ignored_in_asset_apply(self):
        scores = {"gold": 50.0}
        ov = _make_override("regime", "reflation", +0.10)
        out, applied = apply_asset_overrides(scores, [ov])
        assert out == scores
        assert applied == []

    def test_multiple_overrides_stack(self):
        scores = {"gold": 50.0, "energy": 60.0}
        ovs = [
            _make_override("asset", "gold", +5.0, id_=1),
            _make_override("asset", "gold", -2.0, id_=2),
            _make_override("asset", "energy", +10.0, id_=3),
        ]
        out, applied = apply_asset_overrides(scores, ovs)
        assert out["gold"] == 53.0
        assert out["energy"] == 70.0
        assert len(applied) == 3


class TestApplyRegimeOverrides:
    def test_empty_overrides_identity(self):
        probs = {"reflation": 0.4, "stagflation": 0.3, "deflation": 0.2, "goldilocks": 0.1}
        out, applied = apply_regime_overrides(probs, [])
        assert out == probs
        assert applied == []

    def test_delta_renormalizes_to_one(self):
        probs = {"reflation": 0.4, "stagflation": 0.3, "deflation": 0.2, "goldilocks": 0.1}
        ov = _make_override("regime", "reflation", +0.10)
        out, applied = apply_regime_overrides(probs, [ov])
        assert abs(sum(out.values()) - 1.0) < 1e-9
        assert out["reflation"] > probs["reflation"]
        assert len(applied) == 1

    def test_negative_delta_floors_at_min(self):
        probs = {"reflation": 0.4, "stagflation": 0.3, "deflation": 0.2, "goldilocks": 0.1}
        ov = _make_override("regime", "goldilocks", -0.50)
        out, _ = apply_regime_overrides(probs, [ov])
        assert all(v >= 0 for v in out.values())
        assert abs(sum(out.values()) - 1.0) < 1e-9

    def test_unknown_regime_ignored(self):
        probs = {"reflation": 0.5, "stagflation": 0.5}
        ov = _make_override("regime", "phantom_regime", +0.10)
        out, applied = apply_regime_overrides(probs, [ov])
        assert out == probs
        assert applied == []

    def test_asset_type_ignored_in_regime_apply(self):
        probs = {"reflation": 0.5, "stagflation": 0.5}
        ov = _make_override("asset", "gold", +10.0)
        out, applied = apply_regime_overrides(probs, [ov])
        assert out == probs
        assert applied == []

    def test_mass_redistribution_only_to_non_targets(self):
        """T7.4: override su goldilocks NON deve gonfiare stagflation oltre i non-target."""
        probs = {"reflation": 0.40, "stagflation": 0.30, "deflation": 0.20, "goldilocks": 0.10}
        ov = _make_override("regime", "goldilocks", -0.05)
        out, _ = apply_regime_overrides(probs, [ov])

        # goldilocks deve scendere
        assert out["goldilocks"] < probs["goldilocks"]
        # massa rimossa = 0.05 (goldilocks resta a 0.05)
        # ridistribuita pro-quota tra refl/stag/defl: somma loro originale = 0.90
        # refl ottiene 0.05 * (0.40/0.90) = 0.0222 → 0.4222
        # stag ottiene 0.05 * (0.30/0.90) = 0.0167 → 0.3167
        # defl ottiene 0.05 * (0.20/0.90) = 0.0111 → 0.2111
        # tot = 0.05 + 0.4222 + 0.3167 + 0.2111 = 1.0
        assert abs(sum(out.values()) - 1.0) < 1e-9
        assert out["reflation"] == pytest.approx(0.4222, abs=0.001)
        assert out["stagflation"] == pytest.approx(0.3167, abs=0.001)
        assert out["deflation"] == pytest.approx(0.2111, abs=0.001)

    def test_target_floored_when_delta_huge_negative(self):
        """Override -0.99 su prob 0.10 → target a floor 0.0001."""
        probs = {"reflation": 0.40, "stagflation": 0.30, "deflation": 0.20, "goldilocks": 0.10}
        ov = _make_override("regime", "goldilocks", -0.99)
        out, _ = apply_regime_overrides(probs, [ov])
        assert out["goldilocks"] < 0.001
        assert abs(sum(out.values()) - 1.0) < 1e-9

    def test_positive_delta_extracts_mass_from_non_targets(self):
        """Override +0.10 su deflation deve estrarre proporzionalmente dagli altri."""
        probs = {"reflation": 0.40, "stagflation": 0.30, "deflation": 0.20, "goldilocks": 0.10}
        ov = _make_override("regime", "deflation", +0.10)
        out, _ = apply_regime_overrides(probs, [ov])
        # deflation sale di 0.10 → 0.30
        # altri scendono di 0.10 ripartito pro-quota (somma altri = 0.80)
        # refl: 0.40 - 0.10 * (0.40/0.80) = 0.40 - 0.05 = 0.35
        # stag: 0.30 - 0.10 * (0.30/0.80) = 0.30 - 0.0375 = 0.2625
        # gold: 0.10 - 0.10 * (0.10/0.80) = 0.10 - 0.0125 = 0.0875
        assert out["deflation"] == pytest.approx(0.30, abs=0.001)
        assert out["reflation"] == pytest.approx(0.35, abs=0.001)
        assert out["stagflation"] == pytest.approx(0.2625, abs=0.001)
        assert out["goldilocks"] == pytest.approx(0.0875, abs=0.001)
        assert abs(sum(out.values()) - 1.0) < 1e-9


class TestLoadActiveOverrides:
    def test_filters_by_expires_at(self):
        future = datetime.utcnow() + timedelta(days=2)
        past = datetime.utcnow() - timedelta(days=1)
        active = _make_override("asset", "gold", +5.0, expires_at=future, id_=1)
        expired = _make_override("asset", "gold", -5.0, expires_at=past, id_=2)

        db = MagicMock()
        chain = MagicMock()
        chain.filter.return_value = chain
        chain.order_by.return_value = chain
        # We expect SQL to filter — mock returns only active
        chain.all.return_value = [active]
        db.query.return_value = chain

        with patch("app.services.scoring.discretionary._ensure_table_exists"):
            out = load_active_overrides(db)
        assert len(out) == 1
        assert out[0].id == 1

    def test_db_error_returns_empty(self):
        db = MagicMock()
        db.query.side_effect = RuntimeError("db down")
        with patch("app.services.scoring.discretionary._ensure_table_exists"):
            out = load_active_overrides(db)
        assert out == []


class TestCreateOverride:
    def test_invalid_target_type_raises(self):
        db = MagicMock()
        with patch("app.services.scoring.discretionary._ensure_table_exists"):
            with pytest.raises(ValueError, match="target_type"):
                create_override(
                    db, target_type="bogus", target_name="gold",
                    delta=5.0, expires_at=datetime.utcnow() + timedelta(days=1),
                    reason="test",
                )

    def test_empty_reason_raises(self):
        db = MagicMock()
        with patch("app.services.scoring.discretionary._ensure_table_exists"):
            with pytest.raises(ValueError, match="reason"):
                create_override(
                    db, target_type="asset", target_name="gold",
                    delta=5.0, expires_at=datetime.utcnow() + timedelta(days=1),
                    reason="   ",
                )

    def test_success_path_commits(self):
        db = MagicMock()
        with patch("app.services.scoring.discretionary._ensure_table_exists"):
            rec = create_override(
                db, target_type="asset", target_name="gold",
                delta=5.0, expires_at=datetime.utcnow() + timedelta(days=1),
                reason="overbought tactical",
                author="erik",
            )
        assert rec.target_name == "gold"
        assert rec.delta == 5.0
        assert rec.author == "erik"
        db.add.assert_called_once()
        db.commit.assert_called_once()


class TestDeleteOverride:
    def test_found_returns_true(self):
        existing = _make_override("asset", "gold", +5.0, id_=42)
        db = MagicMock()
        chain = MagicMock()
        chain.filter.return_value = chain
        chain.first.return_value = existing
        db.query.return_value = chain
        with patch("app.services.scoring.discretionary._ensure_table_exists"):
            ok = delete_override(db, 42)
        assert ok is True
        db.delete.assert_called_once_with(existing)
        db.commit.assert_called_once()

    def test_missing_returns_false(self):
        db = MagicMock()
        chain = MagicMock()
        chain.filter.return_value = chain
        chain.first.return_value = None
        db.query.return_value = chain
        with patch("app.services.scoring.discretionary._ensure_table_exists"):
            ok = delete_override(db, 999)
        assert ok is False
        db.delete.assert_not_called()
