"""Test Tier 9 — Defensive transition mode detector."""
from __future__ import annotations

import pytest

from app.services.regime.transition_detector import (
    DEFENSIVE_ALLOCATION,
    assess_transition,
    # blend_with_defensive deleted 2026-05-28
    get_defensive_allocation,
    shannon_entropy,
)


class TestShannonEntropy:
    def test_uniform_max_entropy(self):
        uniform = {"reflation": 0.25, "stagflation": 0.25, "deflation": 0.25, "goldilocks": 0.25}
        assert shannon_entropy(uniform) == pytest.approx(2.0, abs=0.001)

    def test_concentrated_low_entropy(self):
        concentrated = {"reflation": 0.97, "stagflation": 0.01, "deflation": 0.01, "goldilocks": 0.01}
        assert shannon_entropy(concentrated) < 0.5

    def test_two_regime_split(self):
        split = {"reflation": 0.5, "stagflation": 0.5, "deflation": 0.0, "goldilocks": 0.0}
        # H = -2 * 0.5 * log2(0.5) = 1.0
        assert shannon_entropy(split) == pytest.approx(1.0, abs=0.001)


class TestAssessTransition:
    def test_clear_regime_no_transition(self):
        """Goldilocks chiaro: nessun trigger fired."""
        probs = {"reflation": 0.1, "stagflation": 0.05, "deflation": 0.05, "goldilocks": 0.80}
        out = assess_transition(probs, confidence=0.75)
        assert out.is_transition is False
        assert out.triggers_count == 0

    def test_mixed_regime_triggers_transition(self):
        """T9-AUDIT: 3 regimi quasi pari + low confidence → 3/3 trigger fired."""
        probs = {"reflation": 0.30, "stagflation": 0.28, "deflation": 0.22, "goldilocks": 0.20}
        out = assess_transition(probs, confidence=0.20)  # confidence stretto a 0.30
        assert out.is_transition is True
        assert out.triggers_count == 3

    def test_two_triggers_not_transition_strict(self):
        """T9-AUDIT: require 3/3. Solo 2 trigger fired → NO transition."""
        # max_prob 0.35 < 0.38 → trigger 1
        # entropy ~1.55 borderline (probs 0.35/0.35/0.15/0.15 → ent ~1.88) → trigger 2
        # confidence alta → no trigger 3
        probs = {"reflation": 0.35, "stagflation": 0.35, "deflation": 0.15, "goldilocks": 0.15}
        out = assess_transition(probs, confidence=0.60)
        # 2/3 fired ma require 3/3 → NO transition (T9-AUDIT più conservativo)
        assert out.is_transition is False

    def test_low_confidence_alone_not_transition(self):
        """Probs dominate ma confidence basso: 1 trigger → NON sufficient."""
        probs = {"reflation": 0.70, "stagflation": 0.15, "deflation": 0.10, "goldilocks": 0.05}
        out = assess_transition(probs, confidence=0.20)
        # Solo trigger confidence
        assert out.triggers_count == 1
        assert out.is_transition is False

    def test_triggers_listed_with_values(self):
        probs = {"reflation": 0.25, "stagflation": 0.25, "deflation": 0.25, "goldilocks": 0.25}
        out = assess_transition(probs, confidence=0.10)  # T9: confidence stretto
        # Tutti 3 trigger fired
        assert out.triggers_count == 3
        assert any("max_prob" in t for t in out.triggers_fired)
        assert any("entropy" in t for t in out.triggers_fired)
        assert any("confidence" in t for t in out.triggers_fired)


class TestDefensiveAllocation:
    def test_sum_to_one(self):
        defensive = get_defensive_allocation()
        assert abs(sum(defensive.values()) - 1.0) < 1e-9

    def test_safe_assets_only(self):
        defensive = get_defensive_allocation()
        safe_assets = {
            "cash_money_market", "us_bonds_short", "us_bonds_long",
            "gold", "tips_inflation_bonds",
        }
        assert set(defensive.keys()) == safe_assets

    def test_cash_heavy(self):
        defensive = get_defensive_allocation()
        # Cash + short bonds ≥ 50%
        cash_share = defensive["cash_money_market"] + defensive["us_bonds_short"]
        assert cash_share >= 0.50

    def test_no_speculative_assets(self):
        defensive = get_defensive_allocation()
        assert "bitcoin" not in defensive
        assert "us_equities_growth" not in defensive
        assert "em_equities" not in defensive


# `TestBlendWithDefensive` rimosso 2026-05-28 (code audit: orphan).


class TestScenarios:
    def test_2008_gfc_transition_detected(self):
        """T9-AUDIT: 2008 GFC con confidence VERA bassa + regime mixed → transition.
        Con nuove soglie strette serve confidence < 0.30."""
        probs = {"reflation": 0.20, "stagflation": 0.30, "deflation": 0.35, "goldilocks": 0.15}
        out = assess_transition(probs, confidence=0.20)
        assert out.is_transition is True

    def test_clear_reflation_no_transition(self):
        """2021 reflation pieno: top regime al 70%."""
        probs = {"reflation": 0.70, "stagflation": 0.10, "deflation": 0.05, "goldilocks": 0.15}
        out = assess_transition(probs, confidence=0.80)
        assert out.is_transition is False

    def test_2000_dotcom_transition(self):
        """T9-AUDIT: 2000 dotcom regimi mixed + confidence bassa."""
        probs = {"reflation": 0.27, "stagflation": 0.24, "deflation": 0.30, "goldilocks": 0.19}
        out = assess_transition(probs, confidence=0.20)
        assert out.is_transition is True
        assert out.triggers_count >= 2
