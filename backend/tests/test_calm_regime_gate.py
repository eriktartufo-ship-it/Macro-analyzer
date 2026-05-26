"""Test calm regime gate."""
from __future__ import annotations

import pytest

from app.services.portfolio.calm_regime_gate import (
    CALM_ALLOCATION_60_40,
    apply_calm_gate,
    assess_calm_gate,
)


class TestAssessCalmGate:
    """T9-AUDIT-TA: soglie recalibrate (0.30/0.30) su distribuzione storica.

    Reason: audit empirico DB (1025 records, 2026-05-17):
    - max P(goldilocks) = 0.418 → threshold 0.55 unreachable
    - P(goldi)>0.30 AND conf>0.45 si verifica solo 1 volta (0.1%)
    - P(goldi)>0.30 AND conf>0.30 si verifica 57 volte (5.6%) → operating point
    - Anti-correlazione strutturale: alta P(goldi) implica conf medio-bassa.
    """

    def test_fired_when_goldilocks_dominant(self):
        probs = {"goldilocks": 0.38, "reflation": 0.32, "stagflation": 0.15, "deflation": 0.15}
        out = assess_calm_gate(probs, confidence=0.35)
        assert out.is_fired is True

    def test_not_fired_low_confidence(self):
        probs = {"goldilocks": 0.38, "reflation": 0.32, "stagflation": 0.15, "deflation": 0.15}
        out = assess_calm_gate(probs, confidence=0.25)
        assert out.is_fired is False

    def test_not_fired_goldilocks_low(self):
        probs = {"goldilocks": 0.20, "reflation": 0.50, "stagflation": 0.20, "deflation": 0.10}
        out = assess_calm_gate(probs, confidence=0.60)
        assert out.is_fired is False

    def test_threshold_boundary(self):
        # Exactly at threshold (0.30): NOT triggered (strict >)
        probs = {"goldilocks": 0.30, "reflation": 0.40, "stagflation": 0.20, "deflation": 0.10}
        out = assess_calm_gate(probs, confidence=0.30)
        assert out.is_fired is False  # both at boundary strict

    def test_just_above_threshold(self):
        probs = {"goldilocks": 0.35, "reflation": 0.35, "stagflation": 0.20, "deflation": 0.10}
        out = assess_calm_gate(probs, confidence=0.32)
        assert out.is_fired is True


class TestApplyCalmGate:
    def test_gate_fired_returns_60_40(self):
        top5 = {"us_equities_growth": 75, "bitcoin": 70, "gold": 65, "em_equities": 60, "real_estate_reits": 55}
        probs = {"goldilocks": 0.70, "reflation": 0.20, "stagflation": 0.05, "deflation": 0.05}
        weights, assessment = apply_calm_gate(top5, probs, confidence=0.80)
        assert assessment.is_fired is True
        # 60/40 hardcoded
        assert weights == CALM_ALLOCATION_60_40
        assert weights["us_equities_growth"] == 0.60
        assert weights["us_bonds_long"] == 0.40

    def test_gate_not_fired_returns_equal_weight_top5(self):
        top5 = {"a": 75, "b": 70, "c": 65, "d": 60, "e": 55}
        probs = {"reflation": 0.50, "stagflation": 0.30, "goldilocks": 0.15, "deflation": 0.05}
        weights, assessment = apply_calm_gate(top5, probs, confidence=0.60)
        assert assessment.is_fired is False
        # Equal weight 5 assets = 20% each
        for asset in top5:
            assert weights[asset] == pytest.approx(0.20)

    def test_empty_top5_no_crash(self):
        weights, _ = apply_calm_gate({}, {"reflation": 1.0}, confidence=0.5)
        assert weights == {}

    def test_calm_60_40_sums_to_one(self):
        assert sum(CALM_ALLOCATION_60_40.values()) == pytest.approx(1.0)


class TestCouncilScenarios:
    """Scenari ispirati alle 15 simulazioni random backtest."""

    def test_goldilocks_pure_1996_calm_fires(self):
        """1996 Greenspan pure goldilocks: alta conf + P(gold)>0.6."""
        probs = {"goldilocks": 0.68, "reflation": 0.25, "stagflation": 0.04, "deflation": 0.03}
        out = assess_calm_gate(probs, confidence=0.78)
        assert out.is_fired is True

    def test_stagflation_2022_no_gate(self):
        """2022 stagflation: bassa P(goldi), no gate."""
        probs = {"stagflation": 0.60, "reflation": 0.25, "goldilocks": 0.10, "deflation": 0.05}
        out = assess_calm_gate(probs, confidence=0.70)
        assert out.is_fired is False

    def test_2008_gfc_no_gate(self):
        """2008 GFC deflation: no goldilocks, no gate."""
        probs = {"deflation": 0.65, "stagflation": 0.20, "reflation": 0.10, "goldilocks": 0.05}
        out = assess_calm_gate(probs, confidence=0.75)
        assert out.is_fired is False

    def test_current_2026_mixed_no_gate(self):
        """Current 2026 reflation 42%, goldilocks 22%: no gate."""
        probs = {"reflation": 0.42, "goldilocks": 0.22, "stagflation": 0.18, "deflation": 0.18}
        out = assess_calm_gate(probs, confidence=0.50)
        assert out.is_fired is False
