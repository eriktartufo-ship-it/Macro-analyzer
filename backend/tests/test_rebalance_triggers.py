"""Test T19 Trigger-based rebalance — council action #1 sessione 26."""
from __future__ import annotations

import pytest

from app.services.backtest.rebalance_triggers import (
    HY_OAS_JUMP_PCT_THRESHOLD,
    NFCI_Z_THRESHOLD,
    REGIME_PROB_SHIFT_THRESHOLD,
    VIX_JUMP_PCT_THRESHOLD,
    any_trigger_fired_simple,
    check_rebalance_triggers,
)


class TestRegimeProbShift:
    def test_large_shift_fires(self):
        curr = {"reflation": 0.5, "stagflation": 0.2, "deflation": 0.2, "goldilocks": 0.1}
        prev = {"reflation": 0.2, "stagflation": 0.4, "deflation": 0.3, "goldilocks": 0.1}
        out = check_rebalance_triggers(curr, prev, None, None, None)
        assert out.rebalance_now
        assert "regime_shift" in out.reason
        assert out.regime_prob_delta >= 0.20  # reflation shifted 0.30

    def test_small_shift_no_fire(self):
        curr = {"reflation": 0.40, "stagflation": 0.30, "deflation": 0.20, "goldilocks": 0.10}
        prev = {"reflation": 0.45, "stagflation": 0.28, "deflation": 0.18, "goldilocks": 0.09}
        out = check_rebalance_triggers(curr, prev, None, None, None)
        assert not out.rebalance_now
        assert out.regime_prob_delta < 0.20

    def test_none_probs_no_fire(self):
        out = check_rebalance_triggers(None, None, None, None, None)
        assert not out.rebalance_now


class TestVixJump:
    def test_panic_jump_fires(self):
        # VIX 18 → 30 = +66% jump
        out = check_rebalance_triggers(None, None, 30.0, 18.0, None)
        assert out.rebalance_now
        assert "vix_jump" in out.reason

    def test_moderate_rise_no_fire(self):
        # VIX 18 → 22 = +22% jump (below 50% threshold)
        out = check_rebalance_triggers(None, None, 22.0, 18.0, None)
        assert not out.rebalance_now

    def test_decline_no_fire(self):
        out = check_rebalance_triggers(None, None, 15.0, 25.0, None)
        assert not out.rebalance_now


class TestNfciZScore:
    def test_high_nfci_fires(self):
        # NFCI 0.8 with mean=0, std=0.5 → z=1.6 > 1.0
        out = check_rebalance_triggers(None, None, None, None, 0.8, nfci_mean=0.0, nfci_std=0.5)
        assert out.rebalance_now
        assert "nfci_z" in out.reason

    def test_neutral_nfci_no_fire(self):
        out = check_rebalance_triggers(None, None, None, None, 0.1, nfci_mean=0.0, nfci_std=0.5)
        assert not out.rebalance_now


class TestHyOasJump:
    def test_extreme_hy_jump_fires(self):
        # HY OAS 4.0 → 5.5 = +37% jump
        out = check_rebalance_triggers(
            None, None, None, None, None,
            current_hy_oas=5.5, previous_hy_oas=4.0,
        )
        assert out.rebalance_now
        assert "hy_oas_jump" in out.reason

    def test_moderate_hy_no_fire(self):
        # HY OAS 4.0 → 4.5 = +12.5% (below 30%)
        out = check_rebalance_triggers(
            None, None, None, None, None,
            current_hy_oas=4.5, previous_hy_oas=4.0,
        )
        assert not out.rebalance_now


class TestMultipleTriggers:
    def test_multiple_fired_concatenated(self):
        curr = {"reflation": 0.6, "stagflation": 0.2, "deflation": 0.1, "goldilocks": 0.1}
        prev = {"reflation": 0.3, "stagflation": 0.4, "deflation": 0.2, "goldilocks": 0.1}
        out = check_rebalance_triggers(curr, prev, 40.0, 18.0, 1.0, nfci_mean=0.0, nfci_std=0.5)
        assert out.rebalance_now
        assert "regime_shift" in out.reason
        assert "vix_jump" in out.reason
        assert "nfci_z" in out.reason


class TestAnyTriggerFiredSimple:
    def test_helper_dict_lookup(self):
        curr_probs = {"reflation": 0.6, "stagflation": 0.2, "deflation": 0.1, "goldilocks": 0.1}
        prev_probs = {"reflation": 0.3, "stagflation": 0.4, "deflation": 0.2, "goldilocks": 0.1}
        ind_curr = {"vix": 30.0, "nfci": 0.2, "hy_credit_spread": 5.0}
        ind_prev = {"vix": 18.0, "nfci": 0.0, "hy_credit_spread": 4.0}
        assert any_trigger_fired_simple(curr_probs, prev_probs, ind_curr, ind_prev)

    def test_helper_no_fire(self):
        curr_probs = {"reflation": 0.40, "stagflation": 0.30, "deflation": 0.20, "goldilocks": 0.10}
        prev_probs = {"reflation": 0.42, "stagflation": 0.28, "deflation": 0.21, "goldilocks": 0.09}
        ind_curr = {"vix": 18.0, "nfci": 0.0, "hy_credit_spread": 4.0}
        ind_prev = {"vix": 17.5, "nfci": 0.05, "hy_credit_spread": 4.05}
        assert not any_trigger_fired_simple(curr_probs, prev_probs, ind_curr, ind_prev)


class TestRealScenarios:
    def test_2008_oct_lehman_scenario(self):
        # Lehman crash week (15 Sep 2008): VIX 25 → 70, regime shift refl → defl
        curr_probs = {"reflation": 0.10, "stagflation": 0.20, "deflation": 0.55, "goldilocks": 0.15}
        prev_probs = {"reflation": 0.30, "stagflation": 0.25, "deflation": 0.30, "goldilocks": 0.15}
        out = check_rebalance_triggers(
            curr_probs, prev_probs, 70.0, 25.0, 1.5,
            nfci_mean=0.0, nfci_std=0.5,
            current_hy_oas=12.0, previous_hy_oas=6.0,
        )
        assert out.rebalance_now
        # Tutti i 4 trigger devono fire in scenario Lehman
        assert "regime_shift" in out.reason
        assert "vix_jump" in out.reason
        assert "nfci_z" in out.reason
        assert "hy_oas_jump" in out.reason

    def test_calm_2017_no_fire(self):
        # Goldilocks calm 2017: nessun shift, VIX low stable
        curr_probs = {"reflation": 0.35, "stagflation": 0.10, "deflation": 0.10, "goldilocks": 0.45}
        prev_probs = {"reflation": 0.33, "stagflation": 0.12, "deflation": 0.11, "goldilocks": 0.44}
        out = check_rebalance_triggers(
            curr_probs, prev_probs, 11.0, 12.0, -0.3,
            nfci_mean=0.0, nfci_std=0.5,
            current_hy_oas=3.5, previous_hy_oas=3.6,
        )
        assert not out.rebalance_now
