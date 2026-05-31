"""Test T15 Tail hedge allocator — council action #3.

Coverage:
- should_activate_hedge (3): threshold trigger
- compute_hedge_weight (5): scaling lineare, edge cases
- simulate_hedge_return (4): payoff buckets
- apply_hedge_to_allocation (4): sum=1, scale-down, edge cases
"""
from __future__ import annotations

import pytest

from app.services.portfolio.tail_hedge import (
    HEDGE_MAX_BUDGET,
    HEDGE_MIN_BUDGET,
    HEDGE_TRIGGER_THRESHOLD,
    apply_hedge_to_allocation,
    compute_hedge_weight,
    should_activate_hedge,
    simulate_hedge_return,
)


class TestShouldActivate:
    def test_above_threshold_fires(self):
        assert should_activate_hedge(0.70)
        assert should_activate_hedge(0.95)

    def test_below_threshold_no_fire(self):
        assert not should_activate_hedge(0.55)
        assert not should_activate_hedge(0.30)

    def test_exactly_at_threshold_fires(self):
        # Default 0.60 → score=0.60 fires
        assert should_activate_hedge(HEDGE_TRIGGER_THRESHOLD)

    def test_custom_threshold(self):
        assert should_activate_hedge(0.50, threshold=0.45)
        assert not should_activate_hedge(0.40, threshold=0.45)


class TestComputeHedgeWeight:
    def test_below_threshold_returns_zero(self):
        assert compute_hedge_weight(0.50) == 0.0

    def test_at_threshold_returns_min_budget(self):
        w = compute_hedge_weight(HEDGE_TRIGGER_THRESHOLD)
        assert w == pytest.approx(HEDGE_MIN_BUDGET, abs=1e-9)

    def test_score_1_returns_max_budget(self):
        w = compute_hedge_weight(1.0)
        assert w == pytest.approx(HEDGE_MAX_BUDGET, abs=1e-9)

    def test_linear_scaling_at_midpoint(self):
        """Score 0.80 = midpoint tra threshold 0.60 e 1.0 → mid weight."""
        w = compute_hedge_weight(0.80)
        expected = HEDGE_MIN_BUDGET + 0.5 * (HEDGE_MAX_BUDGET - HEDGE_MIN_BUDGET)
        assert w == pytest.approx(expected, abs=1e-9)

    def test_custom_max_budget(self):
        w = compute_hedge_weight(1.0, max_budget=0.05)
        assert w == 0.05


class TestSimulateHedgeReturn:
    def test_deep_crash_pays_50pct(self):
        # SPY -15% → hedge +50%
        assert simulate_hedge_return(-0.15) == 0.50
        assert simulate_hedge_return(-0.20) == 0.50

    def test_moderate_crash_pays_20pct(self):
        # SPY -8% → hedge +20%
        assert simulate_hedge_return(-0.08) == 0.20
        assert simulate_hedge_return(-0.06) == 0.20

    def test_minor_dip_loses_5pct(self):
        # SPY -3% → hedge -5%
        assert simulate_hedge_return(-0.03) == -0.05
        assert simulate_hedge_return(-0.01) == -0.05

    def test_calm_or_up_loses_8pct(self):
        # SPY 0% o positivo → hedge -8%
        assert simulate_hedge_return(0.0) == -0.08
        assert simulate_hedge_return(0.05) == -0.08
        assert simulate_hedge_return(0.15) == -0.08


class TestApplyHedgeToAllocation:
    def test_no_hedge_returns_copy(self):
        base = {"a": 0.5, "b": 0.5}
        out = apply_hedge_to_allocation(base, hedge_weight=0.0)
        assert out == base
        assert out is not base  # è copia, non riferimento

    def test_hedge_2pct_scales_existing(self):
        base = {"a": 0.5, "b": 0.5}
        out = apply_hedge_to_allocation(base, hedge_weight=0.02)
        assert out["tail_hedge_spy_put"] == pytest.approx(0.02)
        assert out["a"] == pytest.approx(0.5 * 0.98)
        assert out["b"] == pytest.approx(0.5 * 0.98)
        assert sum(out.values()) == pytest.approx(1.0)

    def test_custom_asset_key(self):
        base = {"a": 1.0}
        out = apply_hedge_to_allocation(
            base, hedge_weight=0.05, hedge_asset_key="vix_long",
        )
        assert "vix_long" in out
        assert "tail_hedge_spy_put" not in out

    def test_hedge_full_edge_case(self):
        # hedge_weight >= 1.0 → tutto hedge (edge robustness)
        out = apply_hedge_to_allocation({"a": 1.0}, hedge_weight=1.0)
        assert out == {"tail_hedge_spy_put": 1.0}


class TestFullScenarioIntegration:
    def test_2008_scenario_activates_max_hedge(self):
        """Scenario crisis score 0.95 → activated, near max budget."""
        score = 0.95
        assert should_activate_hedge(score)
        w = compute_hedge_weight(score)
        assert HEDGE_MIN_BUDGET < w <= HEDGE_MAX_BUDGET

    def test_calm_2017_no_hedge_no_drag(self):
        """Crisis score 0.20 → NO hedge → portfolio non zappato."""
        score = 0.20
        assert not should_activate_hedge(score)
        w = compute_hedge_weight(score)
        assert w == 0.0

    def test_hedge_pays_in_crash_drag_in_calm(self):
        """Verifica convex payoff: hedge ratio crash/calm > 5x."""
        crash_return = simulate_hedge_return(-0.15)
        calm_return = simulate_hedge_return(0.05)
        # 0.50 / abs(-0.08) = 6.25
        assert crash_return / abs(calm_return) > 5


class TestVixyProxy:
    """T15.4 — VIXY proxy alternative payoffs."""

    def test_vixy_deeper_crash_payoff(self):
        """VIXY payoff in deep crash > OTM put payoff."""
        otm = simulate_hedge_return(-0.15, proxy="otm_put")
        vixy = simulate_hedge_return(-0.15, proxy="vixy")
        assert vixy > otm  # 0.80 > 0.50

    def test_vixy_higher_calm_decay(self):
        """VIXY decay in calm più aggressivo (roll cost)."""
        otm = simulate_hedge_return(0.05, proxy="otm_put")
        vixy = simulate_hedge_return(0.05, proxy="vixy")
        assert vixy < otm  # -0.10 < -0.08

    def test_vixy_moderate_crash_higher(self):
        otm = simulate_hedge_return(-0.07, proxy="otm_put")
        vixy = simulate_hedge_return(-0.07, proxy="vixy")
        assert vixy > otm  # 0.35 > 0.20

    def test_default_proxy_is_otm_put(self):
        """No proxy arg → otm_put default."""
        default = simulate_hedge_return(-0.15)
        otm = simulate_hedge_return(-0.15, proxy="otm_put")
        assert default == otm
