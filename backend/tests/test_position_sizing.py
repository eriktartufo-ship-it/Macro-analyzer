"""Test Tier 6.4 — Position sizing layer.

Coverage:
- 4 algoritmi: equal_weight, score_weighted, risk_parity, kelly_fractional.
- Output invariants: weights sum=1, all keys subset of top-N, no negative.
- Edge cases: empty scores, top_n=0, score_baseline edge, missing vols.
- Dispatch via compute_allocation.
"""
from __future__ import annotations

import pytest

from app.services.portfolio.position_sizing import (
    PortfolioAllocation,
    compute_allocation,
    equal_weight,
    kelly_fractional,
    list_methods,
    risk_parity,
    score_weighted,
)


def _scores_sample() -> dict[str, float]:
    """8 asset con scores ranging 30-85."""
    return {
        "us_equities_growth": 75.0,
        "us_equities_value": 68.0,
        "us_bonds_long": 45.0,
        "gold": 80.0,
        "broad_commodities": 55.0,
        "cash_money_market": 50.0,
        "real_estate_reits": 40.0,
        "bitcoin": 85.0,
    }


def _vols_sample() -> dict[str, float]:
    return {
        "us_equities_growth": 0.17,
        "us_equities_value": 0.15,
        "us_bonds_long": 0.12,
        "gold": 0.18,
        "broad_commodities": 0.22,
        "cash_money_market": 0.01,
        "real_estate_reits": 0.20,
        "bitcoin": 0.65,
    }


class TestEqualWeight:
    def test_top_n_5_returns_5_weights(self):
        out = equal_weight(_scores_sample(), top_n=5)
        assert len(out.weights) == 5
        assert all(abs(w - 0.2) < 1e-9 for w in out.weights.values())
        assert out.total_weight == pytest.approx(1.0)

    def test_top_n_3_picks_highest_scores(self):
        out = equal_weight(_scores_sample(), top_n=3)
        # Top 3 score: bitcoin 85, gold 80, growth 75
        assert set(out.weights.keys()) == {"bitcoin", "gold", "us_equities_growth"}

    def test_empty_scores_returns_empty(self):
        out = equal_weight({}, top_n=5)
        assert out.weights == {}
        assert out.total_weight == 0.0

    def test_top_n_zero_returns_empty(self):
        out = equal_weight(_scores_sample(), top_n=0)
        assert out.weights == {}

    def test_top_n_exceeds_assets(self):
        out = equal_weight({"a": 10.0, "b": 20.0}, top_n=10)
        assert len(out.weights) == 2
        assert all(w == 0.5 for w in out.weights.values())


class TestScoreWeighted:
    def test_weight_proportional_to_score(self):
        scores = {"a": 80.0, "b": 40.0, "c": 20.0}
        out = score_weighted(scores, top_n=3)
        total = 80 + 40 + 20
        assert out.weights["a"] == pytest.approx(80 / total)
        assert out.weights["b"] == pytest.approx(40 / total)
        assert out.weights["c"] == pytest.approx(20 / total)

    def test_sum_to_one(self):
        out = score_weighted(_scores_sample(), top_n=5)
        assert sum(out.weights.values()) == pytest.approx(1.0)

    def test_all_zero_falls_back_to_equal(self):
        scores = {"a": 0.0, "b": 0.0}
        out = score_weighted(scores, top_n=2)
        # Fallback: equal_weight result
        assert all(w == 0.5 for w in out.weights.values())

    def test_higher_score_gets_higher_weight(self):
        scores = {"high": 90.0, "low": 30.0}
        out = score_weighted(scores, top_n=2)
        assert out.weights["high"] > out.weights["low"]


class TestRiskParity:
    def test_low_vol_gets_higher_weight(self):
        """Cash (vol 0.01) deve avere peso più alto di equities (vol 0.17)."""
        scores = {"cash": 50.0, "equity": 75.0}
        vols = {"cash": 0.01, "equity": 0.17}
        out = risk_parity(scores, vols, top_n=2)
        # Cash 1/0.01 = 100, equity 1/0.17 ≈ 5.88 → cash domina
        assert out.weights["cash"] > out.weights["equity"]

    def test_sum_to_one(self):
        out = risk_parity(_scores_sample(), _vols_sample(), top_n=5)
        assert sum(out.weights.values()) == pytest.approx(1.0)

    def test_missing_vol_uses_floor(self):
        """Asset senza vol nel dict usa min_vol floor."""
        scores = {"a": 70.0, "b": 60.0}
        vols = {"a": 0.10}  # missing b
        out = risk_parity(scores, vols, top_n=2, min_vol=0.01)
        # b avrà weight con vol=0.01 (floor) → 1/0.01=100 >> 1/0.10=10
        assert out.weights["b"] > out.weights["a"]

    def test_empty_returns_empty(self):
        out = risk_parity({}, {}, top_n=5)
        assert out.weights == {}


class TestKellyFractional:
    def test_higher_score_higher_weight(self):
        scores = {"high": 90.0, "low": 55.0}
        vols = {"high": 0.15, "low": 0.15}
        out = kelly_fractional(scores, vols, top_n=2, fraction=0.25)
        # Edge high = 0.4, low = 0.05 → high² >> low²
        assert out.weights["high"] > out.weights["low"]

    def test_all_below_baseline_falls_back_equal(self):
        """Tutti score < 50 (baseline) → no positive edge → fallback equal."""
        scores = {"a": 40.0, "b": 30.0}
        vols = {"a": 0.15, "b": 0.15}
        out = kelly_fractional(scores, vols, top_n=2)
        assert all(w == 0.5 for w in out.weights.values())

    def test_sum_to_one(self):
        out = kelly_fractional(_scores_sample(), _vols_sample(), top_n=5)
        assert sum(out.weights.values()) == pytest.approx(1.0)

    def test_lower_vol_higher_weight_at_same_score(self):
        scores = {"low_vol": 75.0, "high_vol": 75.0}
        vols = {"low_vol": 0.10, "high_vol": 0.30}
        out = kelly_fractional(scores, vols, top_n=2)
        # Stessa edge ma vol diversa → 1/vol² premia low_vol
        assert out.weights["low_vol"] > out.weights["high_vol"]

    def test_fraction_does_not_change_proportions(self):
        scores = {"a": 80.0, "b": 65.0}
        vols = {"a": 0.15, "b": 0.15}
        out_low = kelly_fractional(scores, vols, top_n=2, fraction=0.1)
        out_high = kelly_fractional(scores, vols, top_n=2, fraction=0.5)
        # Fraction è uniforme → proportional weights stessi (sum=1 entrambi)
        assert out_low.weights["a"] == pytest.approx(out_high.weights["a"])


class TestComputeAllocationDispatch:
    def test_equal_weight_dispatch(self):
        out = compute_allocation("equal_weight", _scores_sample(), top_n=5)
        assert out.method == "equal_weight"

    def test_score_weighted_dispatch(self):
        out = compute_allocation("score_weighted", _scores_sample(), top_n=5)
        assert out.method == "score_weighted"

    def test_risk_parity_requires_vols(self):
        with pytest.raises(ValueError, match="risk_parity richiede"):
            compute_allocation("risk_parity", _scores_sample(), top_n=5)

    def test_risk_parity_with_vols(self):
        out = compute_allocation(
            "risk_parity", _scores_sample(), vols=_vols_sample(), top_n=5
        )
        assert out.method == "risk_parity"

    def test_kelly_with_vols_and_fraction(self):
        out = compute_allocation(
            "kelly_fractional", _scores_sample(), vols=_vols_sample(),
            top_n=5, fraction=0.3,
        )
        assert out.method == "kelly_fractional"

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="non supportato"):
            compute_allocation("garbage_method", _scores_sample(), top_n=5)

    def test_case_insensitive(self):
        out = compute_allocation("EQUAL_WEIGHT", _scores_sample(), top_n=5)
        assert out.method == "equal_weight"


class TestListMethods:
    def test_lists_all_four_methods(self):
        out = list_methods()
        assert set(out) == {"equal_weight", "score_weighted", "risk_parity", "kelly_fractional"}


class TestInvariants:
    """Invarianti globali per qualsiasi algoritmo."""

    @pytest.fixture
    def scenarios(self):
        return [
            ("equal_weight", {"scores": _scores_sample(), "top_n": 5}),
            ("score_weighted", {"scores": _scores_sample(), "top_n": 5}),
            ("risk_parity", {"scores": _scores_sample(), "vols": _vols_sample(), "top_n": 5}),
            ("kelly_fractional", {"scores": _scores_sample(), "vols": _vols_sample(), "top_n": 5}),
        ]

    def test_all_weights_non_negative(self, scenarios):
        for method, kwargs in scenarios:
            out = compute_allocation(method, **kwargs)
            assert all(w >= 0 for w in out.weights.values()), \
                f"{method} ha pesi negativi"

    def test_sum_close_to_one(self, scenarios):
        for method, kwargs in scenarios:
            out = compute_allocation(method, **kwargs)
            assert sum(out.weights.values()) == pytest.approx(1.0, abs=1e-6), \
                f"{method} non somma a 1"

    def test_excluded_assets_disjoint_from_weights(self, scenarios):
        for method, kwargs in scenarios:
            out = compute_allocation(method, **kwargs)
            assert set(out.excluded_assets) & set(out.weights.keys()) == set()

    def test_weights_subset_of_input_assets(self, scenarios):
        for method, kwargs in scenarios:
            out = compute_allocation(method, **kwargs)
            assert set(out.weights.keys()).issubset(set(kwargs["scores"].keys()))


# ============================================================================
# Tier 6.10 — Vol targeting + leverage tests
# ============================================================================


from app.services.portfolio.position_sizing import (
    apply_vol_targeting,
    compute_portfolio_vol,
)


class TestComputePortfolioVol:
    def test_empty_weights_returns_zero(self):
        assert compute_portfolio_vol({}, {"a": 0.15}) == 0.0

    def test_empty_vols_returns_zero(self):
        assert compute_portfolio_vol({"a": 1.0}, {}) == 0.0

    def test_single_asset_returns_asset_vol(self):
        """1 asset 100% → portfolio vol = asset vol."""
        v = compute_portfolio_vol({"a": 1.0}, {"a": 0.20})
        assert v == pytest.approx(0.20)

    def test_zero_correlation_diversification(self):
        """2 asset equipesati 1:1, vols 0.20 ciascuno, corr=0:
        σ_p² = 0.5² · 0.20² + 0.5² · 0.20² = 0.02 → σ_p = 0.1414."""
        v = compute_portfolio_vol(
            {"a": 0.5, "b": 0.5}, {"a": 0.20, "b": 0.20}, avg_correlation=0.0
        )
        assert v == pytest.approx(0.1414, abs=0.001)

    def test_full_correlation_no_diversification(self):
        """corr=1 → σ_p = weighted avg σ_i (no diversification benefit)."""
        v = compute_portfolio_vol(
            {"a": 0.5, "b": 0.5}, {"a": 0.20, "b": 0.20}, avg_correlation=1.0
        )
        assert v == pytest.approx(0.20, abs=0.001)

    def test_partial_correlation_intermediate(self):
        """corr=0.3 → vol intermedio tra full diversification e full corr."""
        v_zero = compute_portfolio_vol({"a": 0.5, "b": 0.5}, {"a": 0.20, "b": 0.20}, 0.0)
        v_full = compute_portfolio_vol({"a": 0.5, "b": 0.5}, {"a": 0.20, "b": 0.20}, 1.0)
        v_03 = compute_portfolio_vol({"a": 0.5, "b": 0.5}, {"a": 0.20, "b": 0.20}, 0.3)
        assert v_zero < v_03 < v_full


class TestApplyVolTargeting:
    def _alloc_high_vol(self) -> PortfolioAllocation:
        """Allocation con vol portfolio alta (~20% pre-targeting)."""
        return PortfolioAllocation(
            method="equal_weight", top_n=2,
            weights={"equity": 0.5, "crypto": 0.5}, total_weight=1.0,
        )

    def _alloc_low_vol(self) -> PortfolioAllocation:
        """Allocation con vol portfolio bassa (~5% pre-targeting)."""
        return PortfolioAllocation(
            method="equal_weight", top_n=2,
            weights={"bonds": 0.5, "cash": 0.5}, total_weight=1.0,
        )

    def test_empty_allocation_returns_empty(self):
        alloc = PortfolioAllocation(method="equal_weight", top_n=0, weights={})
        out = apply_vol_targeting(alloc, vols={"a": 0.20}, target_vol=0.10)
        assert out.weights == {}
        assert "Empty" in out.description

    def test_high_vol_scales_down_with_cash_buffer(self):
        """Portfolio vol 18% > target 10% → scale 0.55 → cash buffer ~45%."""
        alloc = self._alloc_high_vol()
        vols = {"equity": 0.20, "crypto": 0.60}  # high vol portfolio
        out = apply_vol_targeting(alloc, vols, target_vol=0.10, avg_correlation=0.3)
        assert out.scale_factor < 1.0
        assert out.leverage_used == 1.0
        assert out.cash_buffer > 0
        assert sum(out.weights.values()) < 1.0
        # Realized vol post ≈ target
        assert out.realized_vol_post == pytest.approx(0.10, abs=0.01)

    def test_low_vol_scales_up_with_leverage(self):
        """Portfolio vol 3% < target 10% → scale ~3.3 → leverage used."""
        alloc = self._alloc_low_vol()
        vols = {"bonds": 0.05, "cash": 0.01}  # very low vol
        out = apply_vol_targeting(alloc, vols, target_vol=0.10, avg_correlation=0.3, max_leverage=4.0)
        assert out.scale_factor > 1.0
        assert out.leverage_used > 1.0
        assert out.cash_buffer == 0.0
        # Realized vol post ≈ target (cappato a max_leverage)
        assert out.realized_vol_post == pytest.approx(0.10, abs=0.01) or out.scale_factor == 4.0

    def test_max_leverage_cap_applied(self):
        """Even with very low realized vol, scale capped at max_leverage."""
        alloc = self._alloc_low_vol()
        vols = {"bonds": 0.001, "cash": 0.001}  # absurdly low
        out = apply_vol_targeting(alloc, vols, target_vol=0.10, max_leverage=2.0)
        assert out.scale_factor <= 2.0

    def test_no_vol_data_identity_scaling(self):
        """Vols dict vuoto → identity (scale=1, no change)."""
        alloc = self._alloc_high_vol()
        out = apply_vol_targeting(alloc, vols={}, target_vol=0.10)
        assert out.scale_factor == 1.0
        assert out.weights == alloc.weights

    def test_post_targeting_vol_close_to_target(self):
        """Sanity: realized_vol_post deve essere ≈ target_vol entro ±2%."""
        alloc = PortfolioAllocation(
            method="risk_parity", top_n=3,
            weights={"a": 0.4, "b": 0.4, "c": 0.2}, total_weight=1.0,
        )
        vols = {"a": 0.18, "b": 0.12, "c": 0.30}
        target = 0.10
        out = apply_vol_targeting(alloc, vols, target_vol=target, max_leverage=3.0)
        if out.scale_factor < 3.0:  # not capped
            assert abs(out.realized_vol_post - target) < 0.02

    def test_scale_factor_proportional_inverse_to_vol_ratio(self):
        """scale = target / realized_pre."""
        alloc = PortfolioAllocation(
            method="equal_weight", top_n=1, weights={"a": 1.0}, total_weight=1.0,
        )
        vols = {"a": 0.20}
        target = 0.10
        out = apply_vol_targeting(alloc, vols, target_vol=target)
        # Realized = 0.20, target = 0.10 → scale = 0.5
        assert out.scale_factor == pytest.approx(0.5, abs=0.01)
        assert out.cash_buffer == pytest.approx(0.5, abs=0.01)
