"""Test Tier 5.6 LOOP CLOSURE — Asset returns → regime Bayesian feedback.

Coverage:
- Flag OFF: identity (default).
- Flag ON: Bayesian shift verso regime coerente con returns osservati.
- Soft blend (alpha 0.05): max shift limitato.
- Invarianti: Σ=1, no negative probs, all keys preserved.
- Empty/zero asset_returns → no-op.
- Asset sconosciuti ignorati silenziosamente.
"""
from __future__ import annotations

import math

import pytest

from app.services.regime.asset_feedback import (
    _gaussian_log_pdf,
    _softmax_normalize,
    apply_asset_feedback,
    compute_likelihoods,
)


def _approx_sum_one(probs: dict, tol: float = 1e-6) -> bool:
    return abs(sum(probs.values()) - 1.0) < tol


class TestGaussianLogPdf:
    def test_peak_at_mean(self):
        """Log-pdf massima quando x = mu."""
        ll_peak = _gaussian_log_pdf(0.10, 0.10, 0.15)
        ll_off = _gaussian_log_pdf(-0.30, 0.10, 0.15)
        assert ll_peak > ll_off

    def test_floor_on_small_sigma(self):
        """Sigma molto piccolo non deve causare crash o overflow assurdo."""
        ll = _gaussian_log_pdf(0.10, 0.10, 0.001)
        assert isinstance(ll, float)
        assert math.isfinite(ll)


class TestComputeLikelihoods:
    def test_all_keys_returned(self):
        ll = compute_likelihoods({"gold": 0.15})
        assert set(ll.keys()) == {"reflation", "stagflation", "deflation", "goldilocks"}

    def test_unknown_asset_ignored(self):
        ll = compute_likelihoods({"unknown_asset_xyz": 0.50})
        # Nessun asset noto → log_lik tutti zero
        assert all(v == 0.0 for v in ll.values())

    def test_gold_high_return_favors_stagflation(self):
        """Gold +30% YoY → log_lik(stagflation) > log_lik(goldilocks)."""
        ll = compute_likelihoods({"gold": 0.30})
        assert ll["stagflation"] > ll["goldilocks"]

    def test_bonds_long_negative_favors_stagflation_over_deflation(self):
        """Bonds long -20% → favorisce stagflation (peggior asset stag) vs
        deflation (dove invece bonds long performano bene)."""
        ll = compute_likelihoods({"us_bonds_long": -0.20})
        assert ll["stagflation"] > ll["deflation"]

    def test_growth_high_return_favors_reflation_or_goldilocks(self):
        """Equities growth +20% → favorisce reflation/goldilocks."""
        ll = compute_likelihoods({"us_equities_growth": 0.20})
        assert max(ll["reflation"], ll["goldilocks"]) > max(ll["stagflation"], ll["deflation"])


class TestSoftmaxNormalize:
    def test_uniform_input(self):
        out = _softmax_normalize({"a": 0.0, "b": 0.0, "c": 0.0})
        assert out["a"] == pytest.approx(1/3)

    def test_sum_to_one(self):
        out = _softmax_normalize({"a": 1.0, "b": 2.0, "c": 3.0})
        assert sum(out.values()) == pytest.approx(1.0, abs=1e-6)

    def test_numerical_stability_large_logs(self):
        """Subtraction del max evita overflow exp()."""
        out = _softmax_normalize({"a": 1000.0, "b": 999.0, "c": 998.0})
        assert sum(out.values()) == pytest.approx(1.0, abs=1e-6)
        assert out["a"] > out["b"] > out["c"]


class TestApplyAssetFeedbackFlagGating:
    def test_flag_off_is_identity(self, monkeypatch):
        monkeypatch.delenv("USE_ASSET_FEEDBACK", raising=False)
        probs = {"reflation": 0.4, "stagflation": 0.2, "deflation": 0.2, "goldilocks": 0.2}
        returns = {"gold": 0.30, "us_bonds_long": -0.20}
        out = apply_asset_feedback(probs, returns)
        assert out == probs

    def test_flag_on_empty_returns_is_identity(self, monkeypatch):
        monkeypatch.setenv("USE_ASSET_FEEDBACK", "1")
        probs = {"reflation": 0.4, "stagflation": 0.2, "deflation": 0.2, "goldilocks": 0.2}
        out = apply_asset_feedback(probs, {})
        assert out == probs

    def test_flag_on_only_unknown_assets_is_identity(self, monkeypatch):
        monkeypatch.setenv("USE_ASSET_FEEDBACK", "1")
        probs = {"reflation": 0.4, "stagflation": 0.2, "deflation": 0.2, "goldilocks": 0.2}
        out = apply_asset_feedback(probs, {"unknown_xyz": 0.5})
        # Nessun asset noto → log_lik tutti zero → no-op
        for k in probs:
            assert abs(out[k] - probs[k]) < 1e-9

    def test_alpha_zero_is_identity(self, monkeypatch):
        monkeypatch.setenv("USE_ASSET_FEEDBACK", "1")
        probs = {"reflation": 0.4, "stagflation": 0.2, "deflation": 0.2, "goldilocks": 0.2}
        out = apply_asset_feedback(probs, {"gold": 0.30}, alpha=0.0)
        assert out == probs


class TestApplyAssetFeedbackScenarios:
    def test_stagflation_returns_boost_stagflation(self, monkeypatch):
        """Gold +30% + bonds long -20% + energy +25% → boost stagflation."""
        monkeypatch.setenv("USE_ASSET_FEEDBACK", "1")
        probs = {"reflation": 0.25, "stagflation": 0.25, "deflation": 0.25, "goldilocks": 0.25}
        returns = {
            "gold": 0.30,
            "us_bonds_long": -0.20,
            "energy": 0.40,
            "broad_commodities": 0.25,
        }
        out = apply_asset_feedback(probs, returns)
        assert out["stagflation"] > probs["stagflation"]
        assert _approx_sum_one(out)

    def test_deflation_returns_boost_deflation(self, monkeypatch):
        """Bonds long +25% + equities growth -30% + commodities -20% → defl."""
        monkeypatch.setenv("USE_ASSET_FEEDBACK", "1")
        probs = {"reflation": 0.25, "stagflation": 0.25, "deflation": 0.25, "goldilocks": 0.25}
        returns = {
            "us_bonds_long": 0.20,
            "us_equities_growth": -0.30,
            "us_equities_value": -0.25,
            "broad_commodities": -0.20,
        }
        out = apply_asset_feedback(probs, returns)
        assert out["deflation"] > probs["deflation"]
        assert _approx_sum_one(out)

    def test_reflation_returns_boost_reflation_or_goldilocks(self, monkeypatch):
        """Equities growth +20% + bonds long flat → reflation/goldilocks."""
        monkeypatch.setenv("USE_ASSET_FEEDBACK", "1")
        probs = {"reflation": 0.25, "stagflation": 0.25, "deflation": 0.25, "goldilocks": 0.25}
        returns = {
            "us_equities_growth": 0.20,
            "us_equities_value": 0.15,
            "us_bonds_long": -0.02,
        }
        out = apply_asset_feedback(probs, returns)
        boost = (out["reflation"] - probs["reflation"]) + (out["goldilocks"] - probs["goldilocks"])
        assert boost > 0


class TestApplyAssetFeedbackInvariants:
    def test_probs_sum_to_one(self, monkeypatch):
        monkeypatch.setenv("USE_ASSET_FEEDBACK", "1")
        probs = {"reflation": 0.4, "stagflation": 0.3, "deflation": 0.2, "goldilocks": 0.1}
        out = apply_asset_feedback(probs, {"gold": 0.40, "us_bonds_long": -0.30})
        assert _approx_sum_one(out)

    def test_no_negative_probs(self, monkeypatch):
        monkeypatch.setenv("USE_ASSET_FEEDBACK", "1")
        probs = {"reflation": 0.95, "stagflation": 0.02, "deflation": 0.02, "goldilocks": 0.01}
        out = apply_asset_feedback(probs, {"gold": 0.50, "us_bonds_long": -0.40})
        for k, v in out.items():
            assert v >= 0.0

    def test_all_keys_preserved(self, monkeypatch):
        monkeypatch.setenv("USE_ASSET_FEEDBACK", "1")
        probs = {"reflation": 0.25, "stagflation": 0.25, "deflation": 0.25, "goldilocks": 0.25}
        out = apply_asset_feedback(probs, {"gold": 0.30})
        assert set(out.keys()) == set(probs.keys())

    def test_alpha_caps_shift_under_5pp(self, monkeypatch):
        """Soft blend con alpha=0.05 → max shift <5pp per regime singolo."""
        monkeypatch.setenv("USE_ASSET_FEEDBACK", "1")
        probs = {"reflation": 0.25, "stagflation": 0.25, "deflation": 0.25, "goldilocks": 0.25}
        # Scenario estremo: tutti gli asset urlano "stagflation"
        returns = {
            "gold": 0.50, "us_bonds_long": -0.40, "energy": 0.60,
            "broad_commodities": 0.40, "us_equities_growth": -0.30,
            "us_equities_value": -0.20,
        }
        out = apply_asset_feedback(probs, returns, alpha=0.05)
        for r in probs:
            assert abs(out[r] - probs[r]) < 0.05, f"{r}: shift {abs(out[r]-probs[r]):.4f} > 5pp"
