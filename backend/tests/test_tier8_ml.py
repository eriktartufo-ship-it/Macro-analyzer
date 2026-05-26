"""Test Tier 8 — ML regime classifier + ensemble blend + forecast.

Coverage:
- T8.1 Feature extraction: shape, NaN safety, derived features.
- T8.2 ML classifier: training, prediction, classes, LOO CV.
- T8.3 Ensemble blend: rule + ML, alpha weighting, 2001 dotcom correttamente catturato.
- T8.4 Forecast horizons: transition matrix, propagation, asset allocation.
"""
from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.services.regime.historical_episodes import (
    HISTORICAL_EPISODES,
    get_episode_by_name,
    regime_distribution,
)
from app.services.regime.ml_classifier import (
    REGIMES,
    extract_features,
    loo_cross_validation,
    predict_regime,
    train_classifier,
)
from app.services.regime.ml_ensemble import (
    blend_with_rule_based,
    reset_cache,
)
from app.services.regime.ml_forecast import (
    estimate_transition_matrix,
    forecast_horizons,
    shannon_entropy,
)


class TestHistoricalEpisodes:
    def test_dataset_size_minimum(self):
        assert len(HISTORICAL_EPISODES) >= 18

    def test_all_regimes_represented(self):
        dist = regime_distribution()
        assert set(dist.keys()) >= {"reflation", "stagflation", "deflation", "goldilocks"}
        for regime, count in dist.items():
            assert count >= 2, f"{regime} has only {count} episodes"

    def test_get_by_name_lookup(self):
        ep = get_episode_by_name("2008")
        assert ep is not None
        assert "GFC" in ep.name
        assert ep.true_regime == "deflation"

    def test_lookup_missing_returns_none(self):
        assert get_episode_by_name("nonexistent_year") is None


class TestFeatureExtraction:
    def test_feature_vector_shape(self):
        ep = HISTORICAL_EPISODES[0]
        feat, names = extract_features(ep.indicators)
        assert feat.ndim == 1
        assert len(feat) == len(names)
        assert len(feat) >= 29  # 19 raw + 10 z + derived

    def test_nan_safe(self):
        feat, _ = extract_features({"gdp_roc": float("nan"), "cpi_yoy": 2.0})
        assert not any(math.isnan(v) for v in feat)

    def test_inf_safe(self):
        feat, _ = extract_features({"gdp_roc": float("inf")})
        assert not any(math.isinf(v) for v in feat)

    def test_missing_indicators_use_anchor_mean(self):
        feat, names = extract_features({})
        # Tutti i feature finite
        assert all(math.isfinite(v) for v in feat)

    def test_derived_features_present(self):
        _, names = extract_features({})
        for name in (
            "gdp_minus_cpi", "real_fed_funds", "growth_z", "inflation_z",
            "stress_z", "is_recession_signal", "is_inflation_high",
        ):
            assert name in names


class TestMLClassifier:
    def test_train_produces_bundle(self):
        bundle = train_classifier()
        assert bundle.rf_model is not None
        assert bundle.gb_model is not None
        assert len(bundle.classes) >= 4

    def test_predict_returns_full_distribution(self):
        bundle = train_classifier()
        ep = HISTORICAL_EPISODES[0]
        probs = predict_regime(ep.indicators, bundle)
        for r in REGIMES:
            assert r in probs
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_2008_gfc_predicted_deflation(self):
        bundle = train_classifier()
        ep = get_episode_by_name("GFC")
        probs = predict_regime(ep.indicators, bundle)
        # Deflation deve essere il regime più probabile per 2008 GFC
        assert max(probs, key=probs.get) == "deflation"
        assert probs["deflation"] > 0.5

    def test_1973_opec_predicted_stagflation(self):
        bundle = train_classifier()
        ep = get_episode_by_name("OPEC")
        probs = predict_regime(ep.indicators, bundle)
        assert max(probs, key=probs.get) == "stagflation"

    def test_loo_cv_accuracy_above_threshold(self):
        """Soglia minima accettabile: 65% accuracy (era 0% rule-based su 2001)."""
        res = loo_cross_validation()
        assert res["accuracy"] >= 0.65, f"LOO accuracy {res['accuracy']} below threshold"
        assert res["brier_score"] < 0.20, f"Brier {res['brier_score']} too high"

    def test_loo_cv_includes_all_episodes(self):
        res = loo_cross_validation()
        assert len(res["per_episode"]) == len(HISTORICAL_EPISODES)


class TestEnsembleBlend:
    def setup_method(self):
        reset_cache()

    def test_blend_returns_full_distribution(self):
        ep = HISTORICAL_EPISODES[0]
        rb_probs = {"reflation": 0.25, "stagflation": 0.25, "deflation": 0.25, "goldilocks": 0.25}
        blended, ml_only = blend_with_rule_based(rb_probs, ep.indicators)
        for r in REGIMES:
            assert r in blended
        assert abs(sum(blended.values()) - 1.0) < 1e-6

    def test_alpha_zero_returns_rule_based(self):
        ep = HISTORICAL_EPISODES[0]
        rb_probs = {"reflation": 0.7, "stagflation": 0.1, "deflation": 0.1, "goldilocks": 0.1}
        blended, _ = blend_with_rule_based(rb_probs, ep.indicators, alpha=0.0)
        # Alpha=0 → blended = rule_based renormalized
        assert abs(blended["reflation"] - 0.7) < 1e-6

    def test_alpha_one_returns_ml_only(self):
        ep = get_episode_by_name("GFC")
        rb_probs = {"reflation": 1.0, "stagflation": 0.0, "deflation": 0.0, "goldilocks": 0.0}
        blended, ml_only = blend_with_rule_based(rb_probs, ep.indicators, alpha=1.0)
        # Alpha=1 → solo ML, deve predire deflation per 2008 GFC anche se rb=reflation
        assert blended["deflation"] > 0.4

    def test_2001_dotcom_blend_catches_deflation(self):
        """Test critico T8.3: ensemble deve catturare 2001 dotcom come deflation."""
        ep = get_episode_by_name("dotcom")
        rb_probs = {"reflation": 0.26, "stagflation": 0.31, "deflation": 0.23, "goldilocks": 0.20}
        blended, _ = blend_with_rule_based(rb_probs, ep.indicators, alpha=0.4)
        # Con blend, deflation deve essere il top
        top = max(blended, key=blended.get)
        assert top == "deflation"


class TestForecast:
    def test_shannon_entropy_uniform_max(self):
        uniform = {"reflation": 0.25, "stagflation": 0.25, "deflation": 0.25, "goldilocks": 0.25}
        ent = shannon_entropy(uniform)
        assert abs(ent - 2.0) < 0.01

    def test_shannon_entropy_concentrated_low(self):
        concentrated = {"reflation": 0.97, "stagflation": 0.01, "deflation": 0.01, "goldilocks": 0.01}
        ent = shannon_entropy(concentrated)
        assert ent < 0.5

    def test_estimate_transition_matrix_fallback_uniform(self):
        """DB con pochi records → matrix uniform con self-persistence."""
        db = MagicMock()
        db.query.return_value.order_by.return_value.all.return_value = []
        T = estimate_transition_matrix(db, min_records=50)
        assert T.shape == (4, 4)
        # Diagonale = 0.7 (self-persistence), off-diag = 0.1
        for i in range(4):
            assert T[i][i] == pytest.approx(0.7, abs=0.05)

    def test_estimate_transition_matrix_rows_sum_one(self):
        """Ogni riga di T deve sommare a 1."""
        db = MagicMock()
        db.query.return_value.order_by.return_value.all.return_value = []
        T = estimate_transition_matrix(db)
        for i in range(4):
            assert abs(T[i].sum() - 1.0) < 1e-6

    def test_forecast_horizons_propagation(self):
        T = np.array([
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
            [0.1, 0.1, 0.1, 0.7],
        ])
        current = {"reflation": 1.0, "stagflation": 0.0, "deflation": 0.0, "goldilocks": 0.0}
        forecasts = forecast_horizons(current, T, horizons=(1, 3, 6))
        assert len(forecasts) == 3
        # Horizon 1: prob refl = 0.7
        f1 = forecasts[0]
        assert f1.horizon_months == 1
        assert f1.probs["reflation"] == pytest.approx(0.7, abs=0.001)
        # Horizon 6: prob refl diluita
        f6 = forecasts[2]
        assert f6.probs["reflation"] < f1.probs["reflation"]

    def test_forecast_long_horizon_converges_to_steady_state(self):
        """h grande → distribuzione tende a steady-state."""
        T = np.array([
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
            [0.1, 0.1, 0.1, 0.7],
        ])
        current = {"reflation": 1.0, "stagflation": 0.0, "deflation": 0.0, "goldilocks": 0.0}
        forecasts = forecast_horizons(current, T, horizons=(24,))
        f24 = forecasts[0]
        # Entropy deve essere alta (verso uniforme)
        assert f24.entropy > 1.5

    def test_forecast_degenerate_flag_long_horizon(self):
        """h enorme → distribuzione uniforme → is_degenerate=True."""
        T = np.full((4, 4), 0.25)  # già uniforme
        current = {"reflation": 1.0, "stagflation": 0.0, "deflation": 0.0, "goldilocks": 0.0}
        forecasts = forecast_horizons(current, T, horizons=(60,))
        f60 = forecasts[0]
        assert f60.is_degenerate is True
        assert f60.expected_regime == "uniform"
        assert f60.entropy > 1.95
