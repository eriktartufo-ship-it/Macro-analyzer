"""Test meta-regime classifier: Brier-weighted blend of multi-model predictions.

Tier 1.1 della roadmap Bridgewater. Il meta-classifier combina rule-based +
HMM-Market + MS-VAR (+ in futuro FOMC nudge + term-premium signal) in una
probability finale, con pesi inversamente proporzionali al Brier score
storico di ogni modello.

Goal: trasformare il regime ufficiale del sistema da "single rule-based output"
a "ensemble Brier-weighted" load-bearing per tutti i moduli.
"""
from __future__ import annotations

import pytest

from app.services.regime.classifier import REGIMES
from app.services.regime.meta_classifier import (
    blend_probabilities,
    brier_score,
    compute_brier_weights,
)


class TestBrierScore:
    """Brier = mean squared error tra prob predetta e indicator label vero.
    Per K classi: BS = (1/N) Σ_n Σ_k (p_nk - y_nk)^2. Range [0, 2]."""

    def test_perfect_prediction_zero(self):
        """P(true_class)=1, altri=0 → Brier 0."""
        predictions = [{"reflation": 1.0, "stagflation": 0.0, "deflation": 0.0, "goldilocks": 0.0}]
        labels = ["reflation"]
        assert brier_score(predictions, labels) == pytest.approx(0.0, abs=1e-9)

    def test_opposite_prediction_two(self):
        """P(true)=0, P(false)=1 → Brier = 2 (max teorico)."""
        predictions = [{"reflation": 0.0, "stagflation": 1.0, "deflation": 0.0, "goldilocks": 0.0}]
        labels = ["reflation"]
        # (0-1)^2 + (1-0)^2 + 0 + 0 = 2
        assert brier_score(predictions, labels) == pytest.approx(2.0, abs=1e-9)

    def test_uniform_prediction(self):
        """P uniforme 0.25 ognuno → BS = (1-0.25)^2 + 3·(0.25)^2 = 0.5625 + 0.1875 = 0.75."""
        predictions = [{r: 0.25 for r in REGIMES}]
        labels = ["reflation"]
        assert brier_score(predictions, labels) == pytest.approx(0.75, abs=1e-9)

    def test_average_over_multiple_observations(self):
        """Brier su N obs = media singoli."""
        predictions = [
            {r: 0.25 for r in REGIMES},   # uniforme: BS=0.75
            {"reflation": 1.0, "stagflation": 0.0, "deflation": 0.0, "goldilocks": 0.0},  # perfetta: 0
        ]
        labels = ["reflation", "reflation"]
        assert brier_score(predictions, labels) == pytest.approx(0.375, abs=1e-9)

    def test_unknown_label_skipped(self):
        """Label non in REGIMES viene ignorato (no crash)."""
        predictions = [{r: 0.25 for r in REGIMES}]
        labels = ["bogus"]
        # Tutte le obs invalide → fallback Brier =0 o NaN-safe (assumiamo 0)
        result = brier_score(predictions, labels)
        assert result >= 0.0

    def test_empty_returns_zero(self):
        assert brier_score([], []) == 0.0


class TestComputeBrierWeights:
    """Pesi inversamente proporzionali al Brier. Modello migliore (Brier basso)
    pesa di più. Floor per evitare modelli con peso quasi-zero."""

    def test_perfect_model_dominates(self):
        """Modello con Brier 0 vs altri con Brier > 0 → peso quasi-1 sul perfetto."""
        briers = {"perfect": 0.0, "uniform": 0.75}
        weights = compute_brier_weights(briers, min_weight=0.05)
        assert weights["perfect"] > 0.85
        assert weights["uniform"] >= 0.05  # floor
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_equal_briers_equal_weights(self):
        """Modelli con stesso Brier → pesi uguali."""
        briers = {"a": 0.5, "b": 0.5, "c": 0.5}
        weights = compute_brier_weights(briers)
        for w in weights.values():
            assert w == pytest.approx(1 / 3, abs=1e-9)

    def test_weights_sum_to_one(self):
        briers = {"a": 0.1, "b": 0.5, "c": 0.9}
        weights = compute_brier_weights(briers)
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_min_weight_floor(self):
        """Modello molto peggio degli altri non scende sotto min_weight."""
        briers = {"a": 0.001, "b": 1.5}
        weights = compute_brier_weights(briers, min_weight=0.10)
        assert weights["b"] >= 0.10

    def test_empty_returns_empty(self):
        assert compute_brier_weights({}) == {}

    def test_single_model_full_weight(self):
        weights = compute_brier_weights({"only": 0.3})
        assert weights["only"] == pytest.approx(1.0)


class TestBlendProbabilities:
    """Output = Σ_i w_i · p_i(regime). Normalizzato Σ=1."""

    def test_blend_two_equal_weights(self):
        """w=(0.5, 0.5) → media aritmetica delle probs."""
        model_probs = {
            "a": {"reflation": 0.6, "stagflation": 0.2, "deflation": 0.1, "goldilocks": 0.1},
            "b": {"reflation": 0.2, "stagflation": 0.6, "deflation": 0.1, "goldilocks": 0.1},
        }
        weights = {"a": 0.5, "b": 0.5}
        out = blend_probabilities(model_probs, weights)
        assert out["reflation"] == pytest.approx(0.4)
        assert out["stagflation"] == pytest.approx(0.4)
        assert out["deflation"] == pytest.approx(0.1)
        assert out["goldilocks"] == pytest.approx(0.1)
        assert sum(out.values()) == pytest.approx(1.0)

    def test_blend_dominant_weight(self):
        """w=(0.9, 0.1) → output vicino al modello dominante."""
        model_probs = {
            "dominant": {"reflation": 1.0, "stagflation": 0.0, "deflation": 0.0, "goldilocks": 0.0},
            "minor": {"reflation": 0.0, "stagflation": 1.0, "deflation": 0.0, "goldilocks": 0.0},
        }
        weights = {"dominant": 0.9, "minor": 0.1}
        out = blend_probabilities(model_probs, weights)
        assert out["reflation"] == pytest.approx(0.9)
        assert out["stagflation"] == pytest.approx(0.1)

    def test_blend_drops_models_not_in_weights(self):
        """Modello senza peso non contribuisce."""
        model_probs = {
            "a": {"reflation": 0.5, "stagflation": 0.5, "deflation": 0.0, "goldilocks": 0.0},
            "stale": {"reflation": 0.0, "stagflation": 0.0, "deflation": 1.0, "goldilocks": 0.0},
        }
        weights = {"a": 1.0}  # solo "a"
        out = blend_probabilities(model_probs, weights)
        assert out["reflation"] == pytest.approx(0.5)
        assert out["deflation"] == pytest.approx(0.0)

    def test_blend_renormalizes_after_floor(self):
        """Anche con probabilità che non sommano esattamente a 1 (per floor noise),
        l'output finale somma a 1."""
        model_probs = {
            "a": {"reflation": 0.7, "stagflation": 0.2, "deflation": 0.05, "goldilocks": 0.05},
        }
        weights = {"a": 1.0}
        out = blend_probabilities(model_probs, weights)
        assert sum(out.values()) == pytest.approx(1.0, abs=1e-9)

    def test_blend_empty_weights_uniform_fallback(self):
        """Nessun modello valido → distribuzione uniforme."""
        out = blend_probabilities({}, {})
        assert sum(out.values()) == pytest.approx(1.0)
        for r in REGIMES:
            assert out[r] == pytest.approx(0.25)

    def test_blend_nonzero_floor_after_blend(self):
        """Anche se tutti i modelli danno 0 a un regime, output finale ha
        almeno un piccolo floor (evita Brier saturation futuro)."""
        model_probs = {
            "a": {"reflation": 1.0, "stagflation": 0.0, "deflation": 0.0, "goldilocks": 0.0},
            "b": {"reflation": 1.0, "stagflation": 0.0, "deflation": 0.0, "goldilocks": 0.0},
        }
        weights = {"a": 0.5, "b": 0.5}
        out = blend_probabilities(model_probs, weights, prob_floor=0.01)
        # Tutti i regimi (anche con prob 0 in input) hanno floor
        for r in REGIMES:
            assert out[r] >= 0.01 - 1e-9
        assert sum(out.values()) == pytest.approx(1.0, abs=1e-9)
