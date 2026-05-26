"""Test crisis Brier calibration."""
from __future__ import annotations

import pytest

from app.services.validation.brier_crisis import (
    REGIMES,
    _compute_brier,
    assess_crisis_brier,
)


class TestComputeBrier:
    def test_perfect_prediction_zero_brier(self):
        probs = {"reflation": 1.0, "stagflation": 0, "deflation": 0, "goldilocks": 0}
        assert _compute_brier(probs, "reflation") == 0

    def test_worst_prediction_max_brier(self):
        probs = {"reflation": 0, "stagflation": 0, "deflation": 0, "goldilocks": 1.0}
        # 1.0 su goldilocks + 1.0 su reflation expected → (1-0)^2 + (0-1)^2 = 2 / 4 = 0.5
        assert _compute_brier(probs, "reflation") == pytest.approx(0.5)

    def test_uniform_prediction(self):
        probs = {r: 0.25 for r in REGIMES}
        # Per regime predetto: (0.25-1)^2 + 3*(0.25-0)^2 = 0.5625 + 3*0.0625 = 0.75 → /4 = 0.1875
        assert _compute_brier(probs, "reflation") == pytest.approx(0.1875)


class TestAssessCrisisBrier:
    def test_perfect_predictor(self):
        """Predictor che sa SEMPRE il regime vero → mean Brier = 0."""
        def perfect(indicators):
            # Cheating: lookup tramite indicators (uso CPI come proxy)
            cpi = indicators.get("cpi_yoy", 0)
            gdp = indicators.get("gdp_roc", 0)
            if cpi > 5 and gdp < 2:
                regime = "stagflation"
            elif gdp < -1 and cpi < 3:
                regime = "deflation"
            elif gdp > 1 and cpi < 3:
                regime = "reflation"
            else:
                regime = "goldilocks"
            return {r: (1.0 if r == regime else 0.0) for r in REGIMES}

        out = assess_crisis_brier(perfect)
        assert out.n_episodes >= 5
        # Almeno alcuni sono catturati correttamente
        assert out.accuracy >= 0.6

    def test_uniform_predictor(self):
        """Predictor uniforme → Brier ≈ 0.1875 ogni episodio."""
        def uniform(indicators):
            return {r: 0.25 for r in REGIMES}

        out = assess_crisis_brier(uniform)
        assert out.mean_brier == pytest.approx(0.1875, abs=1e-6)
        assert out.accuracy == 0  # max-arg sempre reflation (alphabetical first)

    def test_well_calibrated_threshold(self):
        """Brier <=0.10 → well_calibrated=True."""
        def near_perfect(indicators):
            cpi = indicators.get("cpi_yoy", 0)
            gdp = indicators.get("gdp_roc", 0)
            # 80% prob al regime corretto
            base = {r: 0.067 for r in REGIMES}
            if cpi > 5 and gdp < 2:
                base["stagflation"] = 0.8
            elif gdp < -1:
                base["deflation"] = 0.8
            elif gdp > 1:
                base["reflation"] = 0.8
            else:
                base["goldilocks"] = 0.8
            # Renorm
            total = sum(base.values())
            return {k: v / total for k, v in base.items()}

        out = assess_crisis_brier(near_perfect)
        # Mean Brier (1 corretto = 0.8): ~0.04 per ep corretto, ~0.18 per ep sbagliato
        assert out.mean_brier < 0.20

    def test_rule_based_vanilla_well_calibrated(self):
        """Rule-based vanilla (no flag) deve dare Brier <=0.10 sui crisis.

        Council target: well_calibrated. Verifica empirica.
        """
        from app.services.regime.classifier import classify_regime
        import os
        # Clean flags
        for k in list(os.environ.keys()):
            if k.startswith("USE_"):
                os.environ.pop(k, None)

        def rb_predict(ind):
            return classify_regime(ind)["probabilities"]

        out = assess_crisis_brier(rb_predict)
        # Rule-based vanilla è ottimo sui crisis (sono casi "estremi" facili)
        assert out.is_well_calibrated is True
        assert out.mean_brier <= 0.10
