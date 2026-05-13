"""Test per il Regime Trajectory Analyzer, con focus sui segnali forward-looking."""

from app.services.regime.classifier import REGIMES
from app.services.regime.trajectory import (
    _detect_indicator_trends,
    calculate_trajectory,
)


def _base_probs(regime: str = "goldilocks") -> dict[str, float]:
    """Probabilità quasi-uniformi con lieve bias sul regime passato."""
    probs = {r: 0.2 for r in REGIMES}
    probs[regime] = 0.4
    # normalizza
    total = sum(probs.values())
    return {r: p / total for r, p in probs.items()}


class TestForwardLookingDetection:
    """Verifica che i nuovi trend forward-looking siano rilevati correttamente."""

    def test_breakeven_rising_detected(self):
        trends = _detect_indicator_trends({"breakeven_10y_change_3m": 0.35})
        assert "breakeven_rising" in trends

    def test_breakeven_falling_detected(self):
        trends = _detect_indicator_trends({"breakeven_10y_change_3m": -0.3})
        assert "breakeven_falling" in trends

    def test_breakeven_stable_no_trend(self):
        trends = _detect_indicator_trends({"breakeven_10y_change_3m": 0.05})
        assert "breakeven_rising" not in trends
        assert "breakeven_falling" not in trends

    def test_vix_spike_via_ratio(self):
        trends = _detect_indicator_trends({"vix_ma_ratio": 1.5, "vix": 22})
        assert "vix_spiking" in trends

    def test_vix_spike_via_absolute_level(self):
        trends = _detect_indicator_trends({"vix_ma_ratio": 1.05, "vix": 30})
        assert "vix_spiking" in trends

    def test_vix_compressed(self):
        trends = _detect_indicator_trends({"vix_ma_ratio": 0.75, "vix": 12})
        assert "vix_compressed" in trends

    def test_vix_normal_no_signal(self):
        trends = _detect_indicator_trends({"vix_ma_ratio": 1.0, "vix": 17})
        assert "vix_spiking" not in trends
        assert "vix_compressed" not in trends

    def test_nfci_tightening(self):
        trends = _detect_indicator_trends({"nfci_change_3m": 0.25})
        assert "nfci_tightening" in trends

    def test_nfci_easing(self):
        trends = _detect_indicator_trends({"nfci_change_3m": -0.30})
        assert "nfci_easing" in trends


class TestTrajectoryPressures:
    """Verifica che i nuovi trend influenzino le probabilità proiettate."""

    def test_breakeven_rising_boosts_stagflation(self):
        probs = _base_probs("goldilocks")
        result = calculate_trajectory(
            current_probabilities=probs,
            indicators={"breakeven_10y_change_3m": 0.35, "cpi_yoy": 2.5, "gdp_roc": 2.0},
        )
        # Stagflation deve crescere rispetto al baseline
        assert result["projected_probabilities"]["stagflation"] > probs["stagflation"]

    def test_vix_spike_boosts_deflation(self):
        probs = _base_probs("reflation")
        result = calculate_trajectory(
            current_probabilities=probs,
            indicators={"vix_ma_ratio": 1.6, "vix": 35, "cpi_yoy": 2.5, "gdp_roc": 2.0},
        )
        assert result["projected_probabilities"]["deflation"] > probs["deflation"]

    def test_nfci_easing_boosts_reflation(self):
        probs = _base_probs("goldilocks")
        result = calculate_trajectory(
            current_probabilities=probs,
            indicators={"nfci_change_3m": -0.3, "cpi_yoy": 2.5, "gdp_roc": 2.0},
        )
        assert result["projected_probabilities"]["reflation"] > probs["reflation"]

    def test_forces_include_forward_looking(self):
        probs = _base_probs("goldilocks")
        result = calculate_trajectory(
            current_probabilities=probs,
            indicators={
                "vix_ma_ratio": 1.6,
                "vix": 32,
                "breakeven_10y_change_3m": -0.3,
                "nfci_change_3m": 0.25,
                "cpi_yoy": 2.5,
                "gdp_roc": 2.0,
            },
        )
        names = [f["name"] for f in result["forces"]]
        assert "vix_spiking" in names
        assert "breakeven_falling" in names
        assert "nfci_tightening" in names

    def test_no_forward_indicators_still_works(self):
        probs = _base_probs("goldilocks")
        result = calculate_trajectory(
            current_probabilities=probs,
            indicators={"cpi_yoy": 2.5, "gdp_roc": 2.0},
        )
        # Deve comunque produrre un output valido
        assert "projected_regime" in result
        assert sum(result["projected_probabilities"].values()) > 0.99


class TestInsiderTrajectoryForce:
    """Tier 5.5 LOOP CLOSURE — Insider score come trajectory force.

    Insider è leading indicator (smart money 6-12m ahead). Threshold
    |score|>0.3 = signal forte → boost reflation/goldilocks proiettati
    se positive, deflation/stagflation se negative.
    """

    def _base_neutral_probs(self) -> dict:
        return {
            "reflation": 0.25,
            "goldilocks": 0.25,
            "deflation": 0.25,
            "stagflation": 0.25,
        }

    def _neutral_indicators(self) -> dict:
        return {"cpi_yoy": 2.0, "gdp_roc": 2.0}

    def test_no_insider_score_no_force(self):
        """Senza insider_score nel dict, nessuna force `insider_*` deve apparire."""
        result = calculate_trajectory(
            current_probabilities=self._base_neutral_probs(),
            indicators=self._neutral_indicators(),
        )
        names = [f["name"] for f in result["forces"]]
        assert not any(n.startswith("insider_") for n in names)

    def test_insider_score_neutral_no_force(self):
        """insider_score=0.0 (default) o ±0.2 (sotto threshold) → no force."""
        for score in [0.0, 0.2, -0.2, 0.3, -0.3]:
            result = calculate_trajectory(
                current_probabilities=self._base_neutral_probs(),
                indicators={**self._neutral_indicators(), "insider_score": score},
            )
            names = [f["name"] for f in result["forces"]]
            assert not any(n.startswith("insider_") for n in names), \
                f"score {score} non doveva produrre force"

    def test_insider_strong_buying_boosts_reflation(self):
        """insider_score=+0.6 → reflation projected > baseline."""
        probs = self._base_neutral_probs()
        baseline = calculate_trajectory(
            current_probabilities=probs,
            indicators=self._neutral_indicators(),
        )
        bullish = calculate_trajectory(
            current_probabilities=probs,
            indicators={**self._neutral_indicators(), "insider_score": 0.6},
        )
        # Reflation/goldilocks insieme devono crescere
        delta_refl = bullish["projected_probabilities"]["reflation"] - baseline["projected_probabilities"]["reflation"]
        delta_gold = bullish["projected_probabilities"]["goldilocks"] - baseline["projected_probabilities"]["goldilocks"]
        assert delta_refl + delta_gold > 0
        names = [f["name"] for f in bullish["forces"]]
        assert "insider_strong_buying" in names

    def test_insider_strong_selling_boosts_deflation(self):
        """insider_score=-0.6 → deflation/stagflation projected > baseline."""
        probs = self._base_neutral_probs()
        baseline = calculate_trajectory(
            current_probabilities=probs,
            indicators=self._neutral_indicators(),
        )
        bearish = calculate_trajectory(
            current_probabilities=probs,
            indicators={**self._neutral_indicators(), "insider_score": -0.6},
        )
        delta_defl = bearish["projected_probabilities"]["deflation"] - baseline["projected_probabilities"]["deflation"]
        delta_stag = bearish["projected_probabilities"]["stagflation"] - baseline["projected_probabilities"]["stagflation"]
        assert delta_defl + delta_stag > 0
        names = [f["name"] for f in bearish["forces"]]
        assert "insider_strong_selling" in names

    def test_insider_extreme_score_scales_strength(self):
        """insider_score=+1.0 deve avere strength maggiore di +0.4."""
        probs = self._base_neutral_probs()
        moderate = calculate_trajectory(
            current_probabilities=probs,
            indicators={**self._neutral_indicators(), "insider_score": 0.4},
        )
        extreme = calculate_trajectory(
            current_probabilities=probs,
            indicators={**self._neutral_indicators(), "insider_score": 1.0},
        )
        mod_force = next((f for f in moderate["forces"] if f["name"] == "insider_strong_buying"), None)
        ext_force = next((f for f in extreme["forces"] if f["name"] == "insider_strong_buying"), None)
        assert mod_force is not None and ext_force is not None
        assert abs(ext_force["strength"]) > abs(mod_force["strength"])

    def test_insider_force_does_not_break_probability_sum(self):
        """Probs proiettate devono sempre sommare a ~1.0."""
        probs = self._base_neutral_probs()
        result = calculate_trajectory(
            current_probabilities=probs,
            indicators={**self._neutral_indicators(), "insider_score": 0.7},
        )
        total = sum(result["projected_probabilities"].values())
        assert 0.99 < total < 1.01
