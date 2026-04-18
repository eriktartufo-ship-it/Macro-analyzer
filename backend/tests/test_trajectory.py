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
