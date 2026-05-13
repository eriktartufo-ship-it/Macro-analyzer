"""Test Tier 5.7 LOOP CLOSURE — Vol regime + change-point detection."""
from __future__ import annotations

from app.services.regime.vol_regime import (
    assess_vol_regime,
    classify_vol_regime,
    detect_change_point,
)


class TestClassifyVolRegime:
    def test_vix_low_is_calm(self):
        regime, score = classify_vol_regime(10.0)
        assert regime == "vol_calm"
        assert 0.0 <= score <= 1.0

    def test_vix_boundary_14_is_calm(self):
        regime, _ = classify_vol_regime(14.0)
        assert regime == "vol_calm"

    def test_vix_15_is_normal(self):
        regime, score = classify_vol_regime(15.0)
        assert regime == "vol_normal"
        assert 0.0 < score < 0.5

    def test_vix_20_is_normal_boundary(self):
        regime, _ = classify_vol_regime(20.0)
        assert regime == "vol_normal"

    def test_vix_25_is_elevated(self):
        regime, score = classify_vol_regime(25.0)
        assert regime == "vol_elevated"
        assert score == 0.5

    def test_vix_45_is_spike(self):
        regime, score = classify_vol_regime(45.0)
        assert regime == "vol_spike"
        assert 0.4 < score < 0.6

    def test_vix_extreme_score_capped(self):
        regime, score = classify_vol_regime(80.0)
        assert regime == "vol_spike"
        assert score == 1.0


class TestDetectChangePoint:
    def test_no_change_when_no_ma_data(self):
        cp, mag = detect_change_point(20.0, None, None)
        assert cp == "none"
        assert mag == 0.0

    def test_no_change_zero_std(self):
        cp, mag = detect_change_point(20.0, 18.0, 0.0)
        assert cp == "none"

    def test_breakout_when_vix_above_ma_by_k_sigma(self):
        # VIX 30, ma=18, std=4 → z=(30-18)/4=3.0 → breakout (k=1.5)
        cp, mag = detect_change_point(30.0, 18.0, 4.0, k=1.5)
        assert cp == "vol_breakout"
        assert mag == 3.0

    def test_compression_when_vix_below_ma(self):
        # VIX 10, ma=18, std=4 → z=-2.0 → compression
        cp, mag = detect_change_point(10.0, 18.0, 4.0, k=1.5)
        assert cp == "vol_compression"
        assert mag == 2.0

    def test_no_change_within_threshold(self):
        # VIX 20, ma=18, std=4 → z=0.5 → no change (sotto k=1.5)
        cp, mag = detect_change_point(20.0, 18.0, 4.0, k=1.5)
        assert cp == "none"


class TestAssessVolRegime:
    def test_calm_no_change(self):
        out = assess_vol_regime(11.0, vix_ma20=12.0, vix_std20=2.0)
        assert out.vol_regime == "vol_calm"
        assert out.change_point == "none"
        assert "risk-on" in out.description.lower()

    def test_spike_with_breakout(self):
        out = assess_vol_regime(45.0, vix_ma20=18.0, vix_std20=4.0)
        assert out.vol_regime == "vol_spike"
        assert out.change_point == "vol_breakout"
        assert out.change_magnitude > 1.5
        assert "panic" in out.description.lower() or "spike" in out.description.lower()

    def test_calm_with_compression(self):
        """VIX 9 con ma 14 std 2 → compression + vol_calm = bubble warning."""
        out = assess_vol_regime(9.0, vix_ma20=14.0, vix_std20=2.0)
        assert out.vol_regime == "vol_calm"
        assert out.change_point == "vol_compression"
        assert "compression" in out.description.lower() or "complacency" in out.description.lower()

    def test_normal_without_ma_data(self):
        out = assess_vol_regime(17.0)
        assert out.vol_regime == "vol_normal"
        assert out.change_point == "none"
        assert out.vix_ma20 is None

    def test_output_serializable_fields(self):
        out = assess_vol_regime(22.5, vix_ma20=18.0, vix_std20=3.0)
        # Tutti i campi devono essere float, str o None (no oggetti)
        assert isinstance(out.vix, float)
        assert isinstance(out.vol_score, float)
        assert isinstance(out.vol_regime, str)
        assert isinstance(out.change_point, str)
        assert isinstance(out.description, str)

    def test_2008_oct_panic_scenario(self):
        """2008-10 VIX peaked at 80. Should be extreme spike + breakout."""
        out = assess_vol_regime(80.0, vix_ma20=30.0, vix_std20=8.0)
        assert out.vol_regime == "vol_spike"
        assert out.vol_score == 1.0
        assert out.change_point == "vol_breakout"

    def test_2017_complacency_scenario(self):
        """2017 era ultra-calm: VIX often <10. Should be vol_calm + compression."""
        out = assess_vol_regime(9.5, vix_ma20=12.0, vix_std20=1.5)
        assert out.vol_regime == "vol_calm"
        # Z = (9.5 - 12) / 1.5 = -1.67 → compression
        assert out.change_point == "vol_compression"
