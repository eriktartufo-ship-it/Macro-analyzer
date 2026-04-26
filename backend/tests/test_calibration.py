"""Test del modulo calibration.py — shrinkage Bayesiano e logica caricamento."""
import os
import pytest

from app.services.scoring.calibration import (
    CalibrationParams,
    _shrinkage_weight,
    _shrink,
    _is_outlier,
)


class TestShrinkageWeight:
    def test_below_n_min_zero(self):
        """Sotto n_min: peso 0 (prior intero)."""
        assert _shrinkage_weight(5, 8, 40) == 0.0

    def test_at_or_above_n_full_one(self):
        """A n_full o sopra: peso 1 (misurato intero)."""
        assert _shrinkage_weight(40, 8, 40) == 1.0
        assert _shrinkage_weight(100, 8, 40) == 1.0

    def test_linear_in_between(self):
        """Tra n_min e n_full: lineare. n_min=8, n_full=40, n=24 -> mid = 0.5."""
        w = _shrinkage_weight(24, 8, 40)
        assert 0.49 < w < 0.51


class TestShrink:
    def test_no_measured_returns_prior(self):
        assert _shrink(0.10, None, 0.5) == 0.10

    def test_full_weight_returns_measured(self):
        assert _shrink(0.10, 0.20, 1.0) == 0.20

    def test_zero_weight_returns_prior(self):
        assert _shrink(0.10, 0.20, 0.0) == 0.10

    def test_half_weight_average(self):
        assert _shrink(0.10, 0.20, 0.5) == pytest.approx(0.15)


class TestOutlier:
    def test_extreme_real_return_flagged(self):
        p = CalibrationParams(real_return_max_abs=0.80)
        assert _is_outlier(real_return=1.5, sharpe=0.5, p=p) is True
        assert _is_outlier(real_return=-1.5, sharpe=0.5, p=p) is True

    def test_extreme_sharpe_flagged(self):
        p = CalibrationParams(sharpe_max_abs=4.0)
        assert _is_outlier(real_return=0.10, sharpe=10.0, p=p) is True

    def test_normal_values_pass(self):
        p = CalibrationParams()
        assert _is_outlier(real_return=0.15, sharpe=1.2, p=p) is False


class TestEnvOptIn:
    def test_default_uses_prior(self):
        """Con USE_CALIBRATED_SCORING non settato, scoring usa hardcoded."""
        # Salva env state e ripristina
        prev = os.environ.pop("USE_CALIBRATED_SCORING", None)
        try:
            from app.services.scoring.engine import _calibrated_or_prior, ASSET_REGIME_DATA
            data = _calibrated_or_prior()
            # Deve essere literally il dict hardcoded
            assert data is ASSET_REGIME_DATA
        finally:
            if prev is not None:
                os.environ["USE_CALIBRATED_SCORING"] = prev


class TestCalibrationOutputShape:
    def test_calibrate_produces_expected_keys(self):
        """Smoke test: calibrate() ritorna struttura dict completa."""
        from app.database import engine
        from app.models import RegimeClassification
        from sqlalchemy.orm import Session
        from app.services.scoring.calibration import calibrate

        with Session(engine) as s:
            n = s.query(RegimeClassification).count()
            if n < 30:
                pytest.skip(f"DB ha solo {n} record, troppo pochi per calibrazione")
            payload = calibrate(s)

        assert "version" in payload
        assert "calibrated_on" in payload
        assert "asset_regime_data" in payload
        assert "diagnostics" in payload
        assert payload["n_classifications"] >= 30
        # Almeno alcuni asset class devono esserci
        assert len(payload["asset_regime_data"]) > 5
        # Diagnostics deve avere asset/regime/weight_measured
        if payload["diagnostics"]:
            d = payload["diagnostics"][0]
            assert "weight_measured" in d
            assert 0.0 <= d["weight_measured"] <= 1.0
