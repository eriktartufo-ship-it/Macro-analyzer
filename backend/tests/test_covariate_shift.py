"""Test covariate shift detector."""
from __future__ import annotations

import pytest

from app.services.validation.covariate_shift import (
    CovariateShiftReport,
    assess_covariate_shift,
    assess_persistent_shift,
    ks_test_tail_divergence,
    reset_cache,
)


class TestCovariateShift:
    def setup_method(self):
        reset_cache()

    def test_normal_indicators_no_shift(self):
        """Indicators tipici storici → no shift."""
        normal = {
            "gdp_roc": 2.5, "cpi_yoy": 2.5, "unrate": 5.0, "vix": 18.0,
            "fed_funds": 3.0, "breakeven_10y": 2.2, "baa_spread": 2.0,
            "yield_curve_10y2y": 1.0, "core_pce_yoy": 2.0,
        }
        out = assess_covariate_shift(normal)
        assert out.shift_alert is False
        assert out.n_shifted <= 1  # solo eventuale noise

    def test_hyperinflation_triggers_shift(self):
        """Argentina-style CPI 50%+ → covariate shift critical."""
        extreme = {
            "gdp_roc": -8.0, "cpi_yoy": 60.0, "unrate": 22.0, "vix": 80.0,
            "fed_funds": 18.0, "breakeven_10y": 8.0, "baa_spread": 10.0,
            "yield_curve_10y2y": -2.5, "core_pce_yoy": 50.0,
        }
        out = assess_covariate_shift(extreme)
        assert out.shift_alert is True
        assert out.n_critical >= 3

    def test_single_shifted_no_alert(self):
        """T9-AUDIT P1-B: con threshold 2.0σ (era 2.5), shift più sensibile.
        Test rivisto: indicators MEDI non devono triggerare alert."""
        median = {
            "gdp_roc": 2.5, "cpi_yoy": 2.5,
            "unrate": 5.0, "vix": 18.0, "fed_funds": 3.0,
            "breakeven_10y": 2.5, "baa_spread": 2.0,
            "yield_curve_10y2y": 1.0, "core_pce_yoy": 2.5,
        }
        out = assess_covariate_shift(median)
        assert out.shift_alert is False

    def test_each_indicator_shift_has_z_score(self):
        normal = {"gdp_roc": 2.5, "cpi_yoy": 2.5}
        out = assess_covariate_shift(normal)
        for sh in out.indicators_shift:
            assert isinstance(sh.z_score, float)
            assert sh.severity in {"ok", "warn", "critical"}

    def test_returns_dataclass(self):
        out = assess_covariate_shift({})
        assert isinstance(out, CovariateShiftReport)


class TestCurrentState:
    """Test su dati live attuali (post bug-fix 2026-05-16)."""

    def test_current_2026_no_critical_shift(self):
        """Current macro 2026 dovrebbe avere shift moderato ma NON critico."""
        current = {
            "gdp_roc": 0.49, "cpi_yoy": 3.95, "unrate": 4.3,
            "vix": 17.26, "fed_funds_rate": 4.5, "breakeven_10y": 2.49,
            "baa_spread": 2.0, "yield_curve_10y2y": 0.5,
            "core_pce_yoy": 3.20,
        }
        out = assess_covariate_shift(current)
        # Inflation ~ 4% è elevato vs storico ma non critico
        assert out.n_critical == 0


class TestPersistentShift:
    """T9-AUDIT P1-B persistence check."""

    def setup_method(self):
        reset_cache()

    def _make_report(self, alert: bool, n_shifted: int = 3) -> CovariateShiftReport:
        return CovariateShiftReport(
            n_indicators_evaluated=9, n_shifted=n_shifted, n_critical=0,
            indicators_shift=[], shift_alert=alert,
            summary="test",
        )

    def test_no_persistent_when_isolated(self):
        reports = [self._make_report(True), self._make_report(False), self._make_report(True)]
        is_pers, desc = assess_persistent_shift(reports)
        assert is_pers is False
        assert "1/3" in desc

    def test_persistent_3_consecutive(self):
        reports = [self._make_report(True)] * 3 + [self._make_report(False)]
        is_pers, desc = assess_persistent_shift(reports)
        assert is_pers is True
        assert "PERSISTENT" in desc

    def test_empty_no_persistent(self):
        is_pers, _ = assess_persistent_shift([])
        assert is_pers is False


class TestKSTestDivergence:
    """T9-AUDIT P1-B KS-test tail divergence."""

    def test_same_distribution_no_divergence(self):
        import numpy as np
        rng = np.random.default_rng(42)
        train = rng.normal(0, 1, 100).tolist()
        live = rng.normal(0, 1, 50).tolist()
        is_div, p, desc = ks_test_tail_divergence(live, train)
        assert is_div is False  # stessa distribuzione → no divergence
        assert p > 0.05

    def test_shifted_distribution_divergent(self):
        import numpy as np
        rng = np.random.default_rng(42)
        train = rng.normal(0, 1, 100).tolist()
        live = rng.normal(3, 1, 50).tolist()  # shift di 3σ
        is_div, p, desc = ks_test_tail_divergence(live, train)
        assert is_div is True
        assert p < 0.05

    def test_insufficient_samples(self):
        is_div, p, desc = ks_test_tail_divergence([1.0], [1.0, 2.0, 3.0])
        assert is_div is False
        assert "Insufficient" in desc
