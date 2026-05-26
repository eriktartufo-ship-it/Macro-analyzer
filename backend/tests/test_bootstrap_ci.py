"""Test bootstrap CI implementation."""
from __future__ import annotations

import pytest

from app.services.validation.bootstrap_ci import (
    bootstrap_classification_metrics,
    bootstrap_metric,
)


class TestBootstrapMetric:
    def test_uniform_sample_tight_ci(self):
        """Tutti i valori uguali → CI molto stretto."""
        values = [1.0] * 20
        out = bootstrap_metric(values, n_bootstrap=500, random_state=42)
        assert out.point_estimate == pytest.approx(1.0)
        assert out.ci_lower == pytest.approx(1.0)
        assert out.ci_upper == pytest.approx(1.0)

    def test_binary_50_50_wide_ci(self):
        """50/50 split → CI ~ [0.3, 0.7] su N=10."""
        values = [1, 0] * 5  # N=10, mean=0.5
        out = bootstrap_metric(values, n_bootstrap=1000, random_state=42)
        assert out.point_estimate == pytest.approx(0.5, abs=0.01)
        # CI deve coprire ampia incertezza per N=10
        assert out.ci_lower < 0.4
        assert out.ci_upper > 0.6

    def test_small_sample_wide_ci(self):
        """N=9 (audit OOS reale) → CI molto wide."""
        # 7/9 corretti = 78%
        values = [1, 1, 1, 1, 1, 1, 1, 0, 0]
        out = bootstrap_metric(values, n_bootstrap=1000, random_state=42)
        assert out.point_estimate == pytest.approx(7/9, abs=0.01)
        # CI: con N=9, 7/9 corretti → CI ampio
        assert out.ci_lower > 0.40, "CI lower too low"
        assert out.ci_upper > 0.85, "CI upper not high enough"
        # CI width > 0.30 (statisticamente fragile)
        assert (out.ci_upper - out.ci_lower) > 0.25

    def test_empty_returns_zeros(self):
        out = bootstrap_metric([], n_bootstrap=100)
        assert out.point_estimate == 0
        assert out.n_samples == 0

    def test_is_significant_baseline_outside_ci(self):
        """Random baseline 25% deve essere fuori CI per 7/9 = 78%."""
        values = [1, 1, 1, 1, 1, 1, 1, 0, 0]
        out = bootstrap_metric(values, n_bootstrap=1000, random_state=42)
        assert out.is_significant(baseline=0.25) is True
        # Mentre 50% potrebbe essere borderline
        # 67% (vecchio score) probabilmente dentro CI


class TestBootstrapClassificationMetrics:
    def test_oos_real_audit_metrics(self):
        """Replica audit reale OOS: 7/9 regime correct, top-5 [2,3,3,2,1,2,0,3,3]."""
        correct = [True, True, True, True, False, True, False, True, True]  # 7/9
        overlaps = [2, 3, 3, 2, 1, 2, 0, 3, 3]  # avg 2.11/5
        out = bootstrap_classification_metrics(correct, overlaps)
        # regime accuracy CI
        ra = out["regime_accuracy"]
        assert ra.point_estimate == pytest.approx(7/9, abs=0.01)
        assert ra.ci_lower < 0.55, "CI lower for 7/9 should be < 0.55"
        assert ra.ci_upper > 0.85, "CI upper should reach >0.85"
        # top-5 avg CI
        ta = out["top_n_overlap_avg"]
        assert ta.point_estimate == pytest.approx(2.11, abs=0.05)
        # Top-5 high overlap (>=3) rate
        hr = out["top_n_high_overlap_rate"]
        assert hr.point_estimate == pytest.approx(4/9, abs=0.01)

    def test_all_metrics_present(self):
        correct = [True] * 9
        overlaps = [5] * 9
        out = bootstrap_classification_metrics(correct, overlaps)
        assert set(out.keys()) >= {
            "regime_accuracy", "top_n_overlap_avg",
            "top_n_overlap_pct", "top_n_high_overlap_rate",
        }
