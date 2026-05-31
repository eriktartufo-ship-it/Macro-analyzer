"""Test T14 Event-triggered rebalance — council action #4.

Coverage:
- 4 trigger types: vix_panic, m2_surge, curve_uninvert, fomc_dovish_surprise
- Helper functions any_trigger_fired, max_severity, fired_triggers_summary
- Edge cases: no previous snapshot, all triggers fired, severity bounds.
"""
from __future__ import annotations

import pytest

from app.services.regime.event_triggers import (
    EventTriggerDetector,
    TriggerSnapshot,
    any_trigger_fired,
    fired_triggers_summary,
    max_severity,
)


class TestVixPanic:
    def test_calm_to_panic_fires(self):
        det = EventTriggerDetector()
        prev = TriggerSnapshot(vix=18.0, m2_yoy=5.0, yield_curve_10y2y=1.0)
        curr = TriggerSnapshot(vix=42.0, m2_yoy=5.0, yield_curve_10y2y=1.0)
        out = det.check_window(curr, prev)
        assert out["vix_panic"].fired
        assert out["vix_panic"].severity > 0.0

    def test_already_high_no_fire(self):
        """VIX 30→35 should NOT fire (no calm→panic transition)."""
        det = EventTriggerDetector()
        prev = TriggerSnapshot(vix=30.0, m2_yoy=5.0, yield_curve_10y2y=1.0)
        curr = TriggerSnapshot(vix=35.0, m2_yoy=5.0, yield_curve_10y2y=1.0)
        out = det.check_window(curr, prev)
        assert not out["vix_panic"].fired

    def test_first_check_extreme_fires(self):
        """No previous, VIX > 35 → fire (extreme panic detection)."""
        det = EventTriggerDetector()
        curr = TriggerSnapshot(vix=40.0, m2_yoy=5.0, yield_curve_10y2y=1.0)
        out = det.check_window(curr, None)
        assert out["vix_panic"].fired


class TestM2Surge:
    def test_swing_above_threshold_fires(self):
        det = EventTriggerDetector()
        prev = TriggerSnapshot(vix=20.0, m2_yoy=3.0, yield_curve_10y2y=1.0)
        curr = TriggerSnapshot(vix=20.0, m2_yoy=10.0, yield_curve_10y2y=1.0)
        out = det.check_window(curr, prev)
        assert out["m2_surge"].fired
        assert out["m2_surge"].severity > 0.0

    def test_small_swing_no_fire(self):
        det = EventTriggerDetector()
        prev = TriggerSnapshot(vix=20.0, m2_yoy=5.0, yield_curve_10y2y=1.0)
        curr = TriggerSnapshot(vix=20.0, m2_yoy=7.0, yield_curve_10y2y=1.0)
        out = det.check_window(curr, prev)
        assert not out["m2_surge"].fired


class TestCurveUninvert:
    def test_negative_to_positive_fires(self):
        det = EventTriggerDetector()
        prev = TriggerSnapshot(vix=20.0, m2_yoy=5.0, yield_curve_10y2y=-0.3)
        curr = TriggerSnapshot(vix=20.0, m2_yoy=5.0, yield_curve_10y2y=0.2)
        out = det.check_window(curr, prev)
        assert out["curve_uninvert"].fired

    def test_negative_to_more_negative_no_fire(self):
        det = EventTriggerDetector()
        prev = TriggerSnapshot(vix=20.0, m2_yoy=5.0, yield_curve_10y2y=-0.3)
        curr = TriggerSnapshot(vix=20.0, m2_yoy=5.0, yield_curve_10y2y=-0.5)
        out = det.check_window(curr, prev)
        assert not out["curve_uninvert"].fired

    def test_positive_to_positive_no_fire(self):
        det = EventTriggerDetector()
        prev = TriggerSnapshot(vix=20.0, m2_yoy=5.0, yield_curve_10y2y=0.5)
        curr = TriggerSnapshot(vix=20.0, m2_yoy=5.0, yield_curve_10y2y=0.7)
        out = det.check_window(curr, prev)
        assert not out["curve_uninvert"].fired


class TestFomcDovishSurprise:
    def test_hawkish_to_dovish_swing_fires(self):
        det = EventTriggerDetector()
        prev = TriggerSnapshot(vix=20.0, m2_yoy=5.0, yield_curve_10y2y=1.0, fomc_sentiment=0.3)
        curr = TriggerSnapshot(vix=20.0, m2_yoy=5.0, yield_curve_10y2y=1.0, fomc_sentiment=-0.2)
        out = det.check_window(curr, prev)
        assert out["fomc_dovish_surprise"].fired

    def test_small_shift_no_fire(self):
        det = EventTriggerDetector()
        prev = TriggerSnapshot(vix=20.0, m2_yoy=5.0, yield_curve_10y2y=1.0, fomc_sentiment=0.1)
        curr = TriggerSnapshot(vix=20.0, m2_yoy=5.0, yield_curve_10y2y=1.0, fomc_sentiment=-0.05)
        out = det.check_window(curr, prev)
        assert not out["fomc_dovish_surprise"].fired


class TestHelpers:
    def test_any_trigger_fired_true(self):
        det = EventTriggerDetector()
        prev = TriggerSnapshot(vix=18.0, m2_yoy=5.0, yield_curve_10y2y=1.0)
        curr = TriggerSnapshot(vix=40.0, m2_yoy=5.0, yield_curve_10y2y=1.0)
        out = det.check_window(curr, prev)
        assert any_trigger_fired(out)

    def test_any_trigger_fired_false(self):
        det = EventTriggerDetector()
        prev = TriggerSnapshot(vix=18.0, m2_yoy=5.0, yield_curve_10y2y=1.0)
        curr = TriggerSnapshot(vix=20.0, m2_yoy=5.0, yield_curve_10y2y=1.0)
        out = det.check_window(curr, prev)
        assert not any_trigger_fired(out)

    def test_max_severity_returns_highest(self):
        det = EventTriggerDetector()
        # VIX big jump + M2 swing → 2 trigger fired
        prev = TriggerSnapshot(vix=18.0, m2_yoy=3.0, yield_curve_10y2y=1.0)
        curr = TriggerSnapshot(vix=45.0, m2_yoy=12.0, yield_curve_10y2y=1.0)
        out = det.check_window(curr, prev)
        assert max_severity(out) > 0.5

    def test_max_severity_zero_no_fire(self):
        det = EventTriggerDetector()
        prev = TriggerSnapshot(vix=18.0, m2_yoy=5.0, yield_curve_10y2y=1.0)
        curr = TriggerSnapshot(vix=19.0, m2_yoy=5.0, yield_curve_10y2y=1.0)
        out = det.check_window(curr, prev)
        assert max_severity(out) == 0.0

    def test_fired_summary_lists_descriptions(self):
        det = EventTriggerDetector()
        prev = TriggerSnapshot(vix=18.0, m2_yoy=5.0, yield_curve_10y2y=1.0)
        curr = TriggerSnapshot(vix=40.0, m2_yoy=5.0, yield_curve_10y2y=1.0)
        out = det.check_window(curr, prev)
        summary = fired_triggers_summary(out)
        assert len(summary) == 1
        assert "VIX panic" in summary[0]


class TestAllTriggersIntegration:
    def test_2020_covid_scenario(self):
        """Marzo 2020: VIX spike + M2 pump + curve uninvert + FOMC dovish."""
        det = EventTriggerDetector()
        prev = TriggerSnapshot(
            vix=18.0, m2_yoy=6.0, yield_curve_10y2y=-0.1, fomc_sentiment=0.2,
        )
        curr = TriggerSnapshot(
            vix=65.0, m2_yoy=18.0, yield_curve_10y2y=0.3, fomc_sentiment=-0.4,
        )
        out = det.check_window(curr, prev)
        assert all(r.fired for r in out.values())
        assert max_severity(out) > 0.7

    def test_calm_period_no_trigger(self):
        """Periodo calm tipo 2017-2019: no trigger."""
        det = EventTriggerDetector()
        prev = TriggerSnapshot(
            vix=12.0, m2_yoy=4.0, yield_curve_10y2y=0.8, fomc_sentiment=0.1,
        )
        curr = TriggerSnapshot(
            vix=14.0, m2_yoy=4.5, yield_curve_10y2y=0.6, fomc_sentiment=0.15,
        )
        out = det.check_window(curr, prev)
        assert not any_trigger_fired(out)
