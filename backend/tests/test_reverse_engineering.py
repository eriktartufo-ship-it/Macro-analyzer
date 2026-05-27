"""NEW council 2026-05-27 — Reverse engineering + drift detector tests."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from app.database import Base, SessionLocal, engine
from app.models import PredictionLog


@pytest.fixture
def db_session():
    """Test DB session con prediction_log table fresca."""
    PredictionLog.__table__.create(engine, checkfirst=True)
    sess = SessionLocal()
    # Cleanup
    sess.query(PredictionLog).delete()
    sess.commit()
    yield sess
    sess.query(PredictionLog).delete()
    sess.commit()
    sess.close()


def _seed_predictions(db: Session, scenarios: list[dict]) -> list[int]:
    """Helper: seed N predictions con alpha_3m specifico."""
    ids = []
    for s in scenarios:
        record = PredictionLog(
            date_predicted=s.get("date", date.today() - timedelta(days=120)),
            regime=s["regime"],
            confidence=s.get("confidence", 0.5),
            probabilities_json=json.dumps(s.get("probs", {s["regime"]: 0.5})),
            top5_json=json.dumps(s.get("top5", [{"asset": "gold", "weight": 0.2}])),
            allocation_mode="regime_based",
            alpha_3m=s.get("alpha", -0.08),
            realized_return_3m=s.get("realized", -0.03),
            benchmark_return_3m=s.get("bench", 0.05),
            evaluated_at=datetime.utcnow(),
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        ids.append(record.id)
    return ids


class TestReverseEngineering:
    def test_post_mortem_severe_failure(self, db_session):
        from app.services.validation.reverse_engineering import post_mortem

        [pid] = _seed_predictions(db_session, [{
            "regime": "stagflation",
            "confidence": 0.20,  # low
            "probs": {"stagflation": 0.35, "reflation": 0.30, "deflation": 0.20, "goldilocks": 0.15},  # boundary
            "alpha": -0.12,  # severe
        }])
        pm = post_mortem(db_session, pid, horizon="3m")
        assert pm is not None
        assert pm.severity == "severe"
        assert pm.regime_was_boundary is True  # max prob 0.35
        assert pm.confidence_was_low is True
        assert len(pm.findings) >= 2
        assert len(pm.suggestions) >= 2

    def test_post_mortem_minor_failure(self, db_session):
        from app.services.validation.reverse_engineering import post_mortem

        [pid] = _seed_predictions(db_session, [{
            "regime": "reflation",
            "confidence": 0.65,
            "probs": {"reflation": 0.60, "stagflation": 0.15, "deflation": 0.10, "goldilocks": 0.15},
            "alpha": -0.03,  # minor (< 0.05 absolute)
        }])
        pm = post_mortem(db_session, pid, horizon="3m")
        assert pm.severity == "minor"
        assert pm.regime_was_boundary is False
        assert pm.confidence_was_low is False

    def test_find_patterns_aggregates_regime_failures(self, db_session):
        from app.services.validation.reverse_engineering import find_failure_patterns

        _seed_predictions(db_session, [
            {"regime": "deflation", "alpha": -0.08, "confidence": 0.45},
            {"regime": "deflation", "alpha": -0.10, "confidence": 0.50},
            {"regime": "deflation", "alpha": -0.07, "confidence": 0.40},
            {"regime": "reflation", "alpha": -0.06, "confidence": 0.55},  # single, no pattern
        ])
        patterns = find_failure_patterns(db_session, horizon="3m", min_occurrences=2)
        names = {p.pattern_name for p in patterns}
        assert "regime_deflation_systematic" in names

    def test_no_failures_returns_empty_patterns(self, db_session):
        from app.services.validation.reverse_engineering import find_failure_patterns

        _seed_predictions(db_session, [
            {"regime": "goldilocks", "alpha": 0.05, "confidence": 0.70},  # winner, no failure
        ])
        patterns = find_failure_patterns(db_session, horizon="3m")
        assert patterns == []

    def test_generate_improvement_suggestions_empty(self, db_session):
        from app.services.validation.reverse_engineering import generate_improvement_suggestions
        out = generate_improvement_suggestions(db_session, horizon="3m")
        assert out["n_evaluated"] == 0
        assert out["n_failures"] == 0


class TestDriftDetector:
    def test_psi_zero_for_identical_distributions(self):
        import numpy as np
        from app.services.validation.drift_detector import compute_psi
        rng = np.random.default_rng(0)
        ref = list(rng.normal(0, 1, 500))
        cur = list(rng.normal(0, 1, 500))
        psi = compute_psi(ref, cur)
        assert psi < 0.10  # noise-level, no significant drift

    def test_psi_high_for_shifted_distribution(self):
        import numpy as np
        from app.services.validation.drift_detector import compute_psi
        rng = np.random.default_rng(0)
        ref = list(rng.normal(0, 1, 500))
        cur_shifted = list(rng.normal(3.0, 1, 500))
        psi = compute_psi(ref, cur_shifted)
        assert psi >= 0.25  # significant drift

    def test_detect_drift_report(self):
        import numpy as np
        from app.services.validation.drift_detector import detect_drift
        rng = np.random.default_rng(0)
        ref = list(rng.normal(0, 1, 300))
        ref_stable = list(rng.normal(0, 1, 300))
        cur_drift = list(rng.normal(2.5, 1, 300))

        report = detect_drift(
            reference={"cpi_yoy": ref, "gdp_roc": ref_stable},
            current={"cpi_yoy": cur_drift, "gdp_roc": ref_stable[:100]},
        )
        assert report.n_features == 2
        assert "cpi_yoy" in report.drift_features
        assert "gdp_roc" not in report.drift_features


class TestIndependentCrisisLabels:
    def test_independent_labels_more_episodes(self):
        from app.services.regime.classifier import classify_regime
        from app.services.validation.brier_crisis import assess_crisis_brier

        def fn(ind):
            return classify_regime(ind)["probabilities"]

        r_orig = assess_crisis_brier(fn, use_independent_labels=False)
        r_imf = assess_crisis_brier(fn, use_independent_labels=True)
        # IMF/BIS list ha più episodi matched OR uguale
        assert r_imf.n_episodes >= r_orig.n_episodes
