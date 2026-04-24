"""Test per transition_matrix service."""
from datetime import date, timedelta

from app.services.regime.transition_matrix import (
    project_probabilities,
    compute_transition_matrix,
)


class _FakeRow:
    def __init__(self, d: date, regime: str):
        self.date = d
        self.regime = regime


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def order_by(self, *_):
        return self

    def all(self):
        return self._rows


class _FakeSession:
    def __init__(self, rows):
        self._rows = rows

    def query(self, *_):
        return _FakeQuery(self._rows)


class TestProjectProbabilities:
    def test_identity_matrix_preserves_distribution(self):
        """Se la matrice e' identita, la distribuzione non cambia."""
        identity = {r: {rr: 1.0 if r == rr else 0.0
                        for rr in ["reflation", "stagflation", "deflation", "goldilocks"]}
                    for r in ["reflation", "stagflation", "deflation", "goldilocks"]}
        current = {"reflation": 0.3, "stagflation": 0.25, "deflation": 0.2, "goldilocks": 0.25}
        result = project_probabilities(identity, current, steps=5)
        for r in current:
            assert abs(result[r] - current[r]) < 1e-9

    def test_full_absorbing_transitions(self):
        """Matrice che manda tutto in deflation: dopo 1 passo tutto e' deflation."""
        absorb = {r: {"reflation": 0.0, "stagflation": 0.0, "deflation": 1.0, "goldilocks": 0.0}
                  for r in ["reflation", "stagflation", "deflation", "goldilocks"]}
        current = {"reflation": 0.5, "stagflation": 0.3, "deflation": 0.1, "goldilocks": 0.1}
        result = project_probabilities(absorb, current, steps=1)
        assert abs(result["deflation"] - 1.0) < 1e-9

    def test_sum_to_one_preserved(self):
        """La proiezione deve sempre sommare a 1.0."""
        matrix = {
            "reflation":   {"reflation": 0.7, "stagflation": 0.1, "deflation": 0.05, "goldilocks": 0.15},
            "stagflation": {"reflation": 0.2, "stagflation": 0.6, "deflation": 0.1, "goldilocks": 0.1},
            "deflation":   {"reflation": 0.15, "stagflation": 0.05, "deflation": 0.7, "goldilocks": 0.10},
            "goldilocks":  {"reflation": 0.2, "stagflation": 0.05, "deflation": 0.05, "goldilocks": 0.7},
        }
        current = {"reflation": 0.3, "stagflation": 0.25, "deflation": 0.2, "goldilocks": 0.25}
        for steps in [1, 3, 10]:
            result = project_probabilities(matrix, current, steps=steps)
            total = sum(result.values())
            assert abs(total - 1.0) < 1e-6, f"steps={steps} total={total}"


class TestComputeTransitionMatrix:
    def test_empty_db_returns_zeros(self):
        sess = _FakeSession([])
        result = compute_transition_matrix(sess, horizon_days=30)
        assert result.total_observations == 0
        for r in result.regimes:
            for rr in result.regimes:
                assert result.counts[r][rr] == 0

    def test_single_regime_self_transition(self):
        """Serie tutta reflation: P(reflation|reflation) = 1.0."""
        start = date(2024, 1, 1)
        rows = [_FakeRow(start + timedelta(days=i), "reflation") for i in range(200)]
        sess = _FakeSession(rows)
        result = compute_transition_matrix(sess, horizon_days=30)
        assert result.self_transition_probability["reflation"] > 0.99

    def test_two_regimes_alternating(self):
        """Serie che alterna blocchi di 120 giorni: self-transition dominante (orizzonte 30)."""
        start = date(2024, 1, 1)
        rows = []
        for block in range(3):
            reg = "reflation" if block % 2 == 0 else "goldilocks"
            for i in range(120):
                rows.append(_FakeRow(start + timedelta(days=block * 120 + i), reg))
        sess = _FakeSession(rows)
        result = compute_transition_matrix(sess, horizon_days=30)
        assert result.self_transition_probability["reflation"] > 0.7
        assert result.self_transition_probability["goldilocks"] > 0.7

    def test_avg_duration_computed(self):
        """Run di 90 giorni di reflation + 60 di deflation → medie coerenti."""
        start = date(2024, 1, 1)
        rows = []
        for i in range(90):
            rows.append(_FakeRow(start + timedelta(days=i), "reflation"))
        for i in range(60):
            rows.append(_FakeRow(start + timedelta(days=90 + i), "deflation"))
        sess = _FakeSession(rows)
        result = compute_transition_matrix(sess, horizon_days=30)
        assert result.avg_duration_days["reflation"] >= 85
        assert result.avg_duration_days["deflation"] >= 55
