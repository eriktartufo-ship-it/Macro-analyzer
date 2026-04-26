"""Test Monte Carlo simulator + shock scenarios."""
import numpy as np
import pytest

from app.services.regime.classifier import REGIMES
from app.services.regime.monte_carlo import (
    _compute_regime_bands,
    _sample_initial_states,
    _simulate_paths,
)
from app.services.regime.shock_scenarios import (
    PRESET_SCENARIOS,
    _apply_deltas,
    list_preset_scenarios,
)


class TestSampleInitialStates:
    def test_uniform_when_zero_distribution(self):
        rng = np.random.default_rng(0)
        states = _sample_initial_states({r: 0.0 for r in REGIMES}, 1000, rng)
        # Tutti i regimi devono comparire roughly al 25%
        from collections import Counter
        c = Counter(states)
        for k in range(4):
            assert 200 < c[k] < 300

    def test_concentrated_distribution(self):
        rng = np.random.default_rng(0)
        initial = {"reflation": 1.0, "stagflation": 0.0, "deflation": 0.0, "goldilocks": 0.0}
        states = _sample_initial_states(initial, 100, rng)
        # Tutti gli stati == 0 (reflation)
        assert (states == 0).all()


class TestSimulatePaths:
    def test_persistent_transition_keeps_state(self):
        """Matrice identita': tutti i path restano allo stato iniziale."""
        rng = np.random.default_rng(0)
        K = 4
        A = np.eye(K)
        initial = np.zeros(50, dtype=int)  # tutti in stato 0
        paths = _simulate_paths(initial, A, n_steps=10, rng=rng)
        assert (paths == 0).all()

    def test_uniform_transition_distributes(self):
        """Matrice uniforme: dopo molti step la distribuzione diventa uniforme."""
        rng = np.random.default_rng(42)
        K = 4
        A = np.full((K, K), 1.0 / K)
        initial = np.zeros(2000, dtype=int)
        paths = _simulate_paths(initial, A, n_steps=20, rng=rng)
        # Step finale: ogni stato dovrebbe essere ~25%
        from collections import Counter
        final = Counter(paths[:, -1])
        for k in range(K):
            assert 350 < final[k] < 650


class TestRegimeBands:
    def test_bands_sum_to_one_per_step(self):
        """Ad ogni step le frequenze dei 4 regimi sommano ~1."""
        rng = np.random.default_rng(0)
        paths = rng.integers(0, 4, size=(500, 13))
        bands = _compute_regime_bands(paths)
        for t in range(13):
            total = sum(b.mean[t] for b in bands)
            assert abs(total - 1.0) < 1e-6

    def test_bands_ordered_p10_le_median_le_p90(self):
        rng = np.random.default_rng(0)
        paths = rng.integers(0, 4, size=(500, 13))
        bands = _compute_regime_bands(paths)
        for b in bands:
            for t in range(13):
                assert b.p10[t] <= b.median[t] <= b.p90[t]


class TestShockScenarios:
    def test_preset_scenarios_have_required_fields(self):
        for k, v in PRESET_SCENARIOS.items():
            assert "label" in v
            assert "description" in v
            assert "deltas" in v
            assert isinstance(v["deltas"], dict)
            for ind, (op, val) in v["deltas"].items():
                assert op in ("set", "delta")
                assert isinstance(val, (int, float))

    def test_apply_deltas_set_operation(self):
        baseline = {"vix": 20.0, "cpi_yoy": 3.0}
        deltas = {"vix": ("set", 45.0)}
        out = _apply_deltas(baseline, deltas)
        assert out["vix"] == 45.0
        assert out["cpi_yoy"] == 3.0  # non toccato

    def test_apply_deltas_delta_operation(self):
        baseline = {"fed_funds_rate": 5.0}
        deltas = {"fed_funds_rate": ("delta", -1.0)}
        out = _apply_deltas(baseline, deltas)
        assert out["fed_funds_rate"] == 4.0

    def test_list_preset_scenarios_returns_all(self):
        out = list_preset_scenarios()
        assert len(out) == len(PRESET_SCENARIOS)
        keys = {s["key"] for s in out}
        assert keys == set(PRESET_SCENARIOS.keys())
