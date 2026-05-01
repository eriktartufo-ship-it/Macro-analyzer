"""Test Monte Carlo simulator + shock scenarios."""
import numpy as np
import pytest

from app.services.regime.classifier import REGIMES
from app.services.regime.monte_carlo import (
    INITIAL_SOURCE_ENSEMBLE,
    INITIAL_SOURCE_EXPLICIT,
    INITIAL_SOURCE_RULE_BASED,
    _compute_regime_bands,
    _resolve_initial_distribution,
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


class TestResolveInitialDistribution:
    """initial_source switch tra rule_based (DB) ed ensemble (3 modelli)."""

    def test_explicit_distribution_wins_over_source(self):
        """Se chi chiama passa initial_distribution esplicito, lo usiamo
        anche con initial_source='ensemble' (override deterministico)."""
        explicit = {"reflation": 0.7, "stagflation": 0.1, "deflation": 0.1, "goldilocks": 0.1}
        out = _resolve_initial_distribution(
            db=None,
            initial_distribution=explicit,
            initial_source=INITIAL_SOURCE_ENSEMBLE,
        )
        assert out.distribution == explicit
        assert out.source == INITIAL_SOURCE_EXPLICIT
        assert out.ensemble_disagreement is None
        assert out.ensemble_confidence is None

    def test_ensemble_source_uses_compute_ensemble(self, monkeypatch):
        """initial_source='ensemble' chiama compute_ensemble e ne usa le probabilita'."""
        from app.services.regime import monte_carlo as mc
        from app.services.regime.ensemble import EnsembleResult

        target = {"reflation": 0.10, "stagflation": 0.50, "deflation": 0.35, "goldilocks": 0.05}
        fake = EnsembleResult(
            weights={"rule_based": 1 / 3, "hmm_market": 1 / 3, "msvar": 1 / 3},
            views=[],
            ensemble_probabilities=target,
            confidence=0.42,
            disagreement_score=0.31,
            high_disagreement=True,
            dominant_regime="stagflation",
            notes=["Alto disaccordo (avg JS 0.31)."],
        )
        monkeypatch.setattr(mc, "compute_ensemble", lambda db: fake)

        out = _resolve_initial_distribution(
            db=object(),  # dummy: il monkeypatch ignora db
            initial_distribution=None,
            initial_source=INITIAL_SOURCE_ENSEMBLE,
        )
        assert out.distribution == target
        assert out.source == INITIAL_SOURCE_ENSEMBLE
        assert out.ensemble_disagreement == pytest.approx(0.31)
        assert out.ensemble_confidence == pytest.approx(0.42)
        assert any("disaccordo" in n.lower() for n in out.notes)

    def test_rule_based_source_uses_last_db_classification(self):
        """initial_source='rule_based' (default) pesca l'ultima RegimeClassification."""
        from types import SimpleNamespace

        last = SimpleNamespace(
            probability_reflation=0.40,
            probability_stagflation=0.25,
            probability_deflation=0.10,
            probability_goldilocks=0.25,
        )

        class FakeQuery:
            def order_by(self, *a, **k): return self
            def first(self): return last

        class FakeDB:
            def query(self, *a, **k): return FakeQuery()

        out = _resolve_initial_distribution(
            db=FakeDB(),
            initial_distribution=None,
            initial_source=INITIAL_SOURCE_RULE_BASED,
        )
        assert out.distribution["reflation"] == pytest.approx(0.40)
        assert out.distribution["stagflation"] == pytest.approx(0.25)
        assert out.source == INITIAL_SOURCE_RULE_BASED
        assert out.ensemble_disagreement is None

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError):
            _resolve_initial_distribution(
                db=None, initial_distribution=None, initial_source="bogus_source",
            )
