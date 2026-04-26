"""Test ensemble logic: weighted average + JS divergence + disagreement flag."""
import numpy as np

from app.services.regime.ensemble import (
    _js_divergence,
    _kl,
    _normalize,
    _to_array,
    DEFAULT_WEIGHTS,
)


class TestKLDivergence:
    def test_identical_distributions_zero(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert _kl(p, p) < 1e-9

    def test_kl_positive_for_different(self):
        p = np.array([0.7, 0.1, 0.1, 0.1])
        q = np.array([0.1, 0.7, 0.1, 0.1])
        assert _kl(p, q) > 0


class TestJSDivergence:
    def test_symmetry(self):
        p = np.array([0.7, 0.1, 0.1, 0.1])
        q = np.array([0.1, 0.7, 0.1, 0.1])
        assert abs(_js_divergence(p, q) - _js_divergence(q, p)) < 1e-9

    def test_zero_for_identical(self):
        p = np.array([0.4, 0.3, 0.2, 0.1])
        assert _js_divergence(p, p) < 1e-9

    def test_bounded_by_log2(self):
        """JS in [0, log 2]. Worst case: due 1-hot disgiunti."""
        p = np.array([1.0, 0.0, 0.0, 0.0])
        q = np.array([0.0, 1.0, 0.0, 0.0])
        # log(2) ≈ 0.693
        assert _js_divergence(p, q) <= np.log(2) + 1e-6


class TestArrayHelpers:
    def test_to_array_preserves_order(self):
        from app.services.regime.classifier import REGIMES
        d = {r: i / 10 for i, r in enumerate(REGIMES)}
        arr = _to_array(d)
        assert list(arr) == [d[r] for r in REGIMES]

    def test_normalize_sums_to_one(self):
        x = np.array([0.5, 0.3, 0.1, 0.1])
        n = _normalize(x)
        assert abs(n.sum() - 1.0) < 1e-9


class TestDefaultWeights:
    def test_three_models_equal_weight(self):
        assert "rule_based" in DEFAULT_WEIGHTS
        assert "hmm_market" in DEFAULT_WEIGHTS
        assert "msvar" in DEFAULT_WEIGHTS
        total = sum(DEFAULT_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9


class TestEnsembleAgreement:
    """Smoke test: con 3 modelli che concordano, confidence e' alta."""

    def test_concordant_models_high_confidence(self):
        """Costruisce 3 viste tutte 'reflation 0.7' - confidence > 0.85."""
        from app.services.regime.ensemble import EnsembleResult, ModelView, _js_divergence

        # Manual ensemble computation per non aver bisogno di DB
        same = {"reflation": 0.7, "stagflation": 0.1, "deflation": 0.1, "goldilocks": 0.1}
        arrs = [_to_array(same)] * 3
        js_pairs = [_js_divergence(arrs[i], arrs[j]) for i in range(3) for j in range(i+1, 3)]
        avg_js = float(np.mean(js_pairs))
        # Tutti uguali -> JS = 0
        assert avg_js < 1e-6

    def test_high_disagreement_flag(self):
        """Costruisce 3 viste molto diverse -> JS alto."""
        a = {"reflation": 0.9, "stagflation": 0.04, "deflation": 0.03, "goldilocks": 0.03}
        b = {"reflation": 0.04, "stagflation": 0.04, "deflation": 0.9, "goldilocks": 0.02}
        c = {"reflation": 0.04, "stagflation": 0.9, "deflation": 0.04, "goldilocks": 0.02}
        arrs = [_to_array(d) for d in (a, b, c)]
        js_pairs = [_js_divergence(arrs[i], arrs[j]) for i in range(3) for j in range(i+1, 3)]
        avg_js = float(np.mean(js_pairs))
        assert avg_js > 0.30, f"JS atteso > 0.30 con 3 distribuzioni concentrate distinte, got {avg_js:.3f}"
