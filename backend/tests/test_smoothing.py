"""Test per il modulo smoothing forward-backward."""
import numpy as np

from app.services.regime.smoothing import smooth_probabilities_sequence


class TestSmoothProbabilities:
    def test_output_rows_sum_to_one(self):
        """Ogni posterior smoothed deve sommare a 1."""
        T, K = 50, 4
        rng = np.random.default_rng(0)
        raw = rng.dirichlet(np.ones(K), size=T)
        A = np.full((K, K), 0.2)
        np.fill_diagonal(A, 0.4)
        A = A / A.sum(axis=1, keepdims=True)
        out = smooth_probabilities_sequence(raw, A)
        assert out.shape == (T, K)
        for t in range(T):
            assert abs(out[t].sum() - 1.0) < 1e-6

    def test_identity_raw_preserved(self):
        """Se raw e' gia 1-hot stabile, smoothing non dovrebbe cambiarlo drasticamente
        (matrice persistente)."""
        T, K = 30, 4
        raw = np.full((T, K), 1e-4)
        raw[:, 0] = 1.0 - 3e-4  # tutti su stato 0
        A = np.eye(K) * 0.9 + 0.1 / K  # highly persistent
        A = A / A.sum(axis=1, keepdims=True)
        out = smooth_probabilities_sequence(raw, A)
        # Stato 0 deve rimanere dominante
        assert (out[:, 0] > 0.8).all()

    def test_isolated_noise_attenuated(self):
        """Un punto rumoroso moderatamente convinto (60/40) in una serie persistente
        deve essere attenuato: la prob per lo stato dominante della serie deve
        aumentare dopo lo smoothing."""
        T, K = 21, 4
        raw = np.full((T, K), 0.05)
        raw[:, 0] = 0.85  # serie dominata da stato 0 (prob 0.85 per stato 0, 0.05 altri)
        raw = raw / raw.sum(axis=1, keepdims=True)
        # Punto rumoroso al mezzo: 60% stato 1, 20% stato 0, 10%+10%
        raw[10] = np.array([0.20, 0.60, 0.10, 0.10])
        A = np.eye(K) * 0.85 + 0.15 / K
        A = A / A.sum(axis=1, keepdims=True)
        out = smooth_probabilities_sequence(raw, A)
        # Prob stato 0 deve essere aumentata rispetto al raw nel punto 10
        assert out[10, 0] > raw[10, 0], (
            f"raw[10, 0]={raw[10, 0]:.3f}, smoothed[10, 0]={out[10, 0]:.3f} — "
            f"smoothing non ha attenuato il rumore"
        )
        # Prob stato 1 deve essere diminuita
        assert out[10, 1] < raw[10, 1], (
            f"raw[10, 1]={raw[10, 1]:.3f}, smoothed[10, 1]={out[10, 1]:.3f}"
        )

    def test_transition_detected(self):
        """Transizione vera: primi 10 punti stato 0, ultimi 10 punti stato 2.
        Smoothing deve ancora riconoscere il cambio (non appiattire tutto a una media)."""
        T, K = 20, 4
        raw = np.full((T, K), 1e-4)
        raw[:10, 0] = 0.997
        raw[10:, 2] = 0.997
        A = np.eye(K) * 0.7 + 0.3 / K
        A = A / A.sum(axis=1, keepdims=True)
        out = smooth_probabilities_sequence(raw, A)
        # Prima meta' deve restare dominata da stato 0, seconda da stato 2
        assert out[2, 0] > 0.7
        assert out[17, 2] > 0.7
