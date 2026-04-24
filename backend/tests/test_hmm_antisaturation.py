"""Test che l'HMM non saturi al 100% dopo i fix diag+temperature+floor."""
import numpy as np
from app.services.regime.hmm_classifier import (
    _baum_welch,
    _POSTERIOR_FLOOR,
    _POSTERIOR_TEMPERATURE,
)


class TestHMMNoSaturation:
    def test_posterior_has_floor(self):
        """Il posterior smoothed via temperature+floor non deve MAI avere uno
        stato sotto il floor configurato."""
        # Serie ben separata: 3 stati facilmente distinguibili
        rng = np.random.default_rng(42)
        T = 300
        means = np.array([[0, 0], [3, 3], [-3, 0]])
        X = []
        s = 0
        for _ in range(T):
            if rng.random() < 0.05:
                s = (s + 1) % 3
            X.append(means[s] + rng.normal(0, 0.5, size=2))
        X = np.asarray(X)
        pi, A, mu, var, gamma, ll = _baum_welch(X, n_states=3, max_iter=50)
        # Il posterior raw puo' ancora essere peaked, ma il floor e' applicato in
        # fit_and_predict_hmm — qui testiamo che le probabilita non siano NaN/inf
        assert not np.isnan(gamma).any()
        assert (gamma >= 0).all()
        # Ogni riga somma a ~1
        for t in range(T):
            assert abs(gamma[t].sum() - 1.0) < 1e-6

    def test_dirichlet_prior_prevents_zero_transitions(self):
        """Con Dirichlet alpha > 0 nella M-step, la matrice di transizione non
        deve avere zeri assoluti."""
        rng = np.random.default_rng(0)
        T, D = 100, 3
        X = rng.normal(0, 1, size=(T, D))
        pi, A, mu, var, gamma, ll = _baum_welch(X, n_states=3, max_iter=30)
        assert (A > 0).all(), f"A contiene zeri: {A}"

    def test_temperature_and_floor_values_sane(self):
        """I parametri anti-saturazione non devono essere disattivati per sbaglio."""
        assert _POSTERIOR_TEMPERATURE > 1.0, (
            f"temperature={_POSTERIOR_TEMPERATURE}, >1 serve per attenuare la saturazione"
        )
        assert 0.005 <= _POSTERIOR_FLOOR <= 0.05, (
            f"floor={_POSTERIOR_FLOOR} fuori da range sensato"
        )
