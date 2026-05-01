"""Test soft mapping stato MS-VAR -> distribuzione su regimi.

Il bug originale: `_map_states_via_correlation` faceva bigezione hard greedy.
Con n_states=2, 2 regimi su 4 erano sempre inaccessibili (massa solo da floor).
Il fix: distribuzione soft proporzionale alla correlazione clipped a 0,
con smoothing prior uniforme per evitare zero categorici.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.regime.classifier import REGIMES
from app.services.regime.msvar import _compute_state_to_regime_distribution


def _make_state_probs(values: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Helper: costruisci state_probs DataFrame (T, n_states)."""
    n_states = values.shape[1]
    return pd.DataFrame(values, index=dates, columns=[f"state_{i}" for i in range(n_states)])


def _make_rule_df(probs: dict[str, np.ndarray], dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Helper: costruisci rule_df con colonne REGIMES."""
    return pd.DataFrame(probs, index=dates)


class TestSoftMapping:
    def test_perfect_correlation_concentrates_mass(self):
        """Stato perfettamente correlato a 'deflation' deve assegnare ~tutta
        la massa a deflation, lasciando solo alpha/4 agli altri."""
        T = 60
        dates = pd.date_range("2020-01-31", periods=T, freq="ME")
        state_signal = np.linspace(0.1, 0.9, T)

        # Stato 0 cresce, stato 1 decresce (somma = 1)
        sp = _make_state_probs(np.column_stack([state_signal, 1 - state_signal]), dates)

        # rule-based deflation cresce in sincrono con stato 0; gli altri costanti
        rl = _make_rule_df({
            "reflation": np.full(T, 0.25),
            "stagflation": np.full(T, 0.25),
            "deflation": state_signal,
            "goldilocks": 1 - state_signal - 0.5,
        }, dates)

        out = _compute_state_to_regime_distribution(sp, rl, alpha=0.05)

        # Stato 0 dovrebbe concentrare massa su deflation (corr ~1.0)
        assert out[0]["deflation"] > 0.85
        # Gli altri regimi non devono andare a zero (smoothing prior)
        for r in ("reflation", "stagflation", "goldilocks"):
            assert out[0][r] >= 0.0125 - 1e-9  # alpha/4 = 0.0125
        # Distribuzione deve sommare a 1
        assert abs(sum(out[0].values()) - 1.0) < 1e-9

    def test_two_states_keeps_all_four_regimes_accessible(self):
        """Con n_states=2 nessun regime deve avere probabilita' 0 hard:
        ogni regime riceve almeno alpha/n_regimes via smoothing prior.
        Questo e' il fix del bug strutturale.
        """
        T = 60
        dates = pd.date_range("2020-01-31", periods=T, freq="ME")
        # Stato 0 perfettamente correlato a reflation
        # Stato 1 perfettamente correlato a deflation
        signal = np.linspace(0.0, 1.0, T)
        sp = _make_state_probs(np.column_stack([signal, 1 - signal]), dates)
        rl = _make_rule_df({
            "reflation": signal,
            "stagflation": np.full(T, 0.0),
            "deflation": 1 - signal,
            "goldilocks": np.full(T, 0.0),
        }, dates)

        out = _compute_state_to_regime_distribution(sp, rl, alpha=0.05)

        # Ogni stato e' un dict completo sui 4 regimi (non solo 1)
        for s in range(2):
            assert set(out[s].keys()) == set(REGIMES)
            # Stagflation/goldilocks (corr 0) non vanno a 0 hard
            assert out[s]["stagflation"] > 0
            assert out[s]["goldilocks"] > 0

    def test_split_correlation_distributes_proportionally(self):
        """Stato correlato 50/50 con stagflation+deflation distribuisce
        massa quasi paritaria tra i due."""
        T = 60
        dates = pd.date_range("2020-01-31", periods=T, freq="ME")
        signal = np.linspace(0.1, 0.9, T)
        sp = _make_state_probs(np.column_stack([signal, 1 - signal]), dates)

        # Sia stagflation che deflation crescono col signal (corr alta entrambe)
        # mentre reflation/goldilocks restano costanti
        rl = _make_rule_df({
            "reflation": np.full(T, 0.1),
            "stagflation": signal * 0.45 + 0.05,
            "deflation": signal * 0.45 + 0.05,
            "goldilocks": np.full(T, 0.1),
        }, dates)

        out = _compute_state_to_regime_distribution(sp, rl, alpha=0.05)

        # Stato 0: stag e defl hanno massa simile (entrambi ~0.45 dopo norm)
        assert out[0]["stagflation"] == pytest.approx(out[0]["deflation"], abs=0.02)
        assert out[0]["stagflation"] > 0.40
        assert out[0]["deflation"] > 0.40

    def test_negative_correlations_clipped_to_zero(self):
        """Correlazione negativa non deve dare peso negativo: deve essere
        clipped a 0, lasciando solo lo smoothing prior."""
        T = 60
        dates = pd.date_range("2020-01-31", periods=T, freq="ME")
        signal = np.linspace(0.1, 0.9, T)
        sp = _make_state_probs(np.column_stack([signal, 1 - signal]), dates)

        # reflation perfettamente anti-correlato con stato 0
        rl = _make_rule_df({
            "reflation": 1 - signal,
            "stagflation": signal,
            "deflation": np.full(T, 0.0),
            "goldilocks": np.full(T, 0.0),
        }, dates)

        out = _compute_state_to_regime_distribution(sp, rl, alpha=0.05)

        # Stato 0 deve dare massa a stagflation (corr +1), NON a reflation (corr -1)
        assert out[0]["stagflation"] > out[0]["reflation"]
        # Reflation deve avere solo lo smoothing prior alpha/4 (anti-correlato → clip 0)
        assert out[0]["reflation"] == pytest.approx(0.05 / 4, abs=1e-9)

    def test_no_correlation_falls_back_uniform(self):
        """Se nessun regime ha correlazione positiva (tutti zero o costanti),
        ritorna distribuzione uniforme su tutti i regimi."""
        T = 60
        dates = pd.date_range("2020-01-31", periods=T, freq="ME")
        signal = np.linspace(0.0, 1.0, T)
        sp = _make_state_probs(np.column_stack([signal, 1 - signal]), dates)

        # Tutti i regimi costanti → corr non definita o zero
        rl = _make_rule_df({r: np.full(T, 0.25) for r in REGIMES}, dates)

        out = _compute_state_to_regime_distribution(sp, rl, alpha=0.05)

        for s in range(2):
            for r in REGIMES:
                assert out[s][r] == pytest.approx(0.25, abs=1e-9)

    def test_insufficient_data_returns_uniform(self):
        """Con < 12 osservazioni comuni il fit della correlazione non e' affidabile:
        ritorna distribuzione uniforme su tutti i regimi (no scelte arbitrarie)."""
        T = 8
        dates = pd.date_range("2020-01-31", periods=T, freq="ME")
        signal = np.linspace(0.0, 1.0, T)
        sp = _make_state_probs(np.column_stack([signal, 1 - signal]), dates)
        rl = _make_rule_df({r: np.full(T, 0.25) for r in REGIMES}, dates)

        out = _compute_state_to_regime_distribution(sp, rl, alpha=0.05)

        for s in range(2):
            for r in REGIMES:
                assert out[s][r] == pytest.approx(0.25, abs=1e-9)

    def test_distributions_sum_to_one_per_state(self):
        """Per ogni stato la distribuzione su 4 regimi deve sommare esattamente 1."""
        T = 60
        dates = pd.date_range("2020-01-31", periods=T, freq="ME")
        rng = np.random.default_rng(42)
        signal = rng.random(T)
        sp = _make_state_probs(np.column_stack([signal, 1 - signal]), dates)
        rl = _make_rule_df({r: rng.random(T) for r in REGIMES}, dates)

        out = _compute_state_to_regime_distribution(sp, rl, alpha=0.05)

        for s in range(2):
            assert abs(sum(out[s].values()) - 1.0) < 1e-9
