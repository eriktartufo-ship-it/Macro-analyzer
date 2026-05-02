"""Test soft mapping HMM/MS-VAR state -> regime distribution.

Due varianti:
1. **corr-based** (compute_state_to_regime_distribution_corr):
   per HMM-Market e MS-VAR. Distribuzione proporzionale a correlazione
   clipped a 0 con i posterior rule-based, smoothing prior uniforme.
2. **freq-based** (compute_state_to_regime_distribution_freq):
   per HMM macro. Distribuzione = frequenza empirica dei label rule-based
   per ogni stato + smoothing prior uniforme.

Entrambi risolvono il bug pre-fix del mapping hard 1-a-1 che lasciava regimi
inaccessibili o etichettava in modo arbitrario stati con correlazione debole.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.regime.classifier import REGIMES
from app.services.regime.state_mapping import (
    compute_state_to_regime_distribution_corr,
    compute_state_to_regime_distribution_freq,
)


def _make_state_probs(values: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    n_states = values.shape[1]
    return pd.DataFrame(values, index=dates, columns=[f"state_{i}" for i in range(n_states)])


def _make_rule_df(probs: dict[str, np.ndarray], dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(probs, index=dates)


# ============================================================================
# Variante CORR-based (HMM-Market + MS-VAR)
# ============================================================================


class TestCorrMapping:
    def test_perfect_correlation_concentrates_mass(self):
        T = 60
        dates = pd.date_range("2020-01-31", periods=T, freq="ME")
        state_signal = np.linspace(0.1, 0.9, T)
        sp = _make_state_probs(np.column_stack([state_signal, 1 - state_signal]), dates)
        rl = _make_rule_df({
            "reflation": np.full(T, 0.25),
            "stagflation": np.full(T, 0.25),
            "deflation": state_signal,
            "goldilocks": 1 - state_signal - 0.5,
        }, dates)

        out = compute_state_to_regime_distribution_corr(sp, rl, alpha=0.05)

        assert out[0]["deflation"] > 0.85
        for r in ("reflation", "stagflation", "goldilocks"):
            assert out[0][r] >= 0.0125 - 1e-9
        assert abs(sum(out[0].values()) - 1.0) < 1e-9

    def test_two_states_keeps_all_four_regimes_accessible(self):
        T = 60
        dates = pd.date_range("2020-01-31", periods=T, freq="ME")
        signal = np.linspace(0.0, 1.0, T)
        sp = _make_state_probs(np.column_stack([signal, 1 - signal]), dates)
        rl = _make_rule_df({
            "reflation": signal,
            "stagflation": np.full(T, 0.0),
            "deflation": 1 - signal,
            "goldilocks": np.full(T, 0.0),
        }, dates)

        out = compute_state_to_regime_distribution_corr(sp, rl, alpha=0.05)

        for s in range(2):
            assert set(out[s].keys()) == set(REGIMES)
            assert out[s]["stagflation"] > 0
            assert out[s]["goldilocks"] > 0

    def test_split_correlation_distributes_proportionally(self):
        T = 60
        dates = pd.date_range("2020-01-31", periods=T, freq="ME")
        signal = np.linspace(0.1, 0.9, T)
        sp = _make_state_probs(np.column_stack([signal, 1 - signal]), dates)
        rl = _make_rule_df({
            "reflation": np.full(T, 0.1),
            "stagflation": signal * 0.45 + 0.05,
            "deflation": signal * 0.45 + 0.05,
            "goldilocks": np.full(T, 0.1),
        }, dates)

        out = compute_state_to_regime_distribution_corr(sp, rl, alpha=0.05)

        assert out[0]["stagflation"] == pytest.approx(out[0]["deflation"], abs=0.02)
        assert out[0]["stagflation"] > 0.40
        assert out[0]["deflation"] > 0.40

    def test_negative_correlations_clipped_to_zero(self):
        T = 60
        dates = pd.date_range("2020-01-31", periods=T, freq="ME")
        signal = np.linspace(0.1, 0.9, T)
        sp = _make_state_probs(np.column_stack([signal, 1 - signal]), dates)
        rl = _make_rule_df({
            "reflation": 1 - signal,
            "stagflation": signal,
            "deflation": np.full(T, 0.0),
            "goldilocks": np.full(T, 0.0),
        }, dates)

        out = compute_state_to_regime_distribution_corr(sp, rl, alpha=0.05)

        assert out[0]["stagflation"] > out[0]["reflation"]
        assert out[0]["reflation"] == pytest.approx(0.05 / 4, abs=1e-9)

    def test_no_correlation_falls_back_uniform(self):
        T = 60
        dates = pd.date_range("2020-01-31", periods=T, freq="ME")
        signal = np.linspace(0.0, 1.0, T)
        sp = _make_state_probs(np.column_stack([signal, 1 - signal]), dates)
        rl = _make_rule_df({r: np.full(T, 0.25) for r in REGIMES}, dates)

        out = compute_state_to_regime_distribution_corr(sp, rl, alpha=0.05)

        for s in range(2):
            for r in REGIMES:
                assert out[s][r] == pytest.approx(0.25, abs=1e-9)

    def test_insufficient_data_returns_uniform(self):
        T = 8
        dates = pd.date_range("2020-01-31", periods=T, freq="ME")
        signal = np.linspace(0.0, 1.0, T)
        sp = _make_state_probs(np.column_stack([signal, 1 - signal]), dates)
        rl = _make_rule_df({r: np.full(T, 0.25) for r in REGIMES}, dates)

        out = compute_state_to_regime_distribution_corr(sp, rl, alpha=0.05)

        for s in range(2):
            for r in REGIMES:
                assert out[s][r] == pytest.approx(0.25, abs=1e-9)

    def test_distributions_sum_to_one_per_state(self):
        T = 60
        dates = pd.date_range("2020-01-31", periods=T, freq="ME")
        rng = np.random.default_rng(42)
        signal = rng.random(T)
        sp = _make_state_probs(np.column_stack([signal, 1 - signal]), dates)
        rl = _make_rule_df({r: rng.random(T) for r in REGIMES}, dates)

        out = compute_state_to_regime_distribution_corr(sp, rl, alpha=0.05)

        for s in range(2):
            assert abs(sum(out[s].values()) - 1.0) < 1e-9


# ============================================================================
# Variante FREQ-based (HMM macro)
# ============================================================================


class TestFreqMapping:
    def test_pure_state_concentrates_mass(self):
        """Stato con 100% obs label='stagflation' → quasi tutta la massa
        su stagflation, alpha/4 sugli altri."""
        states = np.zeros(20, dtype=int)
        labels = ["stagflation"] * 20

        out = compute_state_to_regime_distribution_freq(
            states, labels, n_states=1, alpha=0.05,
        )

        assert out[0]["stagflation"] > 0.95
        for r in ("reflation", "deflation", "goldilocks"):
            assert out[0][r] == pytest.approx(0.05 / 4, abs=1e-9)
        assert abs(sum(out[0].values()) - 1.0) < 1e-9

    def test_split_50_50_distributes_proportionally(self):
        """Stato con 50% reflation + 50% goldilocks → distribuzione paritaria
        smoothed (no scelta arbitraria)."""
        states = np.zeros(10, dtype=int)
        labels = ["reflation"] * 5 + ["goldilocks"] * 5

        out = compute_state_to_regime_distribution_freq(
            states, labels, n_states=1, alpha=0.05,
        )

        expected_top = (1 - 0.05) * 0.5 + 0.05 / 4
        expected_other = 0.05 / 4
        assert out[0]["reflation"] == pytest.approx(expected_top)
        assert out[0]["goldilocks"] == pytest.approx(expected_top)
        assert out[0]["stagflation"] == pytest.approx(expected_other)
        assert out[0]["deflation"] == pytest.approx(expected_other)

    def test_two_states_keeps_all_four_regimes_accessible(self):
        """Con n_states=2 ogni regime riceve almeno alpha/n_regimes,
        anche se mai osservato in quello stato."""
        states = np.array([0] * 5 + [1] * 5)
        labels = ["reflation"] * 5 + ["deflation"] * 5

        out = compute_state_to_regime_distribution_freq(
            states, labels, n_states=2, alpha=0.05,
        )

        for s in range(2):
            assert set(out[s].keys()) == set(REGIMES)
            for r in REGIMES:
                assert out[s][r] >= 0.05 / 4 - 1e-9

    def test_concentration_greedy_artifact_resolved(self):
        """Replica il bug pre-fix HMM macro state 3:
        51% reflation + 37% goldilocks + 9% stagflation + 3% deflation.
        Con il vecchio mapping concentration-greedy + bigezione hard,
        il regime "reflation" veniva rubato da uno stato precedente
        e questo stato finiva etichettato 'goldilocks' (seconda preferenza).

        Col soft mapping freq-based, la distribuzione mantiene la
        proporzione vera: refl > gold > stag > defl.
        """
        n = 100
        states = np.zeros(n, dtype=int)
        labels = (
            ["reflation"] * 51 +
            ["goldilocks"] * 37 +
            ["stagflation"] * 9 +
            ["deflation"] * 3
        )

        out = compute_state_to_regime_distribution_freq(
            states, labels, n_states=1, alpha=0.05,
        )

        # Ordine corretto preservato
        assert out[0]["reflation"] > out[0]["goldilocks"] > out[0]["stagflation"] > out[0]["deflation"]
        # Reflation domina, NON goldilocks
        assert out[0]["reflation"] > 0.45
        assert out[0]["goldilocks"] < out[0]["reflation"] - 0.10

    def test_empty_state_falls_back_uniform(self):
        """Stato senza osservazioni → distribuzione uniforme."""
        states = np.zeros(3, dtype=int)
        labels = ["reflation"] * 3

        out = compute_state_to_regime_distribution_freq(
            states, labels, n_states=2, alpha=0.05,
        )

        # State 1 non ha osservazioni → uniforme
        for r in REGIMES:
            assert out[1][r] == pytest.approx(1.0 / len(REGIMES))

    def test_distribution_sums_to_one_per_state(self):
        """La distribuzione per ogni stato deve sommare a 1."""
        rng = np.random.default_rng(42)
        n_states = 4
        states = rng.integers(0, n_states, size=200)
        labels = list(rng.choice(REGIMES, size=200))

        out = compute_state_to_regime_distribution_freq(
            states, labels, n_states=n_states, alpha=0.05,
        )

        for s in range(n_states):
            assert abs(sum(out[s].values()) - 1.0) < 1e-9
