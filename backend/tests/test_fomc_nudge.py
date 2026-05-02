"""Test FOMC nudge applicato alle regime probabilities.

Tier 1.2 della roadmap Bridgewater. Il dict `regime_implication` estratto da
Gemini per ogni FOMC document (es. {reflation: -0.05, stagflation: +0.05, ...})
viene applicato come post-correzione alle probability del meta-classifier:

  P_final(r) = renormalize( P_meta(r) + nudge(r) · confidence · time_decay )

Il decay temporale (half-life 30d) impedisce a uno statement vecchio di
influenzare il regime per sempre.
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from app.services.regime.classifier import REGIMES
from app.services.regime.fomc_nudge import (
    apply_fomc_nudge,
    compute_decay_factor,
)


class TestComputeDecayFactor:
    """Half-life 30d: factor=1.0 se age=0d, 0.5 se age=30d, 0.25 se age=60d."""

    def test_today_no_decay(self):
        assert compute_decay_factor(age_days=0, half_life_days=30) == pytest.approx(1.0)

    def test_one_half_life(self):
        assert compute_decay_factor(age_days=30, half_life_days=30) == pytest.approx(0.5)

    def test_two_half_lives(self):
        assert compute_decay_factor(age_days=60, half_life_days=30) == pytest.approx(0.25)

    def test_negative_age_clamped_to_one(self):
        """Age negativo (timestamp futuro) → factor=1.0 (no boost)."""
        assert compute_decay_factor(age_days=-5, half_life_days=30) == pytest.approx(1.0)

    def test_very_old_decays_to_zero(self):
        """Age >> half_life → factor → 0."""
        assert compute_decay_factor(age_days=365, half_life_days=30) < 0.001


class TestApplyFomcNudge:
    """Pipeline completa: P_meta + nudge × confidence × decay → P_final."""

    def test_no_nudge_returns_input(self):
        """Tutti nudge=0 → output identico all'input."""
        probs = {"reflation": 0.4, "stagflation": 0.3, "deflation": 0.2, "goldilocks": 0.1}
        out = apply_fomc_nudge(
            probs,
            nudge={r: 0.0 for r in REGIMES},
            confidence=0.8,
            doc_date=date.today(),
        )
        for r in REGIMES:
            assert out[r] == pytest.approx(probs[r], abs=1e-9)

    def test_zero_confidence_no_effect(self):
        """confidence=0 azzera l'effetto del nudge."""
        probs = {"reflation": 0.4, "stagflation": 0.3, "deflation": 0.2, "goldilocks": 0.1}
        nudge = {"reflation": -0.10, "stagflation": +0.10, "deflation": 0.0, "goldilocks": 0.0}
        out = apply_fomc_nudge(probs, nudge=nudge, confidence=0.0, doc_date=date.today())
        for r in REGIMES:
            assert out[r] == pytest.approx(probs[r], abs=1e-9)

    def test_full_confidence_today_full_nudge(self):
        """confidence=1, age=0 → applica nudge intero. Output renormalizzato Σ=1."""
        probs = {"reflation": 0.4, "stagflation": 0.3, "deflation": 0.2, "goldilocks": 0.1}
        nudge = {"reflation": -0.10, "stagflation": +0.10, "deflation": 0.0, "goldilocks": 0.0}
        out = apply_fomc_nudge(probs, nudge=nudge, confidence=1.0, doc_date=date.today())

        # Stagflation è cresciuta vs reflation ridotta
        assert out["stagflation"] > probs["stagflation"]
        assert out["reflation"] < probs["reflation"]
        # Σ=1
        assert sum(out.values()) == pytest.approx(1.0, abs=1e-9)

    def test_old_doc_decays_nudge(self):
        """Doc vecchio 60d (2 half-life) → effetto ridotto a 25%."""
        probs = {"reflation": 0.4, "stagflation": 0.3, "deflation": 0.2, "goldilocks": 0.1}
        nudge = {"reflation": 0.0, "stagflation": +0.20, "deflation": 0.0, "goldilocks": -0.20}
        old_date = date.today() - timedelta(days=60)
        out = apply_fomc_nudge(probs, nudge=nudge, confidence=1.0, doc_date=old_date,
                               half_life_days=30)

        # Effetto su stagflation deve essere meno di +0.20 (decay 0.25)
        delta_stag = out["stagflation"] - probs["stagflation"]
        # Il nudge effettivo è +0.20 * 1.0 * 0.25 = +0.05 raw, poi renormalizzato
        # Sta tra 0.04 e 0.06 dopo renormalize
        assert 0.02 < delta_stag < 0.08

    def test_clamp_negative_to_zero(self):
        """Nudge che porterebbe una prob a < 0 viene clamped a 0 prima del renormalize."""
        probs = {"reflation": 0.05, "stagflation": 0.40, "deflation": 0.40, "goldilocks": 0.15}
        # Nudge gigante negativo su reflation
        nudge = {"reflation": -0.20, "stagflation": 0.10, "deflation": 0.05, "goldilocks": 0.05}
        out = apply_fomc_nudge(probs, nudge=nudge, confidence=1.0, doc_date=date.today())

        # Reflation non scende sotto zero
        assert out["reflation"] >= 0.0
        # Σ = 1 (renormalizzato)
        assert sum(out.values()) == pytest.approx(1.0, abs=1e-9)

    def test_output_renormalized_to_one(self):
        """Anche con nudge che NON sommano a 0, output renormalizzato Σ=1."""
        probs = {"reflation": 0.25, "stagflation": 0.25, "deflation": 0.25, "goldilocks": 0.25}
        nudge = {"reflation": +0.10, "stagflation": +0.10, "deflation": -0.05, "goldilocks": 0.0}
        out = apply_fomc_nudge(probs, nudge=nudge, confidence=1.0, doc_date=date.today())
        assert sum(out.values()) == pytest.approx(1.0, abs=1e-9)

    def test_none_nudge_returns_input(self):
        """nudge=None → output identico all'input (no-op safe)."""
        probs = {"reflation": 0.4, "stagflation": 0.3, "deflation": 0.2, "goldilocks": 0.1}
        out = apply_fomc_nudge(probs, nudge=None, confidence=0.8, doc_date=date.today())
        for r in REGIMES:
            assert out[r] == pytest.approx(probs[r], abs=1e-9)
