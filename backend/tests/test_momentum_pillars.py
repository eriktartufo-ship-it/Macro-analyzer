"""T11 Momentum pillars TDD — Council 2026-05-27 verdict.

Aggiunge 5 pillar momentum-based al classifier per catturare ACCELERAZIONE
oltre al livello istantaneo. Flag opt-in `USE_MOMENTUM_PILLARS` (default OFF
finche walk-forward T10c-style validation passa).

Scenari sintetici:
- A: CPI 3.95 costante (level high but no momentum) → flag OFF=ON identici
- B: CPI 2.0→3.95 in 6m, GDP stabile → flag ON eleva stagflation +5-10pp
- C: GDP rallenta -1.0% in 6m, CPI stabile → flag ON aumenta stag/defla
- D: Inflation persistent (min last 6m > 2.5%) → cattura sticky inflation
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_flags(monkeypatch):
    """Disable ML blend + GDP collapse override per asserzioni deterministic."""
    for flag in ("USE_ML_REGIME_BLEND", "USE_GDP_COLLAPSE_OVERRIDE"):
        monkeypatch.setenv(flag, "0")
    monkeypatch.delenv("USE_FRESHNESS_WEIGHTING", raising=False)


def _base_indicators() -> dict[str, float]:
    """Indicators base 2026-05 live (CPI 3.95, GDP 0.49, unrate stabile)."""
    return {
        "gdp_roc": 0.49,
        "pmi": 56.77,
        "cpi_yoy": 3.95,
        "unrate": 4.3,
        "unrate_roc": 0.0,
        "yield_curve_10y2y": 0.49,
        "initial_claims_roc": -6.7,
        "lei_roc": -0.03,
        "fed_funds_rate": 3.64,
        "core_pce_yoy": 3.20,
        "payrolls_roc_12m": 0.16,
        "indpro_roc_12m": 1.35,
        "baa_spread": 1.59,
        "consumer_sentiment": 49.8,
        "vix": 16.59,
        "nfci": -0.523,
        "breakeven_10y": 2.4,
        "housing_starts_roc_12m": 4.64,
        "term_premium_10y": 0.81,
        "gdp_yoy_dfm": 5.0,
        # Momentum derivatives (default 0 = no acceleration)
        "cpi_yoy_change_6m": 0.0,
        "core_pce_yoy_change_6m": 0.0,
        "gdp_roc_change_6m": 0.0,
        "cpi_yoy_min_last_6m": 3.95,
        "gdp_yoy_dfm_change_3m": 0.0,
    }


class TestMomentumPillarsFlagOff:
    """Flag OFF (default): comportamento baseline level-driven invariato."""

    def test_flag_off_no_effect_on_stagflation(self, monkeypatch):
        from app.services.regime.classifier import classify_regime

        monkeypatch.delenv("USE_MOMENTUM_PILLARS", raising=False)
        ind = _base_indicators()
        # Anche con CPI accelerating +2%, flag OFF → no effect
        ind["cpi_yoy_change_6m"] = 2.0
        ind["core_pce_yoy_change_6m"] = 1.5
        out_no_momentum = classify_regime(ind)

        ind_flat = _base_indicators()
        out_flat = classify_regime(ind_flat)

        # Probabilita uguali a sigma 1e-9 (no leak del campo momentum)
        for k in out_flat["probabilities"]:
            assert out_no_momentum["probabilities"][k] == pytest.approx(
                out_flat["probabilities"][k], abs=1e-9
            )


class TestMomentumPillarsFlagOn:
    """Flag ON: pillar momentum aumentano stagflation/deflation in scenari accelerating."""

    def test_inflation_accelerating_lifts_stagflation(self, monkeypatch):
        """Scenario B: CPI 2.0→3.95 in 6m → stagflation +5-10pp."""
        monkeypatch.setenv("USE_MOMENTUM_PILLARS", "1")
        from app.services.regime.classifier import classify_regime

        ind_flat = _base_indicators()
        out_flat = classify_regime(ind_flat)

        ind_accel = _base_indicators()
        ind_accel["cpi_yoy_change_6m"] = 1.95  # 2.0 → 3.95 in 6m
        ind_accel["core_pce_yoy_change_6m"] = 1.4  # 1.8 → 3.2
        out_accel = classify_regime(ind_accel)

        stag_flat = out_flat["probabilities"]["stagflation"]
        stag_accel = out_accel["probabilities"]["stagflation"]

        assert stag_accel > stag_flat, (
            f"Inflation accelerating should lift stagflation. "
            f"Flat={stag_flat:.3f}, Accel={stag_accel:.3f}"
        )
        # Lift atteso 3-12pp (range realistic con rebalance)
        delta_pp = (stag_accel - stag_flat) * 100
        assert delta_pp >= 3.0, f"Lift {delta_pp:.1f}pp < 3pp (council target +5-10pp)"

    def test_gdp_decelerating_lifts_stagflation_and_deflation(self, monkeypatch):
        """Scenario C: GDP rallenta -1.0% in 6m → stag/defla aumentano."""
        monkeypatch.setenv("USE_MOMENTUM_PILLARS", "1")
        from app.services.regime.classifier import classify_regime

        ind_flat = _base_indicators()
        out_flat = classify_regime(ind_flat)

        ind_decel = _base_indicators()
        ind_decel["gdp_roc_change_6m"] = -1.0
        ind_decel["gdp_yoy_dfm_change_3m"] = -0.5
        out_decel = classify_regime(ind_decel)

        # Almeno UNO tra stag e defla deve essere piu alto
        stag_lift = out_decel["probabilities"]["stagflation"] - out_flat["probabilities"]["stagflation"]
        defl_lift = out_decel["probabilities"]["deflation"] - out_flat["probabilities"]["deflation"]
        assert stag_lift + defl_lift > 0.01, (
            f"GDP decelerating should lift stag or defla. "
            f"stag_lift={stag_lift*100:.2f}pp, defl_lift={defl_lift*100:.2f}pp"
        )

    def test_inflation_persistent_strengthens_stagflation(self, monkeypatch):
        """Scenario D: CPI min last 6m > 2.5% (sticky inflation)."""
        monkeypatch.setenv("USE_MOMENTUM_PILLARS", "1")
        from app.services.regime.classifier import classify_regime

        # Caso A: CPI volatile, min 1.5% (no persistence)
        ind_a = _base_indicators()
        ind_a["cpi_yoy_min_last_6m"] = 1.5
        out_a = classify_regime(ind_a)

        # Caso B: CPI sticky, min 3.0% (persistent)
        ind_b = _base_indicators()
        ind_b["cpi_yoy_min_last_6m"] = 3.0
        out_b = classify_regime(ind_b)

        assert out_b["probabilities"]["stagflation"] > out_a["probabilities"]["stagflation"], (
            "Persistent inflation should boost stagflation"
        )

    def test_combined_momentum_scenario_2026(self, monkeypatch):
        """Scenario realistic 2026: CPI accel + GDP slow + sticky → stagflation rises significantly.

        Il council ha previsto stagflation 18% → 24-30% range.
        """
        monkeypatch.setenv("USE_MOMENTUM_PILLARS", "1")
        from app.services.regime.classifier import classify_regime

        ind = _base_indicators()
        # Realistic momentum 2026
        ind["cpi_yoy_change_6m"] = 1.95
        ind["core_pce_yoy_change_6m"] = 1.4
        ind["gdp_roc_change_6m"] = -0.6
        ind["cpi_yoy_min_last_6m"] = 2.8
        ind["gdp_yoy_dfm_change_3m"] = -0.3

        out = classify_regime(ind)
        stag = out["probabilities"]["stagflation"]

        # Council target: 24-30% range con momentum attivo
        assert stag >= 0.20, (
            f"Combined momentum scenario should lift stagflation ≥ 20%. "
            f"Got stag={stag:.3f}, all probs={out['probabilities']}"
        )


class TestMomentumPillarsRegressionGuard:
    """Garantisce che il flag ON non rompa scenari storici noti."""

    def test_2018_q4_not_misclassified_as_stagflation(self, monkeypatch):
        """2018 Q4 era goldilocks-confused (sell-off mercato MA macro sano).
        Flag ON NON deve sopravvalutare stagflation.
        """
        monkeypatch.setenv("USE_MOMENTUM_PILLARS", "1")
        from app.services.regime.classifier import classify_regime

        # 2018 Q4: CPI 1.9% (flat), GDP 2.5% (decelerating from 4%), unrate stable
        ind = {
            "gdp_roc": 2.5, "pmi": 54.3, "cpi_yoy": 1.9, "unrate": 3.9,
            "unrate_roc": 0.0, "yield_curve_10y2y": 0.21, "initial_claims_roc": 5.0,
            "lei_roc": 0.1, "fed_funds_rate": 2.4, "core_pce_yoy": 1.9,
            "payrolls_roc_12m": 1.8, "indpro_roc_12m": 4.1, "baa_spread": 2.5,
            "consumer_sentiment": 95, "vix": 25.4, "breakeven_10y": 1.7,
            # Momentum: CPI flat, GDP decelerating ma da high baseline
            "cpi_yoy_change_6m": -0.1, "core_pce_yoy_change_6m": -0.1,
            "gdp_roc_change_6m": -1.5, "cpi_yoy_min_last_6m": 1.5,
        }
        out = classify_regime(ind)
        # Stagflation NON deve essere top regime
        assert out["regime"] != "stagflation", (
            f"2018 Q4 NOT stagflation. Got {out['regime']}, probs={out['probabilities']}"
        )

    def test_1973_oil_shock_remains_stagflation(self, monkeypatch):
        """1973 oil shock: CPI 8.7%, GDP -0.5%, OPEC. Stagflation classica.
        Flag ON deve preservare classification.
        """
        monkeypatch.setenv("USE_MOMENTUM_PILLARS", "1")
        from app.services.regime.classifier import classify_regime

        ind = {
            "gdp_roc": -0.5, "pmi": 46.0, "cpi_yoy": 8.7, "unrate": 5.5,
            "unrate_roc": 0.5, "yield_curve_10y2y": -0.5, "initial_claims_roc": 15.0,
            "lei_roc": -3.0, "fed_funds_rate": 10.0, "core_pce_yoy": 7.5,
            "payrolls_roc_12m": -0.5, "indpro_roc_12m": -3.0, "baa_spread": 2.8,
            "consumer_sentiment": 60, "vix": 22, "breakeven_10y": 4.0,
            # Momentum: CPI esploso (2%→8.7% in 12m, +3% in 6m), GDP collassato
            "cpi_yoy_change_6m": 3.0, "core_pce_yoy_change_6m": 2.5,
            "gdp_roc_change_6m": -2.5, "cpi_yoy_min_last_6m": 5.0,
        }
        out = classify_regime(ind)
        # Stagflation top regime + probability >= 0.5
        assert out["regime"] == "stagflation", (
            f"1973 oil shock = stagflation. Got {out['regime']}"
        )
        assert out["probabilities"]["stagflation"] >= 0.45, (
            f"1973 stagflation prob too low: {out['probabilities']['stagflation']:.3f}"
        )
