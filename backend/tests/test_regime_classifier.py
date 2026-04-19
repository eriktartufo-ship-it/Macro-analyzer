"""Test TDD per il regime classifier a 4 regimi.

Quadrante macro (crescita x inflazione):
- Reflation:   crescita forte + inflazione in salita
- Stagflation: crescita debole + inflazione alta
- Deflation:   crescita debole + inflazione bassa/in calo
- Goldilocks:  crescita moderata + inflazione bassa
"""

import pytest


class TestRegimeClassifier:
    """Test classificazione regime con scenari macro noti."""

    def test_probabilities_sum_to_one(self):
        """La somma delle probabilita deve essere 1.0."""
        from app.services.regime.classifier import classify_regime

        indicators = {
            "gdp_roc": 2.5,
            "pmi": 55.0,
            "cpi_yoy": 2.0,
            "unrate": 4.0,
            "unrate_roc": -0.1,
            "yield_curve_10y2y": 1.0,
            "initial_claims_roc": -5.0,
            "lei_roc": 0.5,
            "fed_funds_rate": 2.0,
        }
        result = classify_regime(indicators)

        total = sum(result["probabilities"].values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_confidence_between_0_and_1(self):
        """Il confidence score deve essere tra 0 e 1."""
        from app.services.regime.classifier import classify_regime

        indicators = {
            "gdp_roc": 2.5,
            "pmi": 55.0,
            "cpi_yoy": 2.0,
            "unrate": 4.0,
            "unrate_roc": -0.1,
            "yield_curve_10y2y": 1.0,
            "initial_claims_roc": -5.0,
            "lei_roc": 0.5,
            "fed_funds_rate": 2.0,
        }
        result = classify_regime(indicators)

        assert 0.0 <= result["confidence"] <= 1.0

    def test_output_structure(self):
        """L'output deve avere la struttura corretta con 4 regimi."""
        from app.services.regime.classifier import classify_regime

        indicators = {
            "gdp_roc": 2.0,
            "pmi": 52.0,
            "cpi_yoy": 3.0,
            "unrate": 4.5,
            "unrate_roc": 0.0,
            "yield_curve_10y2y": 0.5,
            "initial_claims_roc": 0.0,
            "lei_roc": 0.0,
            "fed_funds_rate": 3.0,
        }
        result = classify_regime(indicators)

        assert "regime" in result
        assert "probabilities" in result
        assert "confidence" in result
        assert "conditions_detail" in result

        expected_regimes = {"reflation", "stagflation", "deflation", "goldilocks"}
        assert set(result["probabilities"].keys()) == expected_regimes

    def test_reflation_scenario(self):
        """GDP forte + PMI alto + inflation in salita = Reflation.

        Tipico di una economia in forte espansione con pressioni inflazionistiche
        (2021 post-COVID recovery, late-cycle boom).
        """
        from app.services.regime.classifier import classify_regime

        indicators = {
            "gdp_roc": 3.5,          # Forte crescita
            "pmi": 58.0,             # Espansione
            "cpi_yoy": 3.5,          # Inflazione in salita
            "unrate": 3.8,           # Basso
            "unrate_roc": -0.3,      # In calo
            "yield_curve_10y2y": 1.5, # Normale/steepening
            "initial_claims_roc": -8.0,  # In calo
            "lei_roc": 1.2,          # Positivo
            "fed_funds_rate": 2.5,
        }
        result = classify_regime(indicators)

        assert result["regime"] == "reflation"
        assert result["probabilities"]["reflation"] > 0.3

    def test_reflation_recovery_style(self):
        """GDP tornando positivo + PMI recovering + credit easing = Reflation.

        Tipico early-cycle recovery con stimolo monetario.
        """
        from app.services.regime.classifier import classify_regime

        indicators = {
            "gdp_roc": 2.0,          # Tornando positivo
            "pmi": 52.0,             # Appena sopra 50 (recovering)
            "cpi_yoy": 3.0,          # Inflazione moderata-alta
            "unrate": 5.5,           # Ancora alto ma in calo
            "unrate_roc": -0.5,      # In netto calo
            "yield_curve_10y2y": 2.0, # Steep (tipico recovery)
            "initial_claims_roc": -15.0,  # In forte calo
            "lei_roc": 2.0,          # Fortemente positivo
            "fed_funds_rate": 0.5,   # Molto accomodante
        }
        result = classify_regime(indicators)

        assert result["regime"] == "reflation"
        assert result["probabilities"]["reflation"] > 0.3

    def test_stagflation_scenario_2022(self):
        """GDP basso + inflation alta + unemployment in salita = Stagflation (2022-style)."""
        from app.services.regime.classifier import classify_regime

        indicators = {
            "gdp_roc": 0.5,          # Quasi flat
            "pmi": 49.0,             # Borderline
            "cpi_yoy": 8.5,          # Inflation molto alta
            "unrate": 4.5,           # Moderato
            "unrate_roc": 0.3,       # Leggero aumento
            "yield_curve_10y2y": -0.2, # Quasi invertita
            "initial_claims_roc": 10.0, # In salita
            "lei_roc": -1.0,         # Negativo
            "fed_funds_rate": 5.0,   # Restrittivo
        }
        result = classify_regime(indicators)

        assert result["regime"] == "stagflation"
        assert result["probabilities"]["stagflation"] > 0.3

    def test_deflation_recession_scenario(self):
        """GDP negativo + PMI sotto 50 + inflation bassa = Deflation.

        Tipico di recessione con pressioni deflazionistiche (2008, COVID crash).
        """
        from app.services.regime.classifier import classify_regime

        indicators = {
            "gdp_roc": -2.0,         # Contrazione
            "pmi": 44.0,             # Sotto 50
            "cpi_yoy": 1.5,          # Bassa
            "unrate": 6.5,           # Alto
            "unrate_roc": 1.5,       # In salita
            "yield_curve_10y2y": -0.5, # Invertita
            "initial_claims_roc": 25.0, # Spike
            "lei_roc": -3.0,         # Negativo
            "fed_funds_rate": 4.0,
        }
        result = classify_regime(indicators)

        assert result["regime"] == "deflation"
        assert result["probabilities"]["deflation"] > 0.3

    def test_deflation_slowdown_scenario(self):
        """GDP decelerante + PMI in calo + inflation moderata = Deflation.

        Tipico di slowdown/late-cycle con forze deflazionistiche.
        """
        from app.services.regime.classifier import classify_regime

        indicators = {
            "gdp_roc": 0.8,          # Positivo ma in calo
            "pmi": 48.0,             # Sotto 50
            "cpi_yoy": 2.0,          # Bassa/moderata
            "unrate": 4.5,           # In lieve aumento
            "unrate_roc": 0.3,       # Leggero aumento
            "yield_curve_10y2y": 0.2, # Piatta
            "initial_claims_roc": 5.0, # In lieve salita
            "lei_roc": -1.5,         # Negativo
            "fed_funds_rate": 4.5,
        }
        result = classify_regime(indicators)

        assert result["regime"] == "deflation"
        assert result["probabilities"]["deflation"] > 0.25

    def test_goldilocks_scenario(self):
        """GDP moderato + inflation bassa + unemployment basso = Goldilocks.

        Il "miglior scenario possibile": crescita sana senza pressioni inflazionistiche.
        """
        from app.services.regime.classifier import classify_regime

        indicators = {
            "gdp_roc": 2.2,          # Moderato
            "pmi": 54.0,             # Espansione moderata
            "cpi_yoy": 1.8,          # Sotto target
            "unrate": 3.5,           # Molto basso
            "unrate_roc": -0.1,      # Stabile/calo
            "yield_curve_10y2y": 1.2, # Normale
            "initial_claims_roc": -3.0, # In calo
            "lei_roc": 0.8,          # Positivo
            "fed_funds_rate": 1.5,   # Accomodante
        }
        result = classify_regime(indicators)

        assert result["regime"] == "goldilocks"
        assert result["probabilities"]["goldilocks"] > 0.3

    def test_high_confidence_clear_signal(self):
        """Con segnali molto chiari, confidence deve essere alta."""
        from app.services.regime.classifier import classify_regime

        # Tutti gli indicatori puntano chiaramente a deflation/recession
        indicators = {
            "gdp_roc": -4.0,
            "pmi": 38.0,
            "cpi_yoy": 1.0,
            "unrate": 9.0,
            "unrate_roc": 3.0,
            "yield_curve_10y2y": -1.0,
            "initial_claims_roc": 50.0,
            "lei_roc": -5.0,
            "fed_funds_rate": 5.0,
        }
        result = classify_regime(indicators)

        assert result["confidence"] > 0.7

    def test_low_confidence_mixed_signals(self):
        """Con segnali contrastanti, confidence deve essere bassa."""
        from app.services.regime.classifier import classify_regime

        # Mix: GDP buono ma PMI basso, inflation alta ma claims bassi
        indicators = {
            "gdp_roc": 3.0,           # Segnale reflation
            "pmi": 45.0,              # Segnale deflation
            "cpi_yoy": 7.0,           # Segnale stagflation
            "unrate": 3.5,            # Segnale goldilocks
            "unrate_roc": -0.2,       # Segnale reflation
            "yield_curve_10y2y": -0.3, # Segnale deflation
            "initial_claims_roc": -5.0, # Segnale reflation
            "lei_roc": -2.0,          # Segnale deflation
            "fed_funds_rate": 3.0,
        }
        result = classify_regime(indicators)

        assert result["confidence"] < 0.5

    def test_only_four_regimes(self):
        """Il sistema deve avere esattamente 4 regimi."""
        from app.services.regime.classifier import REGIMES

        assert len(REGIMES) == 4
        assert set(REGIMES) == {"reflation", "stagflation", "deflation", "goldilocks"}

    def test_new_indicators_tilt_stagflation(self):
        """Core PCE alto + spread BAA largo + sentiment basso devono spingere stagflation.

        Scenario borderline (inflation moderatamente alta) reso più netto dai nuovi
        indicatori: senza di essi stagflation vince appena, con i segnali nuovi
        aumenta ulteriormente la probabilità.
        """
        from app.services.regime.classifier import classify_regime

        base = {
            "gdp_roc": 0.8,
            "pmi": 48.5,
            "cpi_yoy": 4.5,
            "unrate": 4.5,
            "unrate_roc": 0.25,
            "yield_curve_10y2y": 0.0,
            "initial_claims_roc": 6.0,
            "lei_roc": -0.8,
            "fed_funds_rate": 5.0,
        }
        enriched = {
            **base,
            "core_pce_yoy": 4.2,
            "payrolls_roc_12m": 0.5,
            "indpro_roc_12m": -0.2,
            "baa_spread": 2.8,
            "consumer_sentiment": 62.0,
        }

        base_result = classify_regime(base)
        enriched_result = classify_regime(enriched)

        assert enriched_result["regime"] == "stagflation"
        assert (
            enriched_result["probabilities"]["stagflation"]
            > base_result["probabilities"]["stagflation"]
        )

    def test_new_indicators_tilt_goldilocks(self):
        """Core PCE contenuto + spread tight + sentiment alto rafforzano goldilocks."""
        from app.services.regime.classifier import classify_regime

        base = {
            "gdp_roc": 2.2,
            "pmi": 54.0,
            "cpi_yoy": 2.0,
            "unrate": 3.7,
            "unrate_roc": -0.05,
            "yield_curve_10y2y": 1.2,
            "initial_claims_roc": -2.0,
            "lei_roc": 0.4,
            "fed_funds_rate": 2.0,
        }
        enriched = {
            **base,
            "core_pce_yoy": 1.9,
            "payrolls_roc_12m": 1.8,
            "indpro_roc_12m": 2.2,
            "baa_spread": 1.4,
            "consumer_sentiment": 92.0,
        }

        base_result = classify_regime(base)
        enriched_result = classify_regime(enriched)

        assert enriched_result["regime"] == "goldilocks"
        assert (
            enriched_result["probabilities"]["goldilocks"]
            > base_result["probabilities"]["goldilocks"]
        )
