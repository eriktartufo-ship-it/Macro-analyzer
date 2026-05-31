"""Test TDD per il regime classifier a 4 regimi.

Quadrante macro (crescita x inflazione):
- Reflation:   crescita forte + inflazione in salita
- Stagflation: crescita debole + inflazione alta
- Deflation:   crescita debole + inflazione bassa/in calo
- Goldilocks:  crescita moderata + inflazione bassa
"""

import pytest


@pytest.fixture(autouse=True)
def _isolate_ml_blend(monkeypatch):
    """Module-level: test del classifier rule-based puro.
    Disabilita ML blend (T8.3), freshness (T6.9), gdp_collapse (T7.1) per
    asserzioni numeriche deterministiche.

    T10b 2026-05-17: USE_ML_REGIME_BLEND e USE_GDP_COLLAPSE_OVERRIDE promossi
    default-ON. Force "0" explicit (non delenv) per preservare baseline rule-based.
    """
    for flag in ("USE_ML_REGIME_BLEND", "USE_GDP_COLLAPSE_OVERRIDE", "USE_MOMENTUM_PILLARS"):
        monkeypatch.setenv(flag, "0")
    monkeypatch.delenv("USE_FRESHNESS_WEIGHTING", raising=False)


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

    def test_high_confidence_clear_signal(self, monkeypatch):
        """Con segnali molto chiari, confidence deve essere alta.

        Test del classifier base — disabilita T6.9 freshness (che riduce
        intenzionalmente il peso degli indicators stale come GDP/CPI).
        """
        monkeypatch.delenv("USE_FRESHNESS_WEIGHTING", raising=False)
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
            # Segnali finanziari coerenti con deflation severa
            "vix": 45.0,
            "nfci": 0.8,
            "breakeven_10y": 0.8,
            "housing_starts_roc_12m": -25.0,
            # T13 forward inflation (deflation severa: consumer + sticky low)
            "umich_inflation_1y": 1.5,
            "sticky_cpi_yoy": 1.8,
            "breakeven_5y5y": 1.0,
        }
        result = classify_regime(indicators)

        assert result["confidence"] > 0.7

    def test_low_confidence_mixed_signals(self):
        """Con segnali contrastanti, confidence deve essere bassa.

        T9-AUDIT-FIX: scenario aggiornato. CPI 7% non è "mixed signal" è
        chiaramente stagflation (dopo bug-fix bughunter HIGH-1 reflation
        penalty per CPI>3.5). Uso scenario VERAMENTE borderline.
        """
        from app.services.regime.classifier import classify_regime

        # Borderline reflation/goldilocks: GDP medio, CPI vicino target, mixed altri
        indicators = {
            "gdp_roc": 2.0,
            "pmi": 50.0,
            "cpi_yoy": 2.8,
            "unrate": 4.5,
            "unrate_roc": 0.0,
            "yield_curve_10y2y": 0.3,
            "initial_claims_roc": 0.0,
            "lei_roc": 0.0,
            "fed_funds_rate": 3.5,
            "core_pce_yoy": 2.6,
            "consumer_sentiment": 75.0,
            "breakeven_10y": 2.3,
            "baa_10y_spread": 2.0,
            "vix": 18.0,
            "indpro_roc_12m": 1.5,
            "payrolls_roc_12m": 1.5,
        }
        result = classify_regime(indicators)

        # Segnali borderline → confidence < 0.5 (modello incerto)
        assert result["confidence"] < 0.5

    def test_only_four_regimes(self):
        """Il sistema deve avere esattamente 4 regimi."""
        from app.services.regime.classifier import REGIMES

        assert len(REGIMES) == 4
        assert set(REGIMES) == {"reflation", "stagflation", "deflation", "goldilocks"}

    def test_dfm_growth_strong_tilts_reflation(self):
        """DFM nowcast YoY alto (5%) deve aumentare probabilita reflation
        rispetto al neutro 2%. Tier 2.5 roadmap quick win."""
        from app.services.regime.classifier import classify_regime

        base = {
            "gdp_roc": 0.5, "pmi": 50.0, "cpi_yoy": 2.5, "unrate": 4.5,
            "unrate_roc": 0.0, "yield_curve_10y2y": 0.5, "initial_claims_roc": 0.0,
            "lei_roc": 0.0, "fed_funds_rate": 3.0,
        }
        # Default DFM = 2.0 (trend) → segnale neutro
        r_neutral = classify_regime(base)
        # DFM YoY = 5.0 → forte segnale crescita
        r_strong = classify_regime({**base, "gdp_yoy_dfm": 5.0})

        assert r_strong["probabilities"]["reflation"] > r_neutral["probabilities"]["reflation"]

    def test_dfm_growth_weak_tilts_stagflation(self):
        """DFM nowcast YoY basso (-1%) deve aumentare probabilita stagflation
        rispetto al neutro 2%."""
        from app.services.regime.classifier import classify_regime

        base = {
            "gdp_roc": 0.5, "pmi": 50.0, "cpi_yoy": 4.0, "unrate": 5.0,
            "unrate_roc": 0.5, "yield_curve_10y2y": 0.0, "initial_claims_roc": 5.0,
            "lei_roc": -0.5, "fed_funds_rate": 5.0,
        }
        r_neutral = classify_regime(base)
        r_weak = classify_regime({**base, "gdp_yoy_dfm": -1.0})

        assert r_weak["probabilities"]["stagflation"] > r_neutral["probabilities"]["stagflation"]

    def test_dfm_default_neutral_when_missing(self):
        """gdp_yoy_dfm assente → default 2.0% (= trend long-run, segnale neutro).
        Classify funziona normalmente senza la chiave."""
        from app.services.regime.classifier import classify_regime

        ind = {
            "gdp_roc": 2.5, "pmi": 55.0, "cpi_yoy": 2.0, "unrate": 4.0,
            "unrate_roc": -0.1, "yield_curve_10y2y": 1.0, "initial_claims_roc": -5.0,
            "lei_roc": 0.5, "fed_funds_rate": 2.0,
        }
        # Senza gdp_yoy_dfm
        r = classify_regime(ind)
        assert sum(r["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
        assert r["regime"] in {"reflation", "stagflation", "deflation", "goldilocks"}

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


class TestInsiderPillar:
    """Test 7° pillar classifier: SEC Form 4 insider score.

    Verifica:
    - Default insider_score=0 → niente cambiamento (baseline behavior preservata).
    - insider_score=+0.5 (buying strong) → boost reflation/goldilocks.
    - insider_score=-0.5 (selling strong) → boost stagflation/deflation.
    """

    def _base_neutral_indicators(self) -> dict:
        return {
            "gdp_roc": 1.5,
            "pmi": 50.0,
            "cpi_yoy": 2.5,
            "unrate": 4.5,
            "unrate_roc": 0.0,
            "yield_curve_10y2y": 0.5,
            "initial_claims_roc": 0.0,
            "lei_roc": 0.0,
            "fed_funds_rate": 3.0,
        }

    def test_default_insider_score_zero_preserves_baseline(self):
        """Senza insider_score nel dict (= default 0.0), output identico al precedente."""
        from app.services.regime.classifier import classify_regime

        ind = self._base_neutral_indicators()
        # ind senza insider_score key
        result_no = classify_regime(ind)
        # ind con insider_score=0 esplicito
        result_zero = classify_regime({**ind, "insider_score": 0.0})

        for r in result_no["probabilities"]:
            assert result_no["probabilities"][r] == result_zero["probabilities"][r]

    def test_insider_buying_boosts_reflation(self):
        """insider_score=+0.6 (buying strong) deve boostare reflation/goldilocks
        rispetto a insider_score=0."""
        from app.services.regime.classifier import classify_regime

        ind = self._base_neutral_indicators()
        baseline = classify_regime({**ind, "insider_score": 0.0})
        bullish = classify_regime({**ind, "insider_score": 0.6})

        # Almeno uno tra reflation o goldilocks deve crescere
        delta_refl = bullish["probabilities"]["reflation"] - baseline["probabilities"]["reflation"]
        delta_gold = bullish["probabilities"]["goldilocks"] - baseline["probabilities"]["goldilocks"]
        assert delta_refl + delta_gold > 0

    def test_insider_selling_boosts_deflation_or_stagflation(self):
        """insider_score=-0.6 (selling strong) deve boostare deflation/stagflation."""
        from app.services.regime.classifier import classify_regime

        ind = self._base_neutral_indicators()
        baseline = classify_regime({**ind, "insider_score": 0.0})
        bearish = classify_regime({**ind, "insider_score": -0.6})

        delta_defl = bearish["probabilities"]["deflation"] - baseline["probabilities"]["deflation"]
        delta_stag = bearish["probabilities"]["stagflation"] - baseline["probabilities"]["stagflation"]
        assert delta_defl + delta_stag > 0

    def test_insider_score_does_not_flip_regime_with_strong_other_signals(self):
        """Insider è low-weight (0.02-0.03). Non deve flippare un regime con
        segnali macro forti opposti."""
        from app.services.regime.classifier import classify_regime

        # Macro chiaramente deflation: GDP -2, CPI 1, unrate rising, claims rising
        deflation_macro = {
            "gdp_roc": -2.0,
            "pmi": 42.0,
            "cpi_yoy": 1.0,
            "unrate": 6.5,
            "unrate_roc": 1.0,
            "yield_curve_10y2y": -0.3,
            "initial_claims_roc": 15.0,
            "lei_roc": -2.0,
            "fed_funds_rate": 1.0,
            "vix": 35.0,
            "baa_spread": 4.0,
            "insider_score": 0.8,  # forte buying ma macro = deflation severo
        }
        result = classify_regime(deflation_macro)
        # Deflation deve ancora dominare nonostante insider buying
        assert result["regime"] == "deflation"


class TestDedollarPillar:
    """Tier 5.2 LOOP CLOSURE — Dedollar combined come pillar classifier.

    Verifica:
    - Flag OFF (default): comportamento invariato (zero impact pillar).
    - Flag ON: dedollar>0.6 boosta stagflation, dedollar<0.4 boosta reflation.
    - Soft pillar (peso 0.02-0.03) non flippa regime con macro forti opposti.
    """

    def _base_neutral(self) -> dict:
        return {
            "gdp_roc": 1.5, "pmi": 50.0, "cpi_yoy": 2.5, "unrate": 4.5,
            "unrate_roc": 0.0, "yield_curve_10y2y": 0.5, "initial_claims_roc": 0.0,
            "lei_roc": 0.0, "fed_funds_rate": 3.0,
        }

    def test_flag_off_preserves_baseline(self, monkeypatch):
        """Default USE_DEDOLLAR_PILLAR=0: output identico con/senza dedollar_combined."""
        monkeypatch.delenv("USE_DEDOLLAR_PILLAR", raising=False)
        from app.services.regime.classifier import classify_regime

        ind = self._base_neutral()
        r_no = classify_regime(ind)
        r_high = classify_regime({**ind, "dedollar_combined": 0.85})
        r_low = classify_regime({**ind, "dedollar_combined": 0.15})

        # Tutti i probabilities identici
        for k in r_no["probabilities"]:
            assert r_no["probabilities"][k] == r_high["probabilities"][k]
            assert r_no["probabilities"][k] == r_low["probabilities"][k]

    def test_flag_on_dedollar_high_boosts_stagflation(self, monkeypatch):
        """USE_DEDOLLAR_PILLAR=1 + dedollar=0.85 → stagflation prob > baseline."""
        monkeypatch.setenv("USE_DEDOLLAR_PILLAR", "1")
        from app.services.regime.classifier import classify_regime

        ind = self._base_neutral()
        baseline = classify_regime({**ind, "dedollar_combined": 0.5})
        debasement = classify_regime({**ind, "dedollar_combined": 0.85})

        delta_stag = debasement["probabilities"]["stagflation"] - baseline["probabilities"]["stagflation"]
        assert delta_stag > 0

    def test_flag_on_dedollar_low_boosts_reflation(self, monkeypatch):
        """USE_DEDOLLAR_PILLAR=1 + dedollar=0.15 → reflation prob > baseline."""
        monkeypatch.setenv("USE_DEDOLLAR_PILLAR", "1")
        from app.services.regime.classifier import classify_regime

        ind = self._base_neutral()
        baseline = classify_regime({**ind, "dedollar_combined": 0.5})
        strong_dollar = classify_regime({**ind, "dedollar_combined": 0.15})

        delta_refl = strong_dollar["probabilities"]["reflation"] - baseline["probabilities"]["reflation"]
        assert delta_refl > 0

    def test_pillar_does_not_flip_regime_with_strong_macro(self, monkeypatch):
        """Dedollar pillar weight 0.02-0.03 = soft: non deve flippare un regime
        con segnali macro forti opposti."""
        monkeypatch.setenv("USE_DEDOLLAR_PILLAR", "1")
        from app.services.regime.classifier import classify_regime

        # Macro chiaramente deflation
        deflation_macro = {
            "gdp_roc": -2.0, "pmi": 42.0, "cpi_yoy": 1.0, "unrate": 6.5,
            "unrate_roc": 1.0, "yield_curve_10y2y": -0.3, "initial_claims_roc": 15.0,
            "lei_roc": -2.0, "fed_funds_rate": 1.0, "vix": 35.0, "baa_spread": 4.0,
            "dedollar_combined": 0.85,  # debasement forte ma macro = defl severa
        }
        result = classify_regime(deflation_macro)
        assert result["regime"] == "deflation"

    def test_flag_on_neutral_dedollar_small_impact(self, monkeypatch):
        """USE_DEDOLLAR_PILLAR=1 con dedollar=0.5 (neutral) → impact minimo."""
        monkeypatch.setenv("USE_DEDOLLAR_PILLAR", "1")
        from app.services.regime.classifier import classify_regime

        ind = self._base_neutral()
        baseline = classify_regime(ind)  # no key → default 0.5
        neutral = classify_regime({**ind, "dedollar_combined": 0.5})

        # Differenze trascurabili (< 1pp per ogni regime)
        for k in baseline["probabilities"]:
            diff = abs(baseline["probabilities"][k] - neutral["probabilities"][k])
            assert diff < 0.01


class TestNewsPillar:
    """Tier 5.3 LOOP CLOSURE — News avg sentiment come pillar classifier.

    Verifica:
    - Flag OFF: identity (probs invariate con/senza news_sentiment).
    - Flag ON: sentiment>+0.3 boosta reflation, sentiment<-0.3 boosta deflation.
    - Soft pillar (peso 0.02) non flippa regime con macro forti opposti.
    """

    def _base_neutral(self) -> dict:
        return {
            "gdp_roc": 1.5, "pmi": 50.0, "cpi_yoy": 2.5, "unrate": 4.5,
            "unrate_roc": 0.0, "yield_curve_10y2y": 0.5, "initial_claims_roc": 0.0,
            "lei_roc": 0.0, "fed_funds_rate": 3.0,
        }

    def test_flag_off_preserves_baseline(self, monkeypatch):
        monkeypatch.delenv("USE_NEWS_PILLAR", raising=False)
        from app.services.regime.classifier import classify_regime

        ind = self._base_neutral()
        r_no = classify_regime(ind)
        r_pos = classify_regime({**ind, "news_sentiment": 0.7})
        r_neg = classify_regime({**ind, "news_sentiment": -0.7})

        for k in r_no["probabilities"]:
            assert r_no["probabilities"][k] == r_pos["probabilities"][k]
            assert r_no["probabilities"][k] == r_neg["probabilities"][k]

    def test_flag_on_positive_boosts_reflation(self, monkeypatch):
        monkeypatch.setenv("USE_NEWS_PILLAR", "1")
        from app.services.regime.classifier import classify_regime

        ind = self._base_neutral()
        baseline = classify_regime({**ind, "news_sentiment": 0.0})
        positive = classify_regime({**ind, "news_sentiment": 0.6})

        delta_refl = positive["probabilities"]["reflation"] - baseline["probabilities"]["reflation"]
        assert delta_refl > 0

    def test_flag_on_negative_boosts_deflation(self, monkeypatch):
        monkeypatch.setenv("USE_NEWS_PILLAR", "1")
        from app.services.regime.classifier import classify_regime

        ind = self._base_neutral()
        baseline = classify_regime({**ind, "news_sentiment": 0.0})
        panic = classify_regime({**ind, "news_sentiment": -0.6})

        delta_defl = panic["probabilities"]["deflation"] - baseline["probabilities"]["deflation"]
        assert delta_defl > 0

    def test_pillar_does_not_flip_regime_with_strong_macro(self, monkeypatch):
        monkeypatch.setenv("USE_NEWS_PILLAR", "1")
        from app.services.regime.classifier import classify_regime

        # Macro chiaramente deflation, ma news positive (es. policy response)
        ind = {
            "gdp_roc": -2.0, "pmi": 42.0, "cpi_yoy": 1.0, "unrate": 6.5,
            "unrate_roc": 1.0, "yield_curve_10y2y": -0.3, "initial_claims_roc": 15.0,
            "lei_roc": -2.0, "fed_funds_rate": 1.0, "vix": 35.0, "baa_spread": 4.0,
            "news_sentiment": 0.8,  # positive ma macro = deflation severo
        }
        result = classify_regime(ind)
        assert result["regime"] == "deflation"

    def test_flag_on_neutral_sentiment_small_impact(self, monkeypatch):
        monkeypatch.setenv("USE_NEWS_PILLAR", "1")
        from app.services.regime.classifier import classify_regime

        ind = self._base_neutral()
        baseline = classify_regime(ind)  # no key → default 0.0
        neutral = classify_regime({**ind, "news_sentiment": 0.0})

        for k in baseline["probabilities"]:
            assert abs(baseline["probabilities"][k] - neutral["probabilities"][k]) < 1e-6
