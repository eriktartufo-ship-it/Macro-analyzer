"""Test TDD per il layer di dedollarizzazione."""


class TestDedollarizationScore:
    """Test calcolo score dedollarizzazione composito."""

    def test_score_range_0_to_1(self):
        """Lo score finale deve essere tra 0 e 1."""
        from app.services.dedollarization.scorer import calculate_dedollarization

        indicators = {
            "dxy_roc_12m": -5.0,
            "gold_roc_12m": 15.0,
            "gold_oil_ratio": 30.0,
            "debt_gdp": 120.0,
            "real_rate": -1.5,
            "m2_roc_12m": 8.0,
        }
        result = calculate_dedollarization(indicators)
        assert 0.0 <= result["score"] <= 1.0

    def test_high_dedollarization_scenario(self):
        """Scenario forte dedollarizzazione: USD debole, gold forte, debito alto."""
        from app.services.dedollarization.scorer import calculate_dedollarization

        indicators = {
            "dxy_roc_12m": -12.0,   # USD in forte calo
            "gold_roc_12m": 25.0,   # Gold in forte salita
            "gold_oil_ratio": 40.0, # Gold domina oil
            "debt_gdp": 130.0,      # Debito altissimo
            "real_rate": -3.0,      # Tassi reali molto negativi
            "m2_roc_12m": 15.0,     # Espansione monetaria aggressiva
        }
        result = calculate_dedollarization(indicators)
        assert result["score"] > 0.7

    def test_low_dedollarization_scenario(self):
        """Scenario USD forte: dollaro in salita, gold stabile, debito contenuto."""
        from app.services.dedollarization.scorer import calculate_dedollarization

        indicators = {
            "dxy_roc_12m": 8.0,     # USD in forte salita
            "gold_roc_12m": -2.0,   # Gold in calo
            "gold_oil_ratio": 15.0, # Ratio basso
            "debt_gdp": 80.0,       # Debito contenuto
            "real_rate": 2.5,       # Tassi reali positivi
            "m2_roc_12m": 2.0,      # Espansione moderata
        }
        result = calculate_dedollarization(indicators)
        assert result["score"] < 0.3

    def test_components_present(self):
        """Il risultato deve contenere componenti ciclici, strutturali e accelerazione."""
        from app.services.dedollarization.scorer import calculate_dedollarization

        indicators = {
            "dxy_roc_12m": -5.0,
            "gold_roc_12m": 15.0,
            "gold_oil_ratio": 30.0,
            "debt_gdp": 120.0,
            "real_rate": -1.5,
            "m2_roc_12m": 8.0,
        }
        result = calculate_dedollarization(indicators)

        assert "score" in result
        assert "structural_score" in result
        assert "acceleration" in result
        assert "combined_score" in result
        assert "components" in result
        assert "structural" in result
        components = result["components"]
        assert "usd_weakness" in components
        assert "gold_strength" in components
        assert "gold_oil_ratio" in components
        assert "debt_burden" in components
        assert "real_rate_signal" in components
        assert "monetary_debasement" in components
        structural = result["structural"]
        assert "usd_secular" in structural
        assert "gold_secular" in structural
        assert "debt_trajectory" in structural
        assert "m2_cumulative" in structural

    def test_each_component_0_to_1(self):
        """Ogni componente deve essere tra 0 e 1."""
        from app.services.dedollarization.scorer import calculate_dedollarization

        indicators = {
            "dxy_roc_12m": -5.0,
            "gold_roc_12m": 15.0,
            "gold_oil_ratio": 30.0,
            "debt_gdp": 120.0,
            "real_rate": -1.5,
            "m2_roc_12m": 8.0,
        }
        result = calculate_dedollarization(indicators)

        for name, val in result["components"].items():
            assert 0.0 <= val <= 1.0, f"Componente {name} = {val} fuori range"

    def test_missing_indicators_use_defaults(self):
        """Indicatori mancanti devono usare valori neutri, non crashare."""
        from app.services.dedollarization.scorer import calculate_dedollarization

        result = calculate_dedollarization({})
        assert 0.0 <= result["score"] <= 1.0

    def test_deterministic(self):
        """Stesso input = stesso output."""
        from app.services.dedollarization.scorer import calculate_dedollarization

        indicators = {
            "dxy_roc_12m": -5.0,
            "gold_roc_12m": 15.0,
            "gold_oil_ratio": 30.0,
            "debt_gdp": 120.0,
            "real_rate": -1.5,
            "m2_roc_12m": 8.0,
        }
        r1 = calculate_dedollarization(indicators)
        r2 = calculate_dedollarization(indicators)
        assert r1["score"] == r2["score"]


class TestSecularBonus:
    """Test bonus per asset class basato su dedollarizzazione."""

    def test_gold_gets_highest_bonus(self):
        """Gold deve ricevere il bonus piu alto in dedollarizzazione forte."""
        from app.services.dedollarization.scorer import calculate_secular_bonus

        bonus = calculate_secular_bonus(0.9)
        assert bonus["gold"] > bonus["us_equities_growth"]
        assert bonus["gold"] > bonus["us_bonds_long"]
        assert bonus["gold"] >= max(bonus.values()) - 0.01  # gold e' il top o quasi

    def test_bonds_penalized(self):
        """US Bonds Long deve avere bonus negativo (penalty) in dedollarizzazione."""
        from app.services.dedollarization.scorer import calculate_secular_bonus

        bonus = calculate_secular_bonus(0.8)
        assert bonus["us_bonds_long"] < 0

    def test_zero_score_zero_bonus(self):
        """Score dedollarizzazione = 0 → tutti i bonus = 0."""
        from app.services.dedollarization.scorer import calculate_secular_bonus

        bonus = calculate_secular_bonus(0.0)
        for asset, val in bonus.items():
            assert val == 0.0, f"{asset} bonus = {val}, atteso 0"

    def test_all_assets_have_bonus(self):
        """Tutti gli asset class devono avere un bonus definito."""
        from app.services.dedollarization.scorer import calculate_secular_bonus
        from app.services.scoring.engine import ASSET_CLASSES

        bonus = calculate_secular_bonus(0.5)
        for asset in ASSET_CLASSES:
            assert asset in bonus, f"Manca bonus per {asset}"

    def test_em_equities_benefit(self):
        """EM equities beneficiano da USD debole."""
        from app.services.dedollarization.scorer import calculate_secular_bonus

        bonus = calculate_secular_bonus(0.8)
        assert bonus["em_equities"] > 0
        assert bonus["em_equities"] > bonus["us_equities_growth"]

    def test_crypto_benefits(self):
        """Bitcoin e crypto broad beneficiano da dedollarizzazione."""
        from app.services.dedollarization.scorer import calculate_secular_bonus

        bonus = calculate_secular_bonus(0.8)
        assert bonus["bitcoin"] > 0
        assert bonus["crypto_broad"] > 0
