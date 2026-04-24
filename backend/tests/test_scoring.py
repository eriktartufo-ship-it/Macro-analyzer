"""Test TDD per il sistema di scoring finale (4 regimi)."""

class TestFinalScoring:
    """Test calcolo score finale per asset class."""

    def test_score_range_0_to_100(self):
        """Lo score finale deve essere tra 0 e 100."""
        from app.services.scoring.engine import calculate_final_scores

        probabilities = {
            "reflation": 0.50,
            "stagflation": 0.10,
            "deflation": 0.15,
            "goldilocks": 0.25,
        }
        scores = calculate_final_scores(probabilities)

        for asset, score in scores.items():
            assert 0.0 <= score <= 100.0, f"{asset} score {score} fuori range"

    def test_all_asset_classes_present(self):
        """Tutti gli asset class devono avere uno score."""
        from app.services.scoring.engine import calculate_final_scores, ASSET_CLASSES

        probabilities = {
            "reflation": 0.30,
            "stagflation": 0.20,
            "deflation": 0.20,
            "goldilocks": 0.30,
        }
        scores = calculate_final_scores(probabilities)

        for asset in ASSET_CLASSES:
            assert asset in scores, f"Manca {asset}"

    def test_gold_high_in_stagflation(self):
        """Gold deve avere score alto in stagflation."""
        from app.services.scoring.engine import calculate_final_scores

        probabilities = {
            "reflation": 0.10,
            "stagflation": 0.65,
            "deflation": 0.15,
            "goldilocks": 0.10,
        }
        scores = calculate_final_scores(probabilities)

        assert scores["gold"] > 60

    def test_equities_high_in_reflation(self):
        """US Equities Growth deve avere score alto in reflation."""
        from app.services.scoring.engine import calculate_final_scores

        probabilities = {
            "reflation": 0.65,
            "stagflation": 0.05,
            "deflation": 0.10,
            "goldilocks": 0.20,
        }
        scores = calculate_final_scores(probabilities)

        assert scores["us_equities_growth"] > 60

    def test_bonds_long_high_in_deflation(self):
        """US Bonds Long deve performare bene in deflation (flight to safety)."""
        from app.services.scoring.engine import calculate_final_scores

        probabilities = {
            "reflation": 0.05,
            "stagflation": 0.10,
            "deflation": 0.70,
            "goldilocks": 0.15,
        }
        scores = calculate_final_scores(probabilities)

        assert scores["us_bonds_long"] > 55

    def test_cash_high_in_uncertainty(self):
        """Cash deve avere score relativamente alto con segnali misti."""
        from app.services.scoring.engine import calculate_final_scores

        # Distribuzione quasi uniforme = alta incertezza
        probabilities = {
            "reflation": 0.25,
            "stagflation": 0.25,
            "deflation": 0.25,
            "goldilocks": 0.25,
        }
        scores = calculate_final_scores(probabilities)

        assert scores["cash_money_market"] > 40

    def test_probabilities_must_sum_to_one(self):
        """Se le probabilita non sommano a 1, deve normalizzare."""
        from app.services.scoring.engine import calculate_final_scores

        # Probabilita che non sommano a 1
        probabilities = {
            "reflation": 0.50,
            "stagflation": 0.20,
            "deflation": 0.10,
            "goldilocks": 0.15,
        }
        # Somma = 0.95, il sistema deve normalizzare
        scores = calculate_final_scores(probabilities)

        assert len(scores) > 0

    def test_scoring_deterministic(self):
        """Stesso input = stesso output (nessuna randomness)."""
        from app.services.scoring.engine import calculate_final_scores

        probabilities = {
            "reflation": 0.40,
            "stagflation": 0.15,
            "deflation": 0.20,
            "goldilocks": 0.25,
        }
        scores1 = calculate_final_scores(probabilities)
        scores2 = calculate_final_scores(probabilities)

        for asset in scores1:
            assert scores1[asset] == scores2[asset]

    def test_relative_ranking_makes_sense(self):
        """In reflation, equities > bonds > gold (ranking relativo sensato)."""
        from app.services.scoring.engine import calculate_final_scores

        probabilities = {
            "reflation": 0.70,
            "stagflation": 0.05,
            "deflation": 0.10,
            "goldilocks": 0.15,
        }
        scores = calculate_final_scores(probabilities)

        assert scores["us_equities_growth"] > scores["us_bonds_long"]


class TestRegressions:
    """Test di regressione per bug noti corretti (rev. 2)."""

    def test_cash_not_dominant_in_stagflation(self):
        """Bug fix: il cash perde potere d'acquisto in stagflation (real return negativo)
        quindi non deve essere dominante. Deve scorare sotto 40."""
        from app.services.scoring.engine import calculate_final_scores

        probabilities = {
            "reflation": 0.05,
            "stagflation": 0.80,
            "deflation": 0.10,
            "goldilocks": 0.05,
        }
        scores = calculate_final_scores(probabilities)

        assert scores["cash_money_market"] < 40, (
            f"Cash in stagflation = {scores['cash_money_market']}, atteso <40"
        )

    def test_cash_not_dominant_in_reflation(self):
        """Bug fix: il cash in reflation ha real return ~0, non deve battere gli equity."""
        from app.services.scoring.engine import calculate_final_scores

        probabilities = {
            "reflation": 0.80,
            "stagflation": 0.05,
            "deflation": 0.05,
            "goldilocks": 0.10,
        }
        scores = calculate_final_scores(probabilities)

        assert scores["us_equities_growth"] > scores["cash_money_market"] + 15, (
            f"Growth {scores['us_equities_growth']} vs Cash {scores['cash_money_market']}"
        )

    def test_long_bonds_collapse_in_stagflation(self):
        """Bug fix: long bonds sono il peggior asset in stagflation (duration + inflation)."""
        from app.services.scoring.engine import calculate_final_scores

        probabilities = {
            "reflation": 0.05,
            "stagflation": 0.80,
            "deflation": 0.05,
            "goldilocks": 0.10,
        }
        scores = calculate_final_scores(probabilities)

        assert scores["us_bonds_long"] < 25, (
            f"Long bonds in stagflation = {scores['us_bonds_long']}, atteso <25"
        )

    def test_energy_collapse_in_deflation(self):
        """Energia in deflation: 2008 WTI -77%, 2020 WTI negative. Score molto basso."""
        from app.services.scoring.engine import calculate_final_scores

        probabilities = {
            "reflation": 0.05,
            "stagflation": 0.05,
            "deflation": 0.85,
            "goldilocks": 0.05,
        }
        scores = calculate_final_scores(probabilities)

        assert scores["energy"] < 20, (
            f"Energy in deflation = {scores['energy']}, atteso <20"
        )

    def test_cash_king_in_pure_deflation(self):
        """In deflation estrema il cash diventa re (positive real return)."""
        from app.services.scoring.engine import calculate_final_scores

        probabilities = {
            "reflation": 0.05,
            "stagflation": 0.05,
            "deflation": 0.85,
            "goldilocks": 0.05,
        }
        scores = calculate_final_scores(probabilities)

        # Cash deve battere equities e crypto in deflation pura
        assert scores["cash_money_market"] > scores["us_equities_growth"]
        assert scores["cash_money_market"] > scores["bitcoin"]
        assert scores["cash_money_market"] > scores["broad_commodities"]

    def test_gold_strong_in_stagflation(self):
        """Gold e' il classico hedge stagflation: deve scorare alto."""
        from app.services.scoring.engine import calculate_final_scores

        probabilities = {
            "reflation": 0.05,
            "stagflation": 0.85,
            "deflation": 0.05,
            "goldilocks": 0.05,
        }
        scores = calculate_final_scores(probabilities)

        assert scores["gold"] > 65, f"Gold in stagflation = {scores['gold']}"
        assert scores["gold"] > scores["us_bonds_long"]
        assert scores["gold"] > scores["us_equities_growth"]
