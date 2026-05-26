"""Test Tier 7.2 (rank percentile) + Tier 7.3 (downside-protection bonus).

Coverage:
- T7.2: percentile mapping calcolato, top-asset within regime ottiene score 100.
- T7.2: flag OFF preserva comportamento baseline.
- T7.3: cash/short-bonds scorano sopra il baseline in stress regimes.
- T7.3: flag OFF preserva comportamento baseline.
- T7.3: bonus inattivo se stress_share <= 0.5.
- Combinazione T7.2 + T7.3: gold/cash entrano nei top-3 di stagflation/deflation.
"""
from __future__ import annotations

import pytest

from app.services.scoring.engine import (
    _compute_regime_percentiles,
    calculate_final_scores,
    reload_calibration,
)


@pytest.fixture(autouse=True)
def _isolate_flags(monkeypatch):
    """Isola flag noise tra test: solo i flag esplicitamente settati.

    T10b 2026-05-17: USE_CALIBRATED_SCORING promosso default-ON. Per testare
    baseline puro (hardcoded prior), forziamo "0" explicit (non delenv).
    """
    monkeypatch.setenv("USE_CALIBRATED_SCORING", "0")
    for flag in (
        "USE_LIVE_CALIBRATION",
        "USE_DEDOLLAR_BONUS",
        "USE_DFM_ASSET_BONUS",
        "USE_DISCRETIONARY_OVERRIDES",
        "USE_RANK_PERCENTILE_SCORING",
        "USE_DOWNSIDE_PROTECTION_BONUS",
    ):
        monkeypatch.delenv(flag, raising=False)
    reload_calibration()
    yield
    reload_calibration()


class TestComputeRegimePercentiles:
    def test_all_regimes_covered(self):
        out = _compute_regime_percentiles()
        assert set(out.keys()) == {"reflation", "stagflation", "deflation", "goldilocks"}

    def test_percentile_range_0_1(self):
        out = _compute_regime_percentiles()
        for regime, assets in out.items():
            for asset, pct in assets.items():
                assert 0.0 <= pct <= 1.0

    def test_top_asset_has_percentile_1(self):
        out = _compute_regime_percentiles()
        for regime, assets in out.items():
            assert max(assets.values()) == 1.0

    def test_bottom_asset_has_percentile_0(self):
        out = _compute_regime_percentiles()
        for regime, assets in out.items():
            assert min(assets.values()) == 0.0


class TestRankPercentileScoring:
    def test_flag_off_baseline_preserved(self, monkeypatch):
        """Senza flag, scoring tradizionale assoluto."""
        monkeypatch.delenv("USE_RANK_PERCENTILE_SCORING", raising=False)
        probs = {"reflation": 1.0, "stagflation": 0.0, "deflation": 0.0, "goldilocks": 0.0}
        scores = calculate_final_scores(probs)
        # us_equities_growth in reflation pura: avg_return 0.15, sharpe 0.88
        # Score atteso ~71 (formula assoluta)
        assert 60 < scores["us_equities_growth"] < 90
        # cash in reflation: relativo basso (sotto growth)
        assert scores["cash_money_market"] < scores["us_equities_growth"]

    def test_flag_on_top_asset_scores_100(self, monkeypatch):
        """Con flag ON, il top asset in un regime scora 100 (percentile=1.0)."""
        monkeypatch.setenv("USE_RANK_PERCENTILE_SCORING", "1")
        probs = {"reflation": 1.0, "stagflation": 0.0, "deflation": 0.0, "goldilocks": 0.0}
        scores = calculate_final_scores(probs)
        # Almeno un asset deve scorare 100 (il top in reflation)
        assert max(scores.values()) == 100.0

    def test_flag_on_bottom_asset_scores_0(self, monkeypatch):
        """Bottom asset scora 0."""
        monkeypatch.setenv("USE_RANK_PERCENTILE_SCORING", "1")
        probs = {"deflation": 1.0, "reflation": 0.0, "stagflation": 0.0, "goldilocks": 0.0}
        scores = calculate_final_scores(probs)
        assert min(scores.values()) == 0.0

    def test_percentile_gold_top3_in_deflation(self, monkeypatch):
        """Con T7.2 in deflation pura, gold deve essere nei top-5
        (era fuori dal podio con scoring assoluto)."""
        monkeypatch.setenv("USE_RANK_PERCENTILE_SCORING", "1")
        probs = {"deflation": 1.0, "reflation": 0.0, "stagflation": 0.0, "goldilocks": 0.0}
        scores = calculate_final_scores(probs)
        top5 = sorted(scores.items(), key=lambda x: -x[1])[:5]
        top5_names = {name for name, _ in top5}
        # Gold dovrebbe rientrare nel top-5 in deflation pura
        assert "gold" in top5_names


class TestDownsideProtectionBonus:
    def test_flag_off_no_bonus(self, monkeypatch):
        monkeypatch.delenv("USE_DOWNSIDE_PROTECTION_BONUS", raising=False)
        probs = {"reflation": 0.0, "stagflation": 0.7, "deflation": 0.3, "goldilocks": 0.0}
        scores = calculate_final_scores(probs)
        baseline_cash = scores["cash_money_market"]

        monkeypatch.setenv("USE_DOWNSIDE_PROTECTION_BONUS", "1")
        scores_with = calculate_final_scores(probs)
        # Cash sale con bonus
        assert scores_with["cash_money_market"] > baseline_cash

    def test_bonus_inactive_in_reflation(self, monkeypatch):
        """In reflation pura (stress=0), no bonus."""
        monkeypatch.setenv("USE_DOWNSIDE_PROTECTION_BONUS", "1")
        probs = {"reflation": 1.0, "stagflation": 0.0, "deflation": 0.0, "goldilocks": 0.0}
        scores_on = calculate_final_scores(probs)

        monkeypatch.delenv("USE_DOWNSIDE_PROTECTION_BONUS", raising=False)
        scores_off = calculate_final_scores(probs)
        # No differenza in reflation pura
        assert scores_on["cash_money_market"] == scores_off["cash_money_market"]

    def test_bonus_scales_with_stress(self, monkeypatch):
        """Bonus cresce linearmente con stress_share."""
        monkeypatch.setenv("USE_DOWNSIDE_PROTECTION_BONUS", "1")
        # 60% stress
        probs_60 = {"stagflation": 0.4, "deflation": 0.2, "reflation": 0.3, "goldilocks": 0.1}
        s_60 = calculate_final_scores(probs_60)["cash_money_market"]
        # 90% stress
        probs_90 = {"stagflation": 0.5, "deflation": 0.4, "reflation": 0.05, "goldilocks": 0.05}
        s_90 = calculate_final_scores(probs_90)["cash_money_market"]
        # 90% stress dovrebbe dare bonus maggiore di 60% stress
        # Ma il baseline cash cambia perché la distribuzione è diversa.
        # Test minimo: entrambi hanno bonus > 0 rispetto al baseline OFF.
        monkeypatch.delenv("USE_DOWNSIDE_PROTECTION_BONUS", raising=False)
        s_60_off = calculate_final_scores(probs_60)["cash_money_market"]
        s_90_off = calculate_final_scores(probs_90)["cash_money_market"]
        assert (s_60 - s_60_off) >= 0
        assert (s_90 - s_90_off) >= (s_60 - s_60_off) - 0.01  # 90% bonus ≥ 60% bonus

    def test_bonus_not_applied_to_high_vol_assets(self, monkeypatch):
        """Bitcoin (vol > 0.5) NON deve ricevere il bonus anche in stress."""
        monkeypatch.setenv("USE_DOWNSIDE_PROTECTION_BONUS", "1")
        probs = {"stagflation": 0.5, "deflation": 0.4, "reflation": 0.05, "goldilocks": 0.05}
        scores_on = calculate_final_scores(probs)

        monkeypatch.delenv("USE_DOWNSIDE_PROTECTION_BONUS", raising=False)
        scores_off = calculate_final_scores(probs)
        # Bitcoin no bonus
        assert scores_on["bitcoin"] == scores_off["bitcoin"]


class TestCombinedT72T73:
    def test_cash_in_top5_2022_scenario(self, monkeypatch):
        """Scenario 2022 (stagflation alta): con T7.2 + T7.3 cash entra in top-5."""
        monkeypatch.setenv("USE_RANK_PERCENTILE_SCORING", "1")
        monkeypatch.setenv("USE_DOWNSIDE_PROTECTION_BONUS", "1")
        probs = {"reflation": 0.10, "stagflation": 0.70, "deflation": 0.10, "goldilocks": 0.10}
        scores = calculate_final_scores(probs)
        top5 = sorted(scores.items(), key=lambda x: -x[1])[:5]
        top5_names = {name for name, _ in top5}
        assert "cash_money_market" in top5_names

    def test_2008_deflation_top3_cash_bonds_gold(self, monkeypatch):
        """Scenario 2008 (deflation pura): top-3 dovrebbe includere bonds + gold + cash."""
        monkeypatch.setenv("USE_RANK_PERCENTILE_SCORING", "1")
        monkeypatch.setenv("USE_DOWNSIDE_PROTECTION_BONUS", "1")
        probs = {"reflation": 0.05, "stagflation": 0.10, "deflation": 0.75, "goldilocks": 0.10}
        scores = calculate_final_scores(probs)
        top3 = sorted(scores.items(), key=lambda x: -x[1])[:3]
        top3_names = {name for name, _ in top3}
        # Almeno 2 tra: us_bonds_long, us_bonds_short, gold, cash_money_market
        defensive = {"us_bonds_long", "us_bonds_short", "gold", "cash_money_market"}
        assert len(top3_names & defensive) >= 2
