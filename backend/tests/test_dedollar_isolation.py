"""Test che la dedollarizzazione sia OPT-IN via USE_DEDOLLAR_BONUS.

Verifica:
  1. calculate_final_scores con env OFF: ignora secular_bonus, score = pure macro
  2. calculate_final_scores con force_include_dedollar=True: usa secular_bonus
  3. calculate_trajectory con env OFF: skip DEDOLLAR_REGIME_PRESSURE
  4. Default-OFF: il sistema "out of the box" e' data-driven puro
"""
import os
import pytest

from app.services.regime.classifier import classify_regime
from app.services.regime.trajectory import calculate_trajectory
from app.services.scoring.engine import calculate_final_scores


@pytest.fixture
def reset_env():
    """Salva e ripristina USE_DEDOLLAR_BONUS attorno al test."""
    prev = os.environ.pop("USE_DEDOLLAR_BONUS", None)
    yield
    if prev is None:
        os.environ.pop("USE_DEDOLLAR_BONUS", None)
    else:
        os.environ["USE_DEDOLLAR_BONUS"] = prev


def _sample_probs():
    return {"reflation": 0.3, "stagflation": 0.3, "deflation": 0.2, "goldilocks": 0.2}


def _big_secular_bonus():
    """Bonus realistico: gold +5, bonds long -5, etc."""
    return {"gold": 5.0, "us_bonds_long": -5.0, "bitcoin": 3.0, "cash_money_market": -2.0}


class TestScoringEnvFlag:
    def test_default_env_ignores_dedollar_bonus(self, reset_env):
        """Senza env var, secular_bonus non modifica gli score."""
        probs = _sample_probs()
        bonus = _big_secular_bonus()
        no_bonus = calculate_final_scores(probs, secular_bonus=None)
        with_bonus_param_only = calculate_final_scores(probs, secular_bonus=bonus)
        # Deve essere identico: l'env e' OFF, il param viene ignorato
        for asset in ("gold", "us_bonds_long", "bitcoin", "cash_money_market"):
            assert abs(no_bonus[asset] - with_bonus_param_only[asset]) < 0.01, (
                f"{asset}: senza bonus={no_bonus[asset]}, con param={with_bonus_param_only[asset]}"
            )

    def test_env_on_applies_dedollar_bonus(self, reset_env):
        os.environ["USE_DEDOLLAR_BONUS"] = "1"
        probs = _sample_probs()
        bonus = _big_secular_bonus()
        no_bonus = calculate_final_scores(probs, secular_bonus=None)
        with_bonus = calculate_final_scores(probs, secular_bonus=bonus)
        # Gold deve crescere di ~5
        assert with_bonus["gold"] > no_bonus["gold"] + 4.0
        # us_bonds_long deve scendere di ~5
        assert with_bonus["us_bonds_long"] < no_bonus["us_bonds_long"] - 4.0

    def test_force_include_overrides_env_off(self, reset_env):
        """force_include_dedollar=True applica bonus anche con env OFF."""
        probs = _sample_probs()
        bonus = _big_secular_bonus()
        env_off_no_bonus = calculate_final_scores(probs, secular_bonus=bonus)
        forced = calculate_final_scores(probs, secular_bonus=bonus, force_include_dedollar=True)
        assert forced["gold"] > env_off_no_bonus["gold"]

    def test_force_include_false_overrides_env_on(self, reset_env):
        """force_include_dedollar=False ignora bonus anche con env ON."""
        os.environ["USE_DEDOLLAR_BONUS"] = "1"
        probs = _sample_probs()
        bonus = _big_secular_bonus()
        normal = calculate_final_scores(probs, secular_bonus=bonus)
        forced_off = calculate_final_scores(probs, secular_bonus=bonus, force_include_dedollar=False)
        assert forced_off["gold"] < normal["gold"]


class TestTrajectoryEnvFlag:
    def _indicators_for_classifier(self):
        return {
            "gdp_roc": 1.5, "cpi_yoy": 3.0, "unrate": 4.5, "unrate_roc": 0.0,
            "yield_curve_10y2y": 0.5, "initial_claims_roc": 0.0, "lei_roc": 0.5,
            "fed_funds_rate": 4.0,
        }

    def test_env_off_skip_dedollar_pressure(self, reset_env):
        """Con env OFF, DEDOLLAR_REGIME_PRESSURE non viene applicato."""
        probs = _sample_probs()
        ind = self._indicators_for_classifier()
        # dedollar_score alto (0.8) → pressure stagflation +0.20 se env ON
        traj_off = calculate_trajectory(probs, ind, news_sentiment=0.0, dedollar_score=0.8)
        # Verifica che NESSUNA forza sia di tipo "dedollarization"
        dedollar_forces = [f for f in traj_off["forces"] if f["type"] == "dedollarization"]
        assert len(dedollar_forces) == 0, (
            f"Env OFF ma trovate forze dedollar: {dedollar_forces}"
        )

    def test_env_on_includes_dedollar_pressure(self, reset_env):
        os.environ["USE_DEDOLLAR_BONUS"] = "1"
        probs = _sample_probs()
        ind = self._indicators_for_classifier()
        traj_on = calculate_trajectory(probs, ind, news_sentiment=0.0, dedollar_score=0.8)
        dedollar_forces = [f for f in traj_on["forces"] if f["type"] == "dedollarization"]
        assert len(dedollar_forces) > 0, "Env ON ma nessuna forza dedollar applicata"

    def test_force_include_overrides_env_off(self, reset_env):
        probs = _sample_probs()
        ind = self._indicators_for_classifier()
        traj = calculate_trajectory(
            probs, ind, news_sentiment=0.0, dedollar_score=0.8,
            force_include_dedollar=True,
        )
        dedollar_forces = [f for f in traj["forces"] if f["type"] == "dedollarization"]
        assert len(dedollar_forces) > 0


class TestPureDataDrivenDefault:
    """Sanity check: il sistema default-installed e' puro data-driven."""

    def test_no_env_means_dedollar_off(self, reset_env):
        from app.services.config_flags import use_dedollar_bonus
        assert use_dedollar_bonus() is False

    def test_env_falsy_values_means_off(self, reset_env):
        for v in ("0", "false", "no", "off", ""):
            os.environ["USE_DEDOLLAR_BONUS"] = v
            from app.services.config_flags import use_dedollar_bonus
            assert use_dedollar_bonus() is False, f"value '{v}' dovrebbe essere falsy"

    def test_env_truthy_values_means_on(self, reset_env):
        for v in ("1", "true", "yes", "on", "TRUE", "True"):
            os.environ["USE_DEDOLLAR_BONUS"] = v
            from app.services.config_flags import use_dedollar_bonus
            assert use_dedollar_bonus() is True, f"value '{v}' dovrebbe essere truthy"
