"""Test walk-forward retraining annuale (validation/walkforward.py)."""
from __future__ import annotations

import pytest

from app.services.validation.walkforward import (
    WalkForwardAggregate,
    WalkForwardResult,
    walk_forward_evaluation,
    walk_forward_per_year_report,
)


class TestWalkForwardEvaluation:
    def test_aggregate_returns_walk_forward_aggregate(self):
        agg = walk_forward_evaluation()
        assert isinstance(agg, WalkForwardAggregate)

    def test_cutoffs_evaluated_positive(self):
        agg = walk_forward_evaluation()
        assert agg.cutoffs_evaluated > 5

    def test_global_accuracy_in_valid_range(self):
        agg = walk_forward_evaluation()
        assert 0.0 <= agg.global_accuracy <= 1.0

    def test_global_accuracy_above_random(self):
        """Walk-forward accuracy deve battere random (25%)."""
        agg = walk_forward_evaluation()
        assert agg.global_accuracy > 0.40, (
            f"Walk-forward accuracy {agg.global_accuracy:.2%} too low (random=25%)"
        )

    def test_min_max_accuracy_consistent(self):
        agg = walk_forward_evaluation()
        assert agg.min_accuracy <= agg.global_accuracy <= agg.max_accuracy


class TestWalkForwardPerYear:
    def test_returns_list_of_results(self):
        results = walk_forward_per_year_report()
        assert isinstance(results, list)
        assert all(isinstance(r, WalkForwardResult) for r in results)

    def test_each_result_has_episodes(self):
        results = walk_forward_per_year_report()
        for r in results:
            assert r.n_test == r.total
            assert len(r.per_episode) == r.n_test
            assert r.correct <= r.total

    def test_training_size_grows_over_time(self):
        """Walk-forward: cutoff più recente → training set più grande."""
        results = walk_forward_per_year_report()
        if len(results) >= 3:
            first_train = results[0].n_train
            last_train = results[-1].n_train
            assert last_train >= first_train


class TestWalkForwardSpecificEpisodes:
    """Verifica casi specifici noti del walk-forward."""

    def test_2008_gfc_caught_oos(self):
        """2008 GFC: training fino al 2007, deve predire deflation."""
        results = walk_forward_per_year_report(cutoff_start=2007, cutoff_end=2008)
        gfc_results = [r for r in results if r.cutoff_year == 2007]
        assert len(gfc_results) > 0
        gfc = gfc_results[0]
        gfc_ep = [ep for ep in gfc.per_episode if "GFC" in ep["name"]]
        if gfc_ep:
            assert gfc_ep[0]["correct"], "2008 GFC not caught walk-forward"

    def test_2020_covid_caught_oos(self):
        """2020 COVID: training fino al 2019, deve predire deflation."""
        results = walk_forward_per_year_report(cutoff_start=2019, cutoff_end=2020)
        covid_results = [r for r in results if r.cutoff_year == 2019]
        if covid_results:
            covid = covid_results[0]
            covid_ep = [ep for ep in covid.per_episode if "COVID" in ep["name"]]
            if covid_ep:
                assert covid_ep[0]["correct"], "2020 COVID not caught walk-forward"
