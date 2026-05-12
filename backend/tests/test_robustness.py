"""Test robustness analysis (Tier 4.8 roadmap Bridgewater).

Sensitivity grid + walk-forward OOS testano se i risultati (9.2m lead time,
Sharpe 0.88 backtest) sono robusti o data mining. Verifichiamo:
- aggregazione output corretta
- variabilita' stabile su parametri arbitrari
- stability_verdict logic
"""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.services.backtest.robustness import (
    lead_time_sensitivity_grid,
    walk_forward_backtest,
)


def _mock_lead_time_result(hit_rate: float, avg_lead: float | None, n_recessions: int = 8):
    """Crea un LeadTimeReport mock."""
    n_hits = int(round(hit_rate * n_recessions))
    recessions = [
        SimpleNamespace(signal_date=f"date_{i}", lead_months=avg_lead)
        for i in range(n_hits)
    ] + [
        SimpleNamespace(signal_date=None, lead_months=None)
        for _ in range(n_recessions - n_hits)
    ]
    return SimpleNamespace(
        hit_rate=hit_rate,
        avg_lead_months=avg_lead,
        n_recessions_analyzed=n_recessions,
        recessions=recessions,
    )


class TestLeadTimeSensitivityGrid:
    def test_grid_dimensions(self):
        """Grid = |thresholds| × |lookbacks| celle."""
        thresholds = (0.30, 0.40)
        lookbacks = (6, 12, 18)
        with patch(
            "app.services.backtest.robustness.compute_lead_time_report",
            return_value=_mock_lead_time_result(0.75, 9.0),
        ):
            r = lead_time_sensitivity_grid(
                db=None, thresholds=thresholds, lookbacks=lookbacks,
            )
        assert len(r["grid"]) == 2 * 3
        # Verifica che ogni cella abbia i campi richiesti
        for cell in r["grid"]:
            assert "threshold" in cell
            assert "lookback" in cell
            assert "hit_rate" in cell
            assert "avg_lead_months" in cell

    def test_stable_grid_summary_stats(self):
        """Tutte le celle hit_rate=0.75 → mean=0.75, std=0."""
        with patch(
            "app.services.backtest.robustness.compute_lead_time_report",
            return_value=_mock_lead_time_result(0.75, 9.2),
        ):
            r = lead_time_sensitivity_grid(
                db=None, thresholds=(0.30, 0.40, 0.50), lookbacks=(6, 12),
            )
        assert r["hit_rate_mean"] == pytest.approx(0.75)
        assert r["hit_rate_std"] == pytest.approx(0.0, abs=1e-9)
        assert r["avg_lead_mean"] == pytest.approx(9.2)
        assert r["avg_lead_std"] == pytest.approx(0.0, abs=1e-9)

    def test_variable_grid_captures_std(self):
        """Grid con valori diversi → std > 0 (indica sensitivity)."""
        results = [
            _mock_lead_time_result(0.5, 6.0),
            _mock_lead_time_result(0.75, 9.0),
            _mock_lead_time_result(0.9, 12.0),
            _mock_lead_time_result(0.6, 7.0),
        ]
        result_iter = iter(results)
        with patch(
            "app.services.backtest.robustness.compute_lead_time_report",
            side_effect=lambda *a, **kw: next(result_iter),
        ):
            r = lead_time_sensitivity_grid(
                db=None, thresholds=(0.30, 0.40), lookbacks=(6, 12),
            )
        assert r["hit_rate_std"] > 0
        assert r["hit_rate_min"] == pytest.approx(0.5)
        assert r["hit_rate_max"] == pytest.approx(0.9)

    def test_handles_no_hits_for_some_combinations(self):
        """avg_lead_months=None per celle senza hit → escluse da avg_lead stats."""
        results = [
            _mock_lead_time_result(0.0, None),    # nessun hit
            _mock_lead_time_result(0.75, 9.0),
        ]
        result_iter = iter(results)
        with patch(
            "app.services.backtest.robustness.compute_lead_time_report",
            side_effect=lambda *a, **kw: next(result_iter),
        ):
            r = lead_time_sensitivity_grid(
                db=None, thresholds=(0.30,), lookbacks=(6, 12),
            )
        # avg_lead_mean basato solo sulle celle con hit
        assert r["avg_lead_mean"] == pytest.approx(9.0)


class TestWalkForwardBacktest:
    def test_returns_insufficient_data_when_no_periods(self):
        """Finestre 0 → verdict insufficient_data."""
        # Window troppo piccola: start=end stesso giorno
        r = walk_forward_backtest(
            db=None,
            window_start=date(2025, 1, 1),
            window_end=date(2025, 6, 1),  # solo 5 mesi
            test_window_months=36,
        )
        assert r["n_periods"] == 0
        assert r["stability_verdict"] == "insufficient_data"

    def test_window_iteration_logic(self):
        """Con window 24 mesi, step 12, range 60 mesi → 4 periodi
        ([0,24], [12,36], [24,48], [36,60])."""
        def fake_backtest(db, start, end):
            return SimpleNamespace(strategies=[
                SimpleNamespace(
                    name="macro_score_weighted",
                    stats=SimpleNamespace(
                        annualized_return=0.10, sharpe=1.0,
                        max_drawdown=-0.20, calmar=0.5, n_months=24,
                    ),
                ),
            ])

        with patch(
            "app.services.backtest.runner.run_full_backtest",
            side_effect=fake_backtest,
        ):
            r = walk_forward_backtest(
                db=None,
                window_start=date(2020, 1, 1),
                window_end=date(2025, 1, 1),
                test_window_months=24,
                step_months=12,
            )
        # 4 finestre: [2020,2022], [2021,2023], [2022,2024], [2023,2025]
        assert r["n_periods"] == 4

    def test_robust_verdict_when_low_variance(self):
        """Tutti i periodi con Sharpe 0.9 → std=0, verdict robust."""
        def fake_backtest(db, start, end):
            return SimpleNamespace(strategies=[
                SimpleNamespace(
                    name="macro_score_weighted",
                    stats=SimpleNamespace(
                        annualized_return=0.10, sharpe=0.9,
                        max_drawdown=-0.20, calmar=0.5, n_months=24,
                    ),
                ),
            ])

        with patch(
            "app.services.backtest.runner.run_full_backtest",
            side_effect=fake_backtest,
        ):
            r = walk_forward_backtest(
                db=None,
                window_start=date(2020, 1, 1),
                window_end=date(2024, 1, 1),
                test_window_months=24,
                step_months=12,
            )
        assert r["stability_verdict"] == "robust"
        assert r["sharpe_std"] == pytest.approx(0.0, abs=1e-9)

    def test_fragile_verdict_when_high_variance(self):
        """Sharpes oscillanti grosso → fragile."""
        sharpes = iter([1.5, -0.5, 1.2, -0.3])

        def fake_backtest(db, start, end):
            sh = next(sharpes)
            return SimpleNamespace(strategies=[
                SimpleNamespace(
                    name="macro_score_weighted",
                    stats=SimpleNamespace(
                        annualized_return=0.10, sharpe=sh,
                        max_drawdown=-0.20, calmar=0.5, n_months=24,
                    ),
                ),
            ])

        with patch(
            "app.services.backtest.runner.run_full_backtest",
            side_effect=fake_backtest,
        ):
            r = walk_forward_backtest(
                db=None,
                window_start=date(2020, 1, 1),
                window_end=date(2024, 1, 1),
                test_window_months=24,
                step_months=12,
            )
        assert r["stability_verdict"] == "fragile"
        assert r["sharpe_std"] > 0.5
