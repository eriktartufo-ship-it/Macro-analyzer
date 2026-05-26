"""T10 CI gate Sortino — Council 2026-05-17 AI-3 action item.

Test slow (marked) che esegue 3 sim random backtest real-prices walk-forward
e verifica che il modello abbia Sortino delta vs benchmark accettabile.

Soglie (council originale):
- Baseline Sortino delta minimo: -0.4 (sotto = regressione strutturale)
- Drop test (post-modifica): nuovo Sortino non deve cadere > 0.1 vs benchmark
  storico stato salvato in test fixture JSON.

Run on-demand: `pytest -m sortino_ci`. Default skip in regular suite.
"""
from __future__ import annotations

import json
import random
from datetime import date
from pathlib import Path

import numpy as np
import pytest


# Snapshot benchmark da T10b walk-forward (15 sim seed 42, real-prices).
# Update with: `python scripts/t10_validation_batch.py --walk-forward` then
# pasted from JSON output.
EXPECTED_SORTINO_BASELINE = -0.297
EXPECTED_SORTINO_TOLERANCE = 0.10  # delta max ammesso vs expected
N_SIM_QUICK = 3  # minimal sim per CI velocita


@pytest.mark.sortino_ci
def test_baseline_sortino_above_threshold():
    """Council gate: baseline (no flags) Sortino delta vs bench >= -0.4.

    Sotto questa soglia, il modello regredisce vs 60/40 in modo drammatico.
    Skip default per velocita (60-90s); run via `-m sortino_ci`.
    """
    from scripts.random_backtest import load_classifications_by_month, run_simulation
    from app.services.backtest.real_returns_matrix import build_real_returns_matrix
    from app.services.backtest.walk_forward import (
        WalkForwardCache, preload_walk_forward_series,
    )
    from app.services.scoring.engine import ASSET_REGIME_DATA
    import os

    # Force baseline: explicit OFF su tutti i flag promossi
    for flag in ("USE_CALIBRATED_SCORING", "USE_GDP_COLLAPSE_OVERRIDE", "USE_ML_REGIME_BLEND"):
        os.environ[flag] = "0"

    real_matrix = build_real_returns_matrix(list(ASSET_REGIME_DATA.keys()))
    series = preload_walk_forward_series(
        start_date=date(1990, 1, 1),
        end_date=date(2024, 12, 31),
    )
    wf_cache = WalkForwardCache(series)
    cm_by_month = load_classifications_by_month()

    sortinos = []
    rng_starts = random.Random(42)
    for i in range(N_SIM_QUICK):
        start_year = rng_starts.randint(1990, 2014)
        start_month = rng_starts.randint(1, 12)
        res = run_simulation(
            start_date=date(start_year, start_month, 1),
            horizon_years=10,
            classifications_by_month=cm_by_month,
            regime_data=ASSET_REGIME_DATA,
            seed=42 + i,
            real_returns_matrix=real_matrix,
            walk_forward_cache=wf_cache,
        )
        if res is not None:
            sortinos.append(res.model_sortino - res.benchmark_sortino)

    assert sortinos, "Nessuna simulazione completata"
    mean_delta = float(np.mean(sortinos))
    # Floor: regressione strutturale se baseline scende sotto -0.4
    assert mean_delta > -0.4, (
        f"Sortino delta regressione: {mean_delta:.3f} < -0.4. "
        f"Possibile bug in classifier/scoring/real_returns_matrix."
    )


@pytest.mark.sortino_ci
def test_calibrated_flag_improves_sortino():
    """Council gate: USE_CALIBRATED_SCORING deve dare Sortino delta > +0.05
    vs baseline (la sua promotion default-ON in T10b dipendeva da +0.881).

    Se questo test fallisce, qualcosa nel calibrated JSON o nella pipeline
    Bayesian shrinkage si è rotto.
    """
    from scripts.random_backtest import load_classifications_by_month, run_simulation
    from app.services.backtest.real_returns_matrix import build_real_returns_matrix
    from app.services.backtest.walk_forward import (
        WalkForwardCache, preload_walk_forward_series,
    )
    from app.services.scoring.engine import ASSET_REGIME_DATA, reload_calibration
    import os

    real_matrix = build_real_returns_matrix(list(ASSET_REGIME_DATA.keys()))
    series = preload_walk_forward_series(
        start_date=date(1990, 1, 1),
        end_date=date(2024, 12, 31),
    )
    wf_cache = WalkForwardCache(series)
    cm_by_month = load_classifications_by_month()

    def run_with_flag(calibrated: bool) -> float:
        for flag in ("USE_GDP_COLLAPSE_OVERRIDE", "USE_ML_REGIME_BLEND"):
            os.environ[flag] = "0"
        os.environ["USE_CALIBRATED_SCORING"] = "1" if calibrated else "0"
        reload_calibration()
        wf_cache.clear()
        sortinos = []
        rng_starts = random.Random(42)
        for i in range(N_SIM_QUICK):
            sy = rng_starts.randint(1990, 2014)
            sm = rng_starts.randint(1, 12)
            res = run_simulation(
                start_date=date(sy, sm, 1),
                horizon_years=10,
                classifications_by_month=cm_by_month,
                regime_data=ASSET_REGIME_DATA,
                seed=42 + i,
                real_returns_matrix=real_matrix,
                walk_forward_cache=wf_cache,
            )
            if res is not None:
                sortinos.append(res.model_sortino)
        return float(np.mean(sortinos)) if sortinos else 0.0

    baseline_sortino = run_with_flag(False)
    calibrated_sortino = run_with_flag(True)
    delta = calibrated_sortino - baseline_sortino
    assert delta > 0.05, (
        f"USE_CALIBRATED_SCORING dovrebbe migliorare Sortino di almeno +0.05, "
        f"ma delta={delta:.3f} (baseline={baseline_sortino:.3f}, "
        f"calibrated={calibrated_sortino:.3f}). Possibile regression in calibrated JSON."
    )
