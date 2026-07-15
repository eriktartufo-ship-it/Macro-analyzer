"""La cache del regime DEVE distinguere le configurazioni del classifier.

Bug trovato in S42 (stessa famiglia del bug cache Yahoo di S41b): la griglia
settimanale veniva cachata su disco con chiave = SOLA DATA. Cambiando un flag del
classifier, `build_weekly_grid` trovava i venerdì già in cache e restituiva le
classificazioni calcolate con i flag di un ALTRO run, senza errore.

Conseguenza: qualunque A/B su una feature del classifier misurava Δ=0 → falso
negativo ("la feature non serve") impossibile da distinguere da un vero zero.

Prova che il bug era reale (misurata prima del fix, 2026-07-15, finestra 2020Q1):
    classify_at diretto, USE_MOMENTUM_PILLARS ON vs OFF → Δ p_deflation = 0.003175
    build_weekly_grid con cache                          → Δ = 0.0        ← il bug
    build_weekly_grid use_cache=False                    → Δ = 0.005148

Questi test girano SENZA rete: `classify_at` è monkeypatchato con una fake che
legge un flag reale, così misurano la CHIAVE DI CACHE, non il classifier.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from app.services.backtest.daily_regime_grid import build_weekly_grid
from app.services.config_flags import (
    classifier_flags_fingerprint,
    set_runtime_flag,
    use_momentum_pillars,
)

_S, _E = date(2020, 3, 1), date(2020, 3, 31)


@pytest.fixture
def _restore_flags():
    """Ripristina i flag toccati (persist=False: mai scrivere sul DB nei test)."""
    yield
    for f in ("USE_MOMENTUM_PILLARS", "USE_LEVEL_ACCEL_PILLARS",
              "USE_AI_PORTFOLIO_DEPLOY_BOOST"):
        set_runtime_flag(f, None, persist=False)


@pytest.fixture
def _fake_classify(monkeypatch):
    """classify_at fake: l'output DIPENDE da un flag reale del classifier."""
    calls: list[date] = []

    def _fake(series, as_of):
        calls.append(as_of)
        p = 0.90 if use_momentum_pillars() else 0.10
        return {
            "regime": "deflation" if p > 0.5 else "goldilocks",
            "confidence": 0.5,
            "probs": {"reflation": (1 - p) / 3, "stagflation": (1 - p) / 3,
                      "deflation": p, "goldilocks": (1 - p) / 3},
        }

    monkeypatch.setattr("app.services.backtest.walk_forward.classify_at", _fake)
    return calls


def test_grid_cache_distingue_i_flag_del_classifier(tmp_path, _fake_classify, _restore_flags):
    """Il test che era ROSSO col bug: stessa finestra, flag diverso → griglia diversa."""
    cache = tmp_path / "grid.parquet"

    set_runtime_flag("USE_MOMENTUM_PILLARS", True, persist=False)
    on = build_weekly_grid({}, _S, _E, cache_path=cache).frame

    set_runtime_flag("USE_MOMENTUM_PILLARS", False, persist=False)
    off = build_weekly_grid({}, _S, _E, cache_path=cache).frame

    assert not on.empty and len(on) == len(off)
    # col bug: delta esattamente 0.0 perché la 2a build leggeva la cache della 1a
    delta = (on["p_deflation"] - off["p_deflation"]).abs().max()
    assert delta > 0.1, "la cache ha restituito le classificazioni dei flag sbagliati"
    assert set(on["regime"]) == {"deflation"}
    assert set(off["regime"]) == {"goldilocks"}


def test_grid_cache_riusata_a_flag_invariati(tmp_path, _fake_classify, _restore_flags):
    """Il fix non deve rompere la cache: stessi flag → zero ricalcoli alla 2a build."""
    cache = tmp_path / "grid.parquet"
    set_runtime_flag("USE_MOMENTUM_PILLARS", True, persist=False)

    build_weekly_grid({}, _S, _E, cache_path=cache)
    n_dopo_prima = len(_fake_classify)
    assert n_dopo_prima > 0

    build_weekly_grid({}, _S, _E, cache_path=cache)
    assert len(_fake_classify) == n_dopo_prima, "cache non riusata: ricalcolo inutile"


def test_grid_cache_ritorna_alla_griglia_originale(tmp_path, _fake_classify, _restore_flags):
    """A/B ripetuto: tornando ai flag di partenza si riusa la cache, senza ricalcolo."""
    cache = tmp_path / "grid.parquet"

    set_runtime_flag("USE_MOMENTUM_PILLARS", True, persist=False)
    build_weekly_grid({}, _S, _E, cache_path=cache)
    set_runtime_flag("USE_MOMENTUM_PILLARS", False, persist=False)
    build_weekly_grid({}, _S, _E, cache_path=cache)

    n = len(_fake_classify)
    set_runtime_flag("USE_MOMENTUM_PILLARS", True, persist=False)
    back = build_weekly_grid({}, _S, _E, cache_path=cache).frame
    assert len(_fake_classify) == n, "la griglia ON doveva essere ancora in cache"
    assert set(back["regime"]) == {"deflation"}


# --- fingerprint --------------------------------------------------------------

def test_fingerprint_cambia_col_flag_del_classifier(_restore_flags):
    set_runtime_flag("USE_MOMENTUM_PILLARS", True, persist=False)
    a = classifier_flags_fingerprint()
    set_runtime_flag("USE_MOMENTUM_PILLARS", False, persist=False)
    assert classifier_flags_fingerprint() != a


def test_fingerprint_copre_anche_i_flag_assenti_da_get_all_flags_state(_restore_flags):
    """USE_LEVEL_ACCEL_PILLARS manca nelle liste di get_all_flags_state().

    È il flag T20 (livello-vs-accelerazione): un fingerprint costruito su quelle
    liste lo mancherebbe in silenzio. L'introspezione delle use_*() lo copre.
    """
    set_runtime_flag("USE_LEVEL_ACCEL_PILLARS", False, persist=False)
    a = classifier_flags_fingerprint()
    set_runtime_flag("USE_LEVEL_ACCEL_PILLARS", True, persist=False)
    assert classifier_flags_fingerprint() != a


def test_fingerprint_ignora_i_flag_non_del_classifier(_restore_flags):
    """I flag ai_portfolio_* vivono in ai_portfolio/, mai in regime/ (verificato via
    grep in S42) → non devono invalidare una griglia che contiene solo regimi."""
    set_runtime_flag("USE_AI_PORTFOLIO_DEPLOY_BOOST", True, persist=False)
    a = classifier_flags_fingerprint()
    set_runtime_flag("USE_AI_PORTFOLIO_DEPLOY_BOOST", False, persist=False)
    assert classifier_flags_fingerprint() == a


def test_fingerprint_e_stabile_e_corto():
    a = classifier_flags_fingerprint()
    assert a == classifier_flags_fingerprint()
    assert a.isalnum() and 8 <= len(a) <= 16
