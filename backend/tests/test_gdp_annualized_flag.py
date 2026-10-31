"""USE_GDP_ANNUALIZED (S56, 2026-10-31): il PIL trimestrale vs le soglie annue delle condizioni.

Il flag è OFF (A/B walk-forward FAIL, vedi config_flags). Questi test impediscono due difetti:
- un flag che non arriva al classificatore (l'A/B misurerebbe Δ=0 credendo a un vero zero);
- un flag OFF che cambia comunque qualcosa (il sigillo v1.1 sarebbe falsato senza che l'impronta
  lo dica: `compute_classifier_hash` guarda solo REGIME_CONDITIONS).
"""
from __future__ import annotations

import pytest

from app.services.config_flags import annualize_qoq, set_runtime_flag
from app.services.regime.classifier import classify_regime

IND = {"gdp_roc": 0.55, "cpi_yoy": 3.72, "unrate": 4.2}  # PIL e CPI del 31/10/2026


@pytest.fixture(autouse=True)
def _pulito():
    yield
    set_runtime_flag("USE_GDP_ANNUALIZED", None, persist=False)


def _gdp_weak(on: bool | None) -> float:
    set_runtime_flag("USE_GDP_ANNUALIZED", on, persist=False)
    return classify_regime(dict(IND))["conditions_detail"]["stagflation"]["gdp_weak"]["raw_score"]


def test_annualizzato_composto():
    assert annualize_qoq(0.55) == pytest.approx(2.218, abs=1e-3)
    assert annualize_qoq(-1.0) == pytest.approx(-3.940, abs=1e-3)
    assert annualize_qoq(0.0) == 0.0


def test_il_flag_arriva_alle_condizioni():
    # 0,55 trimestrale è «crescita debole» per una soglia annua di 1,5; 2,2 annuo no
    assert _gdp_weak(True) < _gdp_weak(False) - 0.2


def test_di_default_e_spento_e_non_cambia_niente():
    assert _gdp_weak(None) == _gdp_weak(False)
