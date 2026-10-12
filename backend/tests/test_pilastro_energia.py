"""Pilastro shock energetico (S50, flag USE_ENERGY_SHOCK_PILLAR, default OFF).

Lo shock d'offerta del petrolio e' la firma della stagflazione 1973-74 e 2022. Il classificatore lo
vedeva solo nel rapporto oro/petrolio, che nel 2026 (oro quadruplicato) dice "petrolio crollato"
in piena crisi del petrolio. Soglie = quantili storici WTI a 12 mesi 1947-2026 (p90, p75).
"""

import pytest

from app.services.regime import classifier as clf

BASE = {
    "gdp_roc": 0.5, "cpi_yoy": 3.4, "unrate": 4.2, "unrate_roc": 0.0, "yield_curve_10y2y": 0.4,
    "initial_claims_roc": -5.0, "lei_roc": 0.0, "fed_funds_rate": 3.9, "core_pce_yoy": 3.0,
}


def _stag_fit(ind):
    return clf.classify_regime(dict(ind))["fit_scores"]["stagflation"]


def test_flag_spento_il_petrolio_non_cambia_niente(monkeypatch):
    monkeypatch.delenv("USE_ENERGY_SHOCK_PILLAR", raising=False)
    senza = clf.classify_regime(dict(BASE))
    con = clf.classify_regime({**BASE, "oil_yoy": 150.0})
    for r, p in senza["probabilities"].items():
        assert con["probabilities"][r] == pytest.approx(p, abs=1e-9)
    assert "oil_supply_shock" not in con["conditions_detail"]["stagflation"]


def test_flag_acceso_lo_shock_spinge_la_stagflazione(monkeypatch):
    monkeypatch.setenv("USE_ENERGY_SHOCK_PILLAR", "1")
    calmo = _stag_fit({**BASE, "oil_yoy": 0.0})
    shock = _stag_fit({**BASE, "oil_yoy": 150.0})
    assert shock > calmo
    det = clf.classify_regime({**BASE, "oil_yoy": 150.0})["conditions_detail"]["stagflation"]
    assert det["oil_supply_shock"]["raw_score"] > 0.95


def test_al_centro_p90_vale_meta(monkeypatch):
    monkeypatch.setenv("USE_ENERGY_SHOCK_PILLAR", "1")
    det = clf.classify_regime({**BASE, "oil_yoy": clf._OIL_SHOCK_CENTER})["conditions_detail"]["stagflation"]
    assert det["oil_supply_shock"]["raw_score"] == pytest.approx(0.5, abs=1e-6)


def test_dato_assente_e_neutro_basso(monkeypatch):
    monkeypatch.setenv("USE_ENERGY_SHOCK_PILLAR", "1")
    det = clf.classify_regime(dict(BASE))["conditions_detail"]["stagflation"]
    # senza oil_yoy il default e' la mediana storica (0%): niente shock
    assert det["oil_supply_shock"]["raw_score"] < 0.1


def test_il_flag_compare_nel_pannello_dei_flag():
    from app.services.config_flags import get_all_flags_state

    stato = get_all_flags_state()
    assert "USE_ENERGY_SHOCK_PILLAR" in stato
    assert stato["USE_ENERGY_SHOCK_PILLAR"]["source"] in ("default", "env", "runtime")
