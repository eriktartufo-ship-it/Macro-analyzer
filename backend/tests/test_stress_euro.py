"""Trigger stress sovrano euro nell'indicatore di crisi (S50, flag USE_EU_STRESS_TRIGGER, default OFF)."""

from app.services.regime import crisis_indicator as ci

IND = {"gdp_roc": 1.0, "cpi_yoy": 2.5, "vix": 18.0, "baa_spread": 2.0}
PROBS = {"reflation": 0.4, "stagflation": 0.2, "deflation": 0.2, "goldilocks": 0.2}


def _eu(trigs):
    return [t for t in trigs if t["name"] == "eu_sovereign_stress"]


def test_flag_spento_nessun_trigger(monkeypatch):
    monkeypatch.setenv("USE_EU_STRESS_TRIGGER", "0")
    assert _eu(ci._evaluate_triggers({**IND, "ea_sovereign_spread_10y": 1.9}, PROBS)) == []


def test_flag_acceso_scatta_sopra_il_p90(monkeypatch):
    monkeypatch.setenv("USE_EU_STRESS_TRIGGER", "1")
    alto = _eu(ci._evaluate_triggers({**IND, "ea_sovereign_spread_10y": 1.9}, PROBS))
    basso = _eu(ci._evaluate_triggers({**IND, "ea_sovereign_spread_10y": 0.5}, PROBS))
    assert alto and alto[0]["fired"] is True
    assert alto[0]["threshold"] == ci._THRESHOLDS["eu_sovereign_spread_p90"]
    assert basso and basso[0]["fired"] is False


def test_senza_dato_il_trigger_non_esiste(monkeypatch):
    # prima del 2004 non c'e' la curva BCE: niente trigger, niente diluizione del punteggio
    monkeypatch.setenv("USE_EU_STRESS_TRIGGER", "1")
    assert _eu(ci._evaluate_triggers(dict(IND), PROBS)) == []


def test_il_flag_compare_nel_pannello_dei_flag():
    from app.services.config_flags import get_all_flags_state

    assert "USE_EU_STRESS_TRIGGER" in get_all_flags_state()


def test_acceso_di_default_dal_20_10(monkeypatch):
    # S52: cancello I2-bis PASS dopo la correzione della diluizione
    from app.services.config_flags import get_all_flags_state, use_eu_stress_trigger

    monkeypatch.delenv("USE_EU_STRESS_TRIGGER", raising=False)
    assert use_eu_stress_trigger() is True
    assert get_all_flags_state()["USE_EU_STRESS_TRIGGER"]["value"] is True
