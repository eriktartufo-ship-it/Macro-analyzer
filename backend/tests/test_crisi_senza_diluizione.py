"""Il punteggio di crisi non deve DILUIRE i trigger nuovi (S52, 2026-10-20).

Misurato (scripts/s52_diluizione_crisi.py): UN trigger in piu' sempre spento abbassava lo score nel
99% dei 460 punti 2004-2026 e faceva scendere di livello 139 punti, fra cui l'8/9/2008 (elevated ->
moderate). I trigger ADDITIVI entrano nel numeratore quando scattano, mai nel denominatore.
"""
import pytest

from app.services.regime import crisis_indicator as ci

CRITICI = ["vix_spike", "credit_spread_extreme", "deflation_prob_high", "stagflation_prob_high",
           "yield_curve_inverted"]
ALTRI = ["vix_elevated", "vix_compressed", "credit_spread_wide", "cpi_hot", "cpi_collapse", "breakeven_low",
         "breakeven_high", "nfci_tight", "gdp_weak", "unrate_rising"]


def _t(name, fired):
    return {"name": name, "value": 0.0, "threshold": 0.0, "operator": "ge", "fired": fired, "description": ""}


def _base(critici_accesi=2, altri_accesi=2):
    return ([_t(n, i < critici_accesi) for i, n in enumerate(CRITICI)]
            + [_t(n, i < altri_accesi) for i, n in enumerate(ALTRI)])


def test_set_di_oggi_score_invariato():
    # 2 critici (x2) + 2 altri su denominatore 5x2 + 10 = 20 -> 0,30, come prima
    _, score = ci._compute_risk_level(_base(), "stagflation_debasement")
    assert score == pytest.approx(6 / 20)


def test_trigger_additivo_spento_non_cambia_niente():
    prima = ci._compute_risk_level(_base(), "stagflation_debasement")
    dopo = ci._compute_risk_level(_base() + [_t("eu_sovereign_stress", False)], "stagflation_debasement")
    assert dopo == prima


def test_trigger_additivo_acceso_aggiunge_senza_diluire():
    _, prima = ci._compute_risk_level(_base(), "stagflation_debasement")
    _, dopo = ci._compute_risk_level(_base() + [_t("eu_sovereign_stress", True)], "stagflation_debasement")
    assert dopo == pytest.approx(prima + 1 / 20)


def test_un_trigger_NON_additivo_continua_a_pesare_nel_denominatore():
    # la correzione vale solo per i trigger dichiarati additivi: gli altri restano come prima
    _, prima = ci._compute_risk_level(_base(), "stagflation_debasement")
    _, dopo = ci._compute_risk_level(_base() + [_t("un_trigger_qualunque", False)], "stagflation_debasement")
    assert dopo < prima


def test_lo_score_resta_entro_1():
    tutti = _base(5, 10) + [_t("eu_sovereign_stress", True)]
    _, score = ci._compute_risk_level(tutti, "stagflation_debasement")
    assert score == pytest.approx(1.0)
