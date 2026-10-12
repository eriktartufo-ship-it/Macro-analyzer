"""S50: la strategia LLM del portafoglio AI riceve anche i dati di area euro ed energia.

Il prompt dichiara "TUTTI quelli disponibili, Gemini decide cosa e' rilevante", ma la lista delle
chiavi era scritta a mano: i 22 indicatori di europe_energy non ci arrivavano.
"""
from datetime import date

from app.services.ai_portfolio import llm_strategist as S
from app.services.indicators.europe_energy import PANEL_GROUPS


def _prompt(ind):
    return S._build_user_prompt(
        target_date=date(2026, 10, 12), regime="reflation",
        probabilities={"reflation": 0.48, "stagflation": 0.27, "deflation": 0.1, "goldilocks": 0.15},
        confidence=0.24, indicators=ind, positions=[], learnings=[], price_history=None,
    )


def test_gli_indicatori_europei_entrano_nel_prompt():
    p = _prompt({"cpi_yoy": 3.35, "ea_hicp_yoy": 3.8, "oil_yoy": 87.0, "brent_physical_premium": 24.9})
    assert "Europa ed energia" in p
    assert "ea_hicp_yoy: 3.800" in p
    assert "brent_physical_premium: 24.900" in p


def test_ogni_voce_del_pannello_e_anche_nel_prompt():
    # un'unica lista di verita': se il pannello mostra una chiave, il bot la vede
    chiavi = [k for _, _, items in PANEL_GROUPS for k, _, _ in items]
    p = _prompt({k: 1.0 for k in chiavi})
    mancanti = [k for k in chiavi if f"{k}: 1.000" not in p]
    assert mancanti == []


def test_senza_dati_europei_il_gruppo_non_compare():
    assert "Europa ed energia" not in _prompt({"cpi_yoy": 3.35})
