"""Il NULL scenario: se non sta succedendo NIENTE, il contesto non deve inclinare nulla.

S42 — bug strutturale trovato col test del NULL scenario (regola
[[feedback-scoring-contesto-moltiplica-non-somma]], che elencava proprio "Macro Analyzer
(regime detection)" fra i sospetti da auditare).

`gdp_collapse_override` scalava i regimi con `severity = sigmoid(severity_raw)`, ma
**sigmoid(0) = 0.5, non 0**. Il commento del codice dichiarava "severity 0 → no shift
(×1.0)": irraggiungibile per costruzione. Effetto misurato a dati NEUTRI (CPI 2.0 =
target Fed, PMI 50, GDP 2.0, VIX 18, unrate 4.5):

    deflation   × 1.655   (+65% PERMANENTE)
    stagflation × 0.591   (-41% PERMANENTE)
    → il top regime diventava `deflation` su dati che sono goldilocks da manuale.

Peggio: con severity_raw NEGATIVO (economia FORTE) la sigmoid resta > 0 → deflation
veniva premiata (×1.36) proprio mentre l'economia andava benissimo.

Il fix NON tocca la scala di `gdp_collapse_severity` (è output diagnostico e i test
esistenti la verificano su > 0.7 / > 0.80): corregge solo l'INTENSITÀ dello shift.
La proprietà che conta è `0 × k = 0`: il contesto non può creare un segnale dal nulla.
"""
from __future__ import annotations

import pytest

from app.services.regime.classifier import classify_regime

_REGIMES = ("reflation", "stagflation", "deflation", "goldilocks")


def _neutral() -> dict:
    """Economia perfettamente neutra: CPI al target Fed, PMI 50, GDP a trend.

    Questo scenario È goldilocks da manuale (crescita moderata + inflazione ancorata).
    """
    return {
        "gdp_roc": 2.0, "cpi_yoy": 2.0, "core_pce_yoy": 2.0, "unrate": 4.5,
        "unrate_roc": 0.0, "pmi": 50.0, "lei_roc": 0.0, "initial_claims_roc": 0.0,
        "payrolls_roc_12m": 1.5, "indpro_roc_12m": 0.0, "vix": 18.0, "nfci": 0.0,
        "yield_curve_10y2y": 1.0, "yield_curve_10y3m": 1.0, "fed_funds_rate": 2.5,
        "breakeven_10y": 2.0, "housing_starts_roc_12m": 0.0, "gdp_yoy_dfm": 2.0,
    }


def _strong_economy() -> dict:
    """Economia FORTE: severity_raw negativo → nessuno shift verso deflation."""
    ind = _neutral()
    ind.update({"gdp_roc": 4.0, "unrate_roc": -0.5, "vix": 12.0, "breakeven_10y": 2.6})
    return ind


@pytest.fixture(autouse=True)
def _isolate_ml(monkeypatch):
    """Il blend ML è un bias SEPARATO (+0.099 su deflation a dati neutri, misurato in
    S42): va escluso o misurerebbe due cose insieme."""
    monkeypatch.setenv("USE_ML_REGIME_BLEND", "0")
    monkeypatch.setenv("USE_GDP_COLLAPSE_OVERRIDE", "1")


def test_null_scenario_non_produce_deflation():
    """A dati neutri il top NON può essere deflation: era il bug (0.382 vs goldilocks 0.317)."""
    out = classify_regime(_neutral())
    assert out["regime"] != "deflation", (
        f"pavimento: a dati neutri il modello grida deflation "
        f"(probs={ {k: round(out['probabilities'][k], 3) for k in _REGIMES} })"
    )


def test_null_scenario_il_contesto_non_inclina():
    """L'override deve essere ~neutro quando non c'è nessun collasso in corso.

    Confronto ON vs OFF sugli STESSI dati neutri: le probabilità devono coincidere
    (tolleranza per il lieve tilt legittimo: breakeven 2.0 < 2.3 = pressione
    disinflazionaria vera, piccola).
    """
    import os

    on = classify_regime(_neutral())["probabilities"]
    os.environ["USE_GDP_COLLAPSE_OVERRIDE"] = "0"
    try:
        off = classify_regime(_neutral())["probabilities"]
    finally:
        os.environ["USE_GDP_COLLAPSE_OVERRIDE"] = "1"

    delta_defl = on["deflation"] - off["deflation"]
    assert delta_defl < 0.05, (
        f"l'override sposta deflation di {delta_defl:+.3f} su dati neutri "
        f"(col bug era +0.102)"
    )


def test_economia_forte_non_premia_deflation():
    """severity_raw NEGATIVO → shift ZERO. Col bug: sigmoid 0.3 → deflation ×1.36."""
    import os

    on = classify_regime(_strong_economy())["probabilities"]
    os.environ["USE_GDP_COLLAPSE_OVERRIDE"] = "0"
    try:
        off = classify_regime(_strong_economy())["probabilities"]
    finally:
        os.environ["USE_GDP_COLLAPSE_OVERRIDE"] = "1"

    assert on["deflation"] <= off["deflation"] + 1e-6, (
        "con economia FORTE l'override non deve premiare deflation "
        f"(ON {on['deflation']:.3f} vs OFF {off['deflation']:.3f})"
    )


def test_collasso_vero_muove_ancora_verso_deflation():
    """ANTI-REGRESSIONE: il fix non deve rendere l'override inutile quando serve DAVVERO."""
    import os

    collapse = {
        "gdp_roc": -3.0, "cpi_yoy": 4.5, "unrate": 7.0, "unrate_roc": 2.5,
        "pmi": 35.0, "vix": 38.0, "breakeven_10y": 1.5, "core_pce_yoy": 3.5,
        "lei_roc": -5.0, "initial_claims_roc": 40.0,
    }
    on = classify_regime(collapse)["probabilities"]
    os.environ["USE_GDP_COLLAPSE_OVERRIDE"] = "0"
    try:
        off = classify_regime(collapse)["probabilities"]
    finally:
        os.environ["USE_GDP_COLLAPSE_OVERRIDE"] = "1"

    assert on["deflation"] > off["deflation"], "l'override deve ancora agire nei collassi veri"
    assert on["deflation"] > on["stagflation"], "in un collasso disinflazionario deflation deve battere stagflation"
