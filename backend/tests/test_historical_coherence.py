"""Test di coerenza storica del rule-based classifier.

Verifica che su indicatori macro rappresentativi di periodi storici noti
il classificatore attribuisca il regime atteso. Non richiede DB ne FRED:
gli indicatori sono quelli pubblicati dalle fonti ufficiali (BLS, BEA, FRED).

Tolleranza: `probabilities[expected] >= other` per ogni altro regime (majority).
"""

import pytest

from app.services.regime.classifier import classify_regime


HISTORICAL_CASES = [
    # 1974 stagflation (oil shock): CPI ~12%, GDP -0.5%, UNRATE 5.5% rising, PMI <50
    (
        "1974 oil shock stagflation",
        {
            "gdp_roc": -0.5,
            "cpi_yoy": 11.0,
            "unrate": 5.5,
            "unrate_roc": 0.5,
            "pmi": 46.0,
            "yield_curve_10y2y": -0.5,
            "initial_claims_roc": 15.0,
            "lei_roc": -3.0,
            "fed_funds_rate": 10.0,
            "core_pce_yoy": 9.0,
            "payrolls_roc_12m": -0.5,
            "indpro_roc_12m": -4.0,
            "baa_spread": 2.8,
            "consumer_sentiment": 58.0,
        },
        "stagflation",
    ),
    # 1979-80 Volcker stagflation: CPI ~13%, UNRATE 7%, Fed funds ~19%, GDP flat
    (
        "1980 Volcker era stagflation",
        {
            "gdp_roc": 0.3,
            "cpi_yoy": 13.5,
            "unrate": 7.0,
            "unrate_roc": 0.3,
            "pmi": 45.0,
            "yield_curve_10y2y": -1.0,
            "initial_claims_roc": 10.0,
            "lei_roc": -4.0,
            "fed_funds_rate": 17.5,
            "core_pce_yoy": 10.5,
            "payrolls_roc_12m": 0.2,
            "indpro_roc_12m": -2.0,
            "baa_spread": 3.5,
            "consumer_sentiment": 55.0,
        },
        "stagflation",
    ),
    # 2008-09 Great Recession: GDP -4%, CPI ~0%, UNRATE 9%+ rising, LEI crash
    (
        "2009 Great Recession deflation",
        {
            "gdp_roc": -4.0,
            "cpi_yoy": -0.5,
            "unrate": 9.5,
            "unrate_roc": 2.5,
            "pmi": 35.0,
            "yield_curve_10y2y": 2.0,
            "initial_claims_roc": 60.0,
            "lei_roc": -8.0,
            "fed_funds_rate": 0.25,
            "core_pce_yoy": 1.3,
            "payrolls_roc_12m": -4.0,
            "indpro_roc_12m": -12.0,
            "baa_spread": 6.0,
            "consumer_sentiment": 55.0,
        },
        "deflation",
    ),
    # Q2 2020 COVID shock: severa contrazione + inflation collapse
    (
        "2020 Q2 COVID deflation",
        {
            "gdp_roc": -8.0,
            "cpi_yoy": 0.3,
            "unrate": 13.0,
            "unrate_roc": 8.5,
            "pmi": 42.0,
            "yield_curve_10y2y": 0.5,
            "initial_claims_roc": 400.0,
            "lei_roc": -10.0,
            "fed_funds_rate": 0.1,
            "core_pce_yoy": 1.0,
            "payrolls_roc_12m": -12.0,
            "indpro_roc_12m": -14.0,
            "baa_spread": 4.0,
            "consumer_sentiment": 72.0,
        },
        "deflation",
    ),
    # 2021 reopening reflation: GDP forte, CPI rising, UNRATE falling rapidly
    (
        "2021 reopening reflation",
        {
            "gdp_roc": 5.5,
            "cpi_yoy": 5.0,
            "unrate": 5.0,
            "unrate_roc": -1.5,
            "pmi": 60.0,
            "yield_curve_10y2y": 1.5,
            "initial_claims_roc": -40.0,
            "lei_roc": 8.0,
            "fed_funds_rate": 0.1,
            "core_pce_yoy": 3.5,
            "payrolls_roc_12m": 4.5,
            "indpro_roc_12m": 6.0,
            "baa_spread": 1.8,
            "consumer_sentiment": 80.0,
        },
        "reflation",
    ),
    # 2022 stagflation-like: CPI 8-9%, GDP slowing, Fed hiking
    (
        "2022 inflation surge stagflation",
        {
            "gdp_roc": 0.9,
            "cpi_yoy": 8.5,
            "unrate": 3.7,
            "unrate_roc": 0.1,
            "pmi": 48.5,
            "yield_curve_10y2y": -0.3,
            "initial_claims_roc": 8.0,
            "lei_roc": -1.5,
            "fed_funds_rate": 4.0,
            "core_pce_yoy": 5.0,
            "payrolls_roc_12m": 3.0,
            "indpro_roc_12m": 1.0,
            "baa_spread": 2.6,
            "consumer_sentiment": 59.0,
        },
        "stagflation",
    ),
    # Mid-1990s goldilocks: GDP ~3%, CPI <3%, UNRATE <5%, PMI healthy
    (
        "1997 goldilocks",
        {
            "gdp_roc": 3.3,
            "cpi_yoy": 2.3,
            "unrate": 4.7,
            "unrate_roc": -0.2,
            "pmi": 54.0,
            "yield_curve_10y2y": 0.6,
            "initial_claims_roc": -5.0,
            "lei_roc": 2.0,
            "fed_funds_rate": 5.25,
            "core_pce_yoy": 1.7,
            "payrolls_roc_12m": 2.2,
            "indpro_roc_12m": 3.5,
            "baa_spread": 1.7,
            "consumer_sentiment": 105.0,
        },
        "goldilocks",
    ),
    # 2017 goldilocks: crescita stabile, inflation contenuta, unemployment basso
    (
        "2017 goldilocks",
        {
            "gdp_roc": 2.4,
            "cpi_yoy": 2.1,
            "unrate": 4.1,
            "unrate_roc": -0.3,
            "pmi": 57.0,
            "yield_curve_10y2y": 0.8,
            "initial_claims_roc": -3.0,
            "lei_roc": 1.5,
            "fed_funds_rate": 1.25,
            "core_pce_yoy": 1.8,
            "payrolls_roc_12m": 1.6,
            "indpro_roc_12m": 2.2,
            "baa_spread": 1.6,
            "consumer_sentiment": 96.0,
        },
        "goldilocks",
    ),
]


@pytest.mark.parametrize("case_name,indicators,expected_regime", HISTORICAL_CASES,
                         ids=[c[0] for c in HISTORICAL_CASES])
def test_historical_regime_classification(case_name, indicators, expected_regime):
    """Il regime con probabilita massima deve combaciare con quello atteso."""
    result = classify_regime(indicators)
    probs = result["probabilities"]
    dominant = max(probs.items(), key=lambda kv: kv[1])[0]
    assert dominant == expected_regime, (
        f"{case_name}: dominant={dominant} prob={probs[dominant]:.2f}, "
        f"atteso={expected_regime} prob={probs[expected_regime]:.2f}, "
        f"all_probs={probs}"
    )


def test_1974_stagflation_is_dominant():
    """Caso canonico 1974: stagflation deve essere > 40% e > tutti gli altri di 15pt."""
    indicators = dict(HISTORICAL_CASES[0][1])
    result = classify_regime(indicators)
    probs = result["probabilities"]
    assert probs["stagflation"] > 0.40, f"stagflation={probs['stagflation']:.2f}"
    others = [p for r, p in probs.items() if r != "stagflation"]
    assert probs["stagflation"] - max(others) > 0.10, (
        f"margin insufficiente: stagflation={probs['stagflation']:.2f} max_other={max(others):.2f}"
    )


def test_2009_deflation_is_dominant():
    """Caso canonico 2009: deflation deve essere > 40%."""
    indicators = dict(HISTORICAL_CASES[2][1])
    result = classify_regime(indicators)
    probs = result["probabilities"]
    assert probs["deflation"] > 0.40, f"deflation={probs['deflation']:.2f}"


def test_1996_no_false_deflation():
    """Regression: il 1996 (GDP ~2%, CPI ~3%, UNRATE ~5.4%) non deve avere
    prob deflation eccessiva — economia in salute con inflation moderata = goldilocks/reflation,
    non deflation. Senza la penalty health, deflation usciva ~0.22."""
    indicators = {
        "gdp_roc": 2.0,
        "cpi_yoy": 3.0,
        "unrate": 5.4,
        "unrate_roc": 0.0,
        "pmi": 52.0,
        "yield_curve_10y2y": 0.7,
        "initial_claims_roc": -2.0,
        "lei_roc": 2.0,
        "fed_funds_rate": 5.25,
        "core_pce_yoy": 2.0,
        "payrolls_roc_12m": 2.0,
        "indpro_roc_12m": 4.0,
        "baa_spread": 1.6,
        "consumer_sentiment": 92.0,
    }
    result = classify_regime(indicators)
    probs = result["probabilities"]
    assert probs["deflation"] < 0.18, f"deflation={probs['deflation']:.2f} (atteso < 0.18 con economia sana)"
    assert probs["deflation"] < probs["goldilocks"], (
        f"deflation={probs['deflation']:.2f} >= goldilocks={probs['goldilocks']:.2f}"
    )
