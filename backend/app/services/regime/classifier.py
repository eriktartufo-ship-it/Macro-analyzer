"""Regime Classifier — Classifica il regime macro corrente in 4 quadranti.

Quadrante macro (crescita x inflazione):
  - Reflation:   crescita forte + inflazione in salita (growth/recovery)
  - Stagflation: crescita debole + inflazione alta
  - Deflation:   crescita debole/negativa + inflazione bassa/in calo
  - Goldilocks:  crescita moderata + inflazione bassa

Ogni regime ha condizioni pesate. Gli score raw vengono normalizzati
in probabilita (somma = 1.0). Il confidence score misura la concordanza
tra le condizioni.
"""

from typing import Any

REGIMES = ["reflation", "stagflation", "deflation", "goldilocks"]

# Condizioni per ogni regime con pesi (somma per regime = 1.0)
REGIME_CONDITIONS = {
    "reflation": {
        "gdp_strong": {"weight": 0.13, "description": "GDP ROC > 2%"},
        "pmi_expansion": {"weight": 0.11, "description": "PMI > 50"},
        "inflation_rising": {"weight": 0.09, "description": "CPI YoY > 2.5%"},
        "unemployment_low_or_falling": {"weight": 0.08, "description": "Unemployment < 5% o ROC < 0"},
        "claims_falling": {"weight": 0.06, "description": "Initial claims ROC < 0"},
        "lei_positive": {"weight": 0.06, "description": "LEI ROC > 0"},
        "yield_curve_steep": {"weight": 0.06, "description": "10Y-2Y spread > 0.5"},
        "policy_accommodative": {"weight": 0.06, "description": "Fed funds < 3%"},
        "payrolls_growth": {"weight": 0.08, "description": "Nonfarm payrolls YoY > 1.5%"},
        "indpro_growth": {"weight": 0.07, "description": "Industrial production YoY > 2%"},
        "credit_spread_tight": {"weight": 0.06, "description": "BAA-10Y spread < 2%"},
        "housing_expansion": {"weight": 0.05, "description": "Housing starts YoY > 3%"},
        "financial_conditions_loose": {"weight": 0.05, "description": "NFCI < -0.1 (credit easy)"},
        "vix_low": {"weight": 0.04, "description": "VIX < 18 (risk-on)"},
    },
    "stagflation": {
        "inflation_high": {"weight": 0.17, "description": "CPI YoY > 4%"},
        "gdp_weak": {"weight": 0.13, "description": "GDP ROC < 1.5%"},
        "pmi_weak": {"weight": 0.10, "description": "PMI < 50"},
        "unemployment_rising": {"weight": 0.08, "description": "Unemployment ROC > 0.2"},
        "policy_restrictive": {"weight": 0.05, "description": "Fed funds > 4%"},
        "lei_negative": {"weight": 0.05, "description": "LEI ROC < 0"},
        "claims_rising": {"weight": 0.04, "description": "Initial claims ROC > 5%"},
        "yield_curve_stress": {"weight": 0.04, "description": "10Y-2Y spread < 0.5"},
        "core_pce_high": {"weight": 0.08, "description": "Core PCE YoY > 3.5% (Fed-preferred)"},
        "credit_spread_wide": {"weight": 0.06, "description": "BAA-10Y spread > 2.3%"},
        "sentiment_low": {"weight": 0.03, "description": "Consumer sentiment < 70"},
        "breakeven_high": {"weight": 0.08, "description": "Breakeven 10Y > 2.5% (inflation expect)"},
        "vix_elevated": {"weight": 0.05, "description": "VIX > 25 (risk-off)"},
        "housing_slowdown": {"weight": 0.04, "description": "Housing starts YoY < 0"},
    },
    "deflation": {
        "gdp_negative_or_decelerating": {"weight": 0.14, "description": "GDP ROC < 1%"},
        "pmi_contraction": {"weight": 0.12, "description": "PMI < 50"},
        "inflation_low": {"weight": 0.12, "description": "CPI YoY < 2%"},
        "lei_negative": {"weight": 0.09, "description": "LEI ROC < 0"},
        "claims_rising": {"weight": 0.07, "description": "Initial claims ROC > 0"},
        "unemployment_rising": {"weight": 0.07, "description": "Unemployment ROC > 0"},
        "yield_curve_flat_or_inverted": {"weight": 0.05, "description": "10Y-2Y spread < 0.5"},
        "credit_stress": {"weight": 0.04, "description": "Claims spike + yield inversion"},
        "indpro_contraction": {"weight": 0.06, "description": "Industrial production YoY < 0"},
        "payrolls_slowdown": {"weight": 0.05, "description": "Nonfarm payrolls YoY < 0.8%"},
        "credit_spread_wide": {"weight": 0.03, "description": "BAA-10Y spread > 2.5%"},
        "vix_spike": {"weight": 0.06, "description": "VIX > 30 (panic)"},
        "nfci_tight": {"weight": 0.05, "description": "NFCI > 0.3 (credit stress)"},
        "breakeven_collapse": {"weight": 0.05, "description": "Breakeven 10Y < 1.5%"},
    },
    "goldilocks": {
        "gdp_moderate": {"weight": 0.11, "description": "GDP ROC 1.5-3%"},
        "inflation_low": {"weight": 0.13, "description": "CPI YoY < 2.5%"},
        "unemployment_very_low": {"weight": 0.11, "description": "Unemployment < 4%"},
        "pmi_healthy": {"weight": 0.09, "description": "PMI 52-57"},
        "yield_curve_normal": {"weight": 0.07, "description": "10Y-2Y spread 0.5-2.0"},
        "claims_low": {"weight": 0.05, "description": "Initial claims ROC < -3%"},
        "policy_neutral": {"weight": 0.05, "description": "Fed funds 1-3%"},
        "lei_positive": {"weight": 0.04, "description": "LEI ROC > 0"},
        "core_pce_contained": {"weight": 0.07, "description": "Core PCE YoY < 2.3%"},
        "credit_spread_tight": {"weight": 0.06, "description": "BAA-10Y spread < 1.8%"},
        "sentiment_high": {"weight": 0.05, "description": "Consumer sentiment > 82"},
        "vix_calm": {"weight": 0.06, "description": "VIX < 16 (tranquility)"},
        "financial_conditions_easy": {"weight": 0.06, "description": "NFCI < -0.3"},
        "breakeven_stable": {"weight": 0.05, "description": "Breakeven 10Y 1.7-2.3%"},
    },
}


def _evaluate_condition(condition_name: str, regime: str, indicators: dict[str, float]) -> float:
    """Valuta una singola condizione e ritorna score 0-1.

    Usa funzioni sigmoidali soft per evitare cutoff binari.
    """
    gdp = indicators.get("gdp_roc", 0.0)
    pmi = indicators.get("pmi", 50.0)
    cpi = indicators.get("cpi_yoy", 2.0)
    unrate = indicators.get("unrate", 4.5)
    unrate_roc = indicators.get("unrate_roc", 0.0)
    yield_spread = indicators.get("yield_curve_10y2y", 1.0)
    claims_roc = indicators.get("initial_claims_roc", 0.0)
    lei_roc = indicators.get("lei_roc", 0.0)
    fed_funds = indicators.get("fed_funds_rate", 2.5)
    # Nuovi indicatori — default neutri se assenti (core_pce traccia CPI)
    core_pce = indicators.get("core_pce_yoy", cpi)
    payrolls_roc = indicators.get("payrolls_roc_12m", 1.5)
    indpro_roc = indicators.get("indpro_roc_12m", 1.5)
    baa_spread = indicators.get("baa_spread", 2.0)
    sentiment = indicators.get("consumer_sentiment", 75.0)
    # Indicatori finanziari e di aspettative (default neutri)
    vix = indicators.get("vix", 18.0)  # fear gauge, storicamente 12-20 calm
    nfci = indicators.get("nfci", 0.0)  # Chicago Fed FCI, 0 = neutral, + = tight
    breakeven = indicators.get("breakeven_10y", 2.0)  # inflation expectations
    yield_3m = indicators.get("yield_curve_10y3m", 1.0)  # 10y-3m spread
    housing_roc = indicators.get("housing_starts_roc_12m", 0.0)  # housing YoY

    # --- REFLATION conditions ---
    if condition_name == "gdp_strong":
        return _sigmoid(gdp, center=2.0, scale=1.5)
    elif condition_name == "pmi_expansion" and regime == "reflation":
        return _sigmoid(pmi, center=52.0, scale=3.0)
    elif condition_name == "inflation_rising":
        return _sigmoid(cpi, center=2.8, scale=1.0)
    elif condition_name == "unemployment_low_or_falling":
        # Combina: o livello basso o in calo
        low_level = _sigmoid(-unrate, center=-4.5, scale=1.0)
        falling = _sigmoid(-unrate_roc, center=0.1, scale=0.3)
        return max(low_level, falling)
    elif condition_name == "claims_falling" and regime == "reflation":
        return _sigmoid(-claims_roc, center=3.0, scale=5.0)
    elif condition_name == "lei_positive" and regime == "reflation":
        return _sigmoid(lei_roc, center=0.5, scale=1.0)
    elif condition_name == "yield_curve_steep":
        return _sigmoid(yield_spread, center=0.8, scale=0.8)
    elif condition_name == "policy_accommodative":
        return _sigmoid(-fed_funds, center=-2.5, scale=1.5)

    # --- STAGFLATION conditions ---
    elif condition_name == "inflation_high":
        return _sigmoid(cpi, center=4.0, scale=1.5)
    elif condition_name == "gdp_weak":
        return _sigmoid(-gdp, center=-1.0, scale=1.0)
    elif condition_name == "pmi_weak":
        return _sigmoid(-pmi, center=-49.5, scale=1.5)
    elif condition_name == "unemployment_rising" and regime == "stagflation":
        return _sigmoid(unrate_roc, center=0.15, scale=0.2)
    elif condition_name == "policy_restrictive":
        return _sigmoid(fed_funds, center=3.5, scale=1.0)
    elif condition_name == "lei_negative" and regime == "stagflation":
        return _sigmoid(-lei_roc, center=0.3, scale=0.8)
    elif condition_name == "claims_rising" and regime == "stagflation":
        return _sigmoid(claims_roc, center=4.0, scale=4.0)
    elif condition_name == "yield_curve_stress":
        return _sigmoid(-yield_spread, center=-0.3, scale=0.5)

    # --- DEFLATION conditions ---
    elif condition_name == "gdp_negative_or_decelerating":
        return _sigmoid(-gdp, center=-0.5, scale=1.0)
    elif condition_name == "pmi_contraction":
        return _sigmoid(-pmi, center=-49.0, scale=2.5)
    elif condition_name == "inflation_low" and regime == "deflation":
        return _sigmoid(-cpi, center=-2.0, scale=0.8)
    elif condition_name == "lei_negative" and regime == "deflation":
        return _sigmoid(-lei_roc, center=0.5, scale=1.0)
    elif condition_name == "claims_rising" and regime == "deflation":
        return _sigmoid(claims_roc, center=3.0, scale=5.0)
    elif condition_name == "unemployment_rising" and regime == "deflation":
        return _sigmoid(unrate_roc, center=0.1, scale=0.3)
    elif condition_name == "yield_curve_flat_or_inverted":
        return _sigmoid(-yield_spread, center=-0.3, scale=0.5)
    elif condition_name == "credit_stress":
        # Combo: claims spike + yield inversion
        claims_stress = _sigmoid(claims_roc, center=15.0, scale=10.0)
        yield_stress = _sigmoid(-yield_spread, center=0.0, scale=0.3)
        return claims_stress * 0.5 + yield_stress * 0.5

    # --- GOLDILOCKS conditions ---
    elif condition_name == "gdp_moderate":
        return _bell(gdp, center=2.2, width=1.2)
    elif condition_name == "inflation_low" and regime == "goldilocks":
        return _sigmoid(-cpi, center=-2.3, scale=0.8)
    elif condition_name == "unemployment_very_low":
        return _sigmoid(-unrate, center=-3.8, scale=0.6)
    elif condition_name == "pmi_healthy":
        return _bell(pmi, center=54.5, width=3.0)
    elif condition_name == "yield_curve_normal":
        return _bell(yield_spread, center=1.25, width=1.0)
    elif condition_name == "claims_low":
        return _sigmoid(-claims_roc, center=3.0, scale=4.0)
    elif condition_name == "policy_neutral":
        return _bell(fed_funds, center=2.0, width=1.5)
    elif condition_name == "lei_positive" and regime == "goldilocks":
        return _sigmoid(lei_roc, center=0.3, scale=0.8)

    # --- Nuove condizioni (cross-regime) ---
    elif condition_name == "payrolls_growth":
        return _sigmoid(payrolls_roc, center=1.5, scale=0.8)
    elif condition_name == "payrolls_slowdown":
        return _sigmoid(-payrolls_roc, center=-0.8, scale=0.8)
    elif condition_name == "indpro_growth":
        return _sigmoid(indpro_roc, center=2.0, scale=1.5)
    elif condition_name == "indpro_contraction":
        return _sigmoid(-indpro_roc, center=0.0, scale=1.2)
    elif condition_name == "core_pce_high":
        return _sigmoid(core_pce, center=3.5, scale=0.8)
    elif condition_name == "core_pce_contained":
        return _sigmoid(-core_pce, center=-2.3, scale=0.6)
    elif condition_name == "credit_spread_tight":
        return _sigmoid(-baa_spread, center=-1.9, scale=0.5)
    elif condition_name == "credit_spread_wide" and regime == "stagflation":
        return _sigmoid(baa_spread, center=2.3, scale=0.5)
    elif condition_name == "credit_spread_wide" and regime == "deflation":
        return _sigmoid(baa_spread, center=2.5, scale=0.6)
    elif condition_name == "sentiment_high":
        return _sigmoid(sentiment, center=82.0, scale=6.0)
    elif condition_name == "sentiment_low":
        return _sigmoid(-sentiment, center=-70.0, scale=7.0)

    # --- Nuove condizioni: VIX / NFCI / Breakeven / Housing ---
    elif condition_name == "vix_low":
        # Risk-on: VIX sotto 18 (contributo reflation)
        return _sigmoid(-vix, center=-18.0, scale=3.0)
    elif condition_name == "vix_calm":
        # Tranquilita totale: VIX < 16 (goldilocks)
        return _sigmoid(-vix, center=-16.0, scale=2.0)
    elif condition_name == "vix_elevated":
        # Stress: VIX > 25 (stagflation)
        return _sigmoid(vix, center=25.0, scale=3.0)
    elif condition_name == "vix_spike":
        # Panic: VIX > 30 (deflation classica)
        return _sigmoid(vix, center=30.0, scale=4.0)
    elif condition_name == "financial_conditions_loose":
        # NFCI negativo = credit easy (reflation)
        return _sigmoid(-nfci, center=0.1, scale=0.15)
    elif condition_name == "financial_conditions_easy":
        # NFCI < -0.3 = condizioni molto accomodanti (goldilocks)
        return _sigmoid(-nfci, center=0.3, scale=0.2)
    elif condition_name == "nfci_tight":
        # NFCI > 0.3 = stress creditizio (deflation)
        return _sigmoid(nfci, center=0.3, scale=0.2)
    elif condition_name == "breakeven_high":
        # Breakeven > 2.5% = aspettative inflazione (stagflation/reflation)
        return _sigmoid(breakeven, center=2.5, scale=0.3)
    elif condition_name == "breakeven_collapse":
        # Breakeven < 1.5% = deflation expectation
        return _sigmoid(-breakeven, center=-1.5, scale=0.3)
    elif condition_name == "breakeven_stable":
        # Breakeven 1.7-2.3% = Fed target = goldilocks
        return _bell(breakeven, center=2.0, width=0.3)
    elif condition_name == "housing_expansion":
        # Housing YoY > 3% = reflation
        return _sigmoid(housing_roc, center=3.0, scale=3.0)
    elif condition_name == "housing_slowdown":
        # Housing YoY < 0 = stagflation/recession signal
        return _sigmoid(-housing_roc, center=0.0, scale=3.0)

    # Fallback
    return 0.5


def _sigmoid(x: float, center: float, scale: float) -> float:
    """Sigmoid morbida centrata su center. Output 0-1."""
    import math
    z = (x - center) / scale
    z = max(-10, min(10, z))  # Clamp per evitare overflow
    return 1.0 / (1.0 + math.exp(-z))


def _bell(x: float, center: float, width: float) -> float:
    """Curva a campana (gaussiana) centrata su center. Output 0-1."""
    import math
    return math.exp(-0.5 * ((x - center) / width) ** 2)


def classify_regime(indicators: dict[str, float]) -> dict[str, Any]:
    """Classifica il regime macro corrente in 4 quadranti.

    Args:
        indicators: Dict con indicatori macro correnti:
            - gdp_roc: Rate of change GDP (%)
            - pmi: PMI manufacturing (level)
            - cpi_yoy: CPI year-over-year (%)
            - unrate: Unemployment rate (%)
            - unrate_roc: Variazione unemployment
            - yield_curve_10y2y: 10Y-2Y spread
            - initial_claims_roc: ROC initial claims (%)
            - lei_roc: ROC leading economic index (%)
            - fed_funds_rate: Fed funds rate (%)
            - core_pce_yoy: Core PCE YoY % (Fed-preferred inflation)
            - payrolls_roc_12m: Nonfarm payrolls YoY % change
            - indpro_roc_12m: Industrial production YoY %
            - baa_spread: BAA corporate bond spread over 10Y (%)
            - consumer_sentiment: UMich sentiment level
        Indicatori assenti usano default neutri.

    Returns:
        Dict con:
            - regime: nome del regime dominante
            - probabilities: dict {regime: probabilita} (somma = 1.0)
            - confidence: float 0-1
            - conditions_detail: dettaglio condizioni valutate
    """
    raw_scores: dict[str, float] = {}
    conditions_detail: dict[str, dict] = {}

    for regime in REGIMES:
        regime_score = 0.0
        regime_conditions = {}

        for cond_name, cond_config in REGIME_CONDITIONS[regime].items():
            score = _evaluate_condition(cond_name, regime, indicators)
            weighted = score * cond_config["weight"]
            regime_score += weighted
            regime_conditions[cond_name] = {
                "raw_score": round(score, 3),
                "weight": cond_config["weight"],
                "weighted_score": round(weighted, 4),
            }

        raw_scores[regime] = regime_score
        conditions_detail[regime] = regime_conditions

    # Cross-regime adjustments: differenziatori chiave
    cpi = indicators.get("cpi_yoy", 2.0)
    gdp = indicators.get("gdp_roc", 0.0)
    unrate = indicators.get("unrate", 4.5)
    breakeven = indicators.get("breakeven_10y", 2.0)

    # Deflation richiede inflation NON alta. Soglia allineata al Fed target (2%):
    # sopra 2.5% (= target + buffer) la deflation classica e' incompatibile per definizione.
    # Penalty progressiva: -25% per ogni 1% sopra 2.5, floor 0.20 (mai zero).
    # Es. CPI 3.5% -> penalty 0.75; CPI 4.5% -> penalty 0.50; CPI 5.5% -> floor 0.25.
    if cpi > 2.5:
        raw_scores["deflation"] *= max(0.20, 1.0 - (cpi - 2.5) * 0.25)
    # Deflation richiede anche aspettative di inflation basse: se breakeven > 2.3% (mercato
    # pricing inflation forward sopra Fed target), riduci ulteriormente. Questo cattura
    # il caso "CPI scende ma mercato non crede" -> NON e' deflation classica.
    if breakeven > 2.3:
        raw_scores["deflation"] *= max(0.40, 1.0 - (breakeven - 2.3) * 0.40)
    # Deflation richiede economia debole — penalizza se unemployment basso E GDP positivo
    # (il 90s in goldilocks non dovrebbe catturare fit_deflation alto)
    if unrate < 5.0 and gdp > 1.0:
        health_penalty = max(0.4, 1.0 - (5.0 - unrate) * 0.15 - (gdp - 1.0) * 0.10)
        raw_scores["deflation"] *= health_penalty
    # Stagflation richiede inflation ALTA — penalizza se CPI < 3%
    if cpi < 3.0:
        raw_scores["stagflation"] *= max(0.2, cpi / 3.0)
    # Stagflation richiede GDP debole — penalizza se GDP > 3%
    if gdp > 3.0:
        raw_scores["stagflation"] *= max(0.3, 1.0 - (gdp - 3.0) * 0.2)
    # Reflation richiede GDP positivo — penalizza se GDP < 0
    if gdp < 0:
        raw_scores["reflation"] *= max(0.3, 1.0 + gdp * 0.15)
    # Goldilocks richiede inflation bassa — soglia 2.5% (Fed target+buffer), prima era 3.5
    # (troppo permissivo: 3.5% non e' goldilocks per nessuna definizione moderna).
    if cpi > 2.5:
        raw_scores["goldilocks"] *= max(0.25, 1.0 - (cpi - 2.5) * 0.30)

    # Fit score indipendenti [0, 1] per regime: somma pesata condizioni post-penalty
    # ma PRE-normalizzazione. Esprime "quanto lo stato corrente somiglia a questo regime"
    # in valore assoluto, senza competizione zero-sum.
    fit_scores = {r: round(max(0.0, min(1.0, s)), 4) for r, s in raw_scores.items()}

    # Normalizza in probabilita (somma = 1.0)
    total = sum(raw_scores.values())
    if total == 0:
        probabilities = {r: 1.0 / len(REGIMES) for r in REGIMES}
    else:
        probabilities = {r: s / total for r, s in raw_scores.items()}

    # Regime dominante
    regime = max(probabilities, key=probabilities.get)

    # Confidence score: basato su quanto il regime dominante "domina"
    sorted_probs = sorted(probabilities.values(), reverse=True)
    top_prob = sorted_probs[0]
    second_prob = sorted_probs[1]

    # Confidence = differenza tra top e secondo, scalata
    spread = top_prob - second_prob
    spread_confidence = min(1.0, spread * 3.5)  # Scala: spread 0.29 → confidence 1.0

    # Aggiusta per numero condizioni concordanti nel regime top
    top_conditions = conditions_detail[regime]
    conditions_met_count = sum(1 for c in top_conditions.values() if c["raw_score"] > 0.5)
    total_conditions = len(top_conditions)
    agreement_ratio = conditions_met_count / total_conditions if total_conditions > 0 else 0

    # Aggiusta per concentrazione probabilita (HHI-like)
    concentration = sum(p**2 for p in probabilities.values())
    # Uniforme (4 regimi): HHI=0.25, molto concentrato: HHI→1.0
    concentration_score = min(1.0, (concentration - 0.25) / 0.5)

    # Confidence finale: combinazione di spread, agreement, e concentrazione
    confidence = (spread_confidence * 0.4) + (agreement_ratio * 0.35) + (concentration_score * 0.25)
    confidence = max(0.0, min(1.0, confidence))

    return {
        "regime": regime,
        "probabilities": probabilities,
        "fit_scores": fit_scores,
        "confidence": round(confidence, 3),
        "conditions_detail": conditions_detail,
    }
