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

from app.services.config_flags import (
    use_adaptive_thresholds,
    use_cross_asset_pillars,
    use_dedollar_pillar,
    use_forward_inflation_pillar,
    use_freshness_weighting,
    use_gdp_collapse_override,
    use_ml_regime_blend,
    use_momentum_pillars,
    use_news_pillar,
)
from app.services.regime.freshness import effective_weight, freshness_for

REGIMES = ["reflation", "stagflation", "deflation", "goldilocks"]


# T11 (2026-05-27) Momentum pillars — opt-in via USE_MOMENTUM_PILLARS flag.
# Quando attivo, _merge_momentum_pillars() applica:
# - rebalance dei pesi level-based esistenti (per stare a total=1.0)
# - aggiunge i 5 nuovi pillar momentum
_MOMENTUM_PILLARS_ADD: dict[str, dict[str, dict]] = {
    "stagflation": {
        "inflation_accelerating": {"weight": 0.06, "description": "CPI YoY change 6m > 1%"},
        "core_pce_accelerating": {"weight": 0.05, "description": "Core PCE YoY change 6m > 0.8%"},
        "gdp_decelerating": {"weight": 0.04, "description": "GDP ROC change 6m < -0.5%"},
        "inflation_persistent": {"weight": 0.04, "description": "min(CPI YoY last 6m) > 2.5%"},
        "dfm_growth_decelerating": {"weight": 0.03, "description": "DFM nowcast change 3m < -0.4%"},
    },
    "deflation": {
        "gdp_decelerating": {"weight": 0.05, "description": "GDP ROC change 6m < -0.5%"},
        "dfm_growth_decelerating": {"weight": 0.04, "description": "DFM nowcast change 3m < -0.4%"},
    },
}

# Rebalance delta da applicare quando flag ON (pesi vengono SOTTRATTI dai level pillar
# esistenti). Total target stagflation=1.00 (era 1.07), deflation rebalanced.
_MOMENTUM_PILLARS_REBALANCE: dict[str, dict[str, float]] = {
    "stagflation": {
        "inflation_high": -0.04,
        "gdp_weak": -0.02,
        "pmi_weak": -0.01,
        "unemployment_rising": -0.01,
        "dfm_growth_weak": -0.01,
        "gold_oil_low": -0.01,
        "insider_selling_strong": -0.01,
    },
    "deflation": {
        "gdp_negative_or_decelerating": -0.04,
        "indpro_contraction": -0.02,
        "payrolls_slowdown": -0.03,
    },
}


def _merge_momentum_pillars(base: dict) -> dict:
    """T11: applica rebalance + aggiunta pillar momentum, ritorna nuovo dict.

    NON modifica `base` (deep-ish copy a 2 livelli).
    """
    merged = {regime: {k: dict(v) for k, v in conds.items()} for regime, conds in base.items()}
    # Rebalance esistenti (sottrai pesi)
    for regime, deltas in _MOMENTUM_PILLARS_REBALANCE.items():
        for cond_name, delta in deltas.items():
            if cond_name in merged.get(regime, {}):
                merged[regime][cond_name]["weight"] = max(0.0, merged[regime][cond_name]["weight"] + delta)
    # Aggiungi momentum pillar
    for regime, adds in _MOMENTUM_PILLARS_ADD.items():
        for cond_name, cond_data in adds.items():
            merged[regime][cond_name] = dict(cond_data)
    return merged


# T13 (2026-05-31 Council) Forward-looking inflation pillars — fix structural
# "inflation blindness" diagnosticata post bug T12. Discrimina inflation regimes
# usando ANCHE expected/market-priced inflation, non solo CPI level/momentum.
_FORWARD_INFLATION_PILLAR_ADD: dict[str, dict[str, dict]] = {
    "stagflation": {
        "umich_inflation_high": {"weight": 0.04, "description": "UMich 1y inflation expectation > 3.5%"},
        "sticky_inflation_high": {"weight": 0.04, "description": "Atlanta Fed sticky CPI > 4% (hardest to break)"},
        "breakeven_5y5y_high": {"weight": 0.03, "description": "5y5y forward inflation > 2.5% (Fed anchor break)"},
    },
    "deflation": {
        "umich_inflation_low": {"weight": 0.03, "description": "UMich 1y < 2% (consumer expects disinflation)"},
    },
    "goldilocks": {
        "umich_inflation_anchored": {"weight": 0.03, "description": "UMich 1y in 2.0-2.5% (anchored at target)"},
        "sticky_inflation_anchored": {"weight": 0.03, "description": "Sticky CPI in 2.0-2.7% (anchored sticky)"},
    },
}

_FORWARD_INFLATION_PILLAR_REBALANCE: dict[str, dict[str, float]] = {
    "stagflation": {
        # Riduci pesi level inflation per fare spazio ai forward (Σ=1 target)
        "inflation_high": -0.03,
        "core_pce_high": -0.03,
        "breakeven_high": -0.02,
    },
    "deflation": {
        # Riduzione minima per non destabilizzare scenari deflation severa
        "breakeven_collapse": -0.02,
    },
    "goldilocks": {
        "breakeven_stable": -0.02,
        "core_pce_contained": -0.02,
    },
}


def _merge_forward_inflation_pillars(base: dict) -> dict:
    """T13: applica rebalance + aggiunta pillar forward-looking inflation."""
    merged = {regime: {k: dict(v) for k, v in conds.items()} for regime, conds in base.items()}
    for regime, deltas in _FORWARD_INFLATION_PILLAR_REBALANCE.items():
        for cond_name, delta in deltas.items():
            if cond_name in merged.get(regime, {}):
                merged[regime][cond_name]["weight"] = max(
                    0.0, merged[regime][cond_name]["weight"] + delta
                )
    for regime, adds in _FORWARD_INFLATION_PILLAR_ADD.items():
        for cond_name, cond_data in adds.items():
            merged[regime][cond_name] = dict(cond_data)
    return merged

# Condizioni per ogni regime con pesi (somma per regime = 1.0)
REGIME_CONDITIONS = {
    "reflation": {
        "gdp_strong": {"weight": 0.11, "description": "GDP ROC > 2%"},
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
        "financial_conditions_loose": {"weight": 0.03, "description": "NFCI < -0.1 (credit easy)"},
        "vix_low": {"weight": 0.02, "description": "VIX < 18 (risk-on)"},
        "term_premium_low": {"weight": 0.04, "description": "ACM Term Premium 10Y < 0 (risk-on accepted)"},
        "dfm_growth_strong": {"weight": 0.02, "description": "DFM nowcast GDP YoY > 2.5% (real-time growth signal)"},
        "insider_buying_strong": {"weight": 0.03, "description": "SEC Form 4 insider score > 0.3 (smart money bullish)"},
        "dedollar_low": {"weight": 0.02, "description": "Dedollar combined < 0.4 (dollar strong, risk-on global)"},
        "news_positive_strong": {"weight": 0.02, "description": "News avg sentiment > 0.3 (positive narrative)"},
        "copper_gold_strong": {"weight": 0.03, "description": "Copper/gold ratio > 0.45 (cyclical strength)"},
        "hy_ig_tight": {"weight": 0.02, "description": "HY/IG spread ratio < 3 (credit risk-on)"},
    },
    "stagflation": {
        "inflation_high": {"weight": 0.17, "description": "CPI YoY > 4%"},
        "gdp_weak": {"weight": 0.11, "description": "GDP ROC < 1.5%"},
        "pmi_weak": {"weight": 0.10, "description": "PMI < 50"},
        "unemployment_rising": {"weight": 0.08, "description": "Unemployment ROC > 0.2"},
        "policy_restrictive": {"weight": 0.05, "description": "Fed funds > 4%"},
        "lei_negative": {"weight": 0.05, "description": "LEI ROC < 0"},
        "claims_rising": {"weight": 0.04, "description": "Initial claims ROC > 5%"},
        "yield_curve_stress": {"weight": 0.04, "description": "10Y-2Y spread < 0.5"},
        "core_pce_high": {"weight": 0.08, "description": "Core PCE YoY > 3.5% (Fed-preferred)"},
        "credit_spread_wide": {"weight": 0.06, "description": "BAA-10Y spread > 2.3%"},
        "sentiment_low": {"weight": 0.03, "description": "Consumer sentiment < 70"},
        "breakeven_high": {"weight": 0.04, "description": "Breakeven 10Y > 2.5% (inflation expect)"},
        "vix_elevated": {"weight": 0.05, "description": "VIX > 25 (risk-off)"},
        "housing_slowdown": {"weight": 0.04, "description": "Housing starts YoY < 0"},
        "term_premium_high": {"weight": 0.04, "description": "ACM Term Premium 10Y > 0.4% (duration risk priced)"},
        "dfm_growth_weak": {"weight": 0.02, "description": "DFM nowcast GDP YoY < 1.5% (real-time slowdown signal)"},
        "insider_selling_strong": {"weight": 0.03, "description": "SEC Form 4 insider score < -0.3 (smart money bearish)"},
        "dedollar_debasement": {"weight": 0.03, "description": "Dedollar combined > 0.6 (USD debasement, real assets bid)"},
        "gold_oil_low": {"weight": 0.03, "description": "Gold/oil ratio < 15 (oil-driven inflation, stagflation 1973-style)"},
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
        "insider_selling_strong": {"weight": 0.03, "description": "SEC Form 4 insider score < -0.3 (smart money bearish)"},
        "news_panic": {"weight": 0.02, "description": "News avg sentiment < -0.3 (panic narrative)"},
        "copper_gold_weak": {"weight": 0.03, "description": "Copper/gold ratio < 0.30 (cyclical collapse, deflation)"},
        "gold_oil_high": {"weight": 0.03, "description": "Gold/oil ratio > 30 (gold strong vs oil collapse, deflation)"},
        "hy_ig_stress": {"weight": 0.03, "description": "HY/IG spread ratio > 5 (credit stress 2008-style)"},
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
        "insider_buying_strong": {"weight": 0.02, "description": "SEC Form 4 insider score > 0.3 (smart money bullish)"},
    },
}


def _resolve_sigmoid_params(
    condition_name: str,
    default_center: float,
    default_scale: float,
    adaptive_recipes: dict | None,
    sign: int = 1,
) -> tuple[float, float]:
    """Restituisce (center, scale) per il sigmoid.

    Se `use_adaptive_thresholds()` è ON e `adaptive_recipes` ha la key
    per `condition_name`, usa quei params calcolati da quantili rolling.
    Altrimenti default hardcoded.

    `sign=+1` per condizioni "high" (sigmoid(x, center, scale)).
    `sign=-1` per "low" (sigmoid(-x, -center, scale)): center va negato
    perché il classifier passa `-value` al sigmoid; ma il valore quantile
    è positivo (es. q25 di VIX), quindi center=-q25 quando sign=-1.
    """
    if not use_adaptive_thresholds() or not adaptive_recipes:
        return default_center, default_scale
    recipe = adaptive_recipes.get(condition_name)
    if recipe is None:
        return default_center, default_scale
    # recipe.center è positivo (quantile valore originale).
    # Il classifier per condizioni "low" passa _sigmoid(-value, center=-X, scale=Y).
    # Quindi sign=-1 → center negato per matchare quel pattern.
    center = recipe.center * sign
    scale = recipe.scale
    return center, scale


def _evaluate_condition(
    condition_name: str,
    regime: str,
    indicators: dict[str, float],
    adaptive_recipes: dict | None = None,
) -> float:
    """Valuta una singola condizione e ritorna score 0-1.

    Usa funzioni sigmoidali soft per evitare cutoff binari.

    Args:
        adaptive_recipes: dict {condition_name: AdaptiveRecipe} opzionale.
            Se passato e flag T6.1 ON, sostituisce center/scale hardcoded
            per le 13 condizioni in `_CONDITION_RECIPES` (VIX/BAA/CPI/
            yield_curve/NFCI).
    """
    gdp = indicators.get("gdp_roc", 0.0)
    # T9-AUDIT-BUG-4: NAPM (PMI) discontinuato FRED → proxy via indpro_yoy
    from app.services.common.indicator_aliases import derive_pmi_proxy
    pmi = derive_pmi_proxy(indicators)
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
    # T13 Forward-looking inflation features (council 2026-05-31 post bug T12).
    # 5y5y forward (Fed's preferred long-anchor): default 2.0% = at target.
    breakeven_5y5y = indicators.get("breakeven_5y5y", 2.0)
    # UMich 1y consumer expectation (sentiment-driven, monthly): default 2.5% (Fed target+buffer).
    umich_inflation = indicators.get("umich_inflation_1y", 2.5)
    # Atlanta Fed sticky CPI (12m %change, hardest to break): default 2.5% (Fed target+buffer).
    sticky_cpi = indicators.get("sticky_cpi_yoy", 2.5)
    yield_3m = indicators.get("yield_curve_10y3m", 1.0)  # 10y-3m spread
    housing_roc = indicators.get("housing_starts_roc_12m", 0.0)  # housing YoY
    # ACM Term Premium 10Y (Tier 1.3 roadmap): neutral default 0.3% (storica media).
    # > 0.4% = mercato pricing duration risk → +stagflation; < 0 = risk-on → +reflation.
    term_premium = indicators.get("term_premium_10y", 0.3)
    # DFM nowcast GDP YoY (Tier 2.5 roadmap): default 2.0% (= trend long-run USA GDP YoY).
    # > 2.5% = growth above trend → +reflation; < 1.5% = slowdown → +stagflation.
    gdp_yoy_dfm = indicators.get("gdp_yoy_dfm", 2.0)
    # SEC EDGAR Form 4 insider score (Tier 3.6 Fase B): default 0.0 (neutral).
    # Tanh-bounded [-1, +1]. > +0.3 = smart money buying → +reflation/+goldilocks.
    # < -0.3 = smart money selling → +deflation/+stagflation.
    insider_score = indicators.get("insider_score", 0.0)
    # Dedollar combined (Tier 5.2 LOOP CLOSURE): default 0.5 (neutral mid-range).
    # Range 0-1. > 0.6 = USD debasement signal → +stagflation_debasement.
    # < 0.4 = USD strong → +reflation (risk-on global).
    # Opt-in via USE_DEDOLLAR_PILLAR=1 (default OFF per back-compat puro).
    dedollar_combined = indicators.get("dedollar_combined", 0.5)
    # News avg sentiment (Tier 5.3 LOOP CLOSURE): default 0.0 (neutral).
    # Range tipico [-1, +1]. > 0.3 = positive narrative → +reflation.
    # < -0.3 = panic narrative → +deflation.
    # Opt-in via USE_NEWS_PILLAR=1. Pillar è soft (peso 0.02) per gestire
    # rumorosità intraday e LLM-scoring variance.
    news_sentiment = indicators.get("news_sentiment", 0.0)
    # Cross-asset ratios (Tier 6.3): default neutral mid-range storici.
    # Opt-in via USE_CROSS_ASSET_PILLARS. Soft pillar 0.02-0.03 each.
    copper_gold_ratio = indicators.get("copper_gold_ratio", 0.37)
    gold_oil_ratio = indicators.get("gold_oil_ratio", 17.0)
    hy_ig_spread_ratio = indicators.get("hy_ig_spread_ratio", 3.5)

    # --- REFLATION conditions ---
    if condition_name == "gdp_strong":
        return _sigmoid(gdp, center=2.0, scale=1.5)
    elif condition_name == "pmi_expansion" and regime == "reflation":
        return _sigmoid(pmi, center=52.0, scale=3.0)
    elif condition_name == "inflation_rising":
        # T9-AUDIT-BUG-FIX (bughunter): pre-fix sigmoid centro 2.8 →
        # CPI 4% contribuisce 0.76 a reflation. CONCETTUALMENTE SBAGLIATO:
        # CPI > 3.5% NON è reflation salutare, è proto-stagflation.
        # Fix: bell-curve centrata a 2.5%, satura sopra 3.5% (penalty).
        # Reflation "salutare" = CPI 2.0-3.0%. Sopra 3.5% peso decresce.
        c, s = _resolve_sigmoid_params(condition_name, 2.5, 0.6, adaptive_recipes, sign=+1)
        # Bell curve: massimo a CPI=center, decresce sia sotto sia sopra.
        # Per CPI > center+1.5σ → contributo riduce (è "troppa inflation"
        # per essere reflation).
        if cpi > c + 1.5 * s:
            # Decrescita lineare sopra threshold
            excess = (cpi - c - 1.5 * s) / s
            return max(0.0, _sigmoid(cpi, center=c, scale=s) * (1.0 - 0.5 * excess))
        return _sigmoid(cpi, center=c, scale=s)
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
        c, s = _resolve_sigmoid_params(condition_name, 0.8, 0.8, adaptive_recipes, sign=+1)
        return _sigmoid(yield_spread, center=c, scale=s)
    elif condition_name == "policy_accommodative":
        return _sigmoid(-fed_funds, center=-2.5, scale=1.5)

    # --- STAGFLATION conditions ---
    elif condition_name == "inflation_high":
        # T12 BUGFIX 2026-05-31 (council + bughunter): center 4.0 troppo alto.
        # Con CPI=3.95% restituiva 0.49 (border, contributo debole) mentre
        # 3.95% E' inflazione stagflattiva textbook. Center 3.2 risolve il bug
        # "classificato deflation con CPI rising 3 mesi consecutivi" (utente
        # Erik 2026-05-31). Scale 1.0: più sharp anche da 2.5% in su.
        c, s = _resolve_sigmoid_params(condition_name, 3.2, 1.0, adaptive_recipes, sign=+1)
        return _sigmoid(cpi, center=c, scale=s)
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

    # --- T11 MOMENTUM PILLARS (opt-in USE_MOMENTUM_PILLARS) ---
    # Cattura ACCELERAZIONE oltre al livello istantaneo. Council 2026-05-27.
    elif condition_name == "inflation_accelerating":
        # T12 BUGFIX 2026-05-31: center 1.0 troppo alto. CPI rise +0.92pp in
        # 6m era già "accelerating textbook" ma sigmoid restituiva 0.46.
        # Center 0.5 = +0.5pp in 6m è soglia significativa accelerazione.
        cpi_change = indicators.get("cpi_yoy_change_6m", 0.0)
        return _sigmoid(cpi_change, center=0.5, scale=0.4)
    elif condition_name == "core_pce_accelerating":
        # T12 BUGFIX 2026-05-31: center 0.8 → 0.4 stessa logica.
        pce_change = indicators.get("core_pce_yoy_change_6m", 0.0)
        return _sigmoid(pce_change, center=0.4, scale=0.3)
    elif condition_name == "gdp_decelerating":
        gdp_change = indicators.get("gdp_roc_change_6m", 0.0)
        return _sigmoid(-gdp_change, center=0.5, scale=0.5)
    elif condition_name == "inflation_persistent":
        cpi_min = indicators.get("cpi_yoy_min_last_6m", cpi)
        return _sigmoid(cpi_min, center=2.5, scale=0.5)
    elif condition_name == "dfm_growth_decelerating":
        dfm_change = indicators.get("gdp_yoy_dfm_change_3m", 0.0)
        return _sigmoid(-dfm_change, center=0.4, scale=0.4)

    # --- T13 FORWARD-LOOKING INFLATION PILLARS (USE_FORWARD_INFLATION_PILLAR) ---
    # Cattura inflation expectations (market + consumer + sticky) per fix bug
    # "inflation blindness" diagnosticato 2026-05-31. Council T13.
    elif condition_name == "umich_inflation_high":
        # MICH > 3.5% = consumer expects hot inflation (1981-style)
        return _sigmoid(umich_inflation, center=3.5, scale=0.5)
    elif condition_name == "sticky_inflation_high":
        # Sticky CPI > 4% = inflation sticky component embedded
        return _sigmoid(sticky_cpi, center=4.0, scale=0.5)
    elif condition_name == "breakeven_5y5y_high":
        # T5YIFR > 2.5% = Fed anchor break (long-term inflation not anchored)
        return _sigmoid(breakeven_5y5y, center=2.5, scale=0.3)
    elif condition_name == "umich_inflation_low":
        # MICH < 2.0% = consumer doesn't expect inflation (deflation signal)
        return _sigmoid(-umich_inflation, center=-2.0, scale=0.4)
    elif condition_name == "umich_inflation_anchored":
        # MICH 2.0-2.5% = anchored at target (goldilocks)
        return _bell(umich_inflation, center=2.25, width=0.3)
    elif condition_name == "sticky_inflation_anchored":
        # Sticky CPI 2.0-2.7% = sticky component at target (goldilocks)
        return _bell(sticky_cpi, center=2.35, width=0.4)

    # --- DEFLATION conditions ---
    elif condition_name == "gdp_negative_or_decelerating":
        return _sigmoid(-gdp, center=-0.5, scale=1.0)
    elif condition_name == "pmi_contraction":
        return _sigmoid(-pmi, center=-49.0, scale=2.5)
    elif condition_name == "inflation_low" and regime == "deflation":
        c, s = _resolve_sigmoid_params(condition_name, -2.0, 0.8, adaptive_recipes, sign=-1)
        return _sigmoid(-cpi, center=c, scale=s)
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
        # T12 BUGFIX 2026-05-31: center 3.5 → 3.0. Core PCE > 3% è già
        # "sticky inflation" sopra Fed target+buffer → forte segnale stagflation.
        # Live core_pce 3.29% dava 0.43 (border), ora restituisce ~0.65.
        return _sigmoid(core_pce, center=3.0, scale=0.7)
    elif condition_name == "core_pce_contained":
        return _sigmoid(-core_pce, center=-2.3, scale=0.6)
    elif condition_name == "credit_spread_tight":
        c, s = _resolve_sigmoid_params(condition_name, -1.9, 0.5, adaptive_recipes, sign=-1)
        return _sigmoid(-baa_spread, center=c, scale=s)
    elif condition_name == "credit_spread_wide" and regime == "stagflation":
        c, s = _resolve_sigmoid_params(condition_name, 2.3, 0.5, adaptive_recipes, sign=+1)
        return _sigmoid(baa_spread, center=c, scale=s)
    elif condition_name == "credit_spread_wide" and regime == "deflation":
        c, s = _resolve_sigmoid_params(condition_name, 2.5, 0.6, adaptive_recipes, sign=+1)
        return _sigmoid(baa_spread, center=c, scale=s)
    elif condition_name == "sentiment_high":
        return _sigmoid(sentiment, center=82.0, scale=6.0)
    elif condition_name == "sentiment_low":
        return _sigmoid(-sentiment, center=-70.0, scale=7.0)

    # --- Nuove condizioni: VIX / NFCI / Breakeven / Housing ---
    # T6.1 adaptive thresholds: 4 VIX conditions + 3 NFCI usano
    # _resolve_sigmoid_params se flag ON + recipes disponibili.
    elif condition_name == "vix_low":
        # Risk-on: VIX sotto q25 rolling 10y (default -18 hardcoded)
        c, s = _resolve_sigmoid_params(condition_name, -18.0, 3.0, adaptive_recipes, sign=-1)
        return _sigmoid(-vix, center=c, scale=s)
    elif condition_name == "vix_calm":
        # Goldilocks tranquilità: VIX sotto q10
        c, s = _resolve_sigmoid_params(condition_name, -16.0, 2.0, adaptive_recipes, sign=-1)
        return _sigmoid(-vix, center=c, scale=s)
    elif condition_name == "vix_elevated":
        # Stress: VIX sopra q75
        c, s = _resolve_sigmoid_params(condition_name, 25.0, 3.0, adaptive_recipes, sign=+1)
        return _sigmoid(vix, center=c, scale=s)
    elif condition_name == "vix_spike":
        # Panic: VIX sopra q95
        c, s = _resolve_sigmoid_params(condition_name, 30.0, 4.0, adaptive_recipes, sign=+1)
        return _sigmoid(vix, center=c, scale=s)
    elif condition_name == "financial_conditions_loose":
        # NFCI sotto q25 = credit easy (reflation)
        c, s = _resolve_sigmoid_params(condition_name, 0.1, 0.15, adaptive_recipes, sign=-1)
        return _sigmoid(-nfci, center=c, scale=s)
    elif condition_name == "financial_conditions_easy":
        # NFCI sotto q10 = condizioni molto accomodanti (goldilocks)
        c, s = _resolve_sigmoid_params(condition_name, 0.3, 0.2, adaptive_recipes, sign=-1)
        return _sigmoid(-nfci, center=c, scale=s)
    elif condition_name == "nfci_tight":
        # NFCI sopra q75 = stress creditizio (deflation)
        c, s = _resolve_sigmoid_params(condition_name, 0.3, 0.2, adaptive_recipes, sign=+1)
        return _sigmoid(nfci, center=c, scale=s)
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
    elif condition_name == "term_premium_high" and regime == "stagflation":
        # ACM TP 10Y > 0.4% = mercato pricing duration risk → +stagflation.
        # Tier 1.3 roadmap. Storicamente in stagflation TP medio ~0.46% / 96% positive.
        return _sigmoid(term_premium, center=0.4, scale=0.3)
    elif condition_name == "term_premium_low" and regime == "reflation":
        # ACM TP 10Y < 0 = mercato accetta duration risk = risk-on → +reflation.
        # Storicamente in reflation TP medio ~0.14% (solo 60% positive).
        return _sigmoid(-term_premium, center=0.0, scale=0.3)
    elif condition_name == "dfm_growth_strong" and regime == "reflation":
        # DFM nowcast GDP YoY > 2.5% = growth above trend → +reflation.
        # Tier 2.5 roadmap. Real-time signal vs gdp_roc QoQ lagged trimestrale.
        return _sigmoid(gdp_yoy_dfm, center=2.5, scale=1.0)
    elif condition_name == "dfm_growth_weak" and regime == "stagflation":
        # DFM nowcast GDP YoY < 1.5% = slowdown real-time → +stagflation.
        return _sigmoid(-gdp_yoy_dfm, center=-1.5, scale=1.0)
    elif condition_name == "insider_buying_strong" and regime in {"reflation", "goldilocks"}:
        # SEC Form 4 score > 0.3 = officers/directors/10%+ holders aggressive
        # open-market buying → bullish signal → +reflation/+goldilocks.
        # Storicamente leading indicator forte (gennaio 2009 spike, marzo 2020 bottom).
        return _sigmoid(insider_score, center=0.3, scale=0.2)
    elif condition_name == "insider_selling_strong" and regime in {"stagflation", "deflation"}:
        # SEC Form 4 score < -0.3 = aggressive selling → bearish signal.
        # Cautela: sales spesso scheduled 10b5-1 o tax → segnale più rumoroso del buying.
        # Peso 0.03 (low) per gestire il rumore.
        return _sigmoid(-insider_score, center=0.3, scale=0.2)
    elif condition_name == "dedollar_debasement" and regime == "stagflation":
        # Tier 5.2 LOOP CLOSURE: dedollar_combined > 0.6 = USD debasement
        # persistente (currency stress, real assets bid). 2022 saw this clearly.
        # Opt-in via USE_DEDOLLAR_PILLAR; spento = 0.0 (zero impact, NO neutral
        # 0.5 bias) per non sporcare backtest pre-2026.
        if not use_dedollar_pillar():
            return 0.0
        return _sigmoid(dedollar_combined, center=0.6, scale=0.15)
    elif condition_name == "dedollar_low" and regime == "reflation":
        # Tier 5.2 LOOP CLOSURE: dedollar_combined < 0.4 = USD strength
        # (global risk-on, capital flows verso US assets). Pattern '90s + 2014-15.
        if not use_dedollar_pillar():
            return 0.0
        return _sigmoid(-dedollar_combined, center=-0.4, scale=0.15)
    elif condition_name == "news_positive_strong" and regime == "reflation":
        # Tier 5.3 LOOP CLOSURE: news_sentiment > 0.3 = positive narrative
        # (corporate guidance, macro upside, dovish Fed). Opt-in flag-gated.
        # Soft (peso 0.02) per gestire rumorosità intraday + LLM variance.
        if not use_news_pillar():
            return 0.0
        return _sigmoid(news_sentiment, center=0.3, scale=0.15)
    elif condition_name == "news_panic" and regime == "deflation":
        # Tier 5.3 LOOP CLOSURE: news_sentiment < -0.3 = panic narrative.
        # Storicamente: 2008 settembre/ottobre, 2020 marzo COVID, 2022 Q4 banche.
        if not use_news_pillar():
            return 0.0
        return _sigmoid(-news_sentiment, center=0.3, scale=0.15)
    # T6.3 Cross-asset ratios pillars (opt-in USE_CROSS_ASSET_PILLARS).
    # Flag-gated pattern come T5.2/T5.3: spento → return 0.0 (zero impact).
    elif condition_name == "copper_gold_strong" and regime == "reflation":
        if not use_cross_asset_pillars():
            return 0.0
        return _sigmoid(copper_gold_ratio, center=0.45, scale=0.08)
    elif condition_name == "copper_gold_weak" and regime == "deflation":
        if not use_cross_asset_pillars():
            return 0.0
        return _sigmoid(-copper_gold_ratio, center=-0.30, scale=0.08)
    elif condition_name == "gold_oil_high" and regime == "deflation":
        if not use_cross_asset_pillars():
            return 0.0
        return _sigmoid(gold_oil_ratio, center=30.0, scale=10.0)
    elif condition_name == "gold_oil_low" and regime == "stagflation":
        if not use_cross_asset_pillars():
            return 0.0
        return _sigmoid(-gold_oil_ratio, center=-15.0, scale=5.0)
    elif condition_name == "hy_ig_stress" and regime == "deflation":
        if not use_cross_asset_pillars():
            return 0.0
        return _sigmoid(hy_ig_spread_ratio, center=5.0, scale=1.0)
    elif condition_name == "hy_ig_tight" and regime == "reflation":
        if not use_cross_asset_pillars():
            return 0.0
        return _sigmoid(-hy_ig_spread_ratio, center=-3.0, scale=0.5)

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


def classify_regime(
    indicators: dict[str, float],
    adaptive_recipes: dict | None = None,
) -> dict[str, Any]:
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
    apply_freshness = use_freshness_weighting()

    # T11: applica rebalance + aggiunta momentum pillar quando flag ON.
    if use_momentum_pillars():
        active_conditions = _merge_momentum_pillars(REGIME_CONDITIONS)
    else:
        active_conditions = REGIME_CONDITIONS

    # T13 (2026-05-31): forward-looking inflation pillar (MICH/sticky/T5YIFR).
    # Compone con T11 momentum: prima rebalance momentum, poi forward.
    if use_forward_inflation_pillar():
        active_conditions = _merge_forward_inflation_pillars(active_conditions)

    for regime in REGIMES:
        regime_score = 0.0
        regime_conditions = {}

        for cond_name, cond_config in active_conditions[regime].items():
            score = _evaluate_condition(cond_name, regime, indicators, adaptive_recipes=adaptive_recipes)
            base_w = cond_config["weight"]
            if apply_freshness:
                w = effective_weight(cond_name, base_w)
            else:
                w = base_w
            weighted = score * w
            regime_score += weighted
            cond_detail = {
                "raw_score": round(score, 3),
                "weight": base_w,
                "weighted_score": round(weighted, 4),
            }
            if apply_freshness:
                cond_detail["freshness"] = freshness_for(cond_name)
                cond_detail["effective_weight"] = round(w, 4)
            regime_conditions[cond_name] = cond_detail

        raw_scores[regime] = regime_score
        conditions_detail[regime] = regime_conditions

    # Cross-regime adjustments: differenziatori chiave
    cpi = indicators.get("cpi_yoy", 2.0)
    gdp = indicators.get("gdp_roc", 0.0)
    unrate = indicators.get("unrate", 4.5)
    unrate_roc = indicators.get("unrate_roc", 0.0)
    breakeven = indicators.get("breakeven_10y", 2.0)
    vix = indicators.get("vix", 18.0)

    # T7.1 GDP COLLAPSE OVERRIDE (continualizzato post-T9 audit council):
    # cattura le recessioni disinflazionistiche (2008/2020/2001/1990) dove
    # CPI è ancora alto (lagging) ma forward inflation crolla.
    #
    # Pre-T9: 4 condition AND-rigid (data-dredged 2008/2020). Council ha
    # rilevato che non generalizza a 1990/2015/2018 dove uno dei 4 trigger
    # marginalmente non scatta.
    #
    # T9 continualizzato: severity score = sigmoid(z(gdp) + z(unrate_roc) +
    # z(vix) - z(breakeven)) ∈ [0, 1]. Scaling moltiplicatori proporzionale
    # alla severity invece che binary on/off.
    gdp_collapse_triggered = False
    gdp_collapse_severity = 0.0
    if use_gdp_collapse_override():
        # Z-score normalizzato (anchor mean/std come ml_classifier)
        gdp_z = -(gdp - 2.5) / 3.0  # invertito: gdp basso → z positivo
        unrate_roc_z = (unrate_roc - 0.0) / 1.0
        vix_z = (vix - 19.0) / 10.0
        breakeven_z = -(breakeven - 2.3) / 1.0  # invertito: breakeven basso → z positivo
        # Severity composta: media degli z positivi (recession indicators)
        severity_raw = (gdp_z + unrate_roc_z + vix_z + breakeven_z) / 4.0
        # Sigmoid (logistic) per mappare a [0, 1], centro 0.5 → 0.5
        import math
        gdp_collapse_severity = 1.0 / (1.0 + math.exp(-2.0 * severity_raw))
        # Trigger flag se severity > 0.55 (conservativo: evita falsi positivi
        # borderline su baseline neutri che producono ~0.52)
        gdp_collapse_triggered = gdp_collapse_severity > 0.55
        # Scaling moltiplicatori:
        # severity 0 → no shift (×1.0 / ×1.0)
        # severity 0.5 → mild (stag ×0.6, defl ×1.5)
        # severity 1.0 → max (stag ×0.25, defl ×2.2) come pre-T9
        stag_mult = 1.0 - 0.75 * gdp_collapse_severity  # 1.0 → 0.25
        defl_mult = 1.0 + 1.2 * gdp_collapse_severity  # 1.0 → 2.2
        raw_scores["stagflation"] *= stag_mult
        raw_scores["deflation"] *= defl_mult

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
    # T9-AUDIT-BUG-FIX (bughunter HIGH-1 + council P0): reflation richiede CPI
    # sotto target Fed + buffer. CPI > 3.5% NON è reflation salutare.
    # Pre-P0: soft penalty (max 70% reduction at CPI 6%).
    # Post-P0: HARD penalty progressivamente più forte sopra 3.5%.
    # Es. CPI 3.5% -> ×1.0, CPI 4.0% -> ×0.55, CPI 4.5% -> ×0.35, CPI 5.0% -> floor 0.20.
    # Filosofia council: "nessun investitore esperto chiamerebbe reflation un
    # regime con inflazione al 4%".
    if cpi > 3.5:
        raw_scores["reflation"] *= max(0.20, 1.0 - (cpi - 3.5) * 0.90)
    # Goldilocks richiede inflation bassa — soglia 2.5% (Fed target+buffer).
    # T12 BUGFIX 2026-05-31 (council + bughunter): slope 0.30 → 0.60 troppo
    # permissivo, a CPI 3.95% goldilocks veniva ridotto solo del 43%.
    # Floor 0.25 → 0.15 per evitare residual goldilocks in piena stagflation.
    if cpi > 2.5:
        raw_scores["goldilocks"] *= max(0.15, 1.0 - (cpi - 2.5) * 0.60)

    # T12 BUGFIX 2026-05-31: deflation penalty con CPI > 3.0% (NEW).
    # Deflation E' anti-thesis di inflation. Live API mostra deflation 38.7%
    # con CPI 3.95% — economicamente assurdo. Penalty AGGRESSIVA sopra 3.0%.
    # CPI 3.5 → ×0.675, CPI 4.0 → ×0.35, CPI 4.5 → floor 0.20.
    # SKIP solo se gdp_collapse SEVERITY ALTA (>0.80, vero 2008-style crash):
    # GDP < -1.5% + vix > 30 + breakeven crollato. Per severity moderate (0.55-0.80)
    # il classifier T12 vince comunque sulla penalty CPI rising.
    if cpi > 3.0 and gdp_collapse_severity < 0.80:
        raw_scores["deflation"] *= max(0.20, 1.0 - (cpi - 3.0) * 0.65)

    # T12 STAGFLATION TEXTBOOK BONUS 2026-05-31: scenario inequivocabile.
    # CPI > 3.5% AND GDP ROC < 1.5% = signature stagflation (1973-style).
    # Boost ×1.30 evita che reflation/deflation vincano in border cases dove
    # i pillar accumulano molti piccoli positivi cross-regime. Council T12.
    # SKIP solo se gdp_collapse SEVERITY ALTA (>0.80): 2008-style overrides
    # questo bonus, ma per gdp_roc weak ma non collapse si applica.
    if cpi > 3.5 and gdp < 1.5 and gdp_collapse_severity < 0.80:
        raw_scores["stagflation"] *= 1.30
        # Reflation NON è plausibile con CPI > 3.5% AND GDP < 1.5%
        raw_scores["reflation"] *= 0.75

    # Council 2026-05-27 — LIQUIDITY SURGE OVERRIDE (post TEST A 2020 disaster).
    # Fix per V-shape recovery missing: massive QE/stampa valuta = leading
    # indicator del recovery, NON i fondamentali macro.
    # Trigger composite: M2 YoY > +10% AND WALCL ROC 3m > +12% AND real rates declining.
    # T12 GATE 2026-05-31 (council): anti-inflation circuit breaker. Override
    # OFF se CPI accelerating > +0.5pp in 6m OR CPI > 3.5% (= stagflation reale).
    liquidity_surge_triggered = False
    try:
        from app.services.config_flags import use_liquidity_surge_override
        if use_liquidity_surge_override():
            # Circuit breaker T12: blocca override in scenario stagflattivo
            cpi_chg_6m = indicators.get("cpi_yoy_change_6m", 0.0)
            anti_inflation_block = (cpi > 3.5) or (cpi_chg_6m > 0.5)
            if not anti_inflation_block:
                m2_y = indicators.get("m2_yoy")
                walcl_r3m = indicators.get("walcl_roc_3m")
                real_rate_chg = indicators.get("real_rate_change_3m")
                # Real rate "accommodative" criterio flexible:
                # - declining (real_rate_change < 0) OR
                # - already very low (real_rate_now < +1.0) — cattura 2020 deflation
                #   crash dove CPI -drops piu di rate-cuts -> real rate sale a 0 from neg
                ff_now = indicators.get("fed_funds_rate", 2.5)
                cpi_now = indicators.get("cpi_yoy", 2.0)
                real_rate_now = ff_now - cpi_now
                real_rate_accommodative = (
                    (real_rate_chg is not None and real_rate_chg < 0)
                    or real_rate_now < 1.0
                )
                if (m2_y is not None and m2_y > 10.0
                        and walcl_r3m is not None and walcl_r3m > 12.0
                        and real_rate_accommodative):
                    liquidity_surge_triggered = True
                    # Boost reflation +40%, dampen deflation × 0.5 (NON cancellare)
                    raw_scores["reflation"] *= 1.40
                    raw_scores["deflation"] *= 0.5
    except Exception:
        pass

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

    # T8.3 — ML ensemble blend (opt-in). Risolve confusioni rule-based su
    # recessioni disinflazionistiche (2000, 1990) mantenendo stabilità.
    ml_metadata: dict[str, Any] | None = None
    if use_ml_regime_blend():
        try:
            from app.services.regime.ml_ensemble import blend_with_rule_based
            # T9-FIX-2: alpha ridotto da 0.4 a 0.25 post-audit
            # (ratio features/samples sub-determinato → ML deve essere tie-breaker, non dominante)
            blended, ml_only = blend_with_rule_based(probabilities, indicators, alpha=0.25)
            ml_metadata = {
                "ml_only_probs": {k: round(v, 4) for k, v in ml_only.items()},
                "rule_based_probs": {k: round(v, 4) for k, v in probabilities.items()},
                "alpha": 0.25,
            }
            # NB: no rounding qui per preservare Σ=1 strict
            probabilities = blended
            # Ricalcola regime dominante post-blend
            regime = max(probabilities, key=probabilities.get)
        except Exception as e:
            # Audit visibile invece di silent failure (raydalio reco)
            import logging
            logging.getLogger(__name__).warning(
                "ML regime blend failed: %s — falling back to rule-based", e
            )
            ml_metadata = {"error": str(e), "fallback": "rule_based"}

    out = {
        "regime": regime,
        "probabilities": probabilities,
        "fit_scores": fit_scores,
        "confidence": round(confidence, 3),
        "conditions_detail": conditions_detail,
    }
    if gdp_collapse_triggered:
        out["gdp_collapse_triggered"] = True
        out["gdp_collapse_severity"] = round(gdp_collapse_severity, 4)

    if liquidity_surge_triggered:
        out["liquidity_surge_triggered"] = True
    if ml_metadata:
        out["ml_blend"] = ml_metadata

    # T9-AUDIT-FIX: financial-stress veto layer (post-classifier).
    # Cattura 2018 Q4 / 2023 SVB dove macro indicators sembravano ok ma
    # market+credit stress era in essere. Riduce confidence × 0.5.
    from app.services.config_flags import use_financial_stress_veto
    if use_financial_stress_veto():
        try:
            from app.services.regime.financial_stress_veto import apply_veto_to_classification
            out = apply_veto_to_classification(out, indicators)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "financial_stress_veto failed: %s", e
            )
    return out
