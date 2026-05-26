"""Tier 6.9 — Freshness weighting per condition del classifier.

Filosofia: il GDP trimestrale ha lag ~60 giorni rispetto al VIX daily. Trattarli
con peso identico nel classifier sottostima i segnali real-time e sovrappesa
quelli stale. Soluzione: `effective_weight = base_weight × freshness_score`,
dove freshness ∈ [0, 1] dipende dalla cadenza di pubblicazione tipica.

Score di default per frequenza:
- daily (VIX, yield curve, fed funds, ratios cross-asset, term premium): 1.0
- ~daily LLM-scored (news, insider, dedollar combined): 0.95
- weekly (claims, NFCI): 0.9
- monthly tier-1 (CPI, PCE, PMI, payrolls): 0.8
- monthly tier-2 (industrial production, retail sales, sentiment, housing): 0.7
- monthly slow (LEI, breakeven), DFM nowcast: 0.7
- quarterly (GDP YoY/QoQ): 0.4

Pattern: dict `CONDITION_FRESHNESS[cond_name] = score`. Condition non mappate
restano a `DEFAULT_FRESHNESS=0.7` (monthly assumption conservativa).
"""
from __future__ import annotations

DEFAULT_FRESHNESS = 0.7

# Mapping esplicito condition_name → freshness_score (0-1).
# Allineato a `REGIME_CONDITIONS` in classifier.py + alle condition cross-asset.
CONDITION_FRESHNESS: dict[str, float] = {
    # Daily (mercato + policy daily)
    "yield_curve_steep": 1.0,
    "yield_curve_stress": 1.0,
    "yield_curve_flat_or_inverted": 1.0,
    "yield_curve_normal": 1.0,
    "policy_accommodative": 1.0,
    "policy_restrictive": 1.0,
    "policy_neutral": 1.0,
    "credit_spread_tight": 1.0,
    "credit_spread_wide": 1.0,
    "vix_low": 1.0,
    "vix_calm": 1.0,
    "vix_elevated": 1.0,
    "vix_spike": 1.0,
    "term_premium_low": 1.0,
    "term_premium_high": 1.0,
    "copper_gold_strong": 1.0,
    "copper_gold_weak": 1.0,
    "gold_oil_low": 1.0,
    "gold_oil_high": 1.0,
    "hy_ig_tight": 1.0,
    "hy_ig_stress": 1.0,
    "credit_stress": 1.0,
    # ~Daily alt-data LLM scored
    "news_positive_strong": 0.95,
    "news_panic": 0.95,
    "insider_buying_strong": 0.95,
    "insider_selling_strong": 0.95,
    "dedollar_low": 0.90,
    "dedollar_debasement": 0.90,
    # Weekly
    "claims_falling": 0.90,
    "claims_rising": 0.90,
    "claims_low": 0.90,
    "financial_conditions_loose": 0.90,
    "financial_conditions_easy": 0.90,
    "nfci_tight": 0.90,
    "breakeven_high": 0.95,
    "breakeven_collapse": 0.95,
    "breakeven_stable": 0.95,
    # Monthly tier-1 (CPI, PMI, payrolls, unrate)
    "inflation_rising": 0.80,
    "inflation_high": 0.80,
    "inflation_low": 0.80,
    "core_pce_high": 0.80,
    "core_pce_contained": 0.80,
    "pmi_expansion": 0.80,
    "pmi_weak": 0.80,
    "pmi_contraction": 0.80,
    "pmi_healthy": 0.80,
    "payrolls_growth": 0.80,
    "payrolls_slowdown": 0.80,
    "unemployment_low_or_falling": 0.80,
    "unemployment_rising": 0.80,
    "unemployment_very_low": 0.80,
    # Monthly tier-2
    "indpro_growth": 0.70,
    "indpro_contraction": 0.70,
    "housing_expansion": 0.65,
    "housing_slowdown": 0.65,
    "sentiment_high": 0.70,
    "sentiment_low": 0.70,
    "lei_positive": 0.70,
    "lei_negative": 0.70,
    # DFM nowcast (monthly fit)
    "dfm_growth_strong": 0.75,
    "dfm_growth_weak": 0.75,
    # Quarterly
    "gdp_strong": 0.40,
    "gdp_weak": 0.40,
    "gdp_moderate": 0.40,
    "gdp_negative_or_decelerating": 0.40,
}


def freshness_for(cond_name: str) -> float:
    """Returns freshness score per condition; default 0.7 se non mappata."""
    return CONDITION_FRESHNESS.get(cond_name, DEFAULT_FRESHNESS)


def effective_weight(cond_name: str, base_weight: float) -> float:
    """`base_weight × freshness_score`. Per condition non mappate uso default."""
    return base_weight * freshness_for(cond_name)
