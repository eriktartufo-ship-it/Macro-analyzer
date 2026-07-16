"""METADATI degli asset per le viste flussi (S45): area geografica + spettro safe->risky.

Universo completo = 25 prodotti (24 di ASSET_REGIME_DATA + semiconduttori, isolato dallo scoring).
- area: usa | world (DM ex-US) | em (emergenti) | global (oro/commodities/cripto/cash: no geografia)
- risk_rank in [0,1]: scala FISSA safe(0)->risky(1), "dal cash al bitcoin passando per l'azionario".
  Scala editabile a mano (non derivata da vol: e' una convenzione di lettura, dichiarata).
"""
from __future__ import annotations

# ordine = universo canonico delle viste flussi (25)
FLOW_ASSETS = [
    "cash_money_market", "us_bonds_short", "tips_inflation_bonds", "us_bonds_long",
    "sector_staples", "gold", "sector_utilities", "real_estate_reits", "sector_healthcare",
    "factor_quality", "factor_value", "us_equities_value", "international_dm_equities",
    "sector_financials", "factor_momentum", "us_equities_growth", "sector_energy",
    "broad_commodities", "energy", "em_equities", "silver", "sector_technology",
    "sector_semiconductors", "crypto_broad", "bitcoin",
]

# label, area, risk_rank (0=safe .. 1=risky)
META: dict[str, dict] = {
    "cash_money_market":        {"label": "Cash",          "area": "global", "risk": 0.02},
    "us_bonds_short":           {"label": "Bond breve",    "area": "usa",    "risk": 0.08},
    "tips_inflation_bonds":     {"label": "TIPS",          "area": "usa",    "risk": 0.16},
    "us_bonds_long":            {"label": "Bond lungo",    "area": "usa",    "risk": 0.24},
    "sector_staples":           {"label": "Staples",       "area": "usa",    "risk": 0.34},
    "gold":                     {"label": "Oro",           "area": "global", "risk": 0.38},
    "sector_utilities":         {"label": "Utilities",     "area": "usa",    "risk": 0.40},
    "real_estate_reits":        {"label": "REITs",         "area": "usa",    "risk": 0.46},
    "sector_healthcare":        {"label": "Healthcare",    "area": "usa",    "risk": 0.47},
    "factor_quality":           {"label": "Quality",       "area": "usa",    "risk": 0.49},
    "factor_value":             {"label": "Value (fct)",   "area": "usa",    "risk": 0.50},
    "us_equities_value":        {"label": "US Value",      "area": "usa",    "risk": 0.52},
    "international_dm_equities": {"label": "World ex-US",   "area": "world",  "risk": 0.55},
    "sector_financials":        {"label": "Financials",    "area": "usa",    "risk": 0.56},
    "factor_momentum":          {"label": "Momentum",      "area": "usa",    "risk": 0.58},
    "us_equities_growth":       {"label": "US Growth",     "area": "usa",    "risk": 0.62},
    "sector_energy":            {"label": "Sett. Energia", "area": "usa",    "risk": 0.64},
    "broad_commodities":        {"label": "Commodities",   "area": "global", "risk": 0.68},
    "energy":                   {"label": "Energia",       "area": "global", "risk": 0.70},
    "em_equities":              {"label": "Emergenti",     "area": "em",     "risk": 0.74},
    "silver":                   {"label": "Argento",       "area": "global", "risk": 0.78},
    "sector_technology":        {"label": "Technology",    "area": "usa",    "risk": 0.80},
    "sector_semiconductors":    {"label": "Semiconduttori","area": "usa",    "risk": 0.86},
    "crypto_broad":             {"label": "Cripto broad",  "area": "global", "risk": 0.93},
    "bitcoin":                  {"label": "Bitcoin",       "area": "global", "risk": 1.00},
}

AREA_LABEL = {"usa": "USA", "world": "World ex-US", "em": "Emergenti", "global": "Globale"}


def label_of(asset: str) -> str:
    return META.get(asset, {}).get("label", asset)


def area_of(asset: str) -> str:
    return META.get(asset, {}).get("area", "global")


def risk_of(asset: str) -> float:
    return META.get(asset, {}).get("risk", 0.5)
