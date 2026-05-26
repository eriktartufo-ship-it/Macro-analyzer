"""Tier 6.8 — Hierarchical sub-regime classifier.

Secondary classifier rule-light sopra il primario 4-quadranti. Idea: reflation
2009 != reflation 2021 != reflation 2024. Stagflation 1973 != 2022. La
sotto-classificazione cattura la "fase" dentro il quadrante per migliorare
attribution e portfolio tilts.

Sub-regimi:
- reflation: early_cycle | mid_cycle | late_cycle
- stagflation: oil_shock | debasement | wage_spiral
- deflation: financial_crisis | demand_shock | disinflation
- goldilocks: complacency | steady_state

Implementazione: rule-light scoring su indicators chiave. Non sostituisce
il regime primario; ritornato come info layer aggiuntivo (`sub_regime` field).

Pattern di lettura indicators identico a classifier.py (snake_case, fallback
neutrali se assente).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SubRegimeAssessment:
    """Esito sub-classificazione: nome + score + descrizione + breakdown."""
    primary_regime: str
    sub_regime: str
    score: float
    description: str
    breakdown: dict[str, float]


def _safe(indicators: dict[str, Any], key: str, default: float) -> float:
    """Get numeric indicator con fallback per None / non-numerico / NaN / Inf.

    T9-AUDIT-BUG-6: delega ad util globale con alias support (per match
    nomi live JSON come `fed_funds_rate`, `baa_spread`, `claims_roc`).
    """
    from app.services.common.indicator_aliases import safe_indicator
    return safe_indicator(indicators, key, default=default)


def _reflation_sub(indicators: dict[str, Any]) -> SubRegimeAssessment:
    """early/mid/late cycle dentro reflation.

    early: gdp basso ma in accel, claims falling, fed accomod, yield steep, unrate alta in calo.
    late: unrate molto bassa, fed alta, yield flat, breakeven >2.5.
    mid: in mezzo.
    """
    gdp = _safe(indicators, "gdp_roc", 2.0)
    unrate = _safe(indicators, "unrate", 4.5)
    fed = _safe(indicators, "fed_funds", 3.0)
    yc = _safe(indicators, "yield_curve_spread", 1.0)
    breakeven = _safe(indicators, "breakeven_10y", 2.0)
    claims_roc = _safe(indicators, "claims_roc", 0.0)

    early_score = 0.0
    early_score += max(0.0, (6.0 - unrate)) * 0  # peso 0 (no contrib)
    early_score += max(0.0, (5.5 - unrate)) * 0.15  # unrate in calo da alta
    early_score += max(0.0, (3.0 - fed)) * 0.20  # fed accomodante
    early_score += max(0.0, yc - 0.8) * 0.20  # yield steep
    early_score += max(0.0, -claims_roc) * 0.20  # claims in calo
    early_score += max(0.0, gdp - 1.0) * 0.10  # gdp positivo

    late_score = 0.0
    late_score += max(0.0, 4.5 - unrate) * 0.20  # unrate molto bassa
    late_score += max(0.0, fed - 4.0) * 0.20  # fed restrittiva
    late_score += max(0.0, 0.5 - yc) * 0.25  # yield flat/invert
    late_score += max(0.0, breakeven - 2.3) * 0.25  # inflation expect alta
    late_score += max(0.0, gdp - 3.0) * 0.10  # gdp surriscaldato

    mid_score = 0.5  # baseline always

    breakdown = {
        "early_cycle": round(early_score, 3),
        "mid_cycle": round(mid_score, 3),
        "late_cycle": round(late_score, 3),
    }
    winner = max(breakdown, key=breakdown.get)
    descriptions = {
        "early_cycle": "Reflation early: unrate scende da picco, Fed accomodante, yield steep.",
        "mid_cycle": "Reflation mid: ciclo maturo, condizioni miste.",
        "late_cycle": "Reflation late: economia surriscaldata, Fed restrittiva, yield flat.",
    }
    return SubRegimeAssessment(
        primary_regime="reflation",
        sub_regime=winner,
        score=breakdown[winner],
        description=descriptions[winner],
        breakdown=breakdown,
    )


def _stagflation_sub(indicators: dict[str, Any]) -> SubRegimeAssessment:
    """oil_shock / debasement / wage_spiral."""
    cpi = _safe(indicators, "cpi_yoy", 3.0)
    energy_yoy = _safe(indicators, "energy_yoy", 0.0)
    dedollar = _safe(indicators, "dedollar_combined", 0.3)
    gold_oil = _safe(indicators, "gold_oil_ratio", 18.0)
    payrolls_yoy = _safe(indicators, "payrolls_yoy", 1.5)
    unrate = _safe(indicators, "unrate", 4.5)

    # Oil shock: cpi alto + energy yoy positivo + gold_oil basso
    oil_shock = 0.0
    oil_shock += max(0.0, cpi - 4.0) * 0.25
    oil_shock += max(0.0, energy_yoy) / 30.0 * 0.30  # +30% energy → +0.30
    oil_shock += max(0.0, 18.0 - gold_oil) / 18.0 * 0.45  # gold_oil basso

    # Debasement: cpi alto + dedollar elevato + gold strong vs oil
    debasement = 0.0
    debasement += max(0.0, cpi - 4.0) * 0.20
    debasement += max(0.0, dedollar - 0.5) * 1.5  # dedollar bias forte
    debasement += max(0.0, gold_oil - 25.0) / 25.0 * 0.30

    # Wage spiral: cpi alto + payrolls forti + unrate bassa (tight labor)
    wage_spiral = 0.0
    wage_spiral += max(0.0, cpi - 4.0) * 0.20
    wage_spiral += max(0.0, payrolls_yoy - 2.0) * 0.20  # payrolls forti
    wage_spiral += max(0.0, 4.5 - unrate) * 0.30  # unrate bassa
    wage_spiral += max(0.0, payrolls_yoy - 3.0) * 0.20  # extra peso

    breakdown = {
        "oil_shock": round(oil_shock, 3),
        "debasement": round(debasement, 3),
        "wage_spiral": round(wage_spiral, 3),
    }
    winner = max(breakdown, key=breakdown.get)
    descriptions = {
        "oil_shock": "Stagflation oil-driven (1973-style): commodities/energy inflation.",
        "debasement": "Stagflation debasement: dollar weak + gold/real assets strong.",
        "wage_spiral": "Stagflation wage-price spiral: labor tight + inflation entrenched.",
    }
    return SubRegimeAssessment(
        primary_regime="stagflation",
        sub_regime=winner,
        score=breakdown[winner],
        description=descriptions[winner],
        breakdown=breakdown,
    )


def _deflation_sub(indicators: dict[str, Any]) -> SubRegimeAssessment:
    """financial_crisis / demand_shock / disinflation."""
    baa = _safe(indicators, "baa_10y_spread", 2.0)
    vix = _safe(indicators, "vix", 18.0)
    nfci = _safe(indicators, "nfci", 0.0)
    gdp = _safe(indicators, "gdp_roc", 0.0)
    unrate_roc = _safe(indicators, "unrate_roc", 0.0)
    cpi = _safe(indicators, "cpi_yoy", 2.0)
    claims_roc = _safe(indicators, "claims_roc", 0.0)

    # Financial crisis: BAA wide + VIX spike + NFCI tight
    fin_crisis = 0.0
    fin_crisis += max(0.0, baa - 2.5) * 0.30  # BAA wide
    fin_crisis += max(0.0, vix - 25.0) / 25.0 * 0.35  # VIX panic
    fin_crisis += max(0.0, nfci - 0.2) * 0.35  # NFCI tight

    # Demand shock: gdp neg + unrate spike + claims spike
    demand_shock = 0.0
    demand_shock += max(0.0, -gdp) * 0.30  # gdp negativo
    demand_shock += max(0.0, unrate_roc) * 0.30  # unrate in salita
    demand_shock += max(0.0, claims_roc) / 10.0 * 0.40  # claims spike

    # Disinflation soft: cpi in calo verso 0-2, ma economy non in crisi
    disinflation = 0.0
    disinflation += max(0.0, 2.0 - cpi) * 0.30  # cpi sotto target
    disinflation += max(0.0, 2.5 - baa) * 0.25  # BAA contenuto
    disinflation += max(0.0, 22.0 - vix) / 22.0 * 0.25  # VIX normal
    disinflation += max(0.0, gdp + 1.0) * 0.20  # gdp non collassato

    breakdown = {
        "financial_crisis": round(fin_crisis, 3),
        "demand_shock": round(demand_shock, 3),
        "disinflation": round(disinflation, 3),
    }
    winner = max(breakdown, key=breakdown.get)
    descriptions = {
        "financial_crisis": "Deflation financial-driven (2008-style): credit + VIX panic.",
        "demand_shock": "Deflation demand shock (2020-style): GDP collapse + claims spike.",
        "disinflation": "Disinflation soft: CPI sotto target, no crisis features.",
    }
    return SubRegimeAssessment(
        primary_regime="deflation",
        sub_regime=winner,
        score=breakdown[winner],
        description=descriptions[winner],
        breakdown=breakdown,
    )


def _goldilocks_sub(indicators: dict[str, Any]) -> SubRegimeAssessment:
    """complacency / steady_state."""
    vix = _safe(indicators, "vix", 18.0)
    nfci = _safe(indicators, "nfci", -0.1)
    cpi = _safe(indicators, "cpi_yoy", 2.0)
    unrate = _safe(indicators, "unrate", 4.0)
    gdp = _safe(indicators, "gdp_roc", 2.0)

    # Complacency: VIX molto basso + NFCI easy + unrate molto bassa
    complacency = 0.0
    complacency += max(0.0, (16.0 - vix) / 16.0) * 1.5  # VIX al pavimento
    complacency += max(0.0, -nfci - 0.2) * 0.5  # NFCI molto easy
    complacency += max(0.0, 4.2 - unrate) * 0.5  # unrate molto bassa

    # Steady state: tutto nei range
    steady_state = 0.0
    steady_state += 1.0 if 1.0 <= cpi <= 2.5 else 0.0
    steady_state += 1.0 if 1.5 <= gdp <= 3.0 else 0.0
    steady_state += 1.0 if 3.5 <= unrate <= 5.0 else 0.0
    steady_state += 1.0 if 14.0 <= vix <= 22.0 else 0.0
    steady_state *= 0.25  # normalize 0-1

    breakdown = {
        "complacency": round(complacency, 3),
        "steady_state": round(steady_state, 3),
    }
    winner = max(breakdown, key=breakdown.get)
    descriptions = {
        "complacency": "Goldilocks complacency: VIX al pavimento, leverage in salita, fragility latente.",
        "steady_state": "Goldilocks steady-state: tutti gli indicators in range desiderato.",
    }
    return SubRegimeAssessment(
        primary_regime="goldilocks",
        sub_regime=winner,
        score=breakdown[winner],
        description=descriptions[winner],
        breakdown=breakdown,
    )


_DISPATCH = {
    "reflation": _reflation_sub,
    "stagflation": _stagflation_sub,
    "deflation": _deflation_sub,
    "goldilocks": _goldilocks_sub,
}


def classify_sub_regime(
    primary_regime: str,
    indicators: dict[str, Any],
) -> SubRegimeAssessment | None:
    """Classifica sub-regime per `primary_regime` dato indicators.

    Returns None se primary_regime non riconosciuto.
    """
    handler = _DISPATCH.get(primary_regime)
    if handler is None:
        return None
    return handler(indicators)


def list_sub_regimes_for(primary_regime: str) -> list[str]:
    """Lista nomi sub-regime ammessi per un primary regime."""
    samples = {
        "reflation": ["early_cycle", "mid_cycle", "late_cycle"],
        "stagflation": ["oil_shock", "debasement", "wage_spiral"],
        "deflation": ["financial_crisis", "demand_shock", "disinflation"],
        "goldilocks": ["complacency", "steady_state"],
    }
    return samples.get(primary_regime, [])
