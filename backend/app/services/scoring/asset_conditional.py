"""T9-AUDIT — Conditional asset scoring P(asset | regime, stress_level).

Council finding: top-5 overlap fermo a 2.11/5 (vs random 1.67/5 → CI include
random). Asset layer disaccoppiato dal regime layer. Il rank_percentile
within-regime ignora che dentro lo stesso regime ci sono SOTTO-stati con
asset winners diversi:

- **Reflation calm** (VIX < 18, BAA < 1.8): equities growth dominant.
- **Reflation stressed** (VIX 18-25): equities value + REIT, no crypto.
- **Stagflation early** (CPI 4-6, VIX moderate): energy + commodities.
- **Stagflation late** (CPI > 7, recession starting): gold + cash + TIPS.
- **Goldilocks** (CPI < 2.5, VIX < 16): full risk-on.
- **Deflation crash** (VIX > 30, credit blow-out): bonds long + cash + gold.
- **Deflation disinflation** (VIX 18-25, gentle): bonds + utility + value.

Soluzione: introdurre **stress_level** (low/medium/high) come secondo asse,
producendo 12 sub-buckets invece di 4 regimi. Per ogni (regime, stress_level)
una tabella distinta di asset preferiti, derivata empiricamente dai 37
episodi storici labeled.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class StressLevel:
    """Stress macro/financial level: low/medium/high."""
    level: str  # "low" | "medium" | "high"
    score: float  # 0-1 scale
    description: str


# Conditional asset top-N per (regime, stress_level).
# Derivato dagli actual_top5 dei 37 episodi storici. Pesati per recency
# (anni recenti pesano di più) ma con copertura cross-cycle.
CONDITIONAL_ASSET_TOP_N: dict[tuple[str, str], list[str]] = {
    # === REFLATION ===
    ("reflation", "low"): [
        "us_equities_growth", "us_equities_value", "em_equities",
        "real_estate_reits", "international_dm_equities",
        "bitcoin", "gold",  # T9-AUDIT-FIX-2: long-tail (recent 2020-25 reflation)
    ],
    ("reflation", "medium"): [
        "us_equities_value", "us_equities_growth", "real_estate_reits",
        "international_dm_equities", "broad_commodities",
        "tips_inflation_bonds", "gold",
    ],
    ("reflation", "high"): [
        # Reflation con stress alto = late cycle / pre-stagflation
        "us_equities_value", "broad_commodities", "energy",
        "gold", "tips_inflation_bonds", "us_bonds_short", "real_estate_reits",
    ],
    # === STAGFLATION ===
    ("stagflation", "low"): [
        # Stag early/mild: commodities + value
        "broad_commodities", "energy", "gold",
        "us_equities_value", "real_estate_reits",
        "tips_inflation_bonds", "cash_money_market",
    ],
    ("stagflation", "medium"): [
        "energy", "broad_commodities", "gold",
        "cash_money_market", "tips_inflation_bonds",
        "us_equities_value", "us_bonds_short",
    ],
    ("stagflation", "high"): [
        # Stag severa (1973, 1980): commodities/gold ma cash king
        "gold", "energy", "broad_commodities",
        "cash_money_market", "us_bonds_short",
        "tips_inflation_bonds", "us_equities_value",
    ],
    # === DEFLATION ===
    ("deflation", "low"): [
        # Mild disinflation (2015, soft landing): bonds + equities defensive
        "us_bonds_long", "us_bonds_short", "tips_inflation_bonds",
        "us_equities_growth", "gold",
        "cash_money_market", "real_estate_reits",
    ],
    ("deflation", "medium"): [
        # Recession but not crash (2001, 2018Q4): bonds dominant
        "us_bonds_long", "us_bonds_short", "cash_money_market",
        "gold", "tips_inflation_bonds",
        "us_equities_value", "us_equities_growth",
    ],
    ("deflation", "high"): [
        # Crisi acuta (2008, 2020 Q1, 1929): long bonds + cash + gold
        "us_bonds_long", "us_bonds_short", "cash_money_market",
        "gold", "tips_inflation_bonds",
        "us_equities_growth", "us_equities_value",  # post-crash recovery
    ],
    # === GOLDILOCKS ===
    ("goldilocks", "low"): [
        # Full goldilocks (1995-99, 2013-19, 2024): risk-on + bitcoin long-tail
        "us_equities_growth", "us_equities_value", "real_estate_reits",
        "international_dm_equities", "bitcoin",  # T9-AUDIT-FIX-2: 2024 disinflation
        "gold", "em_equities",
    ],
    ("goldilocks", "medium"): [
        # Goldilocks con qualche tensione (2017-19 late, 2019 Q4)
        "us_equities_growth", "us_equities_value", "real_estate_reits",
        "gold", "us_bonds_long",
        "international_dm_equities", "us_bonds_short",
    ],
    ("goldilocks", "high"): [
        # Goldilocks con stress (raro: 1987 black monday, 1998 LTCM)
        "us_equities_value", "us_bonds_long", "gold",
        "cash_money_market", "us_equities_growth",
        "us_bonds_short", "real_estate_reits",
    ],
}


def assess_stress_level(indicators: dict[str, Any]) -> StressLevel:
    """Calcola stress_level low/medium/high da indicators correnti.

    Combina:
    - VIX (market stress, range 9-80)
    - BAA-10Y spread (credit stress, range 0.5-6)
    - NFCI (financial conditions, range -1 to 3)
    - Unrate ROC (labor stress)

    Score finale 0-1, threshold:
    - score < 0.30 → low
    - 0.30-0.55 → medium
    - > 0.55 → high
    """
    # T9-AUDIT-BUG-7: delega ad util globale per match alias live JSON
    from app.services.common.indicator_aliases import safe_indicator

    vix = safe_indicator(indicators, "vix", default=18.0)
    baa = safe_indicator(indicators, "baa_10y_spread", default=1.8)
    nfci = safe_indicator(indicators, "nfci", default=0.0)
    unrate_roc = safe_indicator(indicators, "unrate_roc", default=0.0)

    # Normalize each to 0-1 stress contribution
    vix_stress = max(0.0, min(1.0, (vix - 14.0) / 30.0))  # 14→0, 44→1
    baa_stress = max(0.0, min(1.0, (baa - 1.5) / 3.5))    # 1.5→0, 5.0→1
    nfci_stress = max(0.0, min(1.0, (nfci + 0.3) / 1.3))  # -0.3→0, 1.0→1
    unrate_stress = max(0.0, min(1.0, unrate_roc / 2.0))  # 0→0, 2→1

    # Weighted average (VIX + BAA più importanti)
    composite = (
        vix_stress * 0.35 + baa_stress * 0.30 +
        nfci_stress * 0.20 + unrate_stress * 0.15
    )

    if composite < 0.30:
        level = "low"
        desc = f"Stress low (composite {composite:.2f}): mercato calmo, full risk-on possibile."
    elif composite < 0.55:
        level = "medium"
        desc = f"Stress medium (composite {composite:.2f}): cautela, tilt verso value/defensive."
    else:
        level = "high"
        desc = f"Stress high (composite {composite:.2f}): risk-off, defensive allocation preferita."

    return StressLevel(level=level, score=round(composite, 4), description=desc)


def get_conditional_top_n(
    regime: str,
    stress_level: str,
    top_n: int = 5,
) -> list[str]:
    """Ritorna i top-N asset preferiti per (regime, stress_level).

    Args:
        regime: reflation|stagflation|deflation|goldilocks
        stress_level: low|medium|high
        top_n: numero asset da ritornare (default 5)

    Returns:
        Lista ordered top-N asset names.
    """
    key = (regime, stress_level)
    full_list = CONDITIONAL_ASSET_TOP_N.get(key, [])
    if not full_list:
        # Fallback: usa "medium" stress se key sconosciuta
        key_fb = (regime, "medium")
        full_list = CONDITIONAL_ASSET_TOP_N.get(key_fb, [])
    return full_list[:top_n]


def calculate_conditional_scores(
    probabilities: dict[str, float],
    indicators: dict[str, Any],
    asset_classes: list[str],
    base_scores: dict[str, float] | None = None,
) -> tuple[dict[str, float], StressLevel]:
    """Conditional asset scoring — **additive bonus** sopra base_scores.

    T9-AUDIT-FIX-2 (rev2): blend totale 0.7/0.3 e sostituzione 50/50 hanno
    PEGGIORATO il top-5 OOS (2.11 → 2.00). Cambio strategia: usare conditional
    come **additive bonus chirurgico** sui top-3 conditional asset, +bonus
    proporzionale alla regime prob × stress fit. Base scores invariati per
    gli altri asset → no degradation.

    Bonus formula:
    - top-1 conditional in regime r con prob_r ≥ 0.4: +12 punti
    - top-2 conditional: +8 punti
    - top-3 conditional: +5 punti

    Args:
        probabilities: regime probs (Σ=1)
        indicators: macro indicators per stress assessment
        asset_classes: lista 15 asset universe
        base_scores: scores base (richiesto per modalità bonus).

    Returns:
        (final_scores, stress_level_assessment)
    """
    stress = assess_stress_level(indicators)

    # Se nessun base_scores fornito, fallback al vecchio comportamento
    if base_scores is None:
        base_scores = {a: 50.0 for a in asset_classes}

    final_scores: dict[str, float] = dict(base_scores)

    # Per ogni regime dominante, applica bonus ai top-3 conditional.
    # T9-AUDIT-FIX-3 (council post-audit): bonus ridotti 12/8/5 → 5/3/1.
    # Bonus alti distorcevano lo scoring senza migliorare top-5 OOS.
    # Tie-breaker più leggero = no degradazione, asset layer onesto.
    bonus_per_rank = [5.0, 3.0, 1.0]
    for regime, prob in probabilities.items():
        if prob < 0.30:  # solo regimi rilevanti (no noise)
            continue
        conditional_top = get_conditional_top_n(regime, stress.level, top_n=3)
        for rank, asset in enumerate(conditional_top):
            if asset not in final_scores:
                continue
            # Bonus scalato per prob regime e rank
            bonus = bonus_per_rank[rank] * prob
            final_scores[asset] = min(100.0, final_scores[asset] + bonus)

    # Round
    final_scores = {k: round(v, 2) for k, v in final_scores.items()}
    return final_scores, stress
