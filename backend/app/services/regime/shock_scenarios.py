"""Shock scenarios: what-if su indicatori macro.

Per ogni scenario predefinito (o custom) modifica un sottoinsieme di indicatori
correnti, ri-applica il rule-based classifier e ricalcola asset scores. Mostra
all'utente "se VIX schizza a 45, in quale regime finiamo? Quali asset salgono?".

Differente dalle traiettorie MC: qui non e' una proiezione probabilistica ma una
sensitivity analysis ad-hoc — utile per stress test e tail risk analysis.

Scenari preset documentati con riferimento storico (es. "VIX 45 = panic 2008/2020").
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.models import RegimeClassification
from app.services.regime.classifier import classify_regime
from app.services.scoring.engine import calculate_final_scores


# Scenari preset: ogni voce e' una funzione (current_indicators) → modified_indicators
PRESET_SCENARIOS = {
    "vix_panic_45": {
        "label": "VIX panic spike a 45",
        "description": "Equivalente storico: 2008-09 GFC, marzo 2020 COVID. Risk-off severo.",
        "deltas": {
            "vix": ("set", 45.0),
            "nfci": ("delta", 0.6),       # NFCI tightens
            "baa_spread": ("delta", 1.5), # credit spread widens
            "breakeven_10y": ("delta", -0.4),
        },
    },
    "fed_cut_100bp": {
        "label": "Fed taglia 100bp",
        "description": "Easing aggressivo (es. 2008, 2020). Stimolo monetario.",
        "deltas": {
            "fed_funds_rate": ("delta", -1.0),
            "yield_curve_10y2y": ("delta", 0.4),  # steepener
            "vix": ("delta", -3.0),
            "breakeven_10y": ("delta", 0.2),
        },
    },
    "inflation_shock_plus3": {
        "label": "Inflation shock +3% CPI YoY",
        "description": "Esempio: oil shock '73, supply chain 2022. Stagflation pressure.",
        "deltas": {
            "cpi_yoy": ("delta", 3.0),
            "core_pce_yoy": ("delta", 2.0),
            "breakeven_10y": ("delta", 0.8),
            "fed_funds_rate": ("delta", 1.5),
        },
    },
    "growth_collapse": {
        "label": "GDP collapse (-3% ROC)",
        "description": "Recessione conclamata. Equivalente: Q4 2008, Q2 2020.",
        "deltas": {
            "gdp_roc": ("delta", -3.0),
            "indpro_roc_12m": ("delta", -4.0),
            "payrolls_roc_12m": ("delta", -2.0),
            "unrate_roc": ("delta", 1.5),
            "lei_roc": ("delta", -2.0),
            "vix": ("delta", 8.0),
        },
    },
    "credit_event": {
        "label": "Credit event (HY spread +400bp)",
        "description": "Stress sui mercati credito. Esempio: 2008, energy 2016.",
        "deltas": {
            "baa_spread": ("delta", 2.5),
            "vix": ("delta", 12.0),
            "nfci": ("delta", 1.0),
            "yield_curve_10y2y": ("delta", -0.3),
        },
    },
    "yield_curve_steepener": {
        "label": "Yield curve steepener +100bp",
        "description": "Atteso early-cycle reflation: bull steepener post-pivot Fed.",
        "deltas": {
            "yield_curve_10y2y": ("delta", 1.0),
            "yield_curve_10y3m": ("delta", 1.2),
            "fed_funds_rate": ("delta", -0.5),
        },
    },
}


@dataclass
class ScenarioResult:
    scenario_key: str
    label: str
    description: str
    baseline_indicators: dict[str, float]
    shocked_indicators: dict[str, float]
    baseline_regime: str
    baseline_probabilities: dict[str, float]
    shocked_regime: str
    shocked_probabilities: dict[str, float]
    baseline_scores: dict[str, float]
    shocked_scores: dict[str, float]
    asset_score_deltas: dict[str, float]


def _apply_deltas(
    baseline: dict[str, float], deltas: dict[str, tuple[str, float]],
) -> dict[str, float]:
    out = dict(baseline)
    for key, (op, value) in deltas.items():
        if op == "set":
            out[key] = float(value)
        elif op == "delta":
            out[key] = float(out.get(key, 0.0)) + float(value)
        else:
            raise ValueError(f"Unknown shock op: {op}")
    return out


def _baseline_indicators(db: Session) -> tuple[dict[str, float], str]:
    """Pesca gli ultimi indicatori dal record piu' recente in DB."""
    last = (
        db.query(RegimeClassification)
        .order_by(RegimeClassification.date.desc()).first()
    )
    if last is None:
        raise ValueError("Nessuna classificazione in DB. Esegui /refresh prima.")
    meta = json.loads(last.conditions_met) if last.conditions_met else {}
    indicators = meta.get("indicators", {}) or {}
    return indicators, str(last.date)


def run_scenario(
    db: Session, scenario_key: str,
    custom_deltas: dict[str, tuple[str, float]] | None = None,
    force_include_dedollar: bool | None = None,
) -> ScenarioResult:
    """Applica scenario preset (o custom) e ritorna confronto baseline vs shocked."""
    if scenario_key == "custom":
        if not custom_deltas:
            raise ValueError("custom scenario richiede custom_deltas")
        label = "Custom shock"
        desc = "Custom user-defined deltas"
        deltas = custom_deltas
    else:
        if scenario_key not in PRESET_SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_key}")
        cfg = PRESET_SCENARIOS[scenario_key]
        label, desc = cfg["label"], cfg["description"]
        deltas = cfg["deltas"]

    baseline, _ = _baseline_indicators(db)
    shocked = _apply_deltas(baseline, deltas)

    base_class = classify_regime(baseline)
    shock_class = classify_regime(shocked)

    base_scores = calculate_final_scores(
        base_class["probabilities"], force_include_dedollar=force_include_dedollar,
    )
    shock_scores = calculate_final_scores(
        shock_class["probabilities"], force_include_dedollar=force_include_dedollar,
    )

    deltas_map = {a: shock_scores[a] - base_scores[a] for a in base_scores}

    return ScenarioResult(
        scenario_key=scenario_key,
        label=label,
        description=desc,
        baseline_indicators=baseline,
        shocked_indicators=shocked,
        baseline_regime=base_class["regime"],
        baseline_probabilities=base_class["probabilities"],
        shocked_regime=shock_class["regime"],
        shocked_probabilities=shock_class["probabilities"],
        baseline_scores=base_scores,
        shocked_scores=shock_scores,
        asset_score_deltas=deltas_map,
    )


def list_preset_scenarios() -> list[dict]:
    """Lista degli scenari preset con metadata."""
    return [
        {
            "key": k,
            "label": v["label"],
            "description": v["description"],
            "deltas": {ind: {"op": op, "value": val} for ind, (op, val) in v["deltas"].items()},
        }
        for k, v in PRESET_SCENARIOS.items()
    ]
