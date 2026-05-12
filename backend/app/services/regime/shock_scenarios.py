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
    # ========================================================================
    # Tier 4.9 — Stress historical replay: 6 episodi storici parametrici.
    # Delta calibrati su dati reali (peak-to-trough vs baseline pre-shock).
    # ========================================================================
    "historical_1973_oil_shock": {
        "label": "1973-74 Oil shock (stagflation)",
        "description": (
            "OPEC embargo Ottobre 1973: oil quadruplica, CPI da 3% a 12%, GDP crolla, "
            "unemployment da 4.8% a 9%, Fed funds da 8% a 13%. Birth of 'stagflation'."
        ),
        "deltas": {
            "cpi_yoy": ("delta", 6.0),         # CPI 3% → 9%+
            "core_pce_yoy": ("delta", 4.5),
            "gdp_roc": ("delta", -3.5),        # GDP collapse
            "unrate": ("delta", 3.5),          # 4.8 → ~8
            "unrate_roc": ("delta", 60.0),     # % change YoY positivo grosso
            "fed_funds_rate": ("delta", 5.0),
            "breakeven_10y": ("delta", 2.5),
            "vix": ("delta", 12.0),
            "consumer_sentiment": ("delta", -25.0),
            "indpro_roc_12m": ("delta", -6.0),
        },
    },
    "historical_1987_black_monday": {
        "label": "1987 Black Monday (volatility shock)",
        "description": (
            "19 ottobre 1987: Dow -22.6% in un giorno. VIX-equivalent picco a 150. "
            "Ma economia reale OK: nessuna recession, Greenspan inietta liquidita'. "
            "Pure volatility shock, credit non si rompe."
        ),
        "deltas": {
            "vix": ("delta", 25.0),            # picco volatility
            "fed_funds_rate": ("delta", -0.5), # liquidity injection
            "nfci": ("delta", 0.4),            # tensione finanziaria moderata
            "baa_spread": ("delta", 0.5),      # credit lievemente
            "gdp_roc": ("delta", -0.5),        # rallenta poco
            "consumer_sentiment": ("delta", -8.0),
            "yield_curve_10y2y": ("delta", -0.3),
        },
    },
    "historical_2000_dotcom_bust": {
        "label": "2000-01 Dotcom bust (equity crash)",
        "description": (
            "Marzo 2000-Ott 2002: Nasdaq -78%. GDP rallenta ma non recessione severa "
            "(mild recession Mar-Nov 2001). CPI scende, Fed taglia da 6.5% a 1.75%. "
            "Credit relativamente OK (no banking crisis)."
        ),
        "deltas": {
            "vix": ("delta", 15.0),            # VIX 20 → 35+
            "gdp_roc": ("delta", -2.0),        # rallenta ma non collassa
            "cpi_yoy": ("delta", -1.0),        # disinflation
            "core_pce_yoy": ("delta", -0.8),
            "unrate": ("delta", 2.0),          # 4 → 6
            "unrate_roc": ("delta", 30.0),
            "fed_funds_rate": ("delta", -3.0), # Fed cuts agressive
            "yield_curve_10y2y": ("delta", -0.5),
            "indpro_roc_12m": ("delta", -3.5),
            "consumer_sentiment": ("delta", -15.0),
            "baa_spread": ("delta", 0.8),      # credit modesto
            "breakeven_10y": ("delta", -0.6),
        },
    },
    "historical_2008_gfc": {
        "label": "2008-09 Global Financial Crisis (deflation severa)",
        "description": (
            "Lehman Sept 2008. GDP -4.3% Q4 2008. Unrate 5% → 10%. BAA spread 2% → 6%+. "
            "VIX picco 80. Fed taglia da 5.25% a 0%. CPI da +5.6% a -2% (deflation). "
            "Deflation severa + credit crisis."
        ),
        "deltas": {
            "gdp_roc": ("delta", -5.0),
            "cpi_yoy": ("delta", -3.5),        # CPI collapse (oil + demand)
            "core_pce_yoy": ("delta", -1.5),
            "unrate": ("delta", 5.0),          # 5 → 10
            "unrate_roc": ("delta", 100.0),    # raddoppia YoY
            "fed_funds_rate": ("delta", -4.5), # 5.25 → 0
            "vix": ("delta", 30.0),            # picco 80
            "baa_spread": ("delta", 4.0),      # 2% → 6%+
            "nfci": ("delta", 2.0),            # extreme tightening
            "yield_curve_10y2y": ("delta", -0.5),
            "breakeven_10y": ("delta", -1.5),  # deflation expectations
            "lei_roc": ("delta", -8.0),
            "indpro_roc_12m": ("delta", -12.0),
            "payrolls_roc_12m": ("delta", -5.0),
            "consumer_sentiment": ("delta", -30.0),
            "housing_starts_roc_12m": ("delta", -45.0),
        },
    },
    "historical_2020_covid": {
        "label": "2020 COVID shock (rapid deflation)",
        "description": (
            "Marzo 2020 lockdown globale. GDP -9% Q2. Unrate 3.5% → 14.7% in 2 mesi. "
            "VIX picco 82. Fed cuts 150bp + QE infinito. CPI scende temporaneamente. "
            "Velocita' di shock senza precedenti (settimane, non mesi come 2008)."
        ),
        "deltas": {
            "gdp_roc": ("delta", -8.0),        # Q2 2020 -9% annualized
            "cpi_yoy": ("delta", -2.0),        # disinflation transitoria
            "core_pce_yoy": ("delta", -1.0),
            "unrate": ("delta", 10.0),         # 3.5 → 14.7
            "unrate_roc": ("delta", 280.0),    # 4x YoY
            "fed_funds_rate": ("delta", -1.5), # 1.75 → 0.25
            "vix": ("delta", 35.0),            # picco 82
            "baa_spread": ("delta", 3.0),
            "nfci": ("delta", 1.8),
            "lei_roc": ("delta", -10.0),
            "indpro_roc_12m": ("delta", -15.0),
            "payrolls_roc_12m": ("delta", -8.0),
            "consumer_sentiment": ("delta", -25.0),
            "housing_starts_roc_12m": ("delta", -20.0),
            "initial_claims_roc": ("delta", 300.0),  # claims weekly +x10
        },
    },
    "historical_2022_inflation_surge": {
        "label": "2022 Inflation surge (stagflation soft)",
        "description": (
            "Post-COVID + Russia/Ukraine. CPI da 1.4% a 9.1% picco Giugno 2022. "
            "Fed funds da 0.25% a 5.25% in 14 mesi (piu' rapido in 40 anni). "
            "GDP rallenta ma niente recessione tecnica. Soft stagflation."
        ),
        "deltas": {
            "cpi_yoy": ("delta", 6.0),         # 1.4 → 7+
            "core_pce_yoy": ("delta", 3.5),
            "breakeven_10y": ("delta", 1.0),
            "fed_funds_rate": ("delta", 4.5),  # 0.25 → 4.75
            "gdp_roc": ("delta", -1.5),        # rallenta
            "yield_curve_10y2y": ("delta", -1.5),  # inverte
            "yield_curve_10y3m": ("delta", -2.0),
            "consumer_sentiment": ("delta", -20.0),  # picco minimo storico
            "vix": ("delta", 8.0),
            "housing_starts_roc_12m": ("delta", -15.0),  # mortgage rates 7%
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
