"""Crisis Risk Indicator (Tier 3 roadmap dashboard).

Aggregatore di livello rischio crisi + classificatore TIPO di crisi attesa.
Riconosce 3 modalita' di crisi, non solo i crash deflattivi:

1. **Deflation Crash** (2008, 2020 style): GDP crolla, CPI scende, credit
   spread esplode, VIX spike → bond long/cash/USD vincono, equity/crypto/EM
   crollano.

2. **Stagflation Debasement** (1973, 2022 style): CPI accelera, GDP rallenta,
   dedollar + breakeven salgono, real assets/gold/commodities/TIPS volano,
   bonds long crollano. Mercato puo' "volare" se misurato in fiat ma perdere
   in real terms.

3. **Bubble Goldilocks** (mean-reversion risk, 2000 dotcom pre-burst, 2021
   SPAC mania): VIX compresso + equity valuations stretched + complacency.
   Rischio asimmetrico downside, da defensive tilt.

Output:
- `risk_level`: low / moderate / elevated / extreme (4 livelli)
- `crisis_type`: deflation_crash / stagflation_debasement / bubble_goldilocks / no_crisis
- `positioning`: dict {asset_class: "overweight" | "neutral" | "underweight"}
- `triggers`: lista di segnali attivi (es. "VIX > 25", "credit_spread > 2.5%")
- `historical_analog`: anno episodio storico più simile
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.services.regime.classifier import classify_regime


# Soglie hardcoded per i triggers (calibrate su episodi storici medi).
_THRESHOLDS = {
    "deflation_prob": 0.30,            # >= → deflation pressure
    "stagflation_prob": 0.35,          # >= → stagflation pressure
    "vix_elevated": 22.0,              # >= → market stress
    "vix_spike": 30.0,                 # >= → panic mode
    "vix_compressed": 14.0,            # <= → complacency (bubble risk)
    "credit_spread_wide": 2.3,         # >= → credit stress (BAA)
    "credit_spread_extreme": 4.0,      # >= → 2008-style
    "cpi_hot": 4.0,                    # >= → inflation alta
    "cpi_collapse": 1.0,               # <= → disinflation severa
    "breakeven_anchored_low": 1.7,     # <= → deflation expectations
    "breakeven_unanchored_high": 2.6,  # >= → debasement expectations
    "nfci_tight": 0.3,                 # >= → financial conditions tight
    "yield_curve_inverted": -0.1,      # <= → recession signal
    "dedollar_high": 0.65,             # >= → debasement signal
    "gdp_weak": 1.0,                   # <= QoQ → growth slowdown
    "unrate_rising_fast": 5.0,         # >= unrate_roc → labor market stress
}


# Pattern asset response per crisis type. Triplet: (overweight, neutral, underweight)
_POSITIONING_BY_TYPE: dict[str, dict[str, str]] = {
    "deflation_crash": {
        "us_bonds_long": "overweight",
        "us_bonds_short": "overweight",
        "cash_money_market": "overweight",
        "gold": "neutral",            # ambivalente: oro storia mista in deflation severa
        "tips_inflation_bonds": "neutral",
        "us_equities_growth": "underweight",
        "us_equities_value": "underweight",
        "em_equities": "underweight",
        "international_dm_equities": "underweight",
        "broad_commodities": "underweight",
        "energy": "underweight",
        "silver": "underweight",
        "real_estate_reits": "underweight",
        "bitcoin": "underweight",
        "crypto_broad": "underweight",
    },
    "stagflation_debasement": {
        "gold": "overweight",
        "silver": "overweight",
        "broad_commodities": "overweight",
        "energy": "overweight",
        "tips_inflation_bonds": "overweight",
        "bitcoin": "overweight",       # in debasement narrative, BTC = digital gold
        "us_equities_value": "neutral",
        "em_equities": "neutral",
        "international_dm_equities": "neutral",
        "real_estate_reits": "neutral",
        "us_equities_growth": "underweight",
        "us_bonds_long": "underweight",  # CRUCIALE: bonds long = perdente
        "us_bonds_short": "neutral",
        "cash_money_market": "underweight",  # perde potere d'acquisto
        "crypto_broad": "overweight",
    },
    "bubble_goldilocks": {
        # Posizione difensiva pre-burst, equity tilt verso value/short-vol
        "cash_money_market": "overweight",
        "us_bonds_short": "overweight",
        "tips_inflation_bonds": "neutral",
        "us_equities_value": "neutral",
        "gold": "neutral",
        "us_bonds_long": "neutral",
        "us_equities_growth": "underweight",  # bolla, mean reversion attesa
        "bitcoin": "underweight",
        "crypto_broad": "underweight",
        "broad_commodities": "neutral",
        "energy": "neutral",
        "silver": "neutral",
        "em_equities": "underweight",
        "international_dm_equities": "neutral",
        "real_estate_reits": "underweight",
    },
    "no_crisis": {
        # Baseline: equal-weight diversified (= portfolio target)
        a: "neutral" for a in (
            "us_bonds_long", "us_bonds_short", "cash_money_market",
            "us_equities_growth", "us_equities_value", "em_equities",
            "international_dm_equities", "tips_inflation_bonds", "gold",
            "silver", "broad_commodities", "energy", "real_estate_reits",
            "bitcoin", "crypto_broad",
        )
    },
}


@dataclass
class CrisisRiskAssessment:
    """Output completo del crisis risk indicator."""
    date: str
    risk_level: str                    # low | moderate | elevated | extreme
    risk_score: float                  # 0-1 (aggregato pesato)
    crisis_type: str                   # deflation_crash | stagflation_debasement | bubble_goldilocks | no_crisis
    crisis_type_confidence: float      # 0-1
    positioning: dict[str, str]        # asset → overweight/neutral/underweight
    triggers: list[dict]               # [{name, value, threshold, fired: bool}]
    historical_analog: str | None      # anno episodio storico più simile
    summary: str                       # 1-2 frasi human-readable IT
    indicators_snapshot: dict[str, float]
    notes: list[str] = field(default_factory=list)


def _evaluate_triggers(ind: dict, regime_probs: dict[str, float]) -> list[dict]:
    """Valuta ogni trigger contro le soglie. Output: lista con `fired: bool`."""
    triggers = []
    checks = [
        ("deflation_prob_high", regime_probs.get("deflation", 0.0), _THRESHOLDS["deflation_prob"], "ge", "Probabilita' deflation regime alta"),
        ("stagflation_prob_high", regime_probs.get("stagflation", 0.0), _THRESHOLDS["stagflation_prob"], "ge", "Probabilita' stagflation regime alta"),
        ("vix_elevated", ind.get("vix", 18.0), _THRESHOLDS["vix_elevated"], "ge", "VIX elevato (stress mercati)"),
        ("vix_spike", ind.get("vix", 18.0), _THRESHOLDS["vix_spike"], "ge", "VIX spike (panic)"),
        ("vix_compressed", ind.get("vix", 18.0), _THRESHOLDS["vix_compressed"], "le", "VIX compresso (complacency)"),
        ("credit_spread_wide", ind.get("baa_spread", 2.0), _THRESHOLDS["credit_spread_wide"], "ge", "Credit spread largo"),
        ("credit_spread_extreme", ind.get("baa_spread", 2.0), _THRESHOLDS["credit_spread_extreme"], "ge", "Credit spread estremo (2008-style)"),
        ("cpi_hot", ind.get("cpi_yoy", 2.0), _THRESHOLDS["cpi_hot"], "ge", "CPI YoY caldo (inflation alta)"),
        ("cpi_collapse", ind.get("cpi_yoy", 2.0), _THRESHOLDS["cpi_collapse"], "le", "CPI collasso (disinflation severa)"),
        ("breakeven_low", ind.get("breakeven_10y", 2.0), _THRESHOLDS["breakeven_anchored_low"], "le", "Breakeven 10Y basso (deflation expect)"),
        ("breakeven_high", ind.get("breakeven_10y", 2.0), _THRESHOLDS["breakeven_unanchored_high"], "ge", "Breakeven 10Y alto (debasement expect)"),
        ("nfci_tight", ind.get("nfci", 0.0), _THRESHOLDS["nfci_tight"], "ge", "Financial conditions tight"),
        ("yield_curve_inverted", ind.get("yield_curve_10y2y", 1.0), _THRESHOLDS["yield_curve_inverted"], "le", "Yield curve invertita (recession signal)"),
        ("gdp_weak", ind.get("gdp_roc", 2.0), _THRESHOLDS["gdp_weak"], "le", "GDP QoQ debole"),
        ("unrate_rising", ind.get("unrate_roc", 0.0), _THRESHOLDS["unrate_rising_fast"], "ge", "Unemployment in salita rapida"),
    ]
    for name, value, threshold, op, description in checks:
        fired = value >= threshold if op == "ge" else value <= threshold
        triggers.append({
            "name": name,
            "value": float(value),
            "threshold": float(threshold),
            "operator": op,
            "fired": bool(fired),
            "description": description,
        })
    return triggers


def _classify_crisis_type(
    regime_probs: dict[str, float],
    triggers: list[dict],
    dedollar_combined: float | None,
) -> tuple[str, float, str | None]:
    """Identifica TIPO di crisi (deflation crash / stagflation debasement /
    bubble goldilocks / no crisis). Ritorna anche confidenza e analog storico.

    Logica:
    - Se deflation_prob alta + credit/VIX/yield curve segnali deflation → deflation_crash
    - Se stagflation_prob alta + breakeven_high + dedollar alto → stagflation_debasement
    - Se vix_compressed + reflation/goldilocks alta + no stress macro → bubble_goldilocks
    - Altrimenti → no_crisis
    """
    fired = {t["name"] for t in triggers if t["fired"]}

    deflation_signals = sum(1 for s in [
        "deflation_prob_high", "credit_spread_wide", "credit_spread_extreme",
        "vix_spike", "cpi_collapse", "breakeven_low", "nfci_tight",
        "yield_curve_inverted", "gdp_weak", "unrate_rising",
    ] if s in fired)

    stagflation_signals = sum(1 for s in [
        "stagflation_prob_high", "cpi_hot", "breakeven_high",
        "gdp_weak", "unrate_rising",
    ] if s in fired)
    if dedollar_combined is not None and dedollar_combined >= _THRESHOLDS["dedollar_high"]:
        stagflation_signals += 1

    bubble_signals = sum(1 for s in [
        "vix_compressed",
    ] if s in fired)
    # Bubble cond aggiuntiva: regime reflation/goldilocks dominante (>0.55)
    benign_dom = max(regime_probs.get("reflation", 0), regime_probs.get("goldilocks", 0))

    # Score per tipo
    defl_score = deflation_signals / 10.0
    stag_score = stagflation_signals / 6.0
    bubble_score = 0.0
    # Soglia 0.45: bubble può essere dominata da reflation (2021) non solo
    # goldilocks puro. Cond: VIX compresso + no stress macro deflation/stag.
    if benign_dom > 0.45 and bubble_signals >= 1 and deflation_signals == 0 and stagflation_signals <= 1:
        bubble_score = 0.5 + (bubble_signals * 0.25)

    scores = {
        "deflation_crash": defl_score,
        "stagflation_debasement": stag_score,
        "bubble_goldilocks": bubble_score,
    }
    top_type = max(scores, key=scores.get)
    top_score = scores[top_type]

    # Threshold: serve almeno 0.30 per dichiarare "crisis brewing"
    if top_score < 0.30:
        return "no_crisis", 1.0 - top_score, None

    # Historical analog per tipo
    analogs = {
        "deflation_crash": "2008 GFC" if defl_score >= 0.6 else "2020 COVID" if defl_score >= 0.4 else "2001 Dotcom soft",
        "stagflation_debasement": "1973 Oil shock" if stag_score >= 0.7 else "2022 Inflation surge" if stag_score >= 0.4 else "2008 commodity peak",
        "bubble_goldilocks": "2000 Dotcom pre-burst" if benign_dom > 0.65 else "2021 SPAC mania",
    }
    return top_type, min(1.0, top_score), analogs[top_type]


def _compute_risk_level(triggers: list[dict], crisis_type: str) -> tuple[str, float]:
    """Aggrega trigger fired in livello rischio + score 0-1."""
    fired_count = sum(1 for t in triggers if t["fired"])
    # Weight per criticita': vix_spike, credit_extreme, defl_prob_high pesano di piu'
    critical = {"vix_spike", "credit_spread_extreme", "deflation_prob_high",
                "stagflation_prob_high", "yield_curve_inverted"}
    critical_count = sum(1 for t in triggers if t["fired"] and t["name"] in critical)

    # Score 0-1: combina critical (peso 2) + others (peso 1) / max possible
    score = (critical_count * 2 + (fired_count - critical_count)) / (len(critical) * 2 + (len(triggers) - len(critical)))
    score = max(0.0, min(1.0, score))

    if crisis_type == "no_crisis":
        if score < 0.15:
            level = "low"
        else:
            level = "moderate"
    else:
        if score >= 0.55:
            level = "extreme"
        elif score >= 0.40:
            level = "elevated"
        elif score >= 0.20:
            level = "moderate"
        else:
            level = "low"

    return level, score


def _build_summary(
    crisis_type: str, level: str, analog: str | None,
    regime_probs: dict[str, float],
) -> str:
    """1-2 frasi human-readable in italiano."""
    if crisis_type == "no_crisis":
        dom = max(regime_probs, key=regime_probs.get)
        return (
            f"Nessuna crisi in atto. Rischio sistema {level}. Regime dominante {dom} "
            f"({regime_probs[dom]*100:.0f}%). Posizionamento bilanciato suggerito."
        )
    type_label = {
        "deflation_crash": "crisi deflazionaria (crollo)",
        "stagflation_debasement": "stagflation/debasement (mercati possono volare in nominale, perdere in real terms)",
        "bubble_goldilocks": "bolla di compiacenza (rischio mean-reversion downside)",
    }[crisis_type]
    analog_str = f" Analogia storica: {analog}." if analog else ""
    return (
        f"Rischio {level}: {type_label}.{analog_str} "
        f"Vedere positioning per asset overweight/underweight."
    )


def assess_crisis_risk(
    indicators: dict[str, float],
    dedollar_combined: float | None = None,
    date: str | None = None,
) -> CrisisRiskAssessment:
    """Pipeline completa crisis risk assessment.

    Args:
        indicators: dict indicators correnti (gdp_roc, cpi_yoy, vix, nfci, ecc).
        dedollar_combined: dedollar combined score 0-1 (opzionale).
        date: ISO date string (default oggi).

    Returns: CrisisRiskAssessment con livello, tipo, positioning, triggers.
    """
    if date is None:
        from datetime import date as _date
        date = _date.today().isoformat()

    # Classifica regime per ottenere probs
    regime_result = classify_regime(indicators)
    regime_probs = regime_result["probabilities"]

    # Valuta tutti i triggers
    triggers = _evaluate_triggers(indicators, regime_probs)

    # Identifica tipo crisi
    crisis_type, conf, analog = _classify_crisis_type(
        regime_probs, triggers, dedollar_combined,
    )

    # Livello rischio aggregato
    level, score = _compute_risk_level(triggers, crisis_type)

    # Positioning
    positioning = _POSITIONING_BY_TYPE.get(crisis_type, _POSITIONING_BY_TYPE["no_crisis"])

    # Summary
    summary = _build_summary(crisis_type, level, analog, regime_probs)

    return CrisisRiskAssessment(
        date=date,
        risk_level=level,
        risk_score=score,
        crisis_type=crisis_type,
        crisis_type_confidence=conf,
        positioning=dict(positioning),
        triggers=triggers,
        historical_analog=analog,
        summary=summary,
        indicators_snapshot={k: float(v) for k, v in indicators.items() if isinstance(v, (int, float))},
    )
