"""Regime Trajectory Analyzer — predice la direzione del regime.

Combina:
  1. Trend degli indicatori macro (dove stanno andando i dati)
  2. News sentiment (cosa dice il mercato)
  3. Dedollarizzazione (pressione strutturale)

Produce:
  - Probabilità proiettate (dove saremo tra 3-6 mesi)
  - Forze in gioco (cosa sta spingendo verso ogni regime)
  - Regime più probabile futuro
"""

from typing import Any

from app.services.regime.classifier import REGIMES


# Mapping: quale indicatore in trend spinge verso quale regime
TREND_REGIME_PRESSURE = {
    # Inflazione in salita → stagflation/reflation
    "cpi_rising": {
        "stagflation": 0.30,
        "reflation": 0.15,
        "goldilocks": -0.25,
        "deflation": -0.15,
    },
    # Inflazione in calo → goldilocks/deflation
    "cpi_falling": {
        "goldilocks": 0.25,
        "deflation": 0.15,
        "stagflation": -0.30,
        "reflation": -0.10,
    },
    # GDP in rallentamento → deflation/stagflation
    "gdp_decelerating": {
        "deflation": 0.30,
        "stagflation": 0.15,
        "reflation": -0.25,
        "goldilocks": -0.15,
    },
    # GDP in accelerazione → reflation/goldilocks
    "gdp_accelerating": {
        "reflation": 0.30,
        "goldilocks": 0.15,
        "deflation": -0.25,
        "stagflation": -0.10,
    },
    # Disoccupazione in salita → deflation/stagflation
    "unemployment_rising": {
        "deflation": 0.25,
        "stagflation": 0.15,
        "reflation": -0.20,
        "goldilocks": -0.15,
    },
    # Disoccupazione in calo → reflation/goldilocks
    "unemployment_falling": {
        "reflation": 0.25,
        "goldilocks": 0.15,
        "deflation": -0.20,
        "stagflation": -0.10,
    },
    # Yield curve in inversione → deflation
    "yield_curve_inverting": {
        "deflation": 0.30,
        "stagflation": 0.10,
        "reflation": -0.20,
        "goldilocks": -0.10,
    },
    # Yield curve in normalizzazione → reflation
    "yield_curve_steepening": {
        "reflation": 0.25,
        "goldilocks": 0.10,
        "deflation": -0.20,
        "stagflation": -0.05,
    },
    # Claims in aumento → deflation
    "claims_rising": {
        "deflation": 0.25,
        "stagflation": 0.10,
        "reflation": -0.15,
        "goldilocks": -0.10,
    },
    # Claims in calo → reflation
    "claims_falling": {
        "reflation": 0.25,
        "goldilocks": 0.10,
        "deflation": -0.20,
        "stagflation": -0.05,
    },
    # Breakeven inflation in salita → mercato prezza più inflazione futura
    "breakeven_rising": {
        "stagflation": 0.25,
        "reflation": 0.15,
        "goldilocks": -0.20,
        "deflation": -0.25,
    },
    # Breakeven in calo → aspettative inflazione crollano
    "breakeven_falling": {
        "deflation": 0.25,
        "goldilocks": 0.15,
        "stagflation": -0.25,
        "reflation": -0.10,
    },
    # VIX spike → stress improvviso, pre-crisi/deflation
    "vix_spiking": {
        "deflation": 0.30,
        "stagflation": 0.10,
        "reflation": -0.25,
        "goldilocks": -0.20,
    },
    # VIX compresso → calma, risk-on pro-crescita
    "vix_compressed": {
        "reflation": 0.20,
        "goldilocks": 0.15,
        "deflation": -0.20,
        "stagflation": -0.10,
    },
    # NFCI in tightening → condizioni finanziarie strette, frena crescita
    "nfci_tightening": {
        "deflation": 0.25,
        "stagflation": 0.10,
        "reflation": -0.20,
        "goldilocks": -0.15,
    },
    # NFCI in easing → condizioni loose, supportive per risk
    "nfci_easing": {
        "reflation": 0.20,
        "goldilocks": 0.15,
        "deflation": -0.20,
        "stagflation": -0.05,
    },
}

# Mapping: news sentiment → pressione regime
NEWS_REGIME_PRESSURE = {
    "very_bearish": {  # sentiment < -0.3
        "deflation": 0.20,
        "stagflation": 0.10,
        "reflation": -0.15,
        "goldilocks": -0.10,
    },
    "bearish": {  # -0.3 to -0.1
        "deflation": 0.15,
        "stagflation": 0.05,
        "reflation": -0.10,
    },
    "bullish": {  # 0.1 to 0.3
        "reflation": 0.15,
        "goldilocks": 0.10,
        "deflation": -0.10,
    },
    "very_bullish": {  # > 0.3
        "reflation": 0.20,
        "goldilocks": 0.15,
        "deflation": -0.15,
        "stagflation": -0.10,
    },
}

# Dedollarizzazione → pressione regime
DEDOLLAR_REGIME_PRESSURE = {
    "high": {  # score > 0.6
        "stagflation": 0.20,
        "reflation": 0.05,
        "goldilocks": -0.15,
    },
    "moderate": {  # 0.4 - 0.6
        "stagflation": 0.08,
        "reflation": 0.03,
    },
    "low": {  # < 0.4
        "goldilocks": 0.08,
        "reflation": 0.05,
        "stagflation": -0.05,
    },
}


def _detect_indicator_trends(indicators: dict[str, float]) -> list[str]:
    """Identifica i trend attivi dagli indicatori."""
    trends = []

    cpi = indicators.get("cpi_yoy", 2.0)
    gdp = indicators.get("gdp_roc", 0.0)
    unrate_roc = indicators.get("unrate_roc", 0.0)
    yield_spread = indicators.get("yield_curve_10y2y", 1.0)
    claims_roc = indicators.get("initial_claims_roc", 0.0)

    # CPI trend
    if cpi > 3.5:
        trends.append("cpi_rising")
    elif cpi < 2.0:
        trends.append("cpi_falling")

    # GDP trend
    if gdp < 1.0:
        trends.append("gdp_decelerating")
    elif gdp > 2.5:
        trends.append("gdp_accelerating")

    # Unemployment direction
    if unrate_roc > 0.3:
        trends.append("unemployment_rising")
    elif unrate_roc < -0.3:
        trends.append("unemployment_falling")

    # Yield curve
    if yield_spread < 0.2:
        trends.append("yield_curve_inverting")
    elif yield_spread > 1.0:
        trends.append("yield_curve_steepening")

    # Claims
    if claims_roc > 5.0:
        trends.append("claims_rising")
    elif claims_roc < -5.0:
        trends.append("claims_falling")

    # Breakeven inflation (change 3m): threshold ±0.2 percentage points
    be_change = indicators.get("breakeven_10y_change_3m")
    if be_change is not None:
        if be_change > 0.2:
            trends.append("breakeven_rising")
        elif be_change < -0.2:
            trends.append("breakeven_falling")

    # VIX regime: spike quando ratio su MA 3m > 1.3, compressed < 0.85 e livello < 15
    vix_ratio = indicators.get("vix_ma_ratio")
    vix_level = indicators.get("vix")
    if vix_ratio is not None and vix_level is not None:
        if vix_ratio > 1.3 or vix_level > 25:
            trends.append("vix_spiking")
        elif vix_ratio < 0.85 and vix_level < 15:
            trends.append("vix_compressed")

    # NFCI: >0 tight, <0 loose. Usiamo il change come segnale di direzione.
    nfci_change = indicators.get("nfci_change_3m")
    if nfci_change is not None:
        if nfci_change > 0.15:
            trends.append("nfci_tightening")
        elif nfci_change < -0.15:
            trends.append("nfci_easing")

    return trends


def _get_news_category(avg_sentiment: float) -> str:
    """Classifica il sentiment medio delle news."""
    if avg_sentiment < -0.3:
        return "very_bearish"
    if avg_sentiment < -0.1:
        return "bearish"
    if avg_sentiment > 0.3:
        return "very_bullish"
    if avg_sentiment > 0.1:
        return "bullish"
    return "neutral"


def _get_dedollar_category(score: float) -> str:
    """Classifica il livello di dedollarizzazione."""
    if score > 0.6:
        return "high"
    if score > 0.4:
        return "moderate"
    return "low"


def calculate_trajectory(
    current_probabilities: dict[str, float],
    indicators: dict[str, float],
    news_sentiment: float = 0.0,
    dedollar_score: float = 0.0,
) -> dict[str, Any]:
    """Calcola la traiettoria del regime.

    Args:
        current_probabilities: Probabilità regime attuali
        indicators: Indicatori macro correnti
        news_sentiment: Sentiment medio notizie (-1 to +1)
        dedollar_score: Score dedollarizzazione (0-1)

    Returns:
        {
            "projected_regime": str,
            "projected_probabilities": {regime: float},
            "forces": [
                {"name": str, "description": str, "pushes_toward": str, "strength": float}
            ],
            "transition_risk": float (0-1),
            "summary": str,
        }
    """
    # Accumula pressioni su ogni regime
    pressure: dict[str, float] = {r: 0.0 for r in REGIMES}
    forces: list[dict] = []

    # 1. TREND INDICATORI
    active_trends = _detect_indicator_trends(indicators)
    for trend_name in active_trends:
        mapping = TREND_REGIME_PRESSURE.get(trend_name, {})
        top_regime = max(mapping, key=mapping.get) if mapping else None
        total_pressure = sum(abs(v) for v in mapping.values())

        for regime, weight in mapping.items():
            pressure[regime] += weight

        if top_regime and total_pressure > 0:
            human_name = trend_name.replace("_", " ").capitalize()
            forces.append({
                "name": trend_name,
                "type": "indicator",
                "description": human_name,
                "pushes_toward": top_regime,
                "strength": round(mapping[top_regime], 2),
            })

    # 2. NEWS SENTIMENT
    news_cat = _get_news_category(news_sentiment)
    if news_cat != "neutral":
        mapping = NEWS_REGIME_PRESSURE.get(news_cat, {})
        for regime, weight in mapping.items():
            pressure[regime] += weight

        top_regime = max(mapping, key=mapping.get) if mapping else None
        if top_regime:
            forces.append({
                "name": f"news_{news_cat}",
                "type": "news",
                "description": f"News sentiment: {news_cat.replace('_', ' ')}",
                "pushes_toward": top_regime,
                "strength": round(mapping[top_regime], 2),
            })

    # 3. DEDOLLARIZZAZIONE
    dedollar_cat = _get_dedollar_category(dedollar_score)
    mapping = DEDOLLAR_REGIME_PRESSURE.get(dedollar_cat, {})
    for regime, weight in mapping.items():
        pressure[regime] += weight

    if mapping:
        top_regime = max(mapping, key=mapping.get)
        forces.append({
            "name": f"dedollar_{dedollar_cat}",
            "type": "dedollarization",
            "description": f"Dedollarization: {dedollar_cat}",
            "pushes_toward": top_regime,
            "strength": round(mapping[top_regime], 2),
        })

    # Ordina forze per strength
    forces.sort(key=lambda f: abs(f["strength"]), reverse=True)

    # 4. PROIETTA PROBABILITÀ
    projected = {}
    for regime in REGIMES:
        base = current_probabilities.get(regime, 1.0 / len(REGIMES))
        adjusted = base * (1.0 + pressure[regime])
        projected[regime] = max(0.01, adjusted)

    # Rinormalizza
    total = sum(projected.values())
    projected = {r: round(p / total, 4) for r, p in projected.items()}

    # Regime proiettato
    projected_regime = max(projected, key=projected.get)

    # 5. RISCHIO TRANSIZIONE
    current_regime = max(current_probabilities, key=current_probabilities.get)

    sorted_proj = sorted(projected.values(), reverse=True)
    gap = sorted_proj[0] - sorted_proj[1] if len(sorted_proj) > 1 else 0
    transition_risk = 1.0 - min(1.0, gap * 5.0)
    if projected_regime != current_regime:
        transition_risk = max(transition_risk, 0.6)
    transition_risk = round(max(0.0, min(1.0, transition_risk)), 3)

    # 6. DRIFT
    drift = []
    for regime in REGIMES:
        cur = current_probabilities.get(regime, 0)
        proj = projected.get(regime, 0)
        delta = proj - cur
        drift.append({
            "regime": regime,
            "current": round(cur, 4),
            "projected": round(proj, 4),
            "delta": round(delta, 4),
        })
    drift.sort(key=lambda d: abs(d["delta"]), reverse=True)

    # 7. SUMMARY
    rising = [d for d in drift if d["delta"] > 0.01]
    falling = [d for d in drift if d["delta"] < -0.01]
    parts = []
    if rising:
        top_rise = rising[0]
        parts.append(f"{top_rise['regime']} probability rising (+{top_rise['delta']*100:.1f}pp)")
    if falling:
        top_fall = falling[0]
        parts.append(f"{top_fall['regime']} falling ({top_fall['delta']*100:.1f}pp)")
    if projected_regime != current_regime:
        parts.append(f"potential transition from {current_regime} to {projected_regime}")

    summary = ". ".join(parts) if parts else "Regime stable, no significant forces detected"

    return {
        "current_regime": current_regime,
        "projected_regime": projected_regime,
        "projected_probabilities": projected,
        "forces": forces,
        "drift": drift,
        "transition_risk": transition_risk,
        "summary": summary,
    }
