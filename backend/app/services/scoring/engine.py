"""Engine di scoring finale per asset class.

Combina:
- Probabilita dei regimi (4 quadranti: reflation, stagflation, deflation, goldilocks)
- Performance storiche degli asset per regime
- (Fase 2) Secular trend bonus, news signal, momentum penalty

Formula:
    final_score(asset) = sum(
        regime_probability[r] * asset_regime_score(asset, r)
        for r in all_regimes
    )
"""

ASSET_CLASSES = [
    "us_equities_growth",
    "us_equities_value",
    "international_dm_equities",
    "em_equities",
    "us_bonds_short",
    "us_bonds_long",
    "tips_inflation_bonds",
    "gold",
    "silver",
    "broad_commodities",
    "energy",
    "real_estate_reits",
    "cash_money_market",
    "bitcoin",
    "crypto_broad",
]

# Performance storiche per asset class x regime (4 quadranti)
# Basate su: Bridgewater All Weather research, AQR papers, ricerca accademica
#
# Reflation = media pesata growth (60%) + recovery (40%)
# Deflation = media pesata recession (60%) + slowdown (40%)
# Stagflation e Goldilocks invariati
#
# hit_rate: % periodi con outperformance relativa
# avg_return_12m: rendimento medio annualizzato nel regime
# volatility: deviazione standard nel regime
# sharpe: sharpe ratio nel regime
ASSET_REGIME_DATA: dict[str, dict[str, dict[str, float]]] = {
    "us_equities_growth": {
        "reflation":    {"hit_rate": 0.80, "avg_return": 0.18, "vol": 0.16, "sharpe": 1.08},
        "stagflation":  {"hit_rate": 0.30, "avg_return": -0.05, "vol": 0.22, "sharpe": -0.23},
        "deflation":    {"hit_rate": 0.33, "avg_return": -0.07, "vol": 0.24, "sharpe": -0.24},
        "goldilocks":   {"hit_rate": 0.80, "avg_return": 0.18, "vol": 0.12, "sharpe": 1.50},
    },
    "us_equities_value": {
        "reflation":    {"hit_rate": 0.76, "avg_return": 0.17, "vol": 0.17, "sharpe": 1.01},
        "stagflation":  {"hit_rate": 0.42, "avg_return": 0.02, "vol": 0.20, "sharpe": 0.10},
        "deflation":    {"hit_rate": 0.37, "avg_return": -0.06, "vol": 0.21, "sharpe": -0.21},
        "goldilocks":   {"hit_rate": 0.72, "avg_return": 0.14, "vol": 0.11, "sharpe": 1.27},
    },
    "international_dm_equities": {
        "reflation":    {"hit_rate": 0.71, "avg_return": 0.13, "vol": 0.17, "sharpe": 0.78},
        "stagflation":  {"hit_rate": 0.35, "avg_return": -0.03, "vol": 0.20, "sharpe": -0.15},
        "deflation":    {"hit_rate": 0.33, "avg_return": -0.08, "vol": 0.22, "sharpe": -0.35},
        "goldilocks":   {"hit_rate": 0.70, "avg_return": 0.12, "vol": 0.13, "sharpe": 0.92},
    },
    "em_equities": {
        "reflation":    {"hit_rate": 0.70, "avg_return": 0.20, "vol": 0.23, "sharpe": 0.83},
        "stagflation":  {"hit_rate": 0.30, "avg_return": -0.08, "vol": 0.28, "sharpe": -0.29},
        "deflation":    {"hit_rate": 0.26, "avg_return": -0.13, "vol": 0.29, "sharpe": -0.41},
        "goldilocks":   {"hit_rate": 0.68, "avg_return": 0.16, "vol": 0.20, "sharpe": 0.80},
    },
    "us_bonds_short": {
        "reflation":    {"hit_rate": 0.43, "avg_return": 0.02, "vol": 0.02, "sharpe": 0.80},
        "stagflation":  {"hit_rate": 0.50, "avg_return": 0.02, "vol": 0.03, "sharpe": 0.67},
        "deflation":    {"hit_rate": 0.64, "avg_return": 0.04, "vol": 0.02, "sharpe": 1.43},
        "goldilocks":   {"hit_rate": 0.48, "avg_return": 0.03, "vol": 0.02, "sharpe": 1.50},
    },
    "us_bonds_long": {
        "reflation":    {"hit_rate": 0.33, "avg_return": 0.00, "vol": 0.10, "sharpe": -0.01},
        "stagflation":  {"hit_rate": 0.25, "avg_return": -0.08, "vol": 0.14, "sharpe": -0.57},
        "deflation":    {"hit_rate": 0.73, "avg_return": 0.10, "vol": 0.11, "sharpe": 0.84},
        "goldilocks":   {"hit_rate": 0.55, "avg_return": 0.05, "vol": 0.08, "sharpe": 0.63},
    },
    "tips_inflation_bonds": {
        "reflation":    {"hit_rate": 0.53, "avg_return": 0.04, "vol": 0.06, "sharpe": 0.63},
        "stagflation":  {"hit_rate": 0.72, "avg_return": 0.06, "vol": 0.08, "sharpe": 0.75},
        "deflation":    {"hit_rate": 0.52, "avg_return": 0.04, "vol": 0.08, "sharpe": 0.48},
        "goldilocks":   {"hit_rate": 0.48, "avg_return": 0.03, "vol": 0.05, "sharpe": 0.60},
    },
    "gold": {
        "reflation":    {"hit_rate": 0.41, "avg_return": 0.02, "vol": 0.15, "sharpe": 0.16},
        "stagflation":  {"hit_rate": 0.80, "avg_return": 0.18, "vol": 0.18, "sharpe": 1.00},
        "deflation":    {"hit_rate": 0.59, "avg_return": 0.08, "vol": 0.17, "sharpe": 0.49},
        "goldilocks":   {"hit_rate": 0.38, "avg_return": 0.01, "vol": 0.14, "sharpe": 0.07},
    },
    "silver": {
        "reflation":    {"hit_rate": 0.54, "avg_return": 0.09, "vol": 0.29, "sharpe": 0.31},
        "stagflation":  {"hit_rate": 0.65, "avg_return": 0.12, "vol": 0.30, "sharpe": 0.40},
        "deflation":    {"hit_rate": 0.43, "avg_return": 0.02, "vol": 0.31, "sharpe": 0.08},
        "goldilocks":   {"hit_rate": 0.45, "avg_return": 0.04, "vol": 0.25, "sharpe": 0.16},
    },
    "broad_commodities": {
        "reflation":    {"hit_rate": 0.64, "avg_return": 0.11, "vol": 0.19, "sharpe": 0.56},
        "stagflation":  {"hit_rate": 0.70, "avg_return": 0.14, "vol": 0.22, "sharpe": 0.64},
        "deflation":    {"hit_rate": 0.33, "avg_return": -0.08, "vol": 0.22, "sharpe": -0.34},
        "goldilocks":   {"hit_rate": 0.48, "avg_return": 0.03, "vol": 0.16, "sharpe": 0.19},
    },
    "energy": {
        "reflation":    {"hit_rate": 0.65, "avg_return": 0.15, "vol": 0.29, "sharpe": 0.51},
        "stagflation":  {"hit_rate": 0.75, "avg_return": 0.20, "vol": 0.30, "sharpe": 0.67},
        "deflation":    {"hit_rate": 0.29, "avg_return": -0.14, "vol": 0.33, "sharpe": -0.41},
        "goldilocks":   {"hit_rate": 0.45, "avg_return": 0.04, "vol": 0.25, "sharpe": 0.16},
    },
    "real_estate_reits": {
        "reflation":    {"hit_rate": 0.69, "avg_return": 0.13, "vol": 0.18, "sharpe": 0.74},
        "stagflation":  {"hit_rate": 0.35, "avg_return": -0.04, "vol": 0.22, "sharpe": -0.18},
        "deflation":    {"hit_rate": 0.34, "avg_return": -0.06, "vol": 0.22, "sharpe": -0.24},
        "goldilocks":   {"hit_rate": 0.68, "avg_return": 0.12, "vol": 0.14, "sharpe": 0.86},
    },
    "cash_money_market": {
        "reflation":    {"hit_rate": 0.28, "avg_return": 0.02, "vol": 0.01, "sharpe": 1.60},
        "stagflation":  {"hit_rate": 0.55, "avg_return": 0.03, "vol": 0.01, "sharpe": 3.00},
        "deflation":    {"hit_rate": 0.58, "avg_return": 0.03, "vol": 0.01, "sharpe": 2.60},
        "goldilocks":   {"hit_rate": 0.35, "avg_return": 0.02, "vol": 0.01, "sharpe": 2.00},
    },
    "bitcoin": {
        "reflation":    {"hit_rate": 0.68, "avg_return": 0.48, "vol": 0.72, "sharpe": 0.66},
        "stagflation":  {"hit_rate": 0.40, "avg_return": 0.05, "vol": 0.70, "sharpe": 0.07},
        "deflation":    {"hit_rate": 0.32, "avg_return": -0.19, "vol": 0.78, "sharpe": -0.24},
        "goldilocks":   {"hit_rate": 0.68, "avg_return": 0.35, "vol": 0.65, "sharpe": 0.54},
    },
    "crypto_broad": {
        "reflation":    {"hit_rate": 0.64, "avg_return": 0.43, "vol": 0.80, "sharpe": 0.54},
        "stagflation":  {"hit_rate": 0.35, "avg_return": -0.05, "vol": 0.80, "sharpe": -0.06},
        "deflation":    {"hit_rate": 0.27, "avg_return": -0.27, "vol": 0.88, "sharpe": -0.31},
        "goldilocks":   {"hit_rate": 0.62, "avg_return": 0.30, "vol": 0.70, "sharpe": 0.43},
    },
}


def _asset_regime_score(asset: str, regime: str) -> float:
    """Calcola lo score 0-100 di un asset in un regime specifico.

    Combina hit_rate (peso 40%) e sharpe normalizzato (peso 60%).
    """
    data = ASSET_REGIME_DATA[asset][regime]
    hit_rate = data["hit_rate"]
    sharpe = data["sharpe"]

    # Normalizza sharpe da [-1, 2] a [0, 1]
    sharpe_normalized = max(0.0, min(1.0, (sharpe + 1.0) / 3.0))

    # Composizione: hit_rate pesa 40%, sharpe pesa 60%
    score = (hit_rate * 0.40 + sharpe_normalized * 0.60) * 100

    return max(0.0, min(100.0, score))


def calculate_final_scores(
    probabilities: dict[str, float],
    secular_bonus: dict[str, float] | None = None,
    news_signals: dict[str, float] | None = None,
    momentum_penalty: dict[str, float] | None = None,
) -> dict[str, float]:
    """Calcola lo score finale per ogni asset class.

    Formula:
        final_score(asset) = sum(
            regime_probability[r] * asset_regime_score(asset, r)
            for r in all_regimes
        ) + secular_trend_bonus - momentum_penalty + news_signal

    Args:
        probabilities: Dict {regime: probabilita} (deve sommare a ~1.0)
        secular_bonus: (Fase 2) Dict {asset: bonus 0-10}
        news_signals: (Fase 2) Dict {asset: signal -5 to +5}
        momentum_penalty: Dict {asset: penalty 0-10}

    Returns:
        Dict {asset_class: score 0-100}
    """
    # Normalizza probabilita se non sommano a 1
    total_prob = sum(probabilities.values())
    if total_prob > 0 and abs(total_prob - 1.0) > 0.001:
        probabilities = {r: p / total_prob for r, p in probabilities.items()}

    scores: dict[str, float] = {}

    for asset in ASSET_CLASSES:
        # Score base: media pesata per probabilita regime
        base_score = sum(
            probabilities.get(regime, 0.0) * _asset_regime_score(asset, regime)
            for regime in probabilities
        )

        # Aggiustamenti (Fase 2 — default 0)
        bonus = (secular_bonus or {}).get(asset, 0.0)
        news = (news_signals or {}).get(asset, 0.0)
        penalty = (momentum_penalty or {}).get(asset, 0.0)

        final = base_score + bonus + news - penalty
        scores[asset] = max(0.0, min(100.0, round(final, 1)))

    return scores
