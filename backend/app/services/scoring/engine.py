"""Engine di scoring finale per asset class (rev. 2).

Combina:
- Probabilita dei regimi (4 quadranti: reflation, stagflation, deflation, goldilocks)
- Performance storiche REALI (inflation-adjusted) degli asset per regime
- Secular trend bonus, news signal, momentum penalty (Fase 2)

Revisione (2026-04-23):
- `avg_return` ridefinito come RENDIMENTO REALE annualizzato (nominal − inflation del regime).
  Questo fixa il bug per cui cash_money_market scorava >70 in tutti i regimi: in realta
  il cash in stagflation perde potere d'acquisto (nominal ~5-8%, CPI 8-10%, real -3%).
- Nuova formula con 3 componenti: hit_rate (25%), real_return normalizzato (50%),
  Sharpe normalizzato (25%). Il rendimento reale ora e' la voce dominante — riflette
  che lo scopo dello scoring e' preservare/aumentare potere d'acquisto, non solo Sharpe.
- Tutta la matrice ASSET_REGIME_DATA rivista con riferimenti storici espliciti
  (70s stagflation, 2008 GFC, 2020 COVID, 2022 inflation, anni 90 goldilocks).
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

# Performance storiche REALI (inflation-adjusted) per asset class x regime.
#
# Campi:
#   hit_rate:    probabilita di outperformance reale (>= inflazione di regime) storicamente
#   avg_return:  rendimento REALE annualizzato medio nel regime (nominal − CPI del regime)
#   vol:         volatilita annualizzata (invariata: nominal e real quasi coincidono)
#   sharpe:      Sharpe ratio reale = avg_return / vol (in eccesso sull'inflazione)
#
# Riferimenti usati per calibrare:
#   Reflation: 1983-89, 2003-06, 2009-11, 2020-21 (post-COVID), 2024
#   Stagflation: 1973-75 (OPEC), 1979-82 (Volcker), 2022
#   Deflation: 1990-91 S&L, 2001 dotcom, 2008-09 GFC, 2020 Q1 COVID-shock
#   Goldilocks: 1995-99 Greenspan, 2013-19 QE, 2024 (disinflation)
ASSET_REGIME_DATA: dict[str, dict[str, dict[str, float]]] = {
    "us_equities_growth": {
        # Ref: 2003-06 NASDAQ +13%/a real, 2009-11 +18% real, 2020-21 +25% real
        "reflation":    {"hit_rate": 0.78, "avg_return": 0.15, "vol": 0.17, "sharpe": 0.88},
        # Ref: 1973-74 SPX -43% + 11% CPI; 2022 QQQ -33% + 8% CPI
        "stagflation":  {"hit_rate": 0.22, "avg_return": -0.20, "vol": 0.26, "sharpe": -0.77},
        # Ref: 2001 -13%, 2008 -37%, 2020 Q1 -34% peak-to-trough
        "deflation":    {"hit_rate": 0.30, "avg_return": -0.18, "vol": 0.28, "sharpe": -0.64},
        # Ref: 1995-99 +26%/a real, 2013-19 +13% real
        "goldilocks":   {"hit_rate": 0.82, "avg_return": 0.18, "vol": 0.14, "sharpe": 1.29},
    },
    "us_equities_value": {
        # Value in reflation spesso = growth grazie a cyclicals/financials
        "reflation":    {"hit_rate": 0.74, "avg_return": 0.13, "vol": 0.17, "sharpe": 0.76},
        # Value resiste meglio di growth in stagflation (energy/staples pesano)
        "stagflation":  {"hit_rate": 0.35, "avg_return": -0.10, "vol": 0.22, "sharpe": -0.45},
        # Banks/real estate colpiti duro: 2008 financials -60%
        "deflation":    {"hit_rate": 0.30, "avg_return": -0.15, "vol": 0.25, "sharpe": -0.60},
        "goldilocks":   {"hit_rate": 0.70, "avg_return": 0.12, "vol": 0.13, "sharpe": 0.92},
    },
    "international_dm_equities": {
        "reflation":    {"hit_rate": 0.68, "avg_return": 0.09, "vol": 0.17, "sharpe": 0.53},
        "stagflation":  {"hit_rate": 0.28, "avg_return": -0.14, "vol": 0.22, "sharpe": -0.64},
        # DM colpiti quanto US nel 2008, a volte peggio
        "deflation":    {"hit_rate": 0.28, "avg_return": -0.17, "vol": 0.25, "sharpe": -0.68},
        "goldilocks":   {"hit_rate": 0.65, "avg_return": 0.10, "vol": 0.14, "sharpe": 0.71},
    },
    "em_equities": {
        # Ref: 2003-07 EM +30%/a nominal, +25% real
        "reflation":    {"hit_rate": 0.72, "avg_return": 0.18, "vol": 0.24, "sharpe": 0.75},
        # EM piu fragili: currency + capital flight
        "stagflation":  {"hit_rate": 0.22, "avg_return": -0.25, "vol": 0.32, "sharpe": -0.78},
        # 2008 EM -55% peak-to-trough
        "deflation":    {"hit_rate": 0.20, "avg_return": -0.28, "vol": 0.35, "sharpe": -0.80},
        "goldilocks":   {"hit_rate": 0.65, "avg_return": 0.13, "vol": 0.22, "sharpe": 0.59},
    },
    "us_bonds_short": {
        # Rates che salgono, short duration soffre meno ma perde in real
        "reflation":    {"hit_rate": 0.40, "avg_return": -0.01, "vol": 0.02, "sharpe": -0.50},
        # 2022 SHY ~0% nominal + 8% CPI = -8% real. 70s ancora peggio.
        "stagflation":  {"hit_rate": 0.30, "avg_return": -0.05, "vol": 0.03, "sharpe": -1.67},
        # Flight to quality + rates che scendono
        "deflation":    {"hit_rate": 0.70, "avg_return": 0.04, "vol": 0.02, "sharpe": 2.00},
        "goldilocks":   {"hit_rate": 0.50, "avg_return": 0.02, "vol": 0.02, "sharpe": 1.00},
    },
    "us_bonds_long": {
        # 2022 TLT -31% + 8% CPI = -39% real
        "reflation":    {"hit_rate": 0.30, "avg_return": -0.05, "vol": 0.14, "sharpe": -0.36},
        # Il peggior asset assoluto in stagflation (duration + inflation double-whammy)
        "stagflation":  {"hit_rate": 0.18, "avg_return": -0.20, "vol": 0.18, "sharpe": -1.11},
        # 2008 TLT +34% nominal ~ +35% real; 2020 Q1 TLT +20%
        "deflation":    {"hit_rate": 0.75, "avg_return": 0.15, "vol": 0.14, "sharpe": 1.07},
        "goldilocks":   {"hit_rate": 0.58, "avg_return": 0.04, "vol": 0.10, "sharpe": 0.40},
    },
    "tips_inflation_bonds": {
        # L'indicizzazione paga un po, ma rate-sensitive
        "reflation":    {"hit_rate": 0.55, "avg_return": 0.04, "vol": 0.06, "sharpe": 0.67},
        # TIPS long-duration nel 2022: -12% nominal nonostante indexation. Mix.
        "stagflation":  {"hit_rate": 0.55, "avg_return": -0.02, "vol": 0.08, "sharpe": -0.25},
        # Il loro punto debole: indexation si inverte, capital loss
        "deflation":    {"hit_rate": 0.30, "avg_return": -0.03, "vol": 0.08, "sharpe": -0.38},
        "goldilocks":   {"hit_rate": 0.45, "avg_return": 0.02, "vol": 0.06, "sharpe": 0.33},
    },
    "gold": {
        "reflation":    {"hit_rate": 0.40, "avg_return": 0.01, "vol": 0.16, "sharpe": 0.06},
        # 70s gold +25%/a real. 2022 mixed (Fed aggressivo). Media 70s-today.
        "stagflation":  {"hit_rate": 0.75, "avg_return": 0.15, "vol": 0.20, "sharpe": 0.75},
        # 2008 gold +5%; early-30s +80%; 2015 -10%. Flight-to-quality incostante.
        "deflation":    {"hit_rate": 0.50, "avg_return": 0.03, "vol": 0.18, "sharpe": 0.17},
        "goldilocks":   {"hit_rate": 0.35, "avg_return": -0.01, "vol": 0.14, "sharpe": -0.07},
    },
    "silver": {
        "reflation":    {"hit_rate": 0.55, "avg_return": 0.07, "vol": 0.28, "sharpe": 0.25},
        # 1979 Hunt corner silver $50/oz. 2022 piatto.
        "stagflation":  {"hit_rate": 0.60, "avg_return": 0.10, "vol": 0.32, "sharpe": 0.31},
        # Piu industriale di gold = piu ciclica = piu sofferente in deflation
        "deflation":    {"hit_rate": 0.30, "avg_return": -0.15, "vol": 0.35, "sharpe": -0.43},
        "goldilocks":   {"hit_rate": 0.42, "avg_return": 0.01, "vol": 0.26, "sharpe": 0.04},
    },
    "broad_commodities": {
        "reflation":    {"hit_rate": 0.65, "avg_return": 0.10, "vol": 0.19, "sharpe": 0.53},
        # 70s commodities +15-20%/a real. 2022 DBC +19% + 8% CPI = +11% real.
        "stagflation":  {"hit_rate": 0.75, "avg_return": 0.15, "vol": 0.22, "sharpe": 0.68},
        # 2008 DBC -30%, 2020 Q1 DBC -25%
        "deflation":    {"hit_rate": 0.22, "avg_return": -0.22, "vol": 0.25, "sharpe": -0.88},
        "goldilocks":   {"hit_rate": 0.45, "avg_return": 0.00, "vol": 0.16, "sharpe": 0.00},
    },
    "energy": {
        "reflation":    {"hit_rate": 0.62, "avg_return": 0.12, "vol": 0.28, "sharpe": 0.43},
        # 1973 oil +400%; 2022 XLE +64% real +56%. Star di ogni stagflation.
        "stagflation":  {"hit_rate": 0.75, "avg_return": 0.22, "vol": 0.32, "sharpe": 0.69},
        # 2008 WTI -77% da 147 a 34; 2020 WTI negative
        "deflation":    {"hit_rate": 0.20, "avg_return": -0.32, "vol": 0.40, "sharpe": -0.80},
        "goldilocks":   {"hit_rate": 0.45, "avg_return": 0.02, "vol": 0.25, "sharpe": 0.08},
    },
    "real_estate_reits": {
        "reflation":    {"hit_rate": 0.65, "avg_return": 0.10, "vol": 0.18, "sharpe": 0.56},
        # 2022 VNQ -26% nominal + 8% CPI = -34% real
        "stagflation":  {"hit_rate": 0.25, "avg_return": -0.15, "vol": 0.24, "sharpe": -0.62},
        # 2008 VNQ -37%, financing crisis
        "deflation":    {"hit_rate": 0.28, "avg_return": -0.22, "vol": 0.28, "sharpe": -0.79},
        # Goldilocks + QE = yield-seeking nei REIT
        "goldilocks":   {"hit_rate": 0.70, "avg_return": 0.10, "vol": 0.15, "sharpe": 0.67},
    },
    "cash_money_market": {
        # T-bills 3-5% nominal, CPI ~3%, real ~+1%
        "reflation":    {"hit_rate": 0.35, "avg_return": 0.01, "vol": 0.01, "sharpe": 1.00},
        # 70s T-bills +7% nominal / CPI +9% = real -2%. 2022 T-bills +2% / CPI +8% = real -6%.
        "stagflation":  {"hit_rate": 0.30, "avg_return": -0.03, "vol": 0.01, "sharpe": -3.00},
        # CPI ~0%, T-bills ~2-4%, real +2-4%. Il re quando tutto brucia.
        "deflation":    {"hit_rate": 0.60, "avg_return": 0.02, "vol": 0.01, "sharpe": 2.00},
        # Real ~0% (nominal pareggia CPI)
        "goldilocks":   {"hit_rate": 0.40, "avg_return": 0.00, "vol": 0.01, "sharpe": 0.00},
    },
    "bitcoin": {
        # 2020-21 BTC +300% nominal +290% real; 2024 +120% real
        "reflation":    {"hit_rate": 0.65, "avg_return": 0.45, "vol": 0.75, "sharpe": 0.60},
        # 2022 BTC -65% + 8% CPI = -73% real
        "stagflation":  {"hit_rate": 0.30, "avg_return": -0.40, "vol": 0.85, "sharpe": -0.47},
        # 2020 Mar flash crash -50%, 2022 deleveraging multiplo
        "deflation":    {"hit_rate": 0.25, "avg_return": -0.35, "vol": 0.85, "sharpe": -0.41},
        "goldilocks":   {"hit_rate": 0.60, "avg_return": 0.30, "vol": 0.70, "sharpe": 0.43},
    },
    "crypto_broad": {
        # Amplificato rispetto a BTC: beta 1.3-1.5x su up e down
        "reflation":    {"hit_rate": 0.60, "avg_return": 0.40, "vol": 0.85, "sharpe": 0.47},
        "stagflation":  {"hit_rate": 0.25, "avg_return": -0.50, "vol": 0.90, "sharpe": -0.56},
        "deflation":    {"hit_rate": 0.22, "avg_return": -0.45, "vol": 0.95, "sharpe": -0.47},
        "goldilocks":   {"hit_rate": 0.55, "avg_return": 0.25, "vol": 0.80, "sharpe": 0.31},
    },
}


def _asset_regime_score(asset: str, regime: str) -> float:
    """Calcola lo score 0-100 di un asset in un regime specifico.

    Formula (rev. 2):
        score = hit_rate * 0.25
              + real_return_norm * 0.50
              + sharpe_norm * 0.25

    Dove:
        real_return_norm = clamp((real_return + 0.30) / 0.60, 0, 1)   # range [-30%, +30%]
        sharpe_norm      = clamp((sharpe + 1.0) / 3.0, 0, 1)           # range [-1, +2]

    Il rendimento reale e' la voce dominante (50%): questo perche' lo scopo del
    sistema e' preservare/aumentare potere d'acquisto, non solo ottimizzare Sharpe.
    Nell'implementazione precedente, asset ad alto Sharpe ma rendimento reale negativo
    (es. cash in stagflation) scoravano artificialmente alti.
    """
    data = ASSET_REGIME_DATA[asset][regime]
    hit_rate = data["hit_rate"]
    real_return = data["avg_return"]
    sharpe = data["sharpe"]

    real_return_norm = max(0.0, min(1.0, (real_return + 0.30) / 0.60))
    sharpe_norm = max(0.0, min(1.0, (sharpe + 1.0) / 3.0))

    score = (hit_rate * 0.25 + real_return_norm * 0.50 + sharpe_norm * 0.25) * 100

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
        secular_bonus: Dict {asset: bonus 0-10} (Fase 2)
        news_signals: Dict {asset: signal -5 to +5} (Fase 2)
        momentum_penalty: Dict {asset: penalty 0-10}

    Returns:
        Dict {asset_class: score 0-100}
    """
    total_prob = sum(probabilities.values())
    if total_prob > 0 and abs(total_prob - 1.0) > 0.001:
        probabilities = {r: p / total_prob for r, p in probabilities.items()}

    scores: dict[str, float] = {}

    for asset in ASSET_CLASSES:
        base_score = sum(
            probabilities.get(regime, 0.0) * _asset_regime_score(asset, regime)
            for regime in probabilities
        )

        bonus = (secular_bonus or {}).get(asset, 0.0)
        news = (news_signals or {}).get(asset, 0.0)
        penalty = (momentum_penalty or {}).get(asset, 0.0)

        final = base_score + bonus + news - penalty
        scores[asset] = max(0.0, min(100.0, round(final, 1)))

    return scores
