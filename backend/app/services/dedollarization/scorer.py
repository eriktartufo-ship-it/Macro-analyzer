"""Dedollarization scorer — misura l'intensita della dedollarizzazione globale.

Due livelli di analisi:
  1. CICLICO (12 mesi): segnali a breve termine
  2. STRUTTURALE (3-10 anni): trend secolare + accelerazione

Componenti ciclici (12m):
  1. USD weakness:       ROC 12m del Dollar Index
  2. Gold strength:      ROC 12m dell'oro
  3. Gold/Oil ratio:     Rapporto oro/petrolio corrente
  4. Debt burden:        Debito/PIL USA
  5. Real rate signal:   Fed Funds - CPI YoY
  6. Monetary debasement: ROC 12m M2

Componenti strutturali (pluriennali):
  1. USD secular decline:  ROC 5Y del Dollar Index
  2. Gold secular rise:    ROC 5Y dell'oro
  3. Debt trajectory:      Variazione Debt/GDP su 5Y
  4. M2 cumulative:        Crescita cumulata M2 su 5Y

Accelerazione:
  - Confronta ROC 1Y attuale vs ROC 1Y di 2 anni fa
  - Se il ROC sta peggiorando → trend accelera

Output:
  score: 0.0 (nessuna dedollarizzazione) → 1.0 (estrema)
  structural_score: trend secolare 0-1
  acceleration: -1.0 (decelerazione) → +1.0 (accelerazione forte)
  components: sub-score ciclici
  structural: sub-score strutturali
"""

import math

from app.services.scoring.engine import ASSET_CLASSES


def _sigmoid(x: float, center: float, scale: float) -> float:
    """Sigmoid 0-1. scale > 0: valori sopra center → 1. scale < 0: inverted."""
    if scale == 0.0:
        scale = 0.001
    z = (x - center) / scale
    z = max(-10.0, min(10.0, z))
    return 1.0 / (1.0 + math.exp(-z))


# Pesi per ogni componente nella media pesata ciclica
COMPONENT_WEIGHTS = {
    "usd_weakness": 0.25,
    "gold_strength": 0.20,
    "gold_oil_ratio": 0.10,
    "debt_burden": 0.15,
    "real_rate_signal": 0.15,
    "monetary_debasement": 0.15,
}

# Pesi per componenti strutturali (5Y)
STRUCTURAL_WEIGHTS = {
    "usd_secular": 0.30,
    "gold_secular": 0.25,
    "debt_trajectory": 0.25,
    "m2_cumulative": 0.20,
}

# Pesi per componenti decennali (10Y)
DECADE_WEIGHTS = {
    "usd_decade": 0.30,
    "gold_decade": 0.30,
    "debt_decade": 0.20,
    "m2_decade": 0.20,
}

# Pesi per componenti 20 anni (ventennale)
TWENTY_YEAR_WEIGHTS = {
    "usd_20y": 0.30,
    "gold_20y": 0.30,
    "debt_20y": 0.20,
    "m2_20y": 0.20,
}

# Bonus massimo per asset class quando dedollarization_score = 1.0
ASSET_DEDOLLAR_SENSITIVITY: dict[str, float] = {
    "gold": 10.0,
    "silver": 7.0,
    "bitcoin": 6.0,
    "crypto_broad": 5.0,
    "em_equities": 5.0,
    "broad_commodities": 4.0,
    "international_dm_equities": 4.0,
    "energy": 3.0,
    "tips_inflation_bonds": 3.0,
    "real_estate_reits": 1.0,
    "us_equities_value": 0.0,
    "us_equities_growth": -1.0,
    "us_bonds_short": -2.0,
    "cash_money_market": -4.0,
    "us_bonds_long": -5.0,
}


def calculate_dedollarization(
    indicators: dict[str, float],
    player_history: dict[str, dict[str, float]] | None = None,
) -> dict:
    """Calcola lo score composito di dedollarizzazione (ciclico + strutturale).

    Args:
        indicators: Dict con chiavi:
            # Ciclici (12 mesi)
            dxy_roc_12m: ROC % 12 mesi del Dollar Index
            gold_roc_12m: ROC % 12 mesi dell'oro
            gold_oil_ratio: Prezzo oro / prezzo petrolio
            debt_gdp: Debito/PIL USA in %
            real_rate: Fed Funds Rate - CPI YoY
            m2_roc_12m: ROC % 12 mesi di M2
            # Strutturali (pluriennali)
            dxy_roc_5y: ROC % 5 anni Dollar Index (annualizzato)
            gold_roc_5y: ROC % 5 anni Gold (annualizzato)
            debt_gdp_5y_ago: Debito/PIL USA 5 anni fa
            m2_roc_5y: ROC % 5 anni M2 (annualizzato)
            # Accelerazione
            dxy_roc_12m_2y_ago: ROC 12m del DXY di 2 anni fa (per confronto)
            gold_roc_12m_2y_ago: ROC 12m Gold di 2 anni fa

    Returns:
        {
            "score": float 0-1 (composito ciclico),
            "structural_score": float 0-1 (trend secolare),
            "acceleration": float -1 to +1 (sta accelerando?),
            "combined_score": float 0-1 (media pesata di tutto),
            "components": {name: float 0-1} (ciclici),
            "structural": {name: float 0-1} (strutturali),
        }
    """
    # === COMPONENTI CICLICI (12m) ===
    dxy_roc = indicators.get("dxy_roc_12m", 0.0)
    gold_roc = indicators.get("gold_roc_12m", 0.0)
    gold_oil = indicators.get("gold_oil_ratio", 20.0)
    debt_gdp = indicators.get("debt_gdp", 100.0)
    real_rate = indicators.get("real_rate", 0.0)
    m2_roc = indicators.get("m2_roc_12m", 4.0)

    components: dict[str, float] = {}
    components["usd_weakness"] = _sigmoid(dxy_roc, center=0.0, scale=-5.0)
    components["gold_strength"] = _sigmoid(gold_roc, center=5.0, scale=8.0)
    components["gold_oil_ratio"] = _sigmoid(gold_oil, center=22.0, scale=10.0)
    components["debt_burden"] = _sigmoid(debt_gdp, center=100.0, scale=20.0)
    components["real_rate_signal"] = _sigmoid(real_rate, center=0.0, scale=-2.0)
    components["monetary_debasement"] = _sigmoid(m2_roc, center=5.0, scale=5.0)

    cyclical_score = sum(
        COMPONENT_WEIGHTS[name] * val for name, val in components.items()
    )
    cyclical_score = max(0.0, min(1.0, cyclical_score))

    # === COMPONENTI STRUTTURALI (5Y) ===
    dxy_roc_5y = indicators.get("dxy_roc_5y", 0.0)
    gold_roc_5y = indicators.get("gold_roc_5y", 0.0)
    debt_gdp_5y_ago = indicators.get("debt_gdp_5y_ago", debt_gdp)
    m2_roc_5y = indicators.get("m2_roc_5y", 4.0)

    structural: dict[str, float] = {}

    # USD declino secolare: ROC 5Y annualizzato negativo = trend strutturale
    # center=0, scale=-3: -3%/anno su 5 anni → 0.73
    structural["usd_secular"] = _sigmoid(dxy_roc_5y, center=0.0, scale=-3.0)

    # Gold salita secolare: ROC 5Y annualizzato positivo
    # center=5, scale=5: +10%/anno su 5 anni → 0.73
    structural["gold_secular"] = _sigmoid(gold_roc_5y, center=5.0, scale=5.0)

    # Traiettoria debito: delta Debt/GDP su 5 anni
    debt_delta = debt_gdp - debt_gdp_5y_ago
    # center=10, scale=10: +10pp in 5 anni è neutro, +20pp → 0.73
    structural["debt_trajectory"] = _sigmoid(debt_delta, center=10.0, scale=10.0)

    # M2 cumulata: espansione media su 5 anni
    # center=5, scale=4: +5%/anno è neutro, +9%/anno → 0.73
    structural["m2_cumulative"] = _sigmoid(m2_roc_5y, center=5.0, scale=4.0)

    structural_score = sum(
        STRUCTURAL_WEIGHTS[name] * val for name, val in structural.items()
    )
    structural_score = max(0.0, min(1.0, structural_score))

    # === COMPONENTI DECENNALI (10Y) ===
    dxy_roc_10y = indicators.get("dxy_roc_10y", 0.0)
    gold_roc_10y = indicators.get("gold_roc_10y", 0.0)
    debt_gdp_10y_ago = indicators.get("debt_gdp_10y_ago", debt_gdp)
    m2_roc_10y = indicators.get("m2_roc_10y", 4.0)

    decade: dict[str, float] = {}

    # USD su 10 anni: declino secolare lento ma persistente
    # center=0, scale=-2: -2%/anno su 10 anni → 0.73
    decade["usd_decade"] = _sigmoid(dxy_roc_10y, center=0.0, scale=-2.0)

    # Gold su 10 anni: crescita strutturale
    # center=4, scale=4: +4%/anno è neutro, +8%/anno → 0.73
    decade["gold_decade"] = _sigmoid(gold_roc_10y, center=4.0, scale=4.0)

    # Debt trajectory 10Y: quanto è cresciuto il debito in un decennio
    debt_delta_10y = debt_gdp - debt_gdp_10y_ago
    # center=15, scale=15: +15pp in 10 anni è neutro, +30pp → 0.73
    decade["debt_decade"] = _sigmoid(debt_delta_10y, center=15.0, scale=15.0)

    # M2 cumulata 10Y
    # center=5, scale=3: +5%/anno è neutro, +8%/anno → 0.73
    decade["m2_decade"] = _sigmoid(m2_roc_10y, center=5.0, scale=3.0)

    decade_score = sum(
        DECADE_WEIGHTS[name] * val for name, val in decade.items()
    )
    decade_score = max(0.0, min(1.0, decade_score))

    # === COMPONENTI VENTENNALI (20Y) ===
    dxy_roc_20y = indicators.get("dxy_roc_20y")
    gold_roc_20y = indicators.get("gold_roc_20y")
    debt_gdp_20y_ago = indicators.get("debt_gdp_20y_ago")
    m2_roc_20y = indicators.get("m2_roc_20y")

    twenty_year: dict[str, float] = {}
    twenty_year_score = 0.0
    twenty_year_coverage = 0.0

    if dxy_roc_20y is not None:
        twenty_year["usd_20y"] = _sigmoid(dxy_roc_20y, center=0.0, scale=-1.5)
        twenty_year_score += TWENTY_YEAR_WEIGHTS["usd_20y"] * twenty_year["usd_20y"]
        twenty_year_coverage += TWENTY_YEAR_WEIGHTS["usd_20y"]
    if gold_roc_20y is not None:
        twenty_year["gold_20y"] = _sigmoid(gold_roc_20y, center=5.0, scale=4.0)
        twenty_year_score += TWENTY_YEAR_WEIGHTS["gold_20y"] * twenty_year["gold_20y"]
        twenty_year_coverage += TWENTY_YEAR_WEIGHTS["gold_20y"]
    if debt_gdp_20y_ago is not None:
        debt_delta_20y = debt_gdp - debt_gdp_20y_ago
        twenty_year["debt_20y"] = _sigmoid(debt_delta_20y, center=25.0, scale=20.0)
        twenty_year_score += TWENTY_YEAR_WEIGHTS["debt_20y"] * twenty_year["debt_20y"]
        twenty_year_coverage += TWENTY_YEAR_WEIGHTS["debt_20y"]
    if m2_roc_20y is not None:
        twenty_year["m2_20y"] = _sigmoid(m2_roc_20y, center=5.0, scale=2.5)
        twenty_year_score += TWENTY_YEAR_WEIGHTS["m2_20y"] * twenty_year["m2_20y"]
        twenty_year_coverage += TWENTY_YEAR_WEIGHTS["m2_20y"]

    # Normalizza per coverage se alcuni dati 20Y mancano
    if twenty_year_coverage > 0:
        twenty_year_score = twenty_year_score / twenty_year_coverage
    twenty_year_score = max(0.0, min(1.0, twenty_year_score))

    # === ACCELERAZIONE ===
    # Confronta ROC 1Y corrente vs ROC 1Y di 2 anni fa
    dxy_roc_2y_ago = indicators.get("dxy_roc_12m_2y_ago", dxy_roc)
    gold_roc_2y_ago = indicators.get("gold_roc_12m_2y_ago", gold_roc)

    # DXY: se oggi scende di più → differenza negativa → accelera
    dxy_accel = dxy_roc_2y_ago - dxy_roc  # positivo = DXY peggiora più velocemente
    # Gold: se oggi sale di più → differenza positiva → accelera
    gold_accel = gold_roc - gold_roc_2y_ago  # positivo = Gold sale più velocemente

    # Media accelerazione, normalizzata con sigmoid a [-1, 1]
    raw_accel = (dxy_accel + gold_accel) / 2.0
    accel_sigmoid = _sigmoid(raw_accel, center=0.0, scale=5.0)
    acceleration = (accel_sigmoid - 0.5) * 2.0  # mappa [0,1] → [-1, +1]

    # === SCORE COMBINATO ===
    # Con 20Y: ciclico 20%, 5Y 25%, 10Y 25%, 20Y 20%, accelerazione 10%.
    # Senza dati 20Y: ridistribuisci pesi sulle altre componenti.
    accel_bonus = max(0.0, acceleration) * 0.5

    if twenty_year_coverage > 0:
        combined = (
            cyclical_score * 0.20
            + structural_score * 0.25
            + decade_score * 0.25
            + twenty_year_score * 0.20
            + accel_bonus * 0.10
        )
    else:
        combined = (
            cyclical_score * 0.25
            + structural_score * 0.30
            + decade_score * 0.30
            + accel_bonus * 0.15
        )
    combined = max(0.0, min(1.0, combined))

    # === BY PLAYER (dimensione geopolitica) ===
    by_player = _calculate_player_scores(indicators)
    geopolitical_score = _aggregate_player_score(by_player)

    # === PLAYER HISTORY (score per player per orizzonte) ===
    player_history_scores: dict[str, dict[str, float]] = {}
    player_acceleration: dict[str, float] = {}

    if player_history:
        for horizon, hist_signals in player_history.items():
            hist_player = _calculate_player_scores(hist_signals)
            player_history_scores[horizon] = {
                pid: round(pdata["score"], 4)
                for pid, pdata in hist_player.items()
                if pdata.get("coverage", 0) > 0
            }

        # Per-player acceleration: score attuale − score 1Y fa
        hist_1y = player_history_scores.get("1y", {})
        for pid, current in by_player.items():
            if pid in hist_1y and current.get("coverage", 0) > 0:
                player_acceleration[pid] = round(current["score"] - hist_1y[pid], 4)

    # === ACCELERAZIONE MIGLIORATA ===
    # Blend: 40% indicator-based (DXY/Gold), 60% player-based (se disponibile)
    if player_acceleration:
        weighted_pa = 0.0
        total_pw = 0.0
        for pid, delta in player_acceleration.items():
            w = PLAYER_WEIGHTS.get(pid, 0.0)
            weighted_pa += delta * w
            total_pw += w
        if total_pw > 0:
            player_accel_raw = weighted_pa / total_pw
            # Mappa a [-1, 1]: delta player tipico è [-0.3, +0.3]
            player_accel_scaled = max(-1.0, min(1.0, player_accel_raw * 4.0))
            acceleration = 0.4 * acceleration + 0.6 * player_accel_scaled

    # === SCORE COMBINATO (ricalcolo con accelerazione migliorata) ===
    accel_bonus = max(0.0, acceleration) * 0.5
    if twenty_year_coverage > 0:
        combined = (
            cyclical_score * 0.20
            + structural_score * 0.25
            + decade_score * 0.25
            + twenty_year_score * 0.20
            + accel_bonus * 0.10
        )
    else:
        combined = (
            cyclical_score * 0.25
            + structural_score * 0.30
            + decade_score * 0.30
            + accel_bonus * 0.15
        )
    combined = max(0.0, min(1.0, combined))

    return {
        "score": round(cyclical_score, 4),
        "structural_score": round(structural_score, 4),
        "decade_score": round(decade_score, 4),
        "twenty_year_score": round(twenty_year_score, 4) if twenty_year_coverage > 0 else None,
        "acceleration": round(acceleration, 4),
        "combined_score": round(combined, 4),
        "geopolitical_score": round(geopolitical_score, 4),
        "components": components,
        "structural": structural,
        "decade": decade,
        "twenty_year": twenty_year,
        "by_player": by_player,
        "player_history": player_history_scores,
        "player_acceleration": player_acceleration,
    }


# ============================================================================
# BY-PLAYER (dimensione geopolitica — macro player × indicatori)
# ============================================================================

# Metadata per ogni player: ogni segnale ha
#   key: nome chiave nel dict indicators (input)
#   label: nome visualizzato
#   scorer: funzione (value) -> score 0-1
#   interpret: funzione (value) -> str breve interpretazione
#   red_flag: funzione (value) -> bool
#   weight: peso nel player score

def _score_gold_sp500(v: float) -> float:
    # alto = fuga verso scarsità fisica. Pivot ~0.5
    return _sigmoid(v, center=0.5, scale=0.2)

def _score_copper_gold(v: float) -> float:
    # basso copper/gold = paura. Invertito
    return _sigmoid(v, center=2500.0, scale=-500.0)

def _score_real_yield(v: float) -> float:
    # basso/negativo = debasement. Invertito
    return _sigmoid(v, center=1.5, scale=-1.5)

def _score_yield_curve(v: float) -> float:
    # inversione (<0) = crisi in arrivo. Invertito
    return _sigmoid(v, center=0.0, scale=-0.5)

def _score_interest_tax(v: float) -> float:
    # >20% = punto di rottura
    return _sigmoid(v, center=15.0, scale=5.0)

def _score_foreign_treasury(v: float) -> float:
    # ROC negativo (vendite nette) = dedollar. Invertito
    return _sigmoid(v, center=0.0, scale=-3.0)

def _score_btp_bund(v: float) -> float:
    # spread in bp sul livello. >200 bp = stress. IT-DE è in %, quindi 2.0 = 200bp
    return _sigmoid(v, center=2.0, scale=0.5)

def _score_eur_chf(v: float) -> float:
    # <0.95 = flight to CHF. Invertito
    return _sigmoid(v, center=0.95, scale=-0.05)

def _score_japan_10y(v: float) -> float:
    # >1% = carry unwind
    return _sigmoid(v, center=0.8, scale=0.4)

def _score_jpy_appreciation(v: float) -> float:
    # jpy_usd_roc_3m negativo = yen si rafforza (panico). Invertito
    return _sigmoid(v, center=0.0, scale=-5.0)

def _score_commodity_fx(v: float) -> float:
    # strength 12m positiva = commodity cycle sano = dedollar
    return _sigmoid(v, center=0.0, scale=5.0)

def _score_em_hy_oas(v: float) -> float:
    # FRED BAMLH0A0HYM2 è in % (es. 3.5 = 350bp).
    # Basso OAS = risk-on = EM attraggono capitali (dedollar). Invertito.
    return _sigmoid(v, center=6.0, scale=-2.5)

def _score_defense_gdp(v: float) -> float:
    # >3.5% = riarmo
    return _sigmoid(v, center=3.5, scale=0.5)


# --- Nuovi scorer (Step 2) ---
def _score_current_account(v: float) -> float:
    # CA/GDP più negativo = squilibrio. Invertito, pivot -3%.
    return _sigmoid(v, center=-3.0, scale=-2.0)

def _score_niip(v: float) -> float:
    # NIIP molto negativo = USA debitore netto massiccio. Invertito, pivot -50%.
    return _sigmoid(v, center=-50.0, scale=-20.0)

def _score_fed_debt_pct(v: float) -> float:
    # Fed/GDP elevato = monetizzazione. >20% critico.
    return _sigmoid(v, center=18.0, scale=5.0)

def _score_fed_balance_roc(v: float) -> float:
    # ROC 12m positivo = QE/debasement. >10% aggressivo.
    return _sigmoid(v, center=0.0, scale=8.0)

def _score_reverse_repo(v: float) -> float:
    # RRP alto = liquidità in eccesso nel sistema = segnale dedollar più debole. Invertito.
    # >1500B = moltissima liquidità intrappolata; <500B = scarsità.
    return _sigmoid(v, center=800.0, scale=-500.0)

def _score_breakeven_5y5y(v: float) -> float:
    # >2.5% = de-anchoring aspettative inflazionistiche.
    return _sigmoid(v, center=2.3, scale=0.4)

def _score_term_premium(v: float) -> float:
    # >0.5% = rischio duration ricostruito, il mondo chiede compenso per Treasuries.
    return _sigmoid(v, center=0.2, scale=0.5)

def _score_cny_strength(v: float) -> float:
    # CNY rafforzamento 12m positivo → BRICS forte, USD debole.
    return _sigmoid(v, center=0.0, scale=5.0)

def _score_em_yield(v: float) -> float:
    # Yield EM crescente = pressione inflazionistica / dedollar spinge rates al rialzo.
    # Usato per China/India/Brazil.
    return _sigmoid(v, center=6.0, scale=2.0)

def _score_gold_silver_diff(v: float) -> float:
    # v = gold_roc_12m - silver_roc_12m. Positivo = gold outperforma (paura/flight-to-quality).
    # Score cresce con il gap positivo; gap negativo (silver vince) è catturato dal silver_roc.
    return _sigmoid(v, center=0.0, scale=10.0)

def _score_silver_roc(v: float) -> float:
    # Silver 12m forte conferma il segnale oro + breadth metalli preziosi.
    return _sigmoid(v, center=5.0, scale=10.0)


# --- Step 3: EM breadth + Europa ---
def _score_em_fx_dollar(v: float) -> float:
    # DTWEXEMEGS ROC 12m: USD forte su EM (ROC>0) = stress EM = NO dedollar.
    # USD debole su EM (ROC<0) = EM si rafforzano = dedollar. Invertito.
    return _sigmoid(v, center=0.0, scale=-4.0)

def _score_ecb_balance_roc(v: float) -> float:
    # ECB QE (ROC>0) = debasement euro. Dedollar generale cresce con currency debasement.
    return _sigmoid(v, center=0.0, scale=8.0)

def _score_eur_usd_roc(v: float) -> float:
    # EUR/USD ROC 12m positivo = euro forte vs USD = dedollar via euro.
    return _sigmoid(v, center=0.0, scale=5.0)

def _score_oat_bund(v: float) -> float:
    # Spread OAT-Bund alto = stress core europa. Storicamente <0.5%; >0.8% = stress.
    return _sigmoid(v, center=0.7, scale=0.3)


# --- Nuovi interpret ---
def _interpret_current_account(v: float) -> str:
    if v < -5: return f"CA/GDP {v:+.2f}% — squilibrio strutturale grave"
    if v < -3: return f"CA/GDP {v:+.2f}% — deficit esterno cronico"
    if v < -1: return f"CA/GDP {v:+.2f}% — deficit moderato"
    if v < 1: return f"CA/GDP {v:+.2f}% — bilancia quasi in pareggio"
    return f"CA/GDP {v:+.2f}% — surplus (raro per USA)"

def _interpret_niip(v: float) -> str:
    if v < -70: return f"NIIP/GDP {v:+.0f}% — USA debitore netto record"
    if v < -50: return f"NIIP/GDP {v:+.0f}% — posizione estera molto deteriorata"
    if v < -30: return f"NIIP/GDP {v:+.0f}% — deficit investimento netto significativo"
    if v < -10: return f"NIIP/GDP {v:+.0f}% — debitore netto moderato"
    return f"NIIP/GDP {v:+.0f}% — posizione neutra o creditrice"

def _interpret_fed_debt_pct(v: float) -> str:
    if v > 22: return f"Fed/GDP {v:.1f}% — monetizzazione elevata (QE permanente)"
    if v > 18: return f"Fed/GDP {v:.1f}% — bilancio Fed sopra media post-QE"
    if v > 12: return f"Fed/GDP {v:.1f}% — livello post-2008 normale"
    return f"Fed/GDP {v:.1f}% — contenuto (QT efficace)"

def _interpret_fed_balance_roc(v: float) -> str:
    if v > 15: return f"Balance sheet {v:+.1f}% YoY — QE aggressivo"
    if v > 5: return f"Balance sheet {v:+.1f}% YoY — espansione moderata"
    if v > -5: return f"Balance sheet {v:+.1f}% YoY — stabile"
    return f"Balance sheet {v:+.1f}% YoY — QT in corso"

def _interpret_reverse_repo(v: float) -> str:
    if v > 1500: return f"RRP {v:.0f}$B — liquidità enorme nel sistema"
    if v > 800: return f"RRP {v:.0f}$B — eccesso liquidità residuo"
    if v > 300: return f"RRP {v:.0f}$B — normalizzazione in corso"
    return f"RRP {v:.0f}$B — liquidità scarsa"

def _interpret_breakeven_5y5y(v: float) -> str:
    if v > 2.8: return f"5Y5Y {v:.2f}% — de-anchoring aspettative long-run"
    if v > 2.5: return f"5Y5Y {v:.2f}% — sopra target Fed persistente"
    if v > 2.0: return f"5Y5Y {v:.2f}% — allineato a target 2%"
    return f"5Y5Y {v:.2f}% — aspettative sotto target (pressione deflattiva)"

def _interpret_term_premium(v: float) -> str:
    if v > 1.0: return f"Term premium {v:+.2f}% — mercato richiede premio alto per duration"
    if v > 0.3: return f"Term premium {v:+.2f}% — premio positivo (normale)"
    if v > -0.3: return f"Term premium {v:+.2f}% — neutro"
    return f"Term premium {v:+.2f}% — compresso (forza d'acquisto su Treasury lunghi)"

def _interpret_cny_strength(v: float) -> str:
    if v > 5: return f"CNY {v:+.1f}% 12m — rafforzamento deciso (dedollar BRICS)"
    if v > 0: return f"CNY {v:+.1f}% 12m — lieve apprezzamento"
    if v > -5: return f"CNY {v:+.1f}% 12m — debolezza moderata (stimolo PBoC)"
    return f"CNY {v:+.1f}% 12m — svalutazione competitiva"

def _interpret_india_10y(v: float) -> str:
    if v > 8: return f"India 10Y {v:.2f}% — inflazione/rischio sovrano elevato"
    if v > 7: return f"India 10Y {v:.2f}% — livello storicamente alto"
    if v > 6: return f"India 10Y {v:.2f}% — livello normale EM"
    return f"India 10Y {v:.2f}% — condizioni monetarie accomodanti"

def _interpret_brazil_10y(v: float) -> str:
    if v > 13: return f"Brazil ST rate {v:.2f}% — stress fiscale acuto"
    if v > 11: return f"Brazil ST rate {v:.2f}% — premio alto rischio sovrano"
    if v > 9: return f"Brazil ST rate {v:.2f}% — yield storicamente elevato"
    return f"Brazil ST rate {v:.2f}% — condizioni finanziarie rilassate"

def _interpret_gold_silver_diff(v: float) -> str:
    if v > 20: return f"Gold−Silver {v:+.1f}% 12m — oro domina, paura estrema"
    if v > 5: return f"Gold−Silver {v:+.1f}% 12m — oro outperforma (regime di rischio)"
    if v > -5: return f"Gold−Silver {v:+.1f}% 12m — metalli preziosi in sintonia"
    if v > -20: return f"Gold−Silver {v:+.1f}% 12m — silver outperforma (ciclo industriale)"
    return f"Gold−Silver {v:+.1f}% 12m — silver esplosivo (mania inflazionistica)"

def _interpret_silver_roc(v: float) -> str:
    if v > 30: return f"Silver {v:+.1f}% YoY — breakout metalli preziosi"
    if v > 10: return f"Silver {v:+.1f}% YoY — conferma segnale oro"
    if v > -5: return f"Silver {v:+.1f}% YoY — movimento laterale"
    return f"Silver {v:+.1f}% YoY — metalli preziosi sotto pressione"


def _interpret_em_fx_dollar(v: float) -> str:
    if v > 8: return f"USD vs EM {v:+.1f}% 12m — dollaro dominante, stress EM severo"
    if v > 3: return f"USD vs EM {v:+.1f}% 12m — USD forte, EM in difficoltà"
    if v > -3: return f"USD vs EM {v:+.1f}% 12m — range laterale"
    if v > -8: return f"USD vs EM {v:+.1f}% 12m — EM si rafforzano (dedollar verso EM)"
    return f"USD vs EM {v:+.1f}% 12m — EM in forte rialzo (super-ciclo BRICS)"

def _interpret_ecb_balance_roc(v: float) -> str:
    if v > 15: return f"ECB {v:+.1f}% YoY — QE aggressivo (debasement euro)"
    if v > 5: return f"ECB {v:+.1f}% YoY — espansione moderata"
    if v > -5: return f"ECB {v:+.1f}% YoY — bilancio stabile"
    return f"ECB {v:+.1f}% YoY — QT in corso (euro attraente)"

def _interpret_eur_usd_roc(v: float) -> str:
    if v > 15: return f"EUR/USD {v:+.1f}% 12m — euro in forte apprezzamento (dedollar)"
    if v > 5: return f"EUR/USD {v:+.1f}% 12m — euro guadagna terreno"
    if v > -5: return f"EUR/USD {v:+.1f}% 12m — range laterale"
    if v > -15: return f"EUR/USD {v:+.1f}% 12m — euro debole vs dollaro"
    return f"EUR/USD {v:+.1f}% 12m — crollo euro (crisi EU)"

def _interpret_oat_bund(v: float) -> str:
    if v > 1.2: return f"OAT-Bund {v*100:.0f}bp — stress grave sulla Francia"
    if v > 0.8: return f"OAT-Bund {v*100:.0f}bp — tensione core europa"
    if v > 0.5: return f"OAT-Bund {v*100:.0f}bp — spread sopra media storica"
    return f"OAT-Bund {v*100:.0f}bp — core europa stabile"


# Interpretation helpers (italiano)
def _interpret_gold_sp500(v: float) -> str:
    if v > 0.8: return f"Ratio {v:.2f} — fuga verso scarsità fisica (picco di paura)"
    if v > 0.5: return f"Ratio {v:.2f} — oro premium sull'azionario"
    if v > 0.3: return f"Ratio {v:.2f} — equilibrio oro/equity"
    return f"Ratio {v:.2f} — azionario preferito all'oro"

def _interpret_copper_gold(v: float) -> str:
    if v > 3500: return f"Ratio {v:.0f} — espansione industriale forte"
    if v > 2500: return f"Ratio {v:.0f} — crescita reale moderata"
    if v > 2000: return f"Ratio {v:.0f} — segnali di rallentamento"
    return f"Ratio {v:.0f} — paura / recessione in arrivo"

def _interpret_real_yield(v: float) -> str:
    if v < -0.5: return f"Yield reale {v:+.2f}% — forte debasement"
    if v < 1.0: return f"Yield reale {v:+.2f}% — dollaro poco attrattivo"
    if v < 2.0: return f"Yield reale {v:+.2f}% — livello neutro"
    return f"Yield reale {v:+.2f}% — USD aspirapolvere di capitali"

def _interpret_yield_curve(v: float) -> str:
    if v < -0.2: return f"Curva invertita ({v:+.2f}%) — segnale recessivo forte"
    if v < 0.2: return f"Curva piatta ({v:+.2f}%) — rallentamento atteso"
    if v < 1.0: return f"Curva normale ({v:+.2f}%) — nessun allarme"
    return f"Curva ripida ({v:+.2f}%) — crescita attesa"

def _interpret_interest_tax(v: float) -> str:
    if v > 25: return f"Interest/Tax {v:.1f}% — zona di crisi fiscale"
    if v > 20: return f"Interest/Tax {v:.1f}% — punto di rottura"
    if v > 15: return f"Interest/Tax {v:.1f}% — livello elevato"
    return f"Interest/Tax {v:.1f}% — sostenibile"

def _interpret_foreign_treasury(v: float) -> str:
    if v < -5: return f"Holdings {v:+.1f}% 12m — vendite nette aggressive"
    if v < 0: return f"Holdings {v:+.1f}% 12m — lieve calo"
    if v < 5: return f"Holdings {v:+.1f}% 12m — stabili"
    return f"Holdings {v:+.1f}% 12m — acquisti netti (USD attraente)"

def _interpret_btp_bund(v: float) -> str:
    if v > 3.0: return f"Spread {v*100:.0f}bp — rischio insolvenza periferia"
    if v > 2.0: return f"Spread {v*100:.0f}bp — stress EU evidente"
    if v > 1.0: return f"Spread {v*100:.0f}bp — moderato differenziale"
    return f"Spread {v*100:.0f}bp — eurozona in calma"

def _interpret_eur_chf(v: float) -> str:
    if v < 0.90: return f"EUR/CHF {v:.3f} — panico verso il franco"
    if v < 0.95: return f"EUR/CHF {v:.3f} — flight-to-quality in corso"
    if v < 1.00: return f"EUR/CHF {v:.3f} — CHF leggermente forte"
    return f"EUR/CHF {v:.3f} — eurozona stabile"

def _interpret_japan_10y(v: float) -> str:
    if v > 1.5: return f"JGB 10Y {v:.2f}% — rischio crollo borse globali"
    if v > 1.0: return f"JGB 10Y {v:.2f}% — carry trade in unwind"
    if v > 0.5: return f"JGB 10Y {v:.2f}% — normalizzazione BoJ in corso"
    return f"JGB 10Y {v:.2f}% — yield control ancora attivo"

def _interpret_jpy_appreciation(v: float) -> str:
    if v < -5: return f"Yen si rafforza {v:+.1f}% in 3m — panico azionario"
    if v < -2: return f"Yen {v:+.1f}% in 3m — flight-to-safety JPY"
    if v < 2: return f"Yen {v:+.1f}% in 3m — stabile"
    return f"Yen {v:+.1f}% in 3m — indebolimento / carry attivo"

def _interpret_commodity_fx(v: float) -> str:
    if v > 5: return f"CAD/AUD {v:+.1f}% 12m — super-ciclo commodity attivo"
    if v > 0: return f"CAD/AUD {v:+.1f}% 12m — commodity cycle in espansione"
    if v > -5: return f"CAD/AUD {v:+.1f}% 12m — cycle in rallentamento"
    return f"CAD/AUD {v:+.1f}% 12m — fine super-ciclo industriale"

def _interpret_em_hy_oas(v: float) -> str:
    bp = v * 100  # FRED dati in %, converte a bp per display
    if v > 8.0: return f"HY OAS {bp:.0f}bp — fuga capitali dagli emergenti"
    if v > 5.0: return f"HY OAS {bp:.0f}bp — stress EM in aumento"
    if v > 3.5: return f"HY OAS {bp:.0f}bp — spread neutro"
    return f"HY OAS {bp:.0f}bp — risk-on verso EM"

def _interpret_defense_gdp(v: float) -> str:
    if v > 4.5: return f"Defense/GDP {v:.2f}% — riarmo accelerato"
    if v > 3.5: return f"Defense/GDP {v:.2f}% — fine dividendo della pace"
    if v > 3.0: return f"Defense/GDP {v:.2f}% — spesa elevata storicamente"
    return f"Defense/GDP {v:.2f}% — spesa contenuta"


PLAYERS: dict[str, dict] = {
    "system": {
        "label": "Sistema (Fiducia)",
        "signals": [
            {"key": "gold_sp500_ratio", "label": "Gold / S&P 500", "scorer": _score_gold_sp500,
             "interpret": _interpret_gold_sp500, "weight": 0.40, "red_flag_if": lambda v: v > 0.8},
            {"key": "copper_gold_ratio", "label": "Copper / Gold", "scorer": _score_copper_gold,
             "interpret": _interpret_copper_gold, "weight": 0.30, "red_flag_if": lambda v: v < 2000},
            {"key": "m2_roc_12m", "label": "Global M2 (USA proxy)", "scorer": lambda v: _sigmoid(v, 5.0, 5.0),
             "interpret": lambda v: f"M2 {v:+.1f}% YoY" + (
                 " — espansione aggressiva" if v > 10 else
                 " — crescita normale" if v > 3 else " — contrazione liquidità"),
             "weight": 0.30, "red_flag_if": lambda v: v < 0},
        ],
    },
    "usa": {
        "label": "Stati Uniti (Egemonia)",
        "signals": [
            {"key": "yield_curve_10y2y", "label": "Yield Curve 10Y-2Y", "scorer": _score_yield_curve,
             "interpret": _interpret_yield_curve, "weight": 0.20, "red_flag_if": lambda v: v < 0},
            {"key": "real_yield_10y", "label": "10Y Real Yield", "scorer": _score_real_yield,
             "interpret": _interpret_real_yield, "weight": 0.20, "red_flag_if": lambda v: v > 2.5},
            {"key": "interest_tax_ratio", "label": "Interest / Tax Revenue", "scorer": _score_interest_tax,
             "interpret": _interpret_interest_tax, "weight": 0.20, "red_flag_if": lambda v: v > 25},
            {"key": "foreign_treasury_roc_12m", "label": "Foreign Treasury Holdings (12m)", "scorer": _score_foreign_treasury,
             "interpret": _interpret_foreign_treasury, "weight": 0.20, "red_flag_if": lambda v: v < -5},
            {"key": "current_account_gdp_pct", "label": "Current Account / GDP", "scorer": _score_current_account,
             "interpret": _interpret_current_account, "weight": 0.10, "red_flag_if": lambda v: v < -5},
            {"key": "niip_gdp_pct", "label": "NIIP / GDP", "scorer": _score_niip,
             "interpret": _interpret_niip, "weight": 0.10, "red_flag_if": lambda v: v < -70},
        ],
    },
    "europe": {
        "label": "Europa (Stabilità)",
        "signals": [
            {"key": "btp_bund_spread", "label": "BTP-Bund Spread", "scorer": _score_btp_bund,
             "interpret": _interpret_btp_bund, "weight": 0.25, "red_flag_if": lambda v: v > 3.0},
            {"key": "eur_chf", "label": "EUR / CHF", "scorer": _score_eur_chf,
             "interpret": _interpret_eur_chf, "weight": 0.20, "red_flag_if": lambda v: v < 0.90},
            {"key": "oat_bund_spread", "label": "OAT-Bund Spread (Francia)",
             "scorer": _score_oat_bund, "interpret": _interpret_oat_bund,
             "weight": 0.20, "red_flag_if": lambda v: v > 1.0},
            {"key": "eur_usd_roc_12m", "label": "EUR/USD 12m",
             "scorer": _score_eur_usd_roc, "interpret": _interpret_eur_usd_roc,
             "weight": 0.20, "red_flag_if": lambda v: v > 15 or v < -15},
            {"key": "ecb_balance_roc_12m", "label": "ECB Balance Sheet 12m",
             "scorer": _score_ecb_balance_roc, "interpret": _interpret_ecb_balance_roc,
             "weight": 0.15, "red_flag_if": lambda v: v > 15},
        ],
    },
    "japan": {
        "label": "Giappone (Carry)",
        "signals": [
            {"key": "japan_10y", "label": "JGB 10Y Yield", "scorer": _score_japan_10y,
             "interpret": _interpret_japan_10y, "weight": 0.60, "red_flag_if": lambda v: v > 1.0},
            {"key": "jpy_usd_roc_3m", "label": "Yen appreciation (3m)", "scorer": _score_jpy_appreciation,
             "interpret": _interpret_jpy_appreciation, "weight": 0.40, "red_flag_if": lambda v: v < -5},
        ],
    },
    "commodity_fx": {
        "label": "Commodity FX (CAD/AUD)",
        "signals": [
            {"key": "commodity_fx_strength", "label": "CAD+AUD strength (12m)", "scorer": _score_commodity_fx,
             "interpret": _interpret_commodity_fx, "weight": 1.0, "red_flag_if": lambda v: v < -10},
        ],
    },
    "em": {
        "label": "Emerging Markets (BRICS+)",
        "signals": [
            {"key": "em_hy_oas", "label": "EM HY OAS (EMBI proxy)", "scorer": _score_em_hy_oas,
             "interpret": _interpret_em_hy_oas, "weight": 0.45, "red_flag_if": lambda v: v > 8.0},
            {"key": "em_fx_dollar_roc_12m", "label": "USD vs EM currencies (12m)",
             "scorer": _score_em_fx_dollar, "interpret": _interpret_em_fx_dollar,
             "weight": 0.55, "red_flag_if": lambda v: v > 8 or v < -8},
        ],
    },
    "defense": {
        "label": "Difesa (Geopolitica)",
        "signals": [
            {"key": "defense_gdp_pct", "label": "Defense / GDP (USA)", "scorer": _score_defense_gdp,
             "interpret": _interpret_defense_gdp, "weight": 1.0, "red_flag_if": lambda v: v > 4.5},
        ],
    },
    "fed_liquidity": {
        "label": "Liquidità Fed",
        "signals": [
            {"key": "fed_balance_roc_12m", "label": "Fed Balance Sheet (12m ROC)",
             "scorer": _score_fed_balance_roc, "interpret": _interpret_fed_balance_roc,
             "weight": 0.40, "red_flag_if": lambda v: v > 15},
            {"key": "fed_debt_pct_gdp", "label": "Fed Treasury Holdings / GDP",
             "scorer": _score_fed_debt_pct, "interpret": _interpret_fed_debt_pct,
             "weight": 0.35, "red_flag_if": lambda v: v > 22},
            {"key": "reverse_repo_level_bn", "label": "Overnight Reverse Repo ($bn)",
             "scorer": _score_reverse_repo, "interpret": _interpret_reverse_repo,
             "weight": 0.25, "red_flag_if": lambda v: v < 200},
        ],
    },
    "expectations": {
        "label": "Aspettative (long-run)",
        "signals": [
            {"key": "breakeven_5y5y", "label": "5Y5Y Forward Breakeven",
             "scorer": _score_breakeven_5y5y, "interpret": _interpret_breakeven_5y5y,
             "weight": 0.55, "red_flag_if": lambda v: v > 2.8},
            {"key": "term_premium_10y", "label": "10Y Term Premium (ACM)",
             "scorer": _score_term_premium, "interpret": _interpret_term_premium,
             "weight": 0.45, "red_flag_if": lambda v: v > 1.0},
        ],
    },
    "brics": {
        "label": "BRICS+ (Poli alternativi)",
        "signals": [
            {"key": "cny_strength_roc_12m", "label": "CNY strength (12m)",
             "scorer": _score_cny_strength, "interpret": _interpret_cny_strength,
             "weight": 0.45, "red_flag_if": lambda v: v > 8},
            {"key": "india_10y", "label": "India 10Y Yield",
             "scorer": _score_em_yield, "interpret": _interpret_india_10y,
             "weight": 0.275, "red_flag_if": lambda v: v > 8},
            {"key": "brazil_policy_rate", "label": "Brazil Policy Rate (ST)",
             "scorer": _score_em_yield, "interpret": _interpret_brazil_10y,
             "weight": 0.275, "red_flag_if": lambda v: v > 14},
        ],
    },
    "metals_breadth": {
        "label": "Breadth metalli (Oro/Argento)",
        "signals": [
            {"key": "gold_silver_perf_diff", "label": "Gold−Silver perf diff (12m)",
             "scorer": _score_gold_silver_diff, "interpret": _interpret_gold_silver_diff,
             "weight": 0.50, "red_flag_if": lambda v: v > 20 or v < -20},
            {"key": "silver_roc_12m", "label": "Silver 12m ROC",
             "scorer": _score_silver_roc, "interpret": _interpret_silver_roc,
             "weight": 0.50, "red_flag_if": lambda v: v > 30},
        ],
    },
}

# Pesi relativi dei player nel geopolitical_score
PLAYER_WEIGHTS = {
    "system": 0.18,
    "usa": 0.20,
    "europe": 0.08,
    "japan": 0.08,
    "commodity_fx": 0.06,
    "em": 0.06,
    "defense": 0.07,
    "fed_liquidity": 0.10,
    "expectations": 0.07,
    "brics": 0.07,
    "metals_breadth": 0.03,
}


def _calculate_player_scores(indicators: dict[str, float]) -> dict:
    """Per ogni macro-player calcola il score aggregato e i singoli segnali.

    Returns:
        dict player_id → {
            "label": str,
            "score": float 0-1,
            "signals": list[{key, label, value, score, interpret, red_flag, weight}],
        }
    """
    result: dict = {}
    for player_id, meta in PLAYERS.items():
        signals_out = []
        total_weight = 0.0
        weighted_score = 0.0
        for sig in meta["signals"]:
            value = indicators.get(sig["key"])
            if value is None:
                signals_out.append({
                    "key": sig["key"],
                    "label": sig["label"],
                    "value": None,
                    "score": None,
                    "interpret": "Dato non disponibile",
                    "red_flag": False,
                    "weight": sig["weight"],
                })
                continue
            score = max(0.0, min(1.0, sig["scorer"](value)))
            red_flag = bool(sig["red_flag_if"](value))
            signals_out.append({
                "key": sig["key"],
                "label": sig["label"],
                "value": round(value, 4),
                "score": round(score, 4),
                "interpret": sig["interpret"](value),
                "red_flag": red_flag,
                "weight": sig["weight"],
            })
            weighted_score += score * sig["weight"]
            total_weight += sig["weight"]

        player_score = weighted_score / total_weight if total_weight > 0 else 0.0
        result[player_id] = {
            "label": meta["label"],
            "score": round(player_score, 4),
            "coverage": round(total_weight, 4),
            "signals": signals_out,
        }
    return result


def _aggregate_player_score(by_player: dict) -> float:
    """Media pesata dei player_score in geopolitical_score."""
    weighted = 0.0
    total_weight = 0.0
    for player_id, data in by_player.items():
        w = PLAYER_WEIGHTS.get(player_id, 0.0) * data.get("coverage", 0.0)
        if w > 0:
            weighted += data["score"] * w
            total_weight += w
    return weighted / total_weight if total_weight > 0 else 0.0


def calculate_secular_bonus(
    dedollarization_score: float,
) -> dict[str, float]:
    """Calcola il bonus/penalty per ogni asset class.

    Usa il combined_score (ciclico + strutturale + accelerazione).
    """
    bonus: dict[str, float] = {}
    for asset in ASSET_CLASSES:
        sensitivity = ASSET_DEDOLLAR_SENSITIVITY.get(asset, 0.0)
        bonus[asset] = round(sensitivity * dedollarization_score, 2)
    return bonus
