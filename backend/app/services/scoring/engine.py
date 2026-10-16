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
    # T11 — Sector ETF expansion (Council 2026-05-28: alpha intra-regime)
    "sector_energy",          # XLE — Energy Select Sector SPDR (1998+)
    "sector_financials",      # XLF — Financials (1998+)
    "sector_utilities",       # XLU — Utilities (1998+)
    "sector_technology",      # XLK — Technology (1998+)
    "sector_healthcare",      # XLV — Health Care (1998+)
    "sector_staples",         # XLP — Consumer Staples (1998+)
    # T16 — Factor ETF expansion (Council 2026-05-31 sessione 23): tactical overlay
    "factor_momentum",        # MTUM — iShares MSCI USA Momentum Factor (2013+)
    "factor_quality",         # QUAL — iShares MSCI USA Quality Factor (2013+)
    "factor_value",           # VLUE — iShares MSCI USA Value Factor (2013+)
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
        # T9-AUDIT-FIX: parametri haircut da council (raydalio/taleb/buffett).
        # Sample N=15 anni in regime liquidità infinita = survivorship trap.
        # Pre-T9: avg_return 0.45 sharpe 0.60 in reflation → bitcoin top-1 in 6/9 OOS
        # ma reale in 2/9. Bias structural. Haircut: cap upside, accept asymmetric vol.
        "reflation":    {"hit_rate": 0.55, "avg_return": 0.20, "vol": 0.75, "sharpe": 0.30},
        "stagflation":  {"hit_rate": 0.30, "avg_return": -0.40, "vol": 0.85, "sharpe": -0.47},
        "deflation":    {"hit_rate": 0.25, "avg_return": -0.35, "vol": 0.85, "sharpe": -0.41},
        "goldilocks":   {"hit_rate": 0.50, "avg_return": 0.15, "vol": 0.70, "sharpe": 0.25},
    },
    "crypto_broad": {
        # T9-AUDIT-FIX: amplificato rispetto a BTC ma applico stessa logica haircut.
        # Pre-T9: avg_return 0.40 sharpe 0.47 → crypto_broad anche lui top-N inflato.
        "reflation":    {"hit_rate": 0.52, "avg_return": 0.18, "vol": 0.85, "sharpe": 0.22},
        "stagflation":  {"hit_rate": 0.25, "avg_return": -0.50, "vol": 0.90, "sharpe": -0.56},
        "deflation":    {"hit_rate": 0.22, "avg_return": -0.45, "vol": 0.95, "sharpe": -0.47},
        "goldilocks":   {"hit_rate": 0.48, "avg_return": 0.12, "vol": 0.80, "sharpe": 0.18},
    },
    # ──────────────────────────────────────────────────────────────────────
    # T11 SECTOR ETF — Calibrazione 1998-12-22 inception → 2024
    # Reference: 2008 GFC sector returns (XLF -55%, XLK -42%, XLP -16%),
    # 2020 Q1 COVID (XLU defensive +18% YTD pre-Mar, XLF -32%),
    # 2022 stagflation (XLE +64% real best-of-class, XLK -28%),
    # 1995-99 + 2013-19 goldilocks (XLK +25/+22% real best).
    # ──────────────────────────────────────────────────────────────────────
    "sector_energy": {
        # XLE: cyclical commodity-exposed. Best stagflation winner.
        "reflation":    {"hit_rate": 0.65, "avg_return": 0.18, "vol": 0.30, "sharpe": 0.60},
        "stagflation":  {"hit_rate": 0.78, "avg_return": 0.30, "vol": 0.32, "sharpe": 0.94},
        "deflation":    {"hit_rate": 0.18, "avg_return": -0.35, "vol": 0.42, "sharpe": -0.83},
        "goldilocks":   {"hit_rate": 0.45, "avg_return": 0.04, "vol": 0.26, "sharpe": 0.15},
    },
    "sector_financials": {
        # XLF: rate-sensitive NIM expansion in reflation, crash in 2008/2020.
        "reflation":    {"hit_rate": 0.70, "avg_return": 0.15, "vol": 0.20, "sharpe": 0.75},
        "stagflation":  {"hit_rate": 0.40, "avg_return": -0.05, "vol": 0.24, "sharpe": -0.21},
        "deflation":    {"hit_rate": 0.18, "avg_return": -0.35, "vol": 0.32, "sharpe": -1.09},
        "goldilocks":   {"hit_rate": 0.65, "avg_return": 0.12, "vol": 0.18, "sharpe": 0.67},
    },
    "sector_utilities": {
        # XLU: bond-like, hurt by rising rates, defensive in deflation/goldilocks.
        "reflation":    {"hit_rate": 0.40, "avg_return": 0.02, "vol": 0.13, "sharpe": 0.15},
        "stagflation":  {"hit_rate": 0.40, "avg_return": -0.08, "vol": 0.16, "sharpe": -0.50},
        "deflation":    {"hit_rate": 0.65, "avg_return": 0.08, "vol": 0.13, "sharpe": 0.62},
        "goldilocks":   {"hit_rate": 0.62, "avg_return": 0.08, "vol": 0.11, "sharpe": 0.73},
    },
    "sector_technology": {
        # XLK: cyclical growth. Best goldilocks, worst stagflation/deflation.
        "reflation":    {"hit_rate": 0.75, "avg_return": 0.18, "vol": 0.20, "sharpe": 0.90},
        "stagflation":  {"hit_rate": 0.20, "avg_return": -0.25, "vol": 0.28, "sharpe": -0.89},
        "deflation":    {"hit_rate": 0.25, "avg_return": -0.32, "vol": 0.30, "sharpe": -1.07},
        "goldilocks":   {"hit_rate": 0.85, "avg_return": 0.22, "vol": 0.16, "sharpe": 1.38},
    },
    "sector_healthcare": {
        # XLV: defensive growth. Steady performer cross-regime.
        "reflation":    {"hit_rate": 0.60, "avg_return": 0.09, "vol": 0.14, "sharpe": 0.64},
        "stagflation":  {"hit_rate": 0.55, "avg_return": 0.02, "vol": 0.16, "sharpe": 0.12},
        "deflation":    {"hit_rate": 0.50, "avg_return": -0.10, "vol": 0.18, "sharpe": -0.55},
        "goldilocks":   {"hit_rate": 0.75, "avg_return": 0.13, "vol": 0.12, "sharpe": 1.08},
    },
    "sector_staples": {
        # XLP: defensive value, pricing power, lag cyclicals.
        "reflation":    {"hit_rate": 0.45, "avg_return": 0.05, "vol": 0.12, "sharpe": 0.42},
        "stagflation":  {"hit_rate": 0.55, "avg_return": 0.02, "vol": 0.13, "sharpe": 0.15},
        "deflation":    {"hit_rate": 0.65, "avg_return": 0.05, "vol": 0.13, "sharpe": 0.38},
        "goldilocks":   {"hit_rate": 0.65, "avg_return": 0.09, "vol": 0.10, "sharpe": 0.90},
    },
    # ──────────────────────────────────────────────────────────────────────
    # T16 FACTOR ETF — Calibrazione 2013+ inception → 2024
    # MTUM (Momentum), QUAL (Quality), VLUE (Value): MSCI USA factor series.
    # Reference: 2013-19 goldilocks (MTUM +14% real best), 2022 momentum crash
    # (MTUM -16%), 2020 Q1 quality defensive (+10pp vs SPX), value rally
    # 2021 reopening (VLUE +25% vs MTUM +10%).
    # ──────────────────────────────────────────────────────────────────────
    "factor_momentum": {
        # MTUM: trending winners ride waves, vulnerable a regime shifts.
        # 2018 momentum crash, 2020 March: -25% in 3 weeks, 2022: -16% YTD.
        "reflation":    {"hit_rate": 0.75, "avg_return": 0.20, "vol": 0.20, "sharpe": 1.00},
        "stagflation":  {"hit_rate": 0.55, "avg_return": 0.10, "vol": 0.25, "sharpe": 0.40},
        "deflation":    {"hit_rate": 0.25, "avg_return": -0.25, "vol": 0.30, "sharpe": -0.83},
        "goldilocks":   {"hit_rate": 0.80, "avg_return": 0.22, "vol": 0.18, "sharpe": 1.22},
    },
    "factor_quality": {
        # QUAL: high ROE + low debt + earnings stability. Defensive compound.
        "reflation":    {"hit_rate": 0.65, "avg_return": 0.13, "vol": 0.18, "sharpe": 0.72},
        "stagflation":  {"hit_rate": 0.55, "avg_return": 0.05, "vol": 0.20, "sharpe": 0.25},
        "deflation":    {"hit_rate": 0.50, "avg_return": -0.10, "vol": 0.22, "sharpe": -0.45},
        "goldilocks":   {"hit_rate": 0.78, "avg_return": 0.17, "vol": 0.15, "sharpe": 1.13},
    },
    "factor_value": {
        # VLUE: P/B + earnings yield + price-to-cash. Cyclical recovery winner.
        # 2021 reopening: +28% real best of factors. 2020 Q1: -42% (value trap).
        "reflation":    {"hit_rate": 0.78, "avg_return": 0.18, "vol": 0.21, "sharpe": 0.86},
        "stagflation":  {"hit_rate": 0.55, "avg_return": 0.05, "vol": 0.22, "sharpe": 0.23},
        "deflation":    {"hit_rate": 0.30, "avg_return": -0.20, "vol": 0.28, "sharpe": -0.71},
        "goldilocks":   {"hit_rate": 0.62, "avg_return": 0.10, "vol": 0.14, "sharpe": 0.71},
    },
}


def _calibrated_or_prior() -> dict[str, dict[str, dict[str, float]]]:
    """Carica la calibrazione persistita se presente E se flag abilitati;
    fallback sull'hardcoded.

    Priority lookup (Tier 6.6):
    1. DB `asset_regime_calibrations` table (auto-refresh mensile) → se
       `USE_LIVE_CALIBRATION` ON.
    2. JSON `seed/calibrated_asset_regime.json` (legacy) → se
       `USE_CALIBRATED_SCORING` ON.
    3. Hardcoded `ASSET_REGIME_DATA` (default).

    La calibrazione e' generata da `services/scoring/calibration.py` via shrinkage
    Bayesiano contro rendimenti reali misurati.

    NOTA: la calibrazione attualmente eredita il bias del regime classifier (es. i
    "regimi stagflation" del classifier moderno includono il 2022-24, periodo con
    caratteristiche miste). Per questo motivo e' OPT-IN.
    """
    from app.services.config_flags import use_calibrated_scoring, use_live_calibration

    payload = None

    # T6.6: priorità DB se flag attivo
    if use_live_calibration():
        try:
            from app.database import SessionLocal
            from app.services.scoring.live_calibration import load_latest_from_db
            with SessionLocal() as session:
                payload = load_latest_from_db(session)
        except Exception:
            payload = None

    # Fallback: JSON legacy se flag attivo e DB non disponibile
    if payload is None and use_calibrated_scoring():
        try:
            from app.services.scoring.calibration import load_calibration
            payload = load_calibration()
        except Exception:
            payload = None

    # Né DB né JSON disponibili → hardcoded prior
    if payload is None:
        return ASSET_REGIME_DATA

    if payload and "asset_regime_data" in payload:
        # Costruisci un dict compatibile con ASSET_REGIME_DATA, ignorando
        # i campi extra (n_observations, source).
        out: dict[str, dict[str, dict[str, float]]] = {}
        for asset, regimes in payload["asset_regime_data"].items():
            out[asset] = {}
            for regime, stats in regimes.items():
                out[asset][regime] = {
                    "hit_rate": stats["hit_rate"],
                    "avg_return": stats["avg_return"],
                    "vol": stats["vol"],
                    "sharpe": stats["sharpe"],
                }
        # Per asset/regimi non in calibration, fallback su hardcoded
        for asset, regimes in ASSET_REGIME_DATA.items():
            out.setdefault(asset, {})
            for regime, stats in regimes.items():
                out[asset].setdefault(regime, stats)
        return out
    return ASSET_REGIME_DATA


# Cache modulo: la calibrazione si carica una volta all'import. In caso di
# rigenerazione runtime, chi rigenera deve invocare `reload_calibration()`.
_ACTIVE_DATA = _calibrated_or_prior()


def reload_calibration() -> None:
    """Forza il reload della calibrazione (es. dopo `calibration.calibrate()`)."""
    global _ACTIVE_DATA, _REGIME_PERCENTILES
    _ACTIVE_DATA = _calibrated_or_prior()
    _REGIME_PERCENTILES = {}  # invalida cache T7.2


def _asset_regime_score(asset: str, regime: str) -> float:
    """Calcola lo score 0-100 di un asset in un regime specifico.

    Formula (rev. 2):
        score = hit_rate * 0.25
              + real_return_norm * 0.50
              + sharpe_norm * 0.25

    Usa i parametri calibrati (se presenti) o quelli hardcoded come fallback.

    Dove:
        real_return_norm = clamp((real_return + 0.30) / 0.60, 0, 1)   # range [-30%, +30%]
        sharpe_norm      = clamp((sharpe + 1.0) / 3.0, 0, 1)           # range [-1, +2]
    """
    data = _ACTIVE_DATA[asset][regime]
    hit_rate = data["hit_rate"]
    real_return = data["avg_return"]
    sharpe = data["sharpe"]

    real_return_norm = max(0.0, min(1.0, (real_return + 0.30) / 0.60))
    sharpe_norm = max(0.0, min(1.0, (sharpe + 1.0) / 3.0))

    score = (hit_rate * 0.25 + real_return_norm * 0.50 + sharpe_norm * 0.25) * 100

    return max(0.0, min(100.0, score))


# T9-AUDIT-FIX: asset con prior calibrato su sample troppo piccolo (BTC ~15y,
# crypto_broad ~10y) NON possono dominare il top-rank. Cap percentile a 0.80.
# Quando live_calibration avrà sample_size ≥ 30 cicli, rimuovere cap.
_LIMITED_SAMPLE_ASSETS = {"bitcoin", "crypto_broad"}
_LIMITED_SAMPLE_PERCENTILE_CAP = 0.80


def _compute_regime_percentiles() -> dict[str, dict[str, float]]:
    """Tier 7.2 + T9-AUDIT: pre-compute rank percentile within-regime per asset.

    Per ogni regime, ordina i 24 asset per `_asset_regime_score(asset, regime)`,
    assegna percentile = rank / (n-1) ∈ [0, 1]. Score più alto → percentile 1.0.

    **T9-AUDIT-FIX**: cap percentile a 0.80 per asset in `_LIMITED_SAMPLE_ASSETS`
    (bitcoin, crypto_broad). Senza cap, BTC scora top-1 in 6/9 OOS ma reale 2/9
    (over-confidence su sample N=15 anni). Cap permette comunque top-3 ma non top-1.

    Restituisce dict {regime: {asset: percentile}}.
    """
    out: dict[str, dict[str, float]] = {}
    regimes = ("reflation", "stagflation", "deflation", "goldilocks")
    for regime in regimes:
        scores = [(a, _asset_regime_score(a, regime)) for a in ASSET_CLASSES]
        scores.sort(key=lambda x: x[1])
        n = len(scores)
        if n <= 1:
            out[regime] = {a: 0.5 for a in ASSET_CLASSES}
            continue
        out[regime] = {
            a: min(
                rank / (n - 1),
                _LIMITED_SAMPLE_PERCENTILE_CAP if a in _LIMITED_SAMPLE_ASSETS else 1.0,
            )
            for rank, (a, _) in enumerate(scores)
        }
    return out


# Cache module-level: ricalcolato in `reload_calibration` se _ACTIVE_DATA cambia.
_REGIME_PERCENTILES: dict[str, dict[str, float]] = {}

# T7.6 — Stato dell'ultima invocazione di `calculate_final_scores` con override
# applicati. Permette di esporre l'audit trail tramite
# `calculate_final_scores_with_metadata` SENZA rompere back-compat dei 30+
# callers di `calculate_final_scores` che si aspettano `dict[str, float]`.
_LAST_APPLIED_OVERRIDES: list[dict] = []
_LAST_APPLIED_ASSET_OVERRIDES: list[dict] = []


def _asset_regime_score_percentile(asset: str, regime: str) -> float:
    """T7.2: ritorna il rank percentile within-regime × 100 (0-100 scale)."""
    global _REGIME_PERCENTILES
    if not _REGIME_PERCENTILES:
        _REGIME_PERCENTILES = _compute_regime_percentiles()
    return _REGIME_PERCENTILES.get(regime, {}).get(asset, 0.5) * 100.0


def calculate_final_scores(
    probabilities: dict[str, float],
    secular_bonus: dict[str, float] | None = None,
    news_signals: dict[str, float] | None = None,
    momentum_penalty: dict[str, float] | None = None,
    force_include_dedollar: bool | None = None,
    gdp_yoy_dfm: float | None = None,
    indicators: dict[str, float] | None = None,
) -> dict[str, float]:
    """Calcola lo score finale per ogni asset class.

    Formula:
        final_score(asset) = sum(
            regime_probability[r] * asset_regime_score(asset, r)
            for r in all_regimes
        ) + secular_trend_bonus - momentum_penalty + news_signal

    Args:
        probabilities: Dict {regime: probabilita} (deve sommare a ~1.0)
        secular_bonus: Dict {asset: bonus} - applicato SOLO se env USE_DEDOLLAR_BONUS=1
            o force_include_dedollar=True. Quando OFF, gli score sono pure data-driven.
        news_signals: Dict {asset: signal -5 to +5}
        momentum_penalty: Dict {asset: penalty 0-10}
        force_include_dedollar: override del flag env (utile per esporre vista
            "with dedollar" nell'API senza riavviare). Se None, usa env var.

    Returns:
        Dict {asset_class: score 0-100}
    """
    from app.services.config_flags import (
        use_dedollar_bonus,
        use_dfm_asset_bonus,
        use_discretionary_overrides,
    )

    if force_include_dedollar is None:
        include_dedollar = use_dedollar_bonus()
    else:
        include_dedollar = bool(force_include_dedollar)

    total_prob = sum(probabilities.values())
    if total_prob > 0 and abs(total_prob - 1.0) > 0.001:
        probabilities = {r: p / total_prob for r, p in probabilities.items()}

    # T6.7 — apply regime overrides PRE asset score calc (shift probs).
    active_overrides: list = []
    applied_regime_overrides: list = []
    if use_discretionary_overrides():
        try:
            from app.database import SessionLocal
            from app.services.scoring.discretionary import (
                apply_regime_overrides,
                load_active_overrides,
            )
            with SessionLocal() as session:
                active_overrides = load_active_overrides(session)
                regime_actives = [o for o in active_overrides if o.target_type == "regime"]
            if regime_actives:
                probabilities, applied_regime_overrides = apply_regime_overrides(
                    probabilities, regime_actives
                )
        except Exception:
            active_overrides = []
    # T7.6: traccia di metadata (esposto via calculate_final_scores_with_metadata)
    _LAST_APPLIED_OVERRIDES.clear()
    for ov in applied_regime_overrides:
        _LAST_APPLIED_OVERRIDES.append({
            "id": ov.id,
            "target_type": ov.target_type,
            "target_name": ov.target_name,
            "delta": ov.delta,
            "reason": ov.reason,
            "author": ov.author,
            "expires_at": ov.expires_at.isoformat() if ov.expires_at else None,
        })

    # Tier 5.4 LOOP CLOSURE — DFM nowcast bonus. Opt-in. δ=2.0 default.
    # Pro-cyclical assets bid quando gdp_yoy_dfm > 2.5%, sold quando < 1.5%.
    # Simmetrico verso 2.0% trend USA long-run.
    dfm_bonus: dict[str, float] = {}
    if use_dfm_asset_bonus() and gdp_yoy_dfm is not None:
        # T9-AUDIT P0 (council): DFM nowcast NOT validated (only 2 records in DB).
        # Log warning ogni volta che il flag attivo applica DFM bonus.
        import logging
        logging.getLogger(__name__).warning(
            "USE_DFM_ASSET_BONUS=ON but DFM nowcast has <30 historical "
            "validation pairs. Council recommends backfill before production use. "
            "gdp_yoy_dfm=%s applied with caveat.", gdp_yoy_dfm
        )
        delta = 2.0
        # Strong growth nowcast (>2.5% YoY)
        upside = max(0.0, gdp_yoy_dfm - 2.5)
        # Weak growth nowcast (<1.5% YoY)
        downside = max(0.0, 1.5 - gdp_yoy_dfm)
        pro_cyclical = ("us_equities_growth", "us_equities_value", "em_equities")
        duration_sensitive = ("us_bonds_long",)
        for a in pro_cyclical:
            dfm_bonus[a] = delta * upside - delta * downside
        for a in duration_sensitive:
            dfm_bonus[a] = -delta * upside + delta * downside

    scores: dict[str, float] = {}

    # T7.2 rank_percentile rimosso 2026-05-28 (council KILL T10b: -0.421 Sortino).
    use_percentile = False
    score_fn = _asset_regime_score

    # T7.3 downside_protection rimosso 2026-05-28 (code audit: no validation).
    downside_bonus_active = False

    for asset in ASSET_CLASSES:
        base_score = sum(
            probabilities.get(regime, 0.0) * score_fn(asset, regime)
            for regime in probabilities
        )

        bonus = (secular_bonus or {}).get(asset, 0.0) if include_dedollar else 0.0
        news = (news_signals or {}).get(asset, 0.0)
        penalty = (momentum_penalty or {}).get(asset, 0.0)
        dfm = dfm_bonus.get(asset, 0.0)

        downside = 0.0
        if downside_bonus_active:
            # Calcola vol e avg_return medi cross-regime
            asset_data = _ACTIVE_DATA.get(asset, {})
            avg_vol = sum(d.get("vol", 0.5) for d in asset_data.values()) / max(1, len(asset_data))
            avg_real_return = sum(d.get("avg_return", -0.5) for d in asset_data.values()) / max(1, len(asset_data))
            if avg_vol < 0.05 and avg_real_return > -0.05:
                # Scala bonus per quanto stress domina (40%-100% → 0-25 punti)
                downside = 25.0 * min(1.0, (stress_share - 0.4) / 0.6)

        final = base_score + bonus + news - penalty + dfm + downside
        scores[asset] = max(0.0, min(100.0, round(final, 1)))

    # T9-AUDIT-FIX: Conditional asset scoring P(asset | regime, stress) — blend.
    # use_conditional_asset_scoring rimosso 2026-05-28 (code audit: no validation)

    # T6.7 — apply asset overrides POST score computation (additive delta + clamp).
    _LAST_APPLIED_ASSET_OVERRIDES.clear()
    if use_discretionary_overrides() and active_overrides:
        try:
            from app.services.scoring.discretionary import apply_asset_overrides
            asset_actives = [o for o in active_overrides if o.target_type == "asset"]
            if asset_actives:
                scores, applied_asset = apply_asset_overrides(scores, asset_actives)
                scores = {k: round(v, 1) for k, v in scores.items()}
                for ov in applied_asset:
                    _LAST_APPLIED_ASSET_OVERRIDES.append({
                        "id": ov.id,
                        "target_type": ov.target_type,
                        "target_name": ov.target_name,
                        "delta": ov.delta,
                        "reason": ov.reason,
                        "author": ov.author,
                        "expires_at": ov.expires_at.isoformat() if ov.expires_at else None,
                    })
        except Exception:
            pass

    return scores


def calculate_final_scores_with_metadata(
    probabilities: dict[str, float],
    secular_bonus: dict[str, float] | None = None,
    news_signals: dict[str, float] | None = None,
    momentum_penalty: dict[str, float] | None = None,
    force_include_dedollar: bool | None = None,
    gdp_yoy_dfm: float | None = None,
) -> dict:
    """T7.6 — Wrapper di `calculate_final_scores` che espone l'audit trail
    degli override applicati (Bridgewater radical transparency).

    Returns:
        {
            "scores": dict[str, float],
            "applied_regime_overrides": list[dict],
            "applied_asset_overrides": list[dict],
            "downside_protection_active": bool,
            "rank_percentile_active": bool,
        }
    """
    scores = calculate_final_scores(
        probabilities=probabilities,
        secular_bonus=secular_bonus,
        news_signals=news_signals,
        momentum_penalty=momentum_penalty,
        force_include_dedollar=force_include_dedollar,
        gdp_yoy_dfm=gdp_yoy_dfm,
    )
    return {
        "scores": scores,
        "applied_regime_overrides": list(_LAST_APPLIED_OVERRIDES),
        "applied_asset_overrides": list(_LAST_APPLIED_ASSET_OVERRIDES),
        "downside_protection_active": False,  # T7.3 killed 2026-05-28
        "rank_percentile_active": False,  # T7.2 killed 2026-05-28 (council T10b)
    }
