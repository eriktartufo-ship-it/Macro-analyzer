"""Narrative explainer per lo stato della dedollarizzazione.

Usa Gemini 2.5 Flash via REST per generare un'analisi dettagliata in italiano
dei campanelli d'allarme (red flag) presenti nei dati macro, basandosi su:
  - score ciclico/strutturale/decennale/ventennale + accelerazione
  - indicatori grezzi (DXY, oro, debito/PIL, M2, spread, curva rendimenti, ecc.)
  - segnali per macro-player con interpretazione testuale di ciascuno
  - evoluzione degli score per player lungo gli orizzonti 1Y/5Y/10Y/20Y
  - accelerazione per-player (delta score vs 1Y fa)

L'output è pensato per essere mostrato in testa alla scheda frontend.
"""

from __future__ import annotations

import json

import httpx
from loguru import logger

from app.config import settings

GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/gemini-2.5-flash:generateContent"
)

# Descrizioni sintetiche degli indicatori grezzi per dare contesto a Gemini
INDICATOR_HINTS: dict[str, str] = {
    "dxy_roc_12m": "Dollar Index variazione 12m (%)",
    "dxy_roc_5y": "Dollar Index variazione 5Y annualizzata (%)",
    "dxy_roc_10y": "Dollar Index variazione 10Y annualizzata (%)",
    "dxy_roc_20y": "Dollar Index variazione 20Y annualizzata (%)",
    "gold_roc_12m": "Oro variazione 12m (%)",
    "gold_roc_5y": "Oro variazione 5Y annualizzata (%)",
    "gold_roc_10y": "Oro variazione 10Y annualizzata (%)",
    "gold_roc_20y": "Oro variazione 20Y annualizzata (%)",
    "gold_oil_ratio": "Rapporto Oro/Petrolio (alto = preferenza riserva di valore)",
    "debt_gdp": "Debito USA / PIL (%)",
    "debt_gdp_5y_ago": "Debito/PIL 5 anni fa (%)",
    "debt_gdp_10y_ago": "Debito/PIL 10 anni fa (%)",
    "debt_gdp_20y_ago": "Debito/PIL 20 anni fa (%)",
    "real_rate": "Tasso reale = Fed Funds - CPI YoY (%)",
    "m2_roc_12m": "M2 variazione 12m (%)",
    "m2_roc_5y": "M2 variazione 5Y annualizzata (%)",
    "m2_roc_10y": "M2 variazione 10Y annualizzata (%)",
    "m2_roc_20y": "M2 variazione 20Y annualizzata (%)",
    "yield_curve_10y2y": "Spread 10Y-2Y Treasury (%, <0 = curva invertita)",
    "real_yield_10y": "Rendimento reale 10Y TIPS (%)",
    "interest_tax_ratio": "Interessi sul debito / entrate fiscali (%)",
    "foreign_treasury_roc_12m": "Holdings esteri di Treasury variazione 12m (%)",
    "btp_bund_spread": "Spread BTP-Bund (%, >2 = stress EU)",
    "eur_chf": "Tasso EUR/CHF",
    "japan_10y": "Rendimento JGB 10Y (%)",
    "jpy_usd_roc_3m": "JPY/USD variazione 3m (%, negativo = yen forte)",
    "commodity_fx_strength": "Forza CAD+AUD 12m (%)",
    "em_hy_oas": "ICE BofA HY OAS (%, proxy rischio EM)",
    "defense_gdp_pct": "Spesa difesa USA / PIL (%)",
    "gold_sp500_ratio": "Gold/S&P500 ratio normalizzato",
    "copper_gold_ratio": "Copper/Gold ratio (basso = paura recessione)",
    # --- Step 2: nuovi indicatori multi-temporale ---
    "real_broad_dxy_roc_12m": "Real Broad Dollar variazione 12m (%) — USD aggiustato per inflazione",
    "real_broad_dxy_roc_5y": "Real Broad Dollar variazione 5Y annualizzata (%)",
    "real_broad_dxy_roc_10y": "Real Broad Dollar variazione 10Y annualizzata (%)",
    "real_broad_dxy_roc_20y": "Real Broad Dollar variazione 20Y annualizzata (%)",
    "current_account_gdp_pct": "Bilancia partite correnti / PIL (%) — twin deficit",
    "niip_gdp_pct": "Net International Investment Position / PIL (%) — USA creditore/debitore netto",
    "fed_debt_pct_gdp": "Treasury in pancia alla Fed / PIL (%) — monetizzazione del debito",
    "fed_balance_roc_12m": "Bilancio Fed variazione 12m (%) — QE vs QT",
    "reverse_repo_level_bn": "Reverse Repo overnight ($bn) — liquidità in eccesso nel sistema",
    "breakeven_5y5y": "5Y5Y Forward breakeven inflation (%) — de-anchoring aspettative long-run",
    "term_premium_10y": "10Y Term Premium ACM (%) — premio rischio duration su Treasuries",
    "cny_strength_roc_12m": "Yuan rafforzamento 12m (%, positivo = CNY forte vs USD)",
    "india_10y": "Rendimento India 10Y (%)",
    "brazil_policy_rate": "Tasso breve Brasile (policy rate, %)",
    "silver_roc_12m": "Argento variazione 12m (%)",
    "silver_roc_5y": "Argento variazione 5Y annualizzata (%)",
    "silver_roc_10y": "Argento variazione 10Y annualizzata (%)",
    "silver_roc_20y": "Argento variazione 20Y annualizzata (%)",
    "gold_silver_perf_diff": "Gold-Silver perf diff 12m (%) — positivo=gold vince (paura), negativo=silver vince (mania)",
    "em_fx_dollar_roc_12m": "Dollar Index vs EM currencies 12m (%, DTWEXEMEGS) — positivo=USD forte su EM (stress)",
    "ecb_balance_roc_12m": "ECB balance sheet variazione 12m (%) — QE euro vs QT",
    "eur_usd_roc_12m": "EUR/USD variazione 12m (%) — positivo=euro forte vs dollaro (dedollar via euro)",
    "oat_bund_spread": "Spread OAT-Bund (%, Francia-Germania) — stress core europa",
}

HORIZON_LABELS = {
    "1y": "1 anno fa",
    "5y": "5 anni fa",
    "10y": "10 anni fa",
    "20y": "20 anni fa",
}


def _format_pct(v: float | None) -> str:
    if v is None:
        return "n/d"
    return f"{v * 100:.0f}%"


def _format_value(key: str, v: float) -> str:
    """Formattazione numerica sensibile al tipo di indicatore."""
    if key in ("gold_oil_ratio", "copper_gold_ratio", "gold_sp500_ratio", "eur_chf"):
        return f"{v:.2f}"
    if key == "em_hy_oas":
        return f"{v:.2f}% ({v * 100:.0f}bp)"
    return f"{v:+.2f}"


def _build_prompt(dedollar: dict, raw_indicators: dict[str, float] | None = None) -> str:
    """Prompt compatto centrato sui dati grezzi: dove ci sono problemi vs dove scorre liscio."""
    # Indicatori grezzi (valori reali su cui Gemini ragiona)
    raw_lines: list[str] = []
    raw = raw_indicators or {}
    for key, val in raw.items():
        if val is None:
            continue
        hint = INDICATOR_HINTS.get(key, key)
        raw_lines.append(f"- {hint} [{key}] = {_format_value(key, float(val))}")

    # Segnali per macro-player con interpretazione + red flag (niente score percentuali)
    by_player = dedollar.get("by_player", {}) or {}
    player_blocks: list[str] = []
    all_red_flags: list[str] = []
    for pid, pdata in by_player.items():
        if pdata.get("coverage", 0) <= 0:
            continue
        lines = [f"▸ {pdata['label']}"]
        for sig in pdata.get("signals", []):
            if sig.get("value") is None:
                continue
            flag = " ⚠ RED FLAG" if sig.get("red_flag") else ""
            lines.append(f"   · {sig['label']}: {sig.get('interpret', '')}{flag}")
            if sig.get("red_flag"):
                all_red_flags.append(f"{pdata['label']} → {sig['label']}: {sig.get('interpret','')}")
        player_blocks.append("\n".join(lines))

    # === PROMPT FINALE (compatto, data-driven) ===
    sections = [
        "Sei un analista macro senior. Scrivi un briefing SINTETICO in italiano sulla "
        "dedollarizzazione (400-600 parole, 3-4 paragrafi).",
        "",
        "REGOLE:",
        "- Struttura il testo come: (1) **Dove ci sono problemi** — individua i segnali "
        "effettivamente deteriorati/stressati, citando i valori grezzi; (2) **Dove va liscio** "
        "— indica le aree in regime normale o di solidità, sempre citando i valori; "
        "(3) **Cosa monitorare** — 2-3 segnali da sorvegliare nei prossimi 6-12 mesi.",
        "- NON usare percentuali aggregate come \"score composito\", \"score strutturale\", "
        "\"score geopolitico\". Non citare il concetto di score dell'algoritmo. "
        "Ragiona SOLO sui valori grezzi degli indicatori (yield, ratio, ROC %, spread, ecc.).",
        "- Sii concreto: per ogni tesi cita il numero (es. \"curva 10Y-2Y a -0.25% segnala recessione imminente\").",
        "- Tono: professionale, asciutto. Prosa fluida. Ammessi sottotitoli brevi in **grassetto** "
        "per le sezioni (es. **Dove ci sono problemi**, **Dove va liscio**, **Cosa monitorare**).",
        "- NO \"In sintesi\", NO elenchi puntati nel testo finale.",
        "",
        "═══ INDICATORI GREZZI (valori reali) ═══",
        *raw_lines,
        "",
        "═══ SEGNALI PER MACRO-PLAYER (con interpretazione) ═══",
        *player_blocks,
        "",
        "═══ RED FLAG ATTIVI ═══" if all_red_flags else "═══ NESSUN RED FLAG ATTIVO ═══",
        *(f"⚠ {rf}" for rf in all_red_flags),
    ]
    return "\n".join(s for s in sections if s is not None)


def generate_explanation(
    dedollar: dict,
    raw_indicators: dict[str, float] | None = None,
) -> str | None:
    """Chiama Gemini 2.5 Flash e restituisce il testo narrativo.

    Args:
        dedollar: risultato di `calculate_dedollarization`.
        raw_indicators: dict degli indicatori grezzi (dedollar_indicators nel pipeline).

    Ritorna None se la chiave non è configurata o se la chiamata fallisce
    (non-bloccante: il pipeline deve continuare anche senza spiegazione).
    """
    api_key = settings.gemini_api_key
    if not api_key:
        logger.info("GEMINI_API_KEY non configurata, salto la generazione della spiegazione")
        return None

    prompt = _build_prompt(dedollar, raw_indicators=raw_indicators)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 2000,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }

    try:
        with httpx.Client(timeout=45.0) as client:
            resp = client.post(
                GEMINI_ENDPOINT,
                params={"key": api_key},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        candidates = data.get("candidates") or []
        if not candidates:
            logger.warning(f"Gemini: nessun candidato nella risposta — {json.dumps(data)[:300]}")
            return None
        parts = candidates[0].get("content", {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts).strip()
        return text or None
    except httpx.HTTPStatusError as e:
        logger.warning(f"Gemini HTTP {e.response.status_code}: {e.response.text[:200]}")
        return None
    except Exception as e:
        logger.warning(f"Gemini request failed: {e}")
        return None
