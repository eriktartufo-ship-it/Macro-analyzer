"""LLM-Driven Strategist — Gemini decide cosa comprare/vendere.

3a strategia horserace (sessione 2026-06-13). NON è il commentary del
llm_reasoner: qui Gemini è il DECISORE. Ogni giorno:

1. Riceve in input:
   - regime classifier + confidence + probabilità
   - 25 indicators FRED chiave + ROC
   - score Model-Driven per ogni asset (cosa dice il classifier-based)
   - score Data-Driven per ogni asset (cosa dicono le rule-based su indicators)
   - posizioni aperte con PnL reale e giorni held
   - learnings post-mortem accumulati per strategia llm_driven

2. Ritorna JSON strutturato:
   {"decisions": [
       {"asset": "us_equities_growth", "action": "OPEN_TRANCHE",
        "score": 65, "confidence": 0.7, "size_pct": 4.5, "reason": "..."},
       ...
   ]}

3. Le decisioni sono validate/clampate + applicate via stessa pipeline degli altri
   strategist (open/close/stop/tp/incremental). Pool $100k separato, learning
   post-mortem isolato.

Fallback: se Gemini fail → no-op (nessuna decisione, posizioni invariate).
Mai crash, mai fake data: la 3a strategia salta semplicemente la giornata.

Costo: 1 call Gemini/day = trascurabile.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import date
from typing import Optional

import pandas as pd
import requests
from sqlalchemy.orm import Session

from app.models.ai_portfolio import AiPortfolioDecision, AiPortfolioPosition
from app.models.regime_classifications import RegimeClassification
from app.services.ai_portfolio import learning, momentum, pnl_tracker
from app.services.llm import settings as llm_settings
from app.services.scoring.engine import ASSET_CLASSES

logger = logging.getLogger(__name__)


STRATEGY_TYPE = "llm_driven"
# v2-independent (2026-06-13): rimossi model_scoreboard + data_scoreboard
# dal prompt per evitare anchoring bias. Gemini analizza solo indicators
# raw + regime come second opinion. Bump prompt_version invalida cache.
# v5-roc-forecast (2026-07-15, richiesta Erik): il bot guardava i dati come una
# FOTOGRAFIA (livelli) e giustificava solo le APERTURE. Ora: (a) gli si insegna
# livello-vs-ROC — il livello è già nei prezzi, la derivata è il segnale predittivo
# (stessa lezione strutturale del T20 livello-vs-accelerazione del classifier);
# (b) campo `outlook` = previsione esplicita 3-6m su cui le decisioni devono essere
# coerenti; (c) giustificazione obbligatoria anche sulle CHIUSURE.
#
# v6-market-aware (2026-07-15, richiesta Erik): FIX STRUTTURALE — il bot sceglieva
# 24 asset SENZA VEDERE UN PREZZO. Il `price_history` veniva fetchato DOPO la
# chiamata a Gemini e usato solo per LOGGARE la decisione (momentum_signal/vol_60d).
# Decideva su macro pura, cieco a momentum/RSI/vol/rendimenti degli asset che
# comprava. Ora: (a) fetch prezzi PRIMA del prompt + blocco "lettura di mercato"
# (momentum, RSI, dist MA50, vol, ret 1/3/6m, **rel_3m = forza relativa vs S&P**);
# (b) regime da "second opinion ignorabile" a INPUT DI PRIMA CLASSE con dissenso
# ammesso ma MOTIVATO (campo `regime_dissent`) — scelta Erik, tempera l'anti-anchoring
# di S31 senza perdere il regime-challenge; (c) le reason devono incrociare macro E
# prezzo. rel_3m è la traccia operativa della ROTAZIONE (i flussi veri non sono
# osservabili gratis; la performance relativa e' la loro ombra).
_PROMPT_VERSION = "v6-market-aware"

# Stesse soglie hard di safety (Gemini suggerisce ma noi clampiamo)
MAX_SIZE_PCT_PER_TRANCHE = 0.08  # 8% max per tranche, anti-hallucination
MIN_CONFIDENCE = 0.30
STOP_LOSS_PCT = -0.10
FULL_TP_PCT = 0.30

VALID_ACTIONS = {
    "OPEN_TRANCHE", "CLOSE_TRANCHE", "FULL_CLOSE",
    "STOP_LOSS", "TAKE_PROFIT", "HOLD", "SKIP_NO_SIGNAL",
}

_GEMINI_API_URL_TPL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/{model}:generateContent"
)


_SYSTEM_PROMPT = """Sei un portfolio manager senior di un hedge fund discrezionale.
Fai la TUA analisi INDIPENDENTE dei dati macro raw: niente segnali pre-masticati,
niente scoreboard di altri sistemi. Ti diamo solo gli indicators FRED + un regime
classifier come second opinion (che puoi ignorare se la tua tesi lo giustifica).

**IL TUO LAVORO È PREVEDERE, NON FOTOGRAFARE IL PRESENTE.**
I mercati prezzano già ciò che è successo. Il tuo edge sta nell'anticipare dove
i dati stanno ANDANDO nei prossimi 3-6 mesi.

**COME SI LEGGE UN DATO: LIVELLO vs ROC (regola non negoziabile)**
Ogni indicator ha due letture, e sono cose DIVERSE:
- il **LIVELLO** (`cpi_yoy`, `unrate`, `vix`) dice DOVE SEI OGGI → è già nei prezzi;
- il **ROC / la variazione** (`_roc`, `_roc_12m`, `_change_3m`, `_change_6m`, `_yoy`)
  dice DOVE STAI ANDANDO e a che VELOCITÀ → è lì che sta l'informazione predittiva.

Esempi del ragionamento che ci aspettiamo:
- `cpi_yoy` 4.0 con `cpi_yoy_change_6m` **negativo** = inflazione ALTA ma in
  DECELERAZIONE → disinflazione in arrivo, NON stagflazione persistente.
- `unrate` basso ma `unrate_roc` e `initial_claims_roc` in SALITA = il mercato del
  lavoro sta girando ORA; il livello basso è storia vecchia.
- `gdp_roc_change_6m`, `payrolls_roc_12m`, `lei_roc`, `building_permits_yoy`,
  `heavy_truck_sales_yoy` sono LEADING: girano PRIMA del ciclo. `unrate` e `cpi_yoy`
  in livello sono LAGGING: confermano ciò che è già accaduto.
- Concordanza di più ROC nella stessa direzione = conviction alta. ROC in conflitto
  = dati ambigui → prudenza, non forzare una tesi.

Un livello senza la sua derivata NON è una previsione. Se citi solo livelli, stai
guardando nello specchietto retrovisore.

**LINGUA**: campi `reason` SEMPRE in italiano professionale, brevi (max 180 char).

**UNIVERSE FISSO** (24 asset). Puoi decidere solo su questi:
us_equities_growth, us_equities_value, us_bonds_long, us_bonds_short,
tips_inflation_bonds, gold, silver, cash_money_market, broad_commodities,
energy, real_estate_reits, international_dm_equities, em_equities,
bitcoin, crypto_broad, sector_technology, sector_energy, sector_financials,
sector_utilities, sector_staples, sector_healthcare, factor_momentum,
factor_quality, factor_value

**AZIONI VALIDE**:
- OPEN_TRANCHE: apri/aggiungi tranche (specifica size_pct 1.0-8.0)
- CLOSE_TRANCHE: chiudi parziale (partial TP)
- FULL_CLOSE: chiudi tutto (regime adverse o tesi rotta)
- STOP_LOSS / TAKE_PROFIT: solo se PnL conferma
- HOLD: mantieni posizione esistente invariata
- SKIP_NO_SIGNAL: nessun signal (default per asset senza posizione)

**REGOLE FERREE**:
1. NON aprire più di 8% size per tranche
2. NON suggerire OPEN per asset con confidence < 0.30 nel tuo giudizio
3. STOP_LOSS è automatico a -10% PnL (tu segnalalo solo se vuoi forzarlo prima)
4. TAKE_PROFIT è automatico a +30% PnL
5. Massimo 12 OPEN_TRANCHE per giorno (evita overtrading)
6. Per asset SENZA posizione e SENZA OPEN: ometti dall'output (NON serve SKIP esplicito)
7. Il capitale è 100%: la somma dei pesi delle posizioni aperte NON può
   superarla. Se sei già quasi full-invested, chiudi qualcosa prima di aprire
   (gli OPEN oltre il budget residuo vengono clampati/scartati dal sistema)

**LE DUE FONTI: MACRO E PREZZI (v6)**
Hai DUE corpi di evidenza e devi usarli ENTRAMBI:
1. **Macro** (indicators FRED + i loro ROC + il regime): dice DOVE DOVREBBE andare il capitale.
2. **Prezzi reali** (Yahoo: momentum, RSI, distanza MA50, vol, rendimenti 1/3/6m,
   forza relativa vs S&P): dice dove il capitale sta ANDANDO DAVVERO, adesso.

La macro senza i prezzi ti fa comprare tesi che il mercato smentisce da mesi.
I prezzi senza la macro ti fanno inseguire rally già finiti. **Quando le due fonti
CONCORDANO hai la conviction più alta**; quando confliggono, dillo e riduci la size:
o sei presto, o una delle due letture è sbagliata.

**ROTAZIONE**: `rel_3m` (forza relativa vs S&P) è la traccia di dove si sposta il
capitale. Non esiste un dato gratis dei flussi veri dei fondi, ma la rotazione la
LEGGI nella performance relativa: chi sovraperforma sta attirando capitale.

**OUTPUT JSON STRETTO** (NO markdown):
{
  "thesis": "1 frase: dove siamo OGGI secondo i dati (max 250 char)",
  "outlook": "1 frase: dove stiamo ANDANDO nei prossimi 3-6 mesi e QUALI ROC te lo dicono (max 250 char)",
  "regime_dissent": "vuoto se concordi col regime; se dissenti, i dati che lo giustificano (max 250 char)",
  "decisions": [
    {"asset": "us_equities_growth", "action": "OPEN_TRANCHE", "score": 65,
     "confidence": 0.7, "size_pct": 4.5,
     "reason": "cpi_yoy_change_6m -0.8 + payrolls_roc_12m 1.4 in tenuta -> disinflazione senza recessione, growth beneficia"},
    {"asset": "gold", "action": "FULL_CLOSE", "score": 40, "confidence": 0.6,
     "reason": "real_rate_change_3m +0.9 e breakeven_10y_change_3m -0.3 -> tesi inflazionistica rotta, esco"}
  ]
}

**OGNI decisione va giustificata — APERTURE E CHIUSURE allo stesso modo.**
Il `reason` deve dire *perché ORA* citando gli indicator SPECIFICI (col nome) e
soprattutto i loro ROC/variazioni. Vale per OPEN_TRANCHE quanto per FULL_CLOSE,
CLOSE_TRANCHE, STOP_LOSS, TAKE_PROFIT: una chiusura senza motivo è un errore.
Le decisioni devono essere COERENTI con il tuo `outlook`: se prevedi X, il
portafoglio deve esprimere X.

Decidi come Soros: forma la TUA tesi macro leggendo gli indicators raw e le loro
derivate, poi costruisci il portafoglio coerente con dove i dati stanno ANDANDO.
Aggressivo dove i ROC concordano, prudente dove sono in conflitto. Non delegare il
pensiero al regime classifier — quella è una baseline, non la verità."""


def _compute_input_hash(
    target_date: date,
    regime: str,
    confidence: float,
    n_indicators: int,
    n_positions: int,
    model: str,
) -> str:
    canonical = {
        "date": str(target_date),
        "regime": regime,
        "confidence": round(confidence, 3),
        "n_indicators": n_indicators,
        "n_positions": n_positions,
        "model": model,
        "prompt_version": _PROMPT_VERSION,
    }
    return hashlib.md5(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


# Asset a rendimento monotòno (liquidità / T-bill): il prezzo accumula interessi e
# non "fa trend". Momentum/RSI sono DEGENERI (sempre BULLISH, RSI saturo ~95-100) e
# un LLM li legge come conviction fortissima → sovrappeso del cash per la ragione
# sbagliata. Osservato sul probe reale 2026-07-15: Gemini scrisse "cash_money_market
# ... RSI a 95, perfetto". Per questi asset mostriamo solo vol/rendimenti, MAI il
# segnale di trend.
_MONOTONE_YIELD_ASSETS = {"cash_money_market", "us_bonds_short"}


def _build_market_block(price_history: dict, benchmark_key: str = "us_equities_growth") -> str:
    """Blocco 'lettura di mercato' dai prezzi Yahoo (v6).

    FIX STRUTTURALE 2026-07-15: fino alla v5 Gemini sceglieva 24 asset senza
    vedere UN SOLO PREZZO (il price_history veniva fetchato DOPO la chiamata e
    usato solo per loggare). Qui i prezzi che già scarichiamo diventano input.
    `rel_3m` (forza relativa vs S&P) è la misura operativa della ROTAZIONE.
    """
    bench = price_history.get(benchmark_key)
    rows = []
    for asset in ASSET_CLASSES:
        snap = momentum.market_snapshot(asset, price_history, benchmark=bench)
        if not snap:
            continue

        def f(key, pct=True):
            v = snap.get(key)
            if v is None:
                return "n/d"
            return f"{v*100:+.1f}%" if pct else f"{v:.0f}"

        if asset in _MONOTONE_YIELD_ASSETS:
            # niente momentum/RSI: su questi è rumore che sembra segnale
            rows.append(
                f"  - {asset}: [strumento a rendimento: accumula interessi, "
                f"NON fa trend -> indicatori di trend omessi perche' non "
                f"significativi. Non trattarlo come un momentum play] | "
                f"vol {f('vol_60d')} | "
                f"ret 1m {f('ret_1m')} 3m {f('ret_3m')} 6m {f('ret_6m')}"
            )
            continue

        rows.append(
            f"  - {asset}: {snap.get('signal', 'n/d')} | RSI {f('rsi', pct=False)} | "
            f"vs MA50 {f('dist_ma50')} | vol {f('vol_60d')} | "
            f"ret 1m {f('ret_1m')} 3m {f('ret_3m')} 6m {f('ret_6m')} | "
            f"rel_3m vs S&P {f('rel_strength_3m')}"
        )
    return "\n".join(rows) if rows else "  (prezzi non disponibili)"


def _build_user_prompt(
    target_date: date,
    regime: str,
    probabilities: dict,
    confidence: float,
    indicators: dict,
    positions: list[AiPortfolioPosition],
    learnings: list[dict],
    price_history: dict | None = None,
) -> str:
    probs_str = ", ".join(
        f"{k}={v:.0%}" for k, v in sorted(probabilities.items(), key=lambda x: -x[1])
    )

    # Indicators raw — TUTTI quelli disponibili. Niente pre-selezione del modello.
    # Gemini decide cosa è rilevante. Ordinati per leggibilità (gruppi macro).
    key_inds_grouped = {
        "Crescita / attività": [
            "gdp_roc", "gdp_roc_change_6m", "gdp_yoy_dfm",
            "gdp_nowcast_atlanta",
            "payrolls_roc_12m", "indpro_roc_12m",
            "heavy_truck_sales_yoy", "building_permits_yoy",
            "durable_goods_orders_yoy", "consumer_credit_yoy",
            "empire_fed_activity", "philly_fed_activity",
            "dallas_fed_activity", "fed_surveys_composite",
        ],
        "Lavoro": [
            "unrate", "unrate_roc", "initial_claims_roc",
            "wage_growth_atlanta", "job_openings",
        ],
        "Inflazione": [
            "cpi_yoy", "cpi_yoy_change_6m", "core_pce_yoy",
            "sticky_cpi_yoy", "umich_inflation_1y", "breakeven_5y5y",
            "breakeven_10y", "breakeven_10y_change_3m",
            "energy_yoy", "ulc_yoy", "productivity_yoy",
        ],
        "Politica monetaria / liquidità": [
            "fed_funds_rate", "fed_funds_implied_12m", "fed_path_12m",
            "m2_yoy", "net_liquidity_usd_bn", "net_liquidity_roc_3m",
            "walcl_roc_3m", "term_premium_10y",
            "yield_curve_10y2y", "yield_curve_10y3m",
        ],
        "Stress / sentiment": [
            "vix", "vix_ma_ratio", "move_index", "nfci", "nfci_change_3m",
            "hy_credit_spread", "baa_spread",
            "consumer_sentiment", "news_sentiment",
        ],
        "Cross-asset": [
            "copper_gold_ratio", "gold_oil_ratio", "hy_ig_spread_ratio",
            "lei_roc",
        ],
        "Positioning (contrarian agli estremi |z|>2)": [
            "insider_score", "cot_bonds10y_z", "cot_gold_z",
            "cot_equity_z", "cot_usd_z",
        ],
        "Dedollarization": [
            "dedollar_combined",
        ],
        "Internazionale": [
            "ecb_main_refi_rate", "boe_bank_rate", "boj_policy_rate",
            "korea_exports_yoy",
        ],
    }
    indicator_blocks = []
    for group_name, keys in key_inds_grouped.items():
        lines = []
        for k in keys:
            v = indicators.get(k)
            if v is None:
                continue
            try:
                lines.append(f"  - {k}: {float(v):.3f}")
            except (ValueError, TypeError):
                pass
        if lines:
            indicator_blocks.append(f"### {group_name}\n" + "\n".join(lines))

    # Positions
    pos_lines = []
    for p in positions:
        days = p.days_held if hasattr(p, "days_held") else "?"
        pos_lines.append(
            f"  - {p.asset_class}: peso {p.current_weight_pct*100:.1f}%, "
            f"PnL {p.estimated_pnl_pct*100:+.1f}%, tranches {p.tranches_filled}/{p.tranches_total}, "
            f"days {days}, entry_regime {p.entry_regime}"
        )

    # Learnings (proprio della strategia llm_driven, solo con shift)
    learn_lines = []
    for L in learnings[:5]:
        if L.get("entry_threshold_shift", 0) == 0:
            continue
        learn_lines.append(
            f"  - {L['pattern_key']}: {L['n_wins']}W/{L['n_losses']}L "
            f"(win_rate {L['win_rate']:.0%}, shift {L['entry_threshold_shift']:+.1f})"
        )

    market_block = _build_market_block(price_history or {})

    return f"""## CONTESTO — {target_date}

## INDICATORS MACRO RAW (FRED, valori veri di oggi)
{chr(10).join(indicator_blocks) if indicator_blocks else "  (nessun indicator disponibile)"}

## LETTURA DI MERCATO (prezzi REALI Yahoo, i 24 asset dell'universo)
Per ogni asset: segnale momentum | RSI14 | distanza dalla MA50 | volatilità annua |
rendimenti 1m/3m/6m | **rel_3m = forza relativa vs S&P a 3 mesi**.

{market_block}

**Come usarli**: la macro ti dice DOVE dovrebbe andare il capitale, questi prezzi ti
dicono dove sta ANDANDO davvero. `rel_3m` è la tua misura di **ROTAZIONE**: se un
asset ha rel_3m molto positiva mentre l'S&P arranca, il capitale si sta spostando lì.
NON comprare una tesi macro che i prezzi smentiscono da mesi (rel_3m negativa e
momentum BEARISH = il mercato non è d'accordo con te: o sei presto, o hai torto).
Viceversa: momentum forte senza supporto macro = rischio di inseguire un rally finito.

## REGIME MACRO — classifier proprietario (input di prima classe)
Regime: **{regime.upper()}** (confidence {confidence:.0%})
Probabilità: {probs_str}

**DEVI considerarlo** nella tua tesi: è il nostro modello a soglie sul quadro macro
completo, ed è la baseline con cui il resto del sistema ragiona. Non ignorarlo.
**PUOI dissentire**, ma solo motivando coi dati: se dissenti, dillo ESPLICITAMENTE nel
campo `regime_dissent` citando gli indicator/ROC che ti portano altrove (es. "dissento:
lei_roc +0.3 e building_permits_yoy in ripresa → vedo reflation, non stagflation").
Se concordi, lascia `regime_dissent` vuoto. Un dissenso non motivato dai numeri è
rumore, non conviction.

## TUE POSIZIONI APERTE ({len(positions)})
{chr(10).join(pos_lines) if pos_lines else "  (nessuna posizione)"}

## TUOI LEARNINGS POST-MORTEM (top 5 con shift accumulato)
{chr(10).join(learn_lines) if learn_lines else "  (nessun pattern ancora maturato)"}

## RICHIESTA

1. `thesis`: dove siamo OGGI secondo i dati. 1 frase (max 250 char).
2. `outlook`: **dove stanno ANDANDO i dati nei prossimi 3-6 mesi**, e quali ROC te lo
   dicono. 1 frase (max 250 char). Guarda le derivate (`_roc`, `_change_3m/6m`, `_yoy`)
   e i leading (lei_roc, building_permits_yoy, heavy_truck_sales_yoy, gdp_roc_change_6m,
   initial_claims_roc), non i livelli lagging. Questa è la previsione su cui scommetti.
3. `regime_dissent`: vuoto se concordi col regime del classifier; se dissenti, cita i
   dati che ti portano altrove. Mai dissentire "a sensazione".
4. Per ogni asset dell'universo (24) su cui hai conviction, emetti una decision
   COERENTE con l'`outlook`. Asset senza conviction: ometti (non serve SKIP esplicito).
5. Giustifica OGNI decisione — **aperture E chiusure** — con un `reason` di 1 frase
   che incroci **MACRO e PREZZO**: cita gli indicator per nome con la loro DIREZIONE,
   E cosa dicono i prezzi (momentum / rel_3m / distanza MA50).
   - ottimo: "cpi_yoy_change_6m -0.8 + lei_roc in ripresa -> disinflazione; e i prezzi
     confermano: rel_3m +6.2% e sopra MA50"
   - debole: "CPI alto -> compro oro" (livello senza derivata, e nessuno sguardo al prezzo)
   Se chiudi, di' cosa ha ROTTO la tesi d'ingresso (macro girata? prezzo che smentisce?).

Output JSON: {{ "thesis": "...", "outlook": "...", "regime_dissent": "...", "decisions": [...] }}
"""


def _call_gemini(prompt: str, api_key: str, model: str) -> Optional[str]:
    url = _GEMINI_API_URL_TPL.format(model=model)
    try:
        resp = requests.post(
            url,
            params={"key": api_key},
            json={
                "contents": [
                    {"parts": [{"text": f"{_SYSTEM_PROMPT}\n\n---\n\n{prompt}"}]},
                ],
                "generationConfig": llm_settings.generation_config(
                    model=model, max_output_tokens=6000, temperature=0.3
                ),
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            return None
        parts = candidates[0].get("content", {}).get("parts") or []
        return "".join(p.get("text", "") for p in parts).strip() or None
    except Exception as e:
        logger.warning(f"LLM Strategist Gemini call failed (model={model}): {e}")
        return None


def _fetch_price_history_for_universe() -> dict[str, pd.Series]:
    from datetime import timedelta
    from app.services.prices.yahoo_fetcher import YahooFetcher

    fetcher = YahooFetcher()
    out: dict[str, pd.Series] = {}
    end = date.today()
    start = end - timedelta(days=200)
    for asset in ASSET_CLASSES:
        ticker = momentum.asset_to_ticker(asset)
        if not ticker:
            continue
        try:
            s = fetcher.fetch(ticker, start_date=start, end_date=end)
            if s is not None and not s.empty:
                out[asset] = s.dropna()
        except Exception as e:
            logger.warning(f"llm_strategist price fetch failed for {asset}: {e}")
    return out


def _refresh_real_pnl(position: AiPortfolioPosition) -> float:
    current_price = pnl_tracker.get_latest_price(position.asset_class)
    if current_price is None:
        return position.estimated_pnl_pct
    if position.avg_entry_price is None or position.avg_entry_price == 0:
        position.avg_entry_price = current_price
        position.current_price = current_price
        return 0.0
    position.current_price = current_price
    return pnl_tracker.compute_pnl_pct(position.avg_entry_price, current_price)


def daily_scan(
    db: Session,
    target_date: date | None = None,
) -> dict:
    """Daily scan strategia LLM-driven (Gemini decisore)."""
    target_date = target_date or date.today()

    # 1. Regime + indicators
    latest = (
        db.query(RegimeClassification)
        .order_by(RegimeClassification.date.desc())
        .first()
    )
    if latest is None:
        return {"error": "no regime classification"}

    regime = latest.regime
    confidence = float(latest.confidence or 0.5)
    probabilities = {
        "reflation": float(latest.probability_reflation or 0),
        "stagflation": float(latest.probability_stagflation or 0),
        "deflation": float(latest.probability_deflation or 0),
        "goldilocks": float(latest.probability_goldilocks or 0),
    }
    indicators = {}
    try:
        if latest.conditions_met:
            payload = json.loads(latest.conditions_met)
            indicators = payload.get("indicators", {}) or {}
    except Exception:
        pass

    # v2-independent 2026-06-13: rimossi model/data scoreboard dal prompt per
    # eliminare anchoring bias. Gemini riceve solo indicators raw + regime come
    # second opinion. Decide cosa è rilevante per ogni asset autonomamente.

    # 2. Positions LLM-driven + PnL reale
    existing = {
        p.asset_class: p
        for p in db.query(AiPortfolioPosition)
        .filter(AiPortfolioPosition.strategy_type == STRATEGY_TYPE)
        .all()
    }
    for p in existing.values():
        p.estimated_pnl_pct = _refresh_real_pnl(p)
        # Augment days_held attribute on the fly per prompt
        try:
            p.days_held = (target_date - p.opened_at).days
        except Exception:
            p.days_held = 0

    # 5. Learnings llm_driven
    learnings = learning.list_top_learnings(db, limit=10, strategy_type=STRATEGY_TYPE)

    # 6. Hash + skip se invariato (anti double-call)
    model = llm_settings.get_model()
    h = _compute_input_hash(target_date, regime, confidence, len(indicators), len(existing), model)

    # Check se già fatto oggi
    already_today = (
        db.query(AiPortfolioDecision)
        .filter(
            AiPortfolioDecision.date == target_date,
            AiPortfolioDecision.strategy_type == STRATEGY_TYPE,
        )
        .first()
    )
    if already_today:
        return {"skipped": "already scanned today", "hash": h}

    api_key = llm_settings.get_api_key()
    if not api_key:
        logger.warning("LLM Strategist: no Gemini key, skipping day")
        return {"skipped": "no api key"}

    # 7. Prezzi PRIMA della chiamata (v6 FIX): fino alla v5 il fetch stava DOPO e
    # Gemini decideva cieco sui prezzi — li usavamo solo per loggare la decisione.
    price_history = _fetch_price_history_for_universe()

    prompt = _build_user_prompt(
        target_date=target_date,
        regime=regime,
        probabilities=probabilities,
        confidence=confidence,
        indicators=indicators,
        positions=list(existing.values()),
        learnings=learnings,
        price_history=price_history,
    )
    raw = _call_gemini(prompt, api_key=api_key, model=model)
    if raw is None:
        return {"skipped": "llm call failed"}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning(f"LLM Strategist invalid JSON: {e}, raw: {raw[:200]}")
        return {"skipped": "invalid llm json"}

    raw_decisions = parsed.get("decisions") or []
    if not isinstance(raw_decisions, list):
        return {"skipped": "decisions not list"}

    # 8. (price_history già fetchato al punto 7: serve al prompt, non solo al log)

    summary = {
        "n_decisions": 0, "n_open_tranches": 0, "n_closures": 0,
        "thesis": str(parsed.get("thesis", ""))[:300],
        # v5: la PREVISIONE 3-6m su cui il bot scommette (non solo la foto di oggi)
        "outlook": str(parsed.get("outlook", ""))[:300],
        # v6: se dissente dal regime del classifier, DEVE dire con quali dati
        "regime_dissent": str(parsed.get("regime_dissent", ""))[:300],
    }

    # Cap totale OPEN al giorno (anti-overtrading)
    MAX_OPEN_PER_DAY = 12
    opens_today = 0

    # AUDIT 2026-07-12 (CRITICAL): budget globale 100% del capitale. Il cap
    # era solo per-tranche (8%) e per-giorno (12 open) — le tranche si
    # accumulavano nei giorni oltre il capitale (live: deployed 136% = leva
    # implicita che gonfia il NAV e falsa l'horserace a 3). Clamp di ogni
    # OPEN al budget residuo; skip se resta < 0.5%.
    deployed_total = sum(
        float(p.current_weight_pct or 0.0) for p in existing.values()
    )

    for raw_d in raw_decisions[:50]:
        if not isinstance(raw_d, dict):
            continue
        asset = str(raw_d.get("asset", "")).strip()
        action = str(raw_d.get("action", "")).strip().upper()
        if asset not in ASSET_CLASSES or action not in VALID_ACTIONS:
            continue

        # Clamping
        try:
            llm_score = float(raw_d.get("score", 50))
        except (ValueError, TypeError):
            llm_score = 50.0
        llm_score = max(0.0, min(100.0, llm_score))

        try:
            llm_conf = float(raw_d.get("confidence", 0.5))
        except (ValueError, TypeError):
            llm_conf = 0.5
        llm_conf = max(0.0, min(1.0, llm_conf))

        try:
            size_pct = float(raw_d.get("size_pct", 0)) / 100.0  # arrivava in %, normalizziamo
        except (ValueError, TypeError):
            size_pct = 0.0
        size_pct = max(0.0, min(MAX_SIZE_PCT_PER_TRANCHE, size_pct))

        reason = str(raw_d.get("reason", ""))[:300]
        position = existing.get(asset)
        sig, vol = momentum.get_asset_signal_and_vol(asset, price_history)

        # Safety rails
        if action == "OPEN_TRANCHE":
            if opens_today >= MAX_OPEN_PER_DAY:
                continue
            if llm_conf < MIN_CONFIDENCE:
                continue
            if size_pct <= 0:
                continue
            # Budget globale: mai oltre il 100% del capitale (AUDIT 2026-07-12)
            budget_left = 1.0 - deployed_total
            if budget_left < 0.005:
                continue
            size_pct = min(size_pct, budget_left)
            deployed_total += size_pct
            opens_today += 1
        elif action in ("CLOSE_TRANCHE", "FULL_CLOSE", "STOP_LOSS", "TAKE_PROFIT"):
            if position is None:
                continue  # niente da chiudere
        # HOLD/SKIP_NO_SIGNAL: log per audit ma niente effetto stato

        # Log decision
        dec = AiPortfolioDecision(
            date=target_date,
            asset_class=asset,
            strategy_type=STRATEGY_TYPE,
            action=action,
            size_pct=size_pct,
            tranche_index=(position.tranches_filled + 1) if (position and action == "OPEN_TRANCHE") else (1 if action == "OPEN_TRANCHE" else 0),
            tranches_total=(position.tranches_total if position else 3),
            score=llm_score,
            confidence=llm_conf,
            regime=regime,
            momentum_signal=sig,
            vol_60d=vol,
            estimated_pnl_pct=(position.estimated_pnl_pct if position else None),
            reason_short=reason,
        )
        db.add(dec)
        summary["n_decisions"] += 1

        _apply(db, asset, position, action, size_pct, llm_score, regime, target_date, summary)

    db.commit()
    summary["n_positions_after"] = (
        db.query(AiPortfolioPosition)
        .filter(AiPortfolioPosition.strategy_type == STRATEGY_TYPE)
        .count()
    )
    return summary


def _apply(
    db: Session,
    asset: str,
    position: Optional[AiPortfolioPosition],
    action: str,
    size: float,
    score: float,
    regime: str,
    target_date: date,
    summary: dict,
) -> None:
    if action == "OPEN_TRANCHE":
        entry_price = pnl_tracker.get_latest_price(asset)
        if position is None:
            db.add(AiPortfolioPosition(
                asset_class=asset,
                strategy_type=STRATEGY_TYPE,
                target_weight_pct=size * 3,  # implicit target = 3 tranche
                current_weight_pct=size,
                tranches_filled=1,
                tranches_total=3,
                opened_at=target_date,
                entry_regime=regime,
                avg_entry_score=score,
                avg_entry_price=entry_price,
                current_price=entry_price,
                estimated_pnl_pct=0.0,
                took_partial_tp=False,
                last_action="OPEN_TRANCHE",
                last_action_date=target_date,
            ))
            summary["n_open_tranches"] += 1
        else:
            position.current_weight_pct += size
            n_before = position.tranches_filled
            position.tranches_filled = min(position.tranches_filled + 1, position.tranches_total)
            n = position.tranches_filled
            position.avg_entry_score = (position.avg_entry_score * (n - 1) + score) / n
            if entry_price is not None:
                position.avg_entry_price = pnl_tracker.update_avg_entry_price(
                    position.avg_entry_price, n_before, entry_price,
                )
                position.current_price = entry_price
            position.last_action = "OPEN_TRANCHE"
            position.last_action_date = target_date
            summary["n_open_tranches"] += 1

    elif action == "CLOSE_TRANCHE":
        if position is None:
            return
        per_tranche = position.current_weight_pct / max(position.tranches_filled, 1)
        position.current_weight_pct -= per_tranche
        position.tranches_filled = max(0, position.tranches_filled - 1)
        position.took_partial_tp = True
        position.last_action = "CLOSE_TRANCHE"
        position.last_action_date = target_date
        if position.current_weight_pct <= 0.001:
            db.delete(position)
        summary["n_closures"] += 1

    elif action in ("FULL_CLOSE", "STOP_LOSS", "TAKE_PROFIT"):
        if position is not None:
            try:
                last_decision = AiPortfolioDecision(
                    date=target_date,
                    asset_class=asset,
                    strategy_type=STRATEGY_TYPE,
                    action=action,
                    momentum_signal="NEUTRAL",
                    score=score,
                    confidence=0.0,
                    regime=regime,
                    reason_short="llm-closed",
                )
                learning.record_closed_position(db, last_decision, position)
            except Exception as e:
                logger.warning(f"llm learning record failed: {e}")
            db.delete(position)
            summary["n_closures"] += 1
