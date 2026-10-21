"""Il "perché" di ogni operazione del libro reale, come lo scriverebbe un trader.

Due strati, e l'ordine conta:
1. `why_template` — DETERMINISTICO, costruito solo dalle cause vere della decisione
   (regola scattata, punteggio, regime, condizioni macro, motivo scritto dal bot).
   Esiste sempre: è anche il ripiego.
2. `llm_why_batch` — Gemini riscrive il template in prosa più umana. Passa SOLO se supera
   la guardia `check_narrative`: nessun numero che non stia nei fatti, nessun nome di
   strumento/emittente/ticker/ISIN (il diario pubblico parla di asset class: scelta
   Erik 2026-10-21, per non scrivere raccomandazioni su strumenti specifici).

Perché la guardia sta nel codice e non nel prompt: un racconto scritto DOPO la decisione
tende ad aggiungere cause plausibili che la decisione non aveva. Successo già in fase di
progetto: la prima bozza diceva «RSI alto, per questo non aggiungo altro», ma il bot non
l'aveva mai detto. La guardia sui numeri non prende tutte le cause inventate; prende
quelle con un numero, che sono le più convincenti e quindi le più pericolose.

Il "come" e il "quando" (quote, prezzo, ora, commissione) NON passano dal modello: li
compone il sistema dai campi del movimento.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import requests

from app.services.real_book.instruments import INSTRUMENTS

logger = logging.getLogger(__name__)

REGIME_IT = {
    "reflation": "reflazione",
    "stagflation": "stagflazione",
    "deflation": "deflazione",
    "goldilocks": "goldilocks (crescita senza inflazione)",
}
MOMENTUM_IT = {"BULLISH": "rialzista", "BEARISH": "ribassista", "NEUTRAL": "neutro"}
STRATEGY_IT = {
    "model_driven": "bot Modello",
    "llm_driven": "bot LLM",
    "risk_managed_sp": "bot a rischio gestito",
}

# Sigle che i bot scrivono nei motivi → parole. Solo per il ripiego deterministico:
# la versione Gemini le traduce da sola (e la guardia controlla i numeri, non le parole).
_SIGLE = [
    ("cpi_yoy_change_6m", "variazione dell'inflazione in 6 mesi"),
    ("cpi_yoy", "inflazione annua (CPI)"),
    ("consumer_sentiment", "fiducia dei consumatori"),
    ("hy_credit_spread", "spread dei bond high yield"),
    ("net liquidity roc_3m", "variazione a 3 mesi della liquidità netta"),
    ("net_liquidity", "liquidità netta"),
    ("fed_path", "percorso atteso dei tassi Fed"),
    ("gdp_roc", "crescita del PIL"),
    ("unrate", "disoccupazione"),
    ("rel_3m", "forza relativa a 3 mesi"),
    ("rel_1m", "forza relativa a 1 mese"),
    ("momentum BULLISH", "momentum rialzista"),
    ("momentum BEARISH", "momentum ribassista"),
    ("MA50", "media a 50 giorni"),
    ("MA200", "media a 200 giorni"),
]

_BRANDS = ("invesco", "ishares", "spdr", "state street", "vanguard", "xtrackers",
           "wisdomtree", "vaneck", "bitwise", "amundi")


def _n(x: float | None, d: int = 1) -> str:
    if x is None:
        return "n.d."
    return f"{x:.{d}f}".replace(".", ",")


def humanize(text: str | None) -> str:
    out = (text or "").strip().rstrip(".")
    for sigla, parola in _SIGLE:
        out = re.sub(re.escape(sigla), parola, out, flags=re.IGNORECASE)
    return out


def _regime_sentence(f: dict) -> str:
    s = (f"Il regime macro dominante è la {f.get('regime_it', 'n.d.')} "
         f"({_n(f.get('prob_regime_pct'))}% di probabilità")
    if f.get("regime_alternativo_it"):
        s += f", contro {f['regime_alternativo_it']} al {_n(f.get('prob_alternativo_pct'))}%"
    s += f"; fiducia del modello {_n(f.get('fiducia_pct'), 0)}%)."
    ind = f.get("indicatori") or []
    if ind:
        s += " I dati: " + ", ".join(f"{i['nome']} {_n(i['valore'], 2)}{i.get('unita', '')}" for i in ind[:4]) + "."
    return s


def why_template(f: dict) -> str:
    kind = f.get("evento")
    a = f.get("asset_nome", "l'asset")
    strat = f.get("strategy") or ""
    motivo = humanize(f.get("motivo_del_bot"))

    if kind == "presa_in_carico":
        s = (f"Apro il libro reale: compro la posizione su {a} che il {STRATEGY_IT.get(strat, 'bot')} "
             f"ha già in portafoglio ({f.get('tranche')}/{f.get('tranche_totali')} tranche, aperta il "
             f"{f.get('aperta_il')} in regime di {f.get('regime_di_ingresso_it', 'n.d.')}). Non è un segnale "
             f"nuovo: allineo il portafoglio vero a quello simulato.")
        if motivo:
            s += f" Il motivo dell'ultimo acquisto, nelle parole del bot: «{motivo}»."
        return s
    if kind == "floor":
        return (f"Il {STRATEGY_IT.get(strat, 'bot')} aveva meno del {_n(f.get('floor_pct'), 0)}% del capitale "
                f"investito: la sua regola di investimento minimo schiera la differenza sugli asset col "
                f"punteggio migliore fra quelli favoriti dal regime. {a} ha punteggio {_n(f.get('punteggio'))}/100. "
                f"Questa regola non guarda l'andamento del prezzo.")
    if kind == "riallineamento":
        return ("La posizione del bot è cambiata senza una decisione registrata: riallineo il libro "
                "invece di lasciarlo divergere in silenzio.")

    buy = f.get("lato") == "acquisto"
    k, tot = f.get("tranche"), f.get("tranche_totali")

    if strat == "model_driven":
        if buy:
            if (k or 1) <= 1:
                return (f"{_regime_sentence(f)} In questo regime {a} è fra gli asset storicamente favoriti, "
                        f"il punteggio è {_n(f.get('punteggio'))}/100 (soglia d'ingresso {_n(f.get('soglia_ingresso'), 0)}) "
                        f"e il prezzo è in trend rialzista: sopra la media a 50 giorni con RSI sopra 50. "
                        f"Entro con la prima di {tot} tranche.")
            return (f"Il punteggio di {a} resta {_n(f.get('punteggio'))}/100, sopra la soglia per aggiungere: "
                    f"compro la tranche {k} di {tot}. {_regime_sentence(f)}")
        pnl = f.get("pnl_pct")
        if kind == "stop_loss":
            return (f"Stop: la posizione su {a} perde il {_n(abs(pnl or 0))}% dal prezzo medio "
                    f"(limite {_n(abs(f.get('stop_pct') or 0), 0)}%). Esco per regola, senza discutere.")
        if kind == "take_profit":
            return (f"Obiettivo raggiunto su {a}: +{_n(pnl)}% dal prezzo medio (soglia +{_n(f.get('tp_pct'), 0)}%). "
                    f"Incasso tutto.")
        if kind == "close_tranche":
            return (f"{a} è a +{_n(pnl)}% dal prezzo medio, oltre il +{_n(f.get('tp_parziale_pct'), 0)}%: "
                    f"incasso una tranche e lascio correre il resto.")
        if "regime" in (f.get("motivo_del_bot") or "").lower():
            return (f"Con il regime di {f.get('regime_it', 'n.d.')} {a} non è più fra gli asset favoriti: "
                    f"chiudo la posizione ({_n(pnl)}% dal prezzo medio).")
        if "score" in (f.get("motivo_del_bot") or "").lower():
            return (f"Il punteggio di {a} è sceso a {_n(f.get('punteggio'))}/100, sotto la soglia d'uscita "
                    f"{_n(f.get('soglia_uscita'), 0)}: il segnale si è spento, chiudo ({_n(pnl)}% dal prezzo medio).")
        return f"Chiudo {a}: {motivo or 'decisione del bot'}."

    if strat == "llm_driven":
        reg = (f" Il classificatore indica {f.get('regime_it', 'n.d.')} ({_n(f.get('prob_regime_pct'))}%), "
               f"che il bot tratta come seconda opinione.")
        if buy:
            return f"Il bot LLM motiva così: {motivo or 'n.d.'}. Tranche {k} di {tot}.{reg}"
        return f"Il bot LLM chiude {a}: {motivo or 'n.d.'}."

    if strat == "risk_managed_sp":
        if buy and f.get("asset") == "us_equities_growth":
            return ("Il Nasdaq-100, che questo bot usa come termometro del mercato azionario, è sopra la "
                    "sua media a 200 giorni: sto sulle azioni al 100%.")
        if buy:
            return (f"Il Nasdaq-100 è sotto la media a 200 giorni: esco dalle azioni e mi rifugio su {a}, "
                    f"il migliore degli ultimi 3 mesi fra titoli di Stato lunghi, oro e liquidità.")
        return f"Cambio posizione: {motivo or 'il trend è girato'}."

    return motivo or "Decisione del bot."


def exit_plan(f: dict) -> str | None:
    if f.get("lato") != "acquisto":
        return None
    strat, a = f.get("strategy"), f.get("asset_nome", "l'asset")
    if strat == "model_driven":
        # la nota "soglie misurate in dollari sull'ETF americano" sta UNA volta nel pannello, non in ogni scheda
        return (f"Stop a −{_n(abs(f.get('stop_pct') or 0), 0)}% dal prezzo medio · incasso una tranche a "
                f"+{_n(f.get('tp_parziale_pct'), 0)}% e tutto a +{_n(f.get('tp_pct'), 0)}% · chiudo se il punteggio "
                f"scende sotto {_n(f.get('soglia_uscita'), 0)} o se il regime smette di favorirlo.")
    if strat == "llm_driven":
        return "Nessuna regola fissa: l'uscita la decide il bot LLM giorno per giorno sui dati."
    if strat == "risk_managed_sp":
        if f.get("asset") == "us_equities_growth":
            return "Esco quando il Nasdaq-100 scende sotto la media a 200 giorni."
        return "Rientro sulle azioni quando il Nasdaq-100 torna sopra la media a 200 giorni."
    return None


# ---------------------------------------------------------------------------
# Guardia
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"(?<![\w.,])[-+−]?\d+(?:[.,]\d+)?")


def _numbers(text: str) -> list[float]:
    out = []
    for m in _NUM_RE.findall(text or ""):
        try:
            out.append(abs(float(m.replace("−", "-").replace(",", "."))))
        except ValueError:
            pass
    return out


def forbidden_tokens(instrument_key: str) -> list[str]:
    inst = INSTRUMENTS.get(instrument_key)
    toks = list(_BRANDS)
    if inst:
        toks += [inst.key, inst.ticker, inst.isin, inst.yahoo.split(".")[0]]
    return [t for t in toks if t]


def check_narrative(text: str, allowed_source: str, forbidden: list[str]) -> tuple[bool, str | None]:
    """(ok, motivo del rifiuto). `allowed_source` = fatti + bozza, serializzati."""
    if not text or len(text.strip()) < 20:
        return False, "testo vuoto"
    low = text.lower()
    for tok in forbidden:
        if re.search(rf"(?<![a-z0-9]){re.escape(tok.lower())}(?![a-z0-9])", low):
            return False, f"nomina uno strumento ({tok})"
    allowed = _numbers(allowed_source)
    for n in _numbers(text):
        if not any(abs(n - a) <= max(0.06, 0.012 * a) for a in allowed):
            return False, f"numero {_n(n, 2)} non nei fatti"
    return True, None


# ---------------------------------------------------------------------------
# Gemini (graceful: se manca la chiave o la chiamata fallisce resta il template)
# ---------------------------------------------------------------------------

_URL_TPL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

_SYSTEM = """Sei un gestore che scrive il diario delle proprie operazioni per un lettore non tecnico.
Per ogni operazione riscrivi la BOZZA in 2-3 frasi, prima persona, italiano semplice e concreto.
REGOLE (vincolanti):
1. Usa SOLO i fatti forniti. Nessuna causa, previsione, giudizio o numero che non sia nei fatti o nella bozza.
2. Puoi arrotondare un numero, mai inventarne uno. Niente date, importi in euro o quantità: li scrive il sistema.
3. Traduci le sigle tecniche in parole (es. cpi_yoy_change_6m = variazione dell'inflazione in 6 mesi).
4. Non nominare strumenti, emittenti, ticker o ISIN: parla dell'asset (es. "l'oro", "il settore energia USA").
5. Se il motivo è una regola meccanica (stop, soglia, presa in carico, investimento minimo), dillo come regola.
Output: JSON puro {"<id>": "testo", ...}, nessun markdown."""


def _call_gemini(prompt: str) -> str | None:
    from app.services.llm import settings as llm_settings

    api_key = llm_settings.get_api_key()
    if not api_key:
        return None
    model = llm_settings.get_model()
    try:
        resp = requests.post(
            _URL_TPL.format(model=model),
            params={"key": api_key},
            json={
                "contents": [{"parts": [{"text": f"{_SYSTEM}\n\n---\n\n{prompt}"}]}],
                "generationConfig": llm_settings.generation_config(model=model, max_output_tokens=4000),
            },
            timeout=90,
        )
        resp.raise_for_status()
        parts = ((resp.json().get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
        return "".join(p.get("text", "") for p in parts).strip() or None
    except Exception as e:
        logger.warning(f"real_book narrator: Gemini fallito ({model}): {e}")
        return None


def _parse_json(raw: str) -> dict[str, Any]:
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s[s.find("{"):]
    try:
        return json.loads(s[s.find("{"): s.rfind("}") + 1])
    except Exception:
        return {}


def llm_why_batch(items: list[dict], model_label: str | None = None) -> dict[str, tuple[str, str]]:
    """items: [{id, facts, draft, forbidden}] → {id: (testo, fonte)} solo per i testi che
    passano la guardia. Chi non c'è nel risultato tiene il template."""
    if not items:
        return {}
    payload = [{"id": str(it["id"]), "fatti": it["facts"], "bozza": it["draft"]} for it in items]
    raw = _call_gemini(json.dumps(payload, ensure_ascii=False))
    if not raw:
        return {}
    parsed = _parse_json(raw)
    from app.services.llm import settings as llm_settings

    model = model_label or llm_settings.get_model()
    out: dict[str, tuple[str, str]] = {}
    for it in items:
        text = parsed.get(str(it["id"]))
        if not isinstance(text, str):
            continue
        source = json.dumps(it["facts"], ensure_ascii=False) + " " + it["draft"]
        ok, why = check_narrative(text, source, it["forbidden"])
        if ok:
            out[str(it["id"])] = (text.strip(), model)
        else:
            logger.info(f"real_book narrator: testo {it['id']} scartato dalla guardia: {why}")
            out[str(it["id"])] = ("", f"template (guardia: {why})")
    return out
