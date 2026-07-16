"""CYCLE CLOCK (S46) — Investment Clock: proietta i 4 regimi del classifier su un
orologio del ciclo economico (assi crescita x inflazione) e li collega alle classi
di prodotto.

VISTA INFORMATIVA, NON un trader (assodato S43-S45: regime/flusso non batte QQQ
long-only). Serve a leggere "in che punto del ciclo siamo".

Geometria (verdetto finance_council 2026-07-16): i 4 regimi = i 4 quadranti su
assi crescita(Y) x inflazione(X), rotazione ORARIA del ciclo classico:

    Recovery (crescita+ inflazione-)  ->  Overheat (crescita+ inflazione+)
       [Goldilocks]  TL                      [Reflation]  TR
              ^  y=growth                                 |
              |                                           v
    Reflation/Recessione (cr- infl-)  <-  Stagflation (cr- infl+)
       [Deflation]   BL                      [Stagflation] BR

FIX NOMI (bloccante, council): la nostra etichetta "Reflation" (TR, crescita+
inflazione+) COLLIDE con l'Investment Clock ML dove "Reflation" e' il quadrante
OPPOSTO (BL, recessione/bond). Qui il nome CANONICO ML e' primario (`ml_name`),
il regime del nostro classifier e' il sottotitolo (`regime`).

Corner (x=inflazione, y=crescita), ognuno in [-1,+1]:
  goldilocks  (-1, +1)   reflation   (+1, +1)
  deflation   (-1, -1)   stagflation (+1, -1)
Posizione lancetta = Sigma prob_regime * corner  (media pesata -> dentro il quadrato).
"""
from __future__ import annotations

import math

from app.services.backtest.flow_meta import META

REGIMES = ("reflation", "stagflation", "deflation", "goldilocks")

# corner di ogni regime su (x=inflazione, y=crescita)
_CORNER: dict[str, tuple[float, float]] = {
    "goldilocks": (-1.0, 1.0),
    "reflation": (1.0, 1.0),
    "stagflation": (1.0, -1.0),
    "deflation": (-1.0, -1.0),
}

# metadati delle 4 fasi: nome ML canonico + regime del classifier + leader teorici.
# Leader = classi di prodotto che storicamente guidano nella fase (mappa ML Merrill
# Lynch, layer TEORICO da confrontare col RRG osservato). Chiavi = asset di flow_meta.
PHASE_META: list[dict] = [
    {
        "key": "recovery", "regime": "goldilocks",
        "ml_name": "Recovery", "it": "Ripresa",
        "quadrant": "tl", "corner": [-1.0, 1.0],
        "desc": "Crescita in accelerazione, inflazione bassa. Storicamente guidano le AZIONI.",
        "leaders": ["us_equities_growth", "sector_technology", "sector_financials",
                    "us_equities_value", "factor_momentum"],
    },
    {
        "key": "overheat", "regime": "reflation",
        "ml_name": "Overheat", "it": "Surriscaldamento",
        "quadrant": "tr", "corner": [1.0, 1.0],
        "desc": "Crescita forte, inflazione in salita. Storicamente guidano le COMMODITIES.",
        "leaders": ["broad_commodities", "energy", "sector_energy",
                    "us_equities_value", "silver"],
    },
    {
        "key": "stagflation", "regime": "stagflation",
        "ml_name": "Stagflation", "it": "Stagflazione",
        "quadrant": "br", "corner": [1.0, -1.0],
        "desc": "Crescita debole, inflazione alta. Storicamente guidano CASH e ORO (difensivi).",
        "leaders": ["cash_money_market", "gold", "sector_staples",
                    "sector_utilities", "sector_healthcare"],
    },
    {
        "key": "reflation", "regime": "deflation",
        "ml_name": "Reflation", "it": "Reflation / Recessione",
        "quadrant": "bl", "corner": [-1.0, -1.0],
        "desc": "Crescita negativa, inflazione bassa. Storicamente guidano i BOND (duration).",
        "leaders": ["us_bonds_long", "tips_inflation_bonds", "gold"],
    },
]

_PHASE_BY_REGIME = {p["regime"]: p for p in PHASE_META}
_PHASE_BY_KEY = {p["key"]: p for p in PHASE_META}


def _norm_probs(probs: dict) -> dict:
    """Estrae i 4 regimi, clampa >=0, rinormalizza a somma 1 (fail-safe)."""
    p = {r: max(0.0, float(probs.get(r, 0.0))) for r in REGIMES}
    tot = sum(p.values())
    if tot <= 1e-12:
        return {r: 0.25 for r in REGIMES}
    return {r: v / tot for r, v in p.items()}


def _entropy_sharpness(p: dict) -> float:
    """1 - entropia normalizzata in [0,1]: 1 = prob concentrate (ago netto),
    0 = 4 regimi equiprobabili (ago sfumato)."""
    h = 0.0
    for v in p.values():
        if v > 1e-12:
            h -= v * math.log(v)
    h_norm = h / math.log(len(p))  # log(4)
    return round(max(0.0, min(1.0, 1.0 - h_norm)), 4)


def project(probs: dict, confidence: float | None = None) -> dict:
    """Proietta le probabilita' dei 4 regimi sull'orologio del ciclo.

    Ritorna: x/y in [-1,1] (inflazione, crescita), angolo, raggio, fase (quadrante
    della lancetta), regime dominante, sharpness (opacita' lancetta = confidence),
    transizione (bool: posizione incerta).
    """
    p = _norm_probs(probs)
    x = sum(p[r] * _CORNER[r][0] for r in REGIMES)  # inflazione
    y = sum(p[r] * _CORNER[r][1] for r in REGIMES)  # crescita
    radius = math.hypot(x, y)
    angle = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0

    # fase dalla posizione della lancetta (quadrante di x,y). Sul confine -> usa il
    # regime dominante come tie-break (coerenza con la card regime del dashboard).
    top_regime = max(p, key=p.get)
    top_prob = p[top_regime]
    if abs(x) < 1e-9 and abs(y) < 1e-9:
        phase = _PHASE_BY_REGIME[top_regime]
    else:
        if x >= 0 and y >= 0:
            phase = _PHASE_BY_REGIME["reflation"]   # TR Overheat
        elif x >= 0 and y < 0:
            phase = _PHASE_BY_REGIME["stagflation"]  # BR
        elif x < 0 and y < 0:
            phase = _PHASE_BY_REGIME["deflation"]   # BL Reflation/Recessione
        else:
            phase = _PHASE_BY_REGIME["goldilocks"]  # TL Recovery

    sharpness = _entropy_sharpness(p)
    conf = float(confidence) if confidence is not None else sharpness
    # transizione: nessun regime chiaro (top < 0.40) o confidence bassa (< 0.30)
    transition = bool(top_prob < 0.40 or conf < 0.30)

    return {
        "x": round(x, 4), "y": round(y, 4),
        "angle": round(angle, 1), "radius": round(radius, 4),
        "phase": phase["key"], "phase_ml": phase["ml_name"],
        "top_regime": top_regime, "top_prob": round(top_prob, 4),
        "sharpness": sharpness, "confidence": round(conf, 4),
        "transition": transition,
        "probs": {r: round(p[r], 4) for r in REGIMES},
    }


def _leaders_labels(keys: list[str]) -> list[dict]:
    return [{"asset": k, "label": META.get(k, {}).get("label", k)} for k in keys]


def phase_meta_payload() -> list[dict]:
    """PHASE_META con label leggibili dei leader (per il frontend)."""
    out = []
    for p in PHASE_META:
        out.append({**p, "leaders": _leaders_labels(p["leaders"])})
    return out


def compute_cycle_timeline(
    regime_frame,
    observed_leaders_by_week: dict[str, list[str]] | None = None,
    n_weeks: int = 52,
) -> dict:
    """Serie storica settimanale della posizione sull'orologio.

    Args:
        regime_frame: DataFrame index=venerdi', cols = p_<regime> + (opz) confidence.
            = WeeklyRegimeGrid.frame.
        observed_leaders_by_week: {week_iso: [asset in quadrante RRG 'leading']} dal
            flow timeline, per il confronto teoria<->osservato. Opzionale.
        n_weeks: quante settimane finali tenere.

    Ritorna: {weeks, frames:[{week, ...project, observed_leaders, agreement,
              anomalies}], phase_meta}. NO future leak (ogni riga = griglia <= quel
              venerdi', gia' garantito da build_weekly_grid).
    """
    obs = observed_leaders_by_week or {}
    if regime_frame is None or len(regime_frame) == 0:
        return {"weeks": [], "frames": [], "phase_meta": phase_meta_payload()}

    rows = regime_frame.sort_index()
    tail = rows.iloc[-n_weeks:] if n_weeks and len(rows) > n_weeks else rows

    frames = []
    for idx, row in tail.iterrows():
        week = idx.date().isoformat() if hasattr(idx, "date") else str(idx)
        probs = {r: row.get(f"p_{r}", 0.0) for r in REGIMES}
        conf = row.get("confidence")
        pr = project(probs, confidence=conf)

        # confronto teoria (leader della fase della LANCETTA, = quella mostrata) <-> RRG osservato
        theory = set(_PHASE_BY_KEY[pr["phase"]]["leaders"])
        observed = [a for a in obs.get(week, []) if a in META]
        obs_set = set(observed)
        agreement = sorted(theory & obs_set)               # confermano la fase
        anomalies = sorted(obs_set - theory)               # guidano ma non "dovrebbero"

        frames.append({
            "week": week,
            **pr,
            "observed_leaders": [{"asset": a, "label": META[a]["label"]} for a in observed],
            "agreement": [{"asset": a, "label": META[a]["label"]} for a in agreement],
            "anomalies": [{"asset": a, "label": META[a]["label"]} for a in anomalies],
        })

    return {
        "weeks": [f["week"] for f in frames],
        "frames": frames,
        "phase_meta": phase_meta_payload(),
    }
