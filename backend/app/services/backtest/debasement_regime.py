"""Meta-regime Debasement (R2 DREAM) — l'overlay monetario sopra l'Investment Clock.

Il Debasement e' un regime *monetario* (orizzonte lento, sub-fasi su anni) distinto dai
4 regimi *ciclici economici* (reflation/stagflation/deflation/goldilocks, gia' sull'orologio
S46). Coerente con la regola S46 "orizzonti diversi = widget separati", NON lo fondiamo in
una 5a probabilita' comparabile testa-a-testa: e' un'INTENSITA' 0-1 derivata dal segnale
divergenza oro/tassi reali (R1), con:

  - sotto-fasi Early / Mature / Late lette dalla traiettoria della divergenza;
  - isteresi ASIMMETRICA (facile accendersi, difficile spegnersi): Vito Lops nel video
    dice che il sistema "e' molto rigido per evitare falsi rumori, preferisce restare in
    debasement". Qui: CONFIRM_ON mesi sopra soglia per accendere, CONFIRM_OFF (> CONFIRM_ON)
    mesi sotto per spegnere.

Vista informativa, non trading: nessuna raccomandazione, solo misura (spec R2).

CALIBRAZIONE (R4, scripts/debasement_calibration.py): la sigmoid intensità è
DATA-GROUNDED sulla distribuzione empirica del gap 2008+ (center=mediana +7.9%,
scale=mezzo-IQR). Le soglie ON/OFF + conferme NON sono fit-su-outcome (sarebbe
overfitting: solo ~3 episodi, e sconfinerebbe nel market-timing) ma sono state
VALIDATE come plateau di robustezza: il verdetto "acceso ORA" è stabile su tutte
le 120 combinazioni di soglie testate; i valori scelti stanno al centro del plateau.
Limite dichiarato: TIPS dal 2003 → picchi pre-2003 (1980-81) non riproducibili.
"""
from __future__ import annotations

import math

# Soglie intensita' + conferma (centro del plateau di robustezza, R4)
ON_THRESHOLD = 0.60
OFF_THRESHOLD = 0.40
CONFIRM_ON = 2       # mesi consecutivi sopra ON per accendere
CONFIRM_OFF = 6      # mesi consecutivi sotto OFF per spegnere (asimmetrico = rigido)

# Sotto-fasi
EARLY_MONTHS = 6         # primi mesi dopo l'accensione = Early
LATE_PEAK_FRAC = 0.80    # sotto l'80% del picco + in discesa = Late (verso neutralizzazione)

# Mappatura gap divergenza -> intensita' (sigmoid DATA-GROUNDED sul gap 2008+, R4:
# center = mediana empirica +7.9%, scale = mezzo-IQR -> p25/p75 cadono a ~0.27/0.73)
_INTENSITY_CENTER = 0.08
_INTENSITY_SCALE = 0.20


def gap_to_intensity(gap: float, center: float = _INTENSITY_CENTER,
                     scale: float = _INTENSITY_SCALE) -> float:
    """Fair-value gap (frazione) -> intensita' debasement 0-1 (sigmoid)."""
    z = (gap - center) / scale
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def apply_hysteresis(
    intensity: list[float],
    on: float = ON_THRESHOLD,
    off: float = OFF_THRESHOLD,
    confirm_on: int = CONFIRM_ON,
    confirm_off: int = CONFIRM_OFF,
) -> list[bool]:
    """Serie di stato acceso/spento con isteresi asimmetrica.

    Un singolo scivolone sotto soglia NON spegne (il contatore-off si azzera appena
    l'intensita' risale sopra `off`): rigidita' anti-falso-segnale.
    """
    states: list[bool] = []
    state = False
    up = 0    # mesi consecutivi >= on mentre spento
    down = 0  # mesi consecutivi < off mentre acceso
    for x in intensity:
        if not state:
            if x >= on:
                up += 1
                if up >= confirm_on:
                    state = True
                    down = 0
            else:
                up = 0
        else:
            if x < off:
                down += 1
                if down >= confirm_off:
                    state = False
                    up = 0
            else:
                down = 0
        states.append(state)
    return states


def months_in_current_state(states: list[bool]) -> int:
    """Da quanti mesi consecutivi siamo nello stato corrente (l'ultimo)."""
    if not states:
        return 0
    last = states[-1]
    c = 0
    for s in reversed(states):
        if s == last:
            c += 1
        else:
            break
    return c


def classify_subphase(
    on: bool,
    months_in: int,
    intensity_now: float,
    peak_since_on: float,
    slope: float,
) -> str | None:
    """Early / Mature / Late (o None se spento).

    Early  = acceso da poco (<= EARLY_MONTHS).
    Late   = alto ma sceso sotto LATE_PEAK_FRAC del picco e in discesa (verso neutralita').
    Mature = tutto il resto (alto, sostenuto, anche con piccole correzioni).
    """
    if not on:
        return None
    if months_in <= EARLY_MONTHS:
        return "Early"
    if peak_since_on > 0 and intensity_now < LATE_PEAK_FRAC * peak_since_on and slope < 0:
        return "Late"
    return "Mature"


def compute_debasement_regime(divergence: dict | None, regime_probs: dict | None) -> dict | None:
    """Compone il segnale divergenza (R1) + le prob cicliche in uno snapshot del
    meta-regime. Graceful: None se i dati non bastano.

    `divergence`  = output di compute_divergence (usa il campo `history` = [{date, gap}]).
    `regime_probs`= {reflation, stagflation, deflation, goldilocks} correnti (o None).
    """
    if not divergence:
        return None
    history = divergence.get("history") or []
    if not history:
        return None

    intensity_series = [gap_to_intensity(h["gap"]) for h in history]
    states = apply_hysteresis(intensity_series)
    months_in = months_in_current_state(states)
    on_now = states[-1]
    intensity_now = intensity_series[-1]

    if on_now:
        start = len(states) - months_in
        peak_since_on = max(intensity_series[start:])
    else:
        peak_since_on = intensity_now

    slope = intensity_now - intensity_series[-4] if len(intensity_series) >= 4 else 0.0
    sub_phase = classify_subphase(on_now, months_in, intensity_now, peak_since_on, slope)

    probs = regime_probs or {}
    scores = {
        "reflation": float(probs.get("reflation", 0.0)),
        "stagflation": float(probs.get("stagflation", 0.0)),
        "deflation": float(probs.get("deflation", 0.0)),
        "goldilocks": float(probs.get("goldilocks", 0.0)),
        # intensita' 0-1, NON una probabilita' comparabile testa-a-testa (orizzonte diverso)
        "debasement": intensity_now,
    }

    return {
        "asof": divergence.get("asof"),
        "scores": scores,
        "meta": {
            "active": "debasement" if on_now else "cyclical",
            "sub_phase": sub_phase,
            "intensity": intensity_now,
            "months_in": months_in,
            "slope_3m": slope,
        },
        "divergence": {
            "gap": divergence.get("gap"),
            "gap_pct": divergence.get("gap_pct"),
            "gold": divergence.get("gold"),
            "fair_value": divergence.get("fair_value"),
            "real_yield": divergence.get("real_yield"),
        },
        "intensity_history": [
            {"date": h["date"], "intensity": intensity_series[i], "state": states[i]}
            for i, h in enumerate(history)
        ],
        "thresholds": {
            "on": ON_THRESHOLD, "off": OFF_THRESHOLD,
            "confirm_on": CONFIRM_ON, "confirm_off": CONFIRM_OFF,
        },
    }


def load_live_debasement_regime(regime_probs: dict | None = None) -> dict | None:
    """Carica il segnale divergenza live (R1) e compone il meta-regime. Graceful."""
    from app.services.backtest.debasement_divergence import load_live_divergence

    divergence = load_live_divergence()
    if divergence is None:
        return None
    return compute_debasement_regime(divergence, regime_probs)
