"""Position sizing + dynamic tranche logic per AI Portfolio.

Decisioni user 2026-06-10:
- Capitale base: $100k paper
- Tranche dinamiche 2-5 in base a confidence regime classifier
- Cap singolo asset 25%, floor 3%

Formula target weight:
    base_score = max(0, score - 50)  # only positive edge
    risk_adjusted = base_score / max(vol_60d, 0.05)
    raw = risk_adjusted (normalized across selected universe → 1.0)
    final = clip(raw, 0.03, 0.25)
"""
from __future__ import annotations


# Cap & floor singolo asset
WEIGHT_CAP_MAX = 0.25
WEIGHT_CAP_MIN = 0.03
DEFAULT_VOL = 0.20  # fallback annualized vol se missing


def dynamic_n_tranches(confidence: float) -> int:
    """Numero di tranche DCA in base a confidence regime.

    confidence > 0.80: 2 (alta convinzione, aggressivo)
    confidence > 0.65: 3
    confidence > 0.50: 4
    confidence >= 0.30: 5 (cauto)
    confidence < 0.30: 0 (uncertainty gate skip)

    AUDIT 2026-07-12 (boundary bug, stessa classe del m2_yoy=5.0):
    data_strategist calcola confidence = max(0.30, fired/10) → col floor a
    ESATTAMENTE 0.30 il vecchio `> 0.30` dava 0 tranche → "target weight 0"
    per ogni asset → Data-Driven in deadlock totale (0 posizioni dal reset
    del 19/6). Il gate di incertezza resta a < 0.30, coerente con
    `_decide` che skippa a confidence < 0.30.
    """
    if confidence > 0.80:
        return 2
    if confidence > 0.65:
        return 3
    if confidence > 0.50:
        return 4
    if confidence >= 0.30:
        return 5
    return 0


def raw_target_weight(score: float, vol_60d: float | None) -> float:
    """Pre-normalization raw target weight per asset.

    Returns 0 se score < 50 (no positive edge).
    Risk-adjusted = (score - 50) / vol.
    """
    if score < 50:
        return 0.0
    edge = score - 50.0
    vol = max(vol_60d or DEFAULT_VOL, 0.05)
    return edge / vol


def normalize_and_cap(
    raw_weights: dict[str, float],
    cap_max: float = WEIGHT_CAP_MAX,
    cap_min: float = WEIGHT_CAP_MIN,
) -> dict[str, float]:
    """Normalize raw weights + apply caps con waterfill iterativo.

    Asset con raw=0 vengono mantenuti a 0 (no allocation).
    Per asset con raw>0, normalizza così che sum=1, poi cap [min, max].
    """
    selected = {a: w for a, w in raw_weights.items() if w > 0}
    if not selected:
        return {a: 0.0 for a in raw_weights}

    total = sum(selected.values())
    if total <= 0:
        return {a: 0.0 for a in raw_weights}
    normalized = {a: w / total for a, w in selected.items()}

    # Waterfill caps (max 10 iter)
    n = len(normalized)
    if cap_min * n > 1.0:
        # Floor non rispettabile → equal weight
        eq = 1.0 / n
        return {a: (eq if a in selected else 0.0) for a in raw_weights}

    out = dict(normalized)
    for _ in range(10):
        too_high = [a for a, w in out.items() if w > cap_max + 1e-9]
        too_low = [a for a, w in out.items() if w < cap_min - 1e-9]
        if not too_high and not too_low:
            break

        pinned = {a: cap_max for a in too_high}
        pinned.update({a: cap_min for a in too_low})
        free_assets = [a for a in out if a not in pinned]
        free_budget = 1.0 - sum(pinned.values())

        if not free_assets or free_budget <= 0:
            out = pinned
            for a in free_assets:
                out[a] = cap_min
            break

        free_sum = sum(out[a] for a in free_assets) or 1e-9
        new_out = dict(pinned)
        for a in free_assets:
            new_out[a] = (out[a] / free_sum) * free_budget
        out = new_out

    # Assemble result with all assets (raw=0 → 0)
    return {a: out.get(a, 0.0) for a in raw_weights}


def tranche_size(target_weight: float, n_tranches: int) -> float:
    """Size della singola tranche = target / n_tranches."""
    if n_tranches <= 0:
        return 0.0
    return target_weight / n_tranches
