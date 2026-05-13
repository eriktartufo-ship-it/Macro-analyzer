"""Tier 5.7 LOOP CLOSURE — Volatility regime separato (ultimo task Tier 5).

Migra le boolean "vix_low/elevated/spike" del classifier a sub-regime
continuo con 4 stati + change-point detection rolling.

**4 Vol Regimes** (basati su VIX percentile storico):

- `vol_calm` (VIX ≤ 14): low-vol / risk-on / yield carry / equity bid.
  Storicamente: 1995-99, 2004-07, 2017-19. Bubble-prone fase finale.
- `vol_normal` (14 < VIX ≤ 20): baseline / no stress particolare.
- `vol_elevated` (20 < VIX ≤ 30): correction / stress / risk-off iniziale.
  Storicamente: 2011 EU crisis, 2015-16 China.
- `vol_spike` (VIX > 30): panic / capitulation / systemic event.
  Storicamente: 2008-Q4, 2011-08 Aug, 2020-03, 2022-Sep.

**Change-point detection** (semplice, rolling z-score):

- Se `VIX_today - VIX_ma20 > k·VIX_std20` → vol_breakout (up).
- Se `VIX_today - VIX_ma20 < -k·VIX_std20` → vol_compression (down).
- k=1.5 default.

**Use cases**:

- Sub-regime score informativo separato dal regime 4-quadranti.
- Modulator opzionale per crisis_indicator (vol_spike → boost risk_score).
- Frontend: badge vol_regime accanto a regime principale.

NOTE: questo modulo NON modifica regime probs (il pillar VIX già lo fa
dentro classifier). È **info layer** + ortogonale al regime macro.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VolRegimeAssessment:
    """Snapshot vol regime + change-point signal."""
    vol_regime: str         # vol_calm / vol_normal / vol_elevated / vol_spike
    vol_score: float        # 0-1 normalized intensity within regime
    vix: float
    vix_ma20: float | None
    vix_std20: float | None
    change_point: str       # "none" / "vol_breakout" / "vol_compression"
    change_magnitude: float # z-score abs value, 0 se no change point
    description: str        # human-readable


_VOL_THRESHOLDS = {
    "vol_calm_max": 14.0,
    "vol_normal_max": 20.0,
    "vol_elevated_max": 30.0,
    # >30 = vol_spike
}

_CHANGE_POINT_K = 1.5  # z-score threshold per breakout/compression


def classify_vol_regime(vix: float) -> tuple[str, float]:
    """Classifica VIX in 1 dei 4 vol regimes + intensity score 0-1.

    Score = posizione relativa dentro il bucket (linearmente scalata).
    Es. VIX=18 → vol_normal, score = (18-14)/(20-14) = 0.67.
    """
    if vix <= _VOL_THRESHOLDS["vol_calm_max"]:
        # vol_calm: score 1.0 a VIX=8 (estremo), 0.0 a VIX=14 (boundary)
        score = max(0.0, min(1.0, (14.0 - vix) / 6.0))
        return "vol_calm", score
    if vix <= _VOL_THRESHOLDS["vol_normal_max"]:
        score = (vix - 14.0) / 6.0
        return "vol_normal", score
    if vix <= _VOL_THRESHOLDS["vol_elevated_max"]:
        score = (vix - 20.0) / 10.0
        return "vol_elevated", score
    # vol_spike: score 0.0 a VIX=30 (boundary), 1.0 a VIX=60+ (estremo)
    score = max(0.0, min(1.0, (vix - 30.0) / 30.0))
    return "vol_spike", score


def detect_change_point(
    vix: float,
    vix_ma20: float | None,
    vix_std20: float | None,
    k: float = _CHANGE_POINT_K,
) -> tuple[str, float]:
    """Detect change-point via rolling z-score.

    Returns:
        (change_point, magnitude):
        - "vol_breakout" se vix - ma20 > k·std20 (spike sopra trend)
        - "vol_compression" se vix - ma20 < -k·std20 (compression sotto trend)
        - "none" altrimenti
        - magnitude = |z|, 0 se no change point.
    """
    if vix_ma20 is None or vix_std20 is None or vix_std20 <= 0:
        return "none", 0.0
    z = (vix - vix_ma20) / vix_std20
    if z > k:
        return "vol_breakout", abs(z)
    if z < -k:
        return "vol_compression", abs(z)
    return "none", 0.0


def _describe(vol_regime: str, change_point: str, vix: float) -> str:
    """Descrizione human-readable IT del vol regime + change."""
    base_desc = {
        "vol_calm": f"Vol calm (VIX {vix:.1f} ≤ 14): risk-on, yield carry favorevole",
        "vol_normal": f"Vol normal (VIX {vix:.1f}): no stress particolare",
        "vol_elevated": f"Vol elevated (VIX {vix:.1f}): correction o stress iniziale",
        "vol_spike": f"Vol spike (VIX {vix:.1f} > 30): panic / capitulation / systemic event",
    }
    out = base_desc.get(vol_regime, f"VIX {vix:.1f}")
    if change_point == "vol_breakout":
        out += " · BREAKOUT in salita sopra trend 20gg"
    elif change_point == "vol_compression":
        out += " · COMPRESSION sotto trend 20gg (complacency)"
    return out


def assess_vol_regime(
    vix: float,
    vix_ma20: float | None = None,
    vix_std20: float | None = None,
) -> VolRegimeAssessment:
    """Pipeline completa: classify regime + detect change-point + summary.

    Args:
        vix: VIX corrente.
        vix_ma20: media mobile 20gg (opzionale).
        vix_std20: deviazione std 20gg (opzionale).

    Returns:
        VolRegimeAssessment con regime, score, change-point, descrizione.
    """
    vol_regime, vol_score = classify_vol_regime(vix)
    change_point, magnitude = detect_change_point(vix, vix_ma20, vix_std20)
    description = _describe(vol_regime, change_point, vix)
    return VolRegimeAssessment(
        vol_regime=vol_regime,
        vol_score=round(vol_score, 3),
        vix=float(vix),
        vix_ma20=float(vix_ma20) if vix_ma20 is not None else None,
        vix_std20=float(vix_std20) if vix_std20 is not None else None,
        change_point=change_point,
        change_magnitude=round(magnitude, 2),
        description=description,
    )
