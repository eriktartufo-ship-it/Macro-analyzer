"""Tier 6.3 — Cross-asset signals come INPUT classifier.

Bridgewater pattern: relazioni cross-asset come **leading indicators** del
regime macro, non solo output analitici. Tre ratio prioritari:

1. **copper/gold** — cyclical commodity (Cu, industrial demand) vs safe-haven
   metal (Au). Storicamente:
   - 2009 bottom: 0.20 (GFC depths)
   - 2011 peak: 0.55 (post-stimulus reflation peak)
   - 2020 COVID: 0.18 (deflation panic)
   - 2021-22 recovery: 0.30-0.40
   Alto = reflation/risk-on. Basso = deflation/risk-off.

2. **gold/oil** — gold strong vs oil weak = deflation; gold weak vs oil
   strong = stagflation. Storica avg ~17:
   - 1973 oil shock: 9 (stagflation peak)
   - 2009: 25 (gold $1000, oil $40)
   - 2020-Q1 covid: 75 (gold $1500, oil $20)
   - 2022: 12 (oil $120, gold $1900) → stagflation
   Alto = deflation. Basso = stagflation.

3. **HY/IG spread ratio** — high-yield credit spread / investment-grade
   credit spread. Misura credit risk appetite (rischio "appropriato" vs
   "scarso").
   - Normalmente 3.0-4.5 (range storico tranquillo)
   - >5.5 = stress severo (2008 GFC peak 8, 2020-Q1 picco 7)
   - <2.5 = euforia credit (2007 pre-GFC 2.3, 2021 SPAC 2.5)
   Alto = deflation/stress. Basso = reflation/risk-on.

**Math**: ratio è sempre `numerator / denominator`. Sigmoid soft threshold
con scale ampia per ammortizzare rumore intraday.

**Indicator naming** (per `indicators` dict del classifier):

- `copper_gold_ratio` (float)
- `gold_oil_ratio` (float)
- `hy_ig_spread_ratio` (float)

Pillar conditions (opt-in via `USE_CROSS_ASSET_PILLARS`):

- `copper_gold_strong` → +reflation (ratio > 0.45)
- `copper_gold_weak` → +deflation (ratio < 0.30)
- `gold_oil_high` → +deflation (ratio > 30)
- `gold_oil_low` → +stagflation (ratio < 15)
- `hy_ig_stress` → +deflation (ratio > 5)
- `hy_ig_tight` → +reflation (ratio < 3)

Pesi soft (0.02-0.03 each) per non sovrappesare il signal vs pillar macro
principali. Budget totale T6.3 = ~0.15 sul fit_score.
"""
from __future__ import annotations


# Threshold riferimento (basati su quantili storici 1970-2024)
_THRESHOLDS = {
    "copper_gold_strong": 0.45,   # > = reflation signal
    "copper_gold_weak": 0.30,     # < = deflation signal
    "gold_oil_high": 30.0,        # > = deflation pressure
    "gold_oil_low": 15.0,         # < = stagflation pressure (oil dominates)
    "hy_ig_stress": 5.0,          # > = credit risk-off
    "hy_ig_tight": 3.0,           # < = credit risk-on
}


def compute_copper_gold_ratio(copper_price: float, gold_price: float) -> float | None:
    """Calcola copper/gold ratio. Ritorna None se input invalido.

    Args:
        copper_price: prezzo rame (es. spot CHRIS/CME_HG1 in $/lb o futures HG=F).
        gold_price: prezzo oro (es. spot XAU o futures GC=F in $/oz).

    Returns: ratio float (tipico range 0.15-0.55) o None.
    """
    if copper_price is None or gold_price is None:
        return None
    try:
        copper = float(copper_price)
        gold = float(gold_price)
    except (ValueError, TypeError):
        return None
    if gold <= 0 or copper <= 0:
        return None
    return copper / gold


def compute_gold_oil_ratio(gold_price: float, oil_price: float) -> float | None:
    """Calcola gold/oil ratio.

    Args:
        gold_price: oro $/oz.
        oil_price: oil $/barile (WTI).

    Returns: ratio float (tipico range 8-80) o None.
    """
    if gold_price is None or oil_price is None:
        return None
    try:
        gold = float(gold_price)
        oil = float(oil_price)
    except (ValueError, TypeError):
        return None
    if oil <= 0 or gold <= 0:
        return None
    return gold / oil


def compute_hy_ig_spread_ratio(hy_spread: float, ig_spread: float) -> float | None:
    """Calcola HY OAS / IG OAS ratio. Misura credit risk appetite.

    Args:
        hy_spread: high-yield spread (BAMLH0A0HYM2, % over treasuries).
        ig_spread: investment-grade spread (BAMLC0A0CM, %).

    Returns: ratio float (tipico 2-8) o None.
    """
    if hy_spread is None or ig_spread is None:
        return None
    try:
        hy = float(hy_spread)
        ig = float(ig_spread)
    except (ValueError, TypeError):
        return None
    if ig <= 0 or hy <= 0:
        return None
    return hy / ig


def get_threshold(name: str) -> float:
    """Lookup threshold da `_THRESHOLDS`."""
    return _THRESHOLDS[name]


def list_pillar_conditions() -> list[str]:
    """Lista delle 6 condition T6.3 attivate da `USE_CROSS_ASSET_PILLARS`."""
    return [
        "copper_gold_strong", "copper_gold_weak",
        "gold_oil_high", "gold_oil_low",
        "hy_ig_stress", "hy_ig_tight",
    ]
