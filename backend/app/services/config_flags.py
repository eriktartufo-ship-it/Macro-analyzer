"""Feature flags via env var, centralizzati per consistenza.

Pattern: tutti gli scoring/regime layer "biased" sono OPT-IN. Default = data-driven puro.
Cosi' i numeri base sono trasparenti e l'utente puo' attivare layer aggiuntivi
(calibration shrinkage, dedollar bias) sapendo che modificano il risultato.
"""

from __future__ import annotations

import os

_TRUTHY = ("1", "true", "yes", "on")


def use_calibrated_scoring() -> bool:
    """Se True, scoring engine usa calibrated_asset_regime.json (Bayesian shrinkage).
    Default: False (hardcoded prior values).
    """
    return os.getenv("USE_CALIBRATED_SCORING", "0").lower() in _TRUTHY


def use_dedollar_bonus() -> bool:
    """Se True, secular_bonus dedollar viene applicato a final asset scores
    e DEDOLLAR_REGIME_PRESSURE viene applicato in trajectory.
    Default: False (scoring/trajectory data-driven puri).
    """
    return os.getenv("USE_DEDOLLAR_BONUS", "0").lower() in _TRUTHY
