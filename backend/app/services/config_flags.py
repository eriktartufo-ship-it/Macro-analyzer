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


def use_dedollar_pillar() -> bool:
    """Tier 5.2 LOOP CLOSURE: se True, dedollar_combined entra come pillar
    nel rule_based classifier (`stagflation.dedollar_debasement` weight 0.03,
    `reflation.dedollar_low` weight 0.02). Default: False (back-compat puro).

    Indipendente da `use_dedollar_bonus` (che controlla scoring asset).
    Attivabile separatamente per A/B testing dei due loop.
    """
    return os.getenv("USE_DEDOLLAR_PILLAR", "0").lower() in _TRUTHY


def use_crisis_modulation() -> bool:
    """Tier 5.1 LOOP CLOSURE: se True, le probabilities regime vengono
    modulate post-blend in base al crisis_type + risk_score di
    `assess_crisis_risk`. Default: False.

    Effetto:
    - `deflation_crash` con risk>0.3 → +deflation prob (alpha × risk).
    - `stagflation_debasement` con risk>0.3 → +stagflation prob.
    - `bubble_goldilocks` con risk>0.3 → small shift verso deflation/stag
      (preludio mean-reversion downside).
    - alpha=0.05 default = max ±5pp shift.

    Distinto da `use_dedollar_pillar` (che agisce sui fit_scores dentro
    classifier). Crisis modulation è POST-classifier, sui probs finali.
    """
    return os.getenv("USE_CRISIS_MODULATION", "0").lower() in _TRUTHY


def use_news_pillar() -> bool:
    """Tier 5.3 LOOP CLOSURE: se True, news avg sentiment entra come pillar
    nel rule_based classifier (`deflation.news_panic` weight 0.02,
    `reflation.news_positive_strong` weight 0.02). Default: False.

    News sono oggi consumate solo da trajectory (forces) + scoring asset
    (news_signals). Questo flag chiude il loop news → regime ufficiale.

    Pillar è soft (weight 0.02) per gestire rumorosità news intraday.
    """
    return os.getenv("USE_NEWS_PILLAR", "0").lower() in _TRUTHY


def use_dfm_asset_bonus() -> bool:
    """Tier 5.4 LOOP CLOSURE: se True, gdp_yoy_dfm modula direttamente
    final asset scores (oltre al pillar classifier già attivo).
    Default: False.

    Bonus applicato a:
    - us_equities_growth/value, em_equities: +δ × max(0, gdp_yoy_dfm - 2.5)
      (real-time growth nowcast = pro-cyclical assets bid)
    - us_bonds_long: -δ × max(0, gdp_yoy_dfm - 2.5) (duration risk se yields salgono)
    - Simmetrico negativo se gdp_yoy_dfm < 1.5 (slowdown nowcast).

    δ = 2.0 default → max ±5pp score boost con gdp_yoy_dfm estremo (~5%).
    """
    return os.getenv("USE_DFM_ASSET_BONUS", "0").lower() in _TRUTHY


def use_asset_feedback() -> bool:
    """Tier 5.6 LOOP CLOSURE (sperimentale): se True, applica Bayesian
    update sul regime usando 6m asset returns come likelihood.

    Filosofia: il mercato ha già "votato" sul regime dominante. Performance
    recente di gold/bonds/equities su 6m rolling implica P(regime|returns)
    via likelihood gaussiana da ASSET_REGIME_DATA (avg_return, vol).

    RISCHIO LOOKAHEAD: asset returns recenti possono essere conseguenza
    del regime (causalità inversa). Mitigato:
    - Window 6m (NON 1m troppo rumoroso)
    - Soft blend `alpha=0.05` → max ±5pp shift
    - Validazione: lead-time NBER non deve peggiorare.
    Default: False.
    """
    return os.getenv("USE_ASSET_FEEDBACK", "0").lower() in _TRUTHY
