"""T9-AUDIT — Covariate shift detector.

Council finding (soros): "Manca monitoring riflessivo: come si comporta il
sistema quando i suoi stessi segnali diventano consensus?"

Covariate shift = distribuzione dei INPUT (indicators) cambia tra training e
production. Il modello stesso non cambia ma vede dati con cui non è familiare.

Esempio: training su episodi 1929-2024 (CPI tipico 1-4%, raramente >5%). Se
nel 2026 H2 inizia ipotetico regime fiscal-dominance con CPI 6-8% stabile, il
modello è in covariate-shift → predictions inaffidabili.

Metodologia:
1. Per ogni indicator chiave: mean + std del training set (37 episodi).
2. Per la live observation: z-score = (live_val - train_mean) / train_std.
3. Se |z| > 2.5 → indicator in covariate shift.
4. Se >= 3/19 indicators in shift → alert ALERT_DRIFT.

Differenza con OOD detector (Mahalanobis):
- OOD = punto singolo lontano dal cloud training (multivariata)
- Covariate-shift = drift sistematico nel TEMPO (su finestra recente, non punto)

Per shift temporale serve serie storica live. Per ora: snapshot vs training.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.services.regime.historical_episodes import HISTORICAL_EPISODES


# Indicators chiave per covariate shift detection (subset stabile cross-time)
_SHIFT_INDICATORS = (
    "gdp_roc", "cpi_yoy", "unrate", "vix",
    "fed_funds", "breakeven_10y", "baa_10y_spread",
    "yield_curve_spread", "core_pce_yoy",
)


@dataclass
class IndicatorShift:
    """Shift di un singolo indicator."""
    name: str
    live_value: float
    train_mean: float
    train_std: float
    z_score: float
    is_shifted: bool  # |z| > 2.5
    severity: str  # ok | warn | critical


@dataclass
class CovariateShiftReport:
    """Report shift aggregato."""
    n_indicators_evaluated: int
    n_shifted: int
    n_critical: int  # |z| > 3.0
    indicators_shift: list[IndicatorShift]
    shift_alert: bool  # >= 3 indicators shifted
    summary: str


# Cache module-level (ricomputato se HISTORICAL_EPISODES cambia)
_TRAINING_STATS: dict[str, tuple[float, float]] | None = None


def _compute_training_stats() -> dict[str, tuple[float, float]]:
    """Per ogni indicator chiave: (mean, std) del training set."""
    out: dict[str, tuple[float, float]] = {}
    for ind_name in _SHIFT_INDICATORS:
        values = []
        for ep in HISTORICAL_EPISODES:
            v = ep.indicators.get(ind_name)
            if v is None:
                continue
            try:
                f = float(v)
                import math
                if math.isnan(f) or math.isinf(f):
                    continue
                values.append(f)
            except (TypeError, ValueError):
                continue
        if values:
            arr = np.array(values)
            out[ind_name] = (float(arr.mean()), float(arr.std() or 1.0))
    return out


def _get_training_stats() -> dict[str, tuple[float, float]]:
    global _TRAINING_STATS
    if _TRAINING_STATS is None:
        _TRAINING_STATS = _compute_training_stats()
    return _TRAINING_STATS


def reset_cache() -> None:
    """Forza re-computation (es. dopo update HISTORICAL_EPISODES)."""
    global _TRAINING_STATS
    _TRAINING_STATS = None


def assess_covariate_shift(
    indicators: dict,
    z_warn: float = 2.0,
    z_critical: float = 3.0,
    alert_threshold: int = 3,
) -> CovariateShiftReport:
    """Valuta se gli indicators live sono in covariate shift vs training.

    Args:
        indicators: dict live (alias supportati via safe_indicator).
        z_warn: |z| sopra cui flag warn (default 2.5)
        z_critical: |z| sopra cui flag critical (default 3.0)
        alert_threshold: numero min indicators shifted per shift_alert (default 3)
    """
    from app.services.common.indicator_aliases import safe_indicator

    stats = _get_training_stats()
    shifts: list[IndicatorShift] = []
    n_critical = 0

    for ind_name in _SHIFT_INDICATORS:
        if ind_name not in stats:
            continue
        train_mean, train_std = stats[ind_name]
        live_val = safe_indicator(indicators, ind_name, default=train_mean)
        z = (live_val - train_mean) / max(train_std, 0.01)
        is_shifted = abs(z) > z_warn
        if abs(z) > z_critical:
            severity = "critical"
            n_critical += 1
        elif is_shifted:
            severity = "warn"
        else:
            severity = "ok"
        shifts.append(IndicatorShift(
            name=ind_name,
            live_value=round(live_val, 4),
            train_mean=round(train_mean, 4),
            train_std=round(train_std, 4),
            z_score=round(z, 3),
            is_shifted=is_shifted,
            severity=severity,
        ))

    n_shifted = sum(1 for s in shifts if s.is_shifted)
    shift_alert = n_shifted >= alert_threshold

    if shift_alert:
        summary = (
            f"COVARIATE SHIFT ALERT: {n_shifted}/{len(shifts)} indicators "
            f"in shift (>{z_warn}σ). Modello in zona non testata storicamente."
        )
    else:
        summary = (
            f"NO SIGNIFICANT SHIFT: solo {n_shifted}/{len(shifts)} indicators "
            f"in shift. Indicators dentro training distribution."
        )

    return CovariateShiftReport(
        n_indicators_evaluated=len(shifts),
        n_shifted=n_shifted,
        n_critical=n_critical,
        indicators_shift=shifts,
        shift_alert=shift_alert,
        summary=summary,
    )


def assess_persistent_shift(
    recent_reports: list[CovariateShiftReport],
    min_consecutive_alerts: int = 3,
) -> tuple[bool, str]:
    """T9-AUDIT P1-B (council raydalio): persistence check.

    Pre-fix: alert single-snapshot 2.0σ → falso positivi su shock isolati.
    Post-fix: alert solo se ≥3 settimane consecutive con shift_alert=True.

    Args:
        recent_reports: lista CovariateShiftReport ordinata recent-first.
        min_consecutive_alerts: minimo n. alerts consecutivi (default 3 = 3 weeks).

    Returns:
        (is_persistent, description).
    """
    consecutive = 0
    for r in recent_reports:
        if r.shift_alert:
            consecutive += 1
        else:
            break
    is_persistent = consecutive >= min_consecutive_alerts
    if is_persistent:
        desc = (
            f"PERSISTENT SHIFT: {consecutive} alerts consecutivi "
            f"(threshold {min_consecutive_alerts}). Modello in zona "
            f"non testata persistente — review urgente."
        )
    else:
        desc = (
            f"No persistent shift: {consecutive}/{min_consecutive_alerts} "
            f"alerts consecutivi. OK monitoring."
        )
    return is_persistent, desc


def ks_test_tail_divergence(
    live_values: list[float],
    training_values: list[float],
    alpha: float = 0.05,
) -> tuple[bool, float, str]:
    """T9-AUDIT P1-B (council taleb): KS-test su code distribuzioni.

    Mahalanobis distance assume distribuzione gaussiana; nelle CODE
    (eventi rari) la realtà è diversa. KS-test confronta CDF empiriche.

    Args:
        live_values: campioni recenti (es. 30-60 giorni).
        training_values: campioni training (set historic).
        alpha: significance level (default 0.05).

    Returns:
        (is_divergent, p_value, description).
    """
    if len(live_values) < 10 or len(training_values) < 10:
        return False, 1.0, "Insufficient samples for KS-test (need ≥10 each)"

    try:
        from scipy.stats import ks_2samp
        stat, p_value = ks_2samp(live_values, training_values)
        is_divergent = bool(p_value < alpha)  # cast da np.bool_
        if is_divergent:
            desc = (
                f"TAIL DIVERGENCE: KS-stat={stat:.3f}, p={p_value:.4f} "
                f"(<{alpha}). Distribuzione live diverge da training."
            )
        else:
            desc = (
                f"No tail divergence: KS-stat={stat:.3f}, p={p_value:.4f} "
                f"(>={alpha}). Code coerenti con training."
            )
        return is_divergent, float(p_value), desc
    except ImportError:
        return False, 1.0, "scipy.stats not available — skipping KS-test"
    except Exception as e:
        return False, 1.0, f"KS-test failed: {e}"
