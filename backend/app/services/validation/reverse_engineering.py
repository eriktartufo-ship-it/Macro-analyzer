"""Council 2026-05-27 NEW REQUIREMENT — Reverse engineering automatic.

User Erik: "quando un trade non va a segno bisogna fare reverse engineering
e capire perché non ha funzionato, cosa è sfuggito dall'attenzione e capire
come migliorare il modello rendendolo comunque sempre molto fluido".

Filosofia: il modello deve **imparare dai propri errori**. Quando una
prediction sottoperforma significantly il benchmark (alpha < -5% a 3m horizon),
attivare automatic post-mortem analysis per identificare:

1. **Pillar attribution**: quali pillar avevano triggered? quali erano vicini
   alla soglia? quali NON avevano triggered ma avrebbero dovuto?
2. **Indicator outliers**: quali indicators erano borderline / fuori range
   storico al momento della prediction?
3. **Asset mismatch**: quali asset selected nel top-5 hanno performato peggio?
   c'è un pattern (es. "growth stocks in rising rates")?
4. **Regime confusion**: il regime predicted era boundary case? bassa confidence?
5. **Pattern aggregation**: cross-failures, esiste un pattern strutturale?

Output:
- `post_mortem(prediction_id)`: analisi singola failure
- `find_failure_patterns(min_alpha_threshold=-0.05)`: aggregate patterns
- `generate_improvement_suggestions()`: meta-analisi per council
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from loguru import logger
from sqlalchemy.orm import Session

from app.models import PredictionLog


@dataclass
class PostMortem:
    """Analisi single failure: cosa è andato storto."""
    prediction_id: int
    date_predicted: date | None
    regime: str
    confidence: float
    horizon: str  # "1m" | "3m" | "6m"
    alpha: float
    severity: str  # "minor" | "moderate" | "severe"

    # Root cause analysis
    top5_assets: list[dict] = field(default_factory=list)
    worst_performer: str | None = None
    best_performer: str | None = None
    regime_was_boundary: bool = False  # max_prob < 0.40
    confidence_was_low: bool = False  # conf < 0.30

    # Findings (interpreted)
    findings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


@dataclass
class FailurePattern:
    """Pattern aggregato cross-failures."""
    pattern_name: str
    n_occurrences: int
    mean_alpha: float
    description: str
    affected_regimes: list[str]
    suggested_action: str


def _severity_from_alpha(alpha: float) -> str:
    if alpha <= -0.10:
        return "severe"
    if alpha <= -0.05:
        return "moderate"
    return "minor"


def post_mortem(
    db: Session,
    prediction_id: int,
    horizon: str = "3m",
) -> PostMortem | None:
    """Analizza UN failure specifica.

    Args:
        prediction_id: PK PredictionLog.
        horizon: '1m' | '3m' | '6m' (alpha da analizzare).

    Returns:
        PostMortem con findings + suggestions, o None se prediction non trovata
        o se non ha alpha evaluated nell'horizon.
    """
    record = db.query(PredictionLog).filter_by(id=prediction_id).first()
    if record is None:
        return None

    alpha_attr = f"alpha_{horizon}"
    alpha = getattr(record, alpha_attr, None)
    if alpha is None:
        logger.warning(f"Prediction {prediction_id} non valutata su {horizon}")
        return None

    probs = json.loads(record.probabilities_json) if record.probabilities_json else {}
    top5 = json.loads(record.top5_json) if record.top5_json else []

    max_prob = max(probs.values()) if probs else 0.0
    pm = PostMortem(
        prediction_id=prediction_id,
        date_predicted=record.date_predicted,
        regime=record.regime,
        confidence=record.confidence,
        horizon=horizon,
        alpha=alpha,
        severity=_severity_from_alpha(alpha),
        top5_assets=top5,
        regime_was_boundary=max_prob < 0.40,
        confidence_was_low=record.confidence < 0.30,
    )

    # Findings interpretati
    if pm.regime_was_boundary:
        pm.findings.append(
            f"Regime predicted '{record.regime}' era boundary case "
            f"(max prob {max_prob:.2f} < 0.40). Il classifier non aveva conviction."
        )
        pm.suggestions.append(
            "Considera defensive transition mode quando max_prob < 0.40"
        )
    if pm.confidence_was_low:
        pm.findings.append(
            f"Confidence ({record.confidence:.2f}) era sotto 0.30. "
            f"Il modello era incerto al momento della prediction."
        )
        pm.suggestions.append(
            "Filtra predictions con confidence < 0.30 oppure pesa allocation conservativamente"
        )

    # Asset analysis: confronta realized return vs aspettative
    realized = getattr(record, f"realized_return_{horizon}", None)
    benchmark = getattr(record, f"benchmark_return_{horizon}", None)
    if realized is not None and benchmark is not None:
        gap = realized - benchmark
        if gap < -0.05:
            pm.findings.append(
                f"Portfolio realized {realized*100:.1f}% vs benchmark "
                f"{benchmark*100:.1f}% (gap {gap*100:+.1f}pp)."
            )

    # Regime-specific findings
    if record.regime == "stagflation" and alpha < -0.05:
        pm.findings.append(
            "Stagflation prediction underperformed: probabile che il regime "
            "fosse in realtà reflation late-cycle (CPI alto MA growth resiliente). "
            "USE_MOMENTUM_PILLARS potrebbe aver triggered prematuramente."
        )
        pm.suggestions.append(
            "Cross-check con DFM nowcast: se gdp_yoy_dfm > 3% in stagflation "
            "prediction, downweight signal."
        )
    elif record.regime == "deflation" and alpha < -0.05:
        pm.findings.append(
            "Deflation prediction underperformed: il modello potrebbe aver visto "
            "GDP_COLLAPSE_OVERRIDE trigger su economy resilient. "
            "Rate cut anticipated NON sempre = recession."
        )
        pm.suggestions.append(
            "Aggiungere check su credit spreads + claims: senza confirming "
            "signals, downweight GDP_COLLAPSE."
        )
    elif record.regime == "reflation" and alpha < -0.05:
        pm.findings.append(
            "Reflation prediction underperformed: probabile che il regime "
            "fosse goldilocks o stagflation embrionica. "
            "Confidence boundary case probabile."
        )
        pm.suggestions.append(
            "Reflation con CPI > 3.5% dovrebbe ricevere stagflation overlay "
            "(inflation_persistent pillar)."
        )

    return pm


def find_failure_patterns(
    db: Session,
    horizon: str = "3m",
    min_alpha_threshold: float = -0.05,
    min_occurrences: int = 2,
) -> list[FailurePattern]:
    """Aggregate cross-failures: identifica pattern strutturali.

    Args:
        horizon: '1m' | '3m' | '6m'.
        min_alpha_threshold: alpha sotto questo = failure (default -5%).
        min_occurrences: pattern valido solo se ≥N occurrences.

    Returns:
        Lista di FailurePattern.
    """
    alpha_attr = f"alpha_{horizon}"
    failures = db.query(PredictionLog).filter(
        getattr(PredictionLog, alpha_attr).is_not(None),
        getattr(PredictionLog, alpha_attr) < min_alpha_threshold,
    ).all()

    if not failures:
        return []

    patterns: list[FailurePattern] = []

    # Pattern 1: regime con high failure rate
    by_regime: dict[str, list[float]] = {}
    for f in failures:
        by_regime.setdefault(f.regime, []).append(getattr(f, alpha_attr))
    for regime, alphas in by_regime.items():
        if len(alphas) >= min_occurrences:
            mean_a = sum(alphas) / len(alphas)
            patterns.append(FailurePattern(
                pattern_name=f"regime_{regime}_systematic",
                n_occurrences=len(alphas),
                mean_alpha=round(mean_a, 4),
                description=(
                    f"Regime '{regime}' ha {len(alphas)} failures con mean "
                    f"alpha {mean_a*100:.1f}%. Pattern strutturale: il classifier "
                    f"sbaglia sistematicamente su questo regime."
                ),
                affected_regimes=[regime],
                suggested_action=(
                    f"Audit pesi pillar del regime '{regime}' + consider "
                    f"adding cross-regime cross-check (es. credit spreads, DFM nowcast)."
                ),
            ))

    # Pattern 2: low confidence failures
    low_conf_failures = [f for f in failures if f.confidence < 0.30]
    if len(low_conf_failures) >= min_occurrences:
        alphas = [getattr(f, alpha_attr) for f in low_conf_failures]
        patterns.append(FailurePattern(
            pattern_name="low_confidence_systematic",
            n_occurrences=len(low_conf_failures),
            mean_alpha=round(sum(alphas) / len(alphas), 4),
            description=(
                f"{len(low_conf_failures)} failures con confidence < 0.30. "
                f"Quando il modello è incerto, è MEGLIO non agire."
            ),
            affected_regimes=list({f.regime for f in low_conf_failures}),
            suggested_action=(
                "Filtra automaticamente predictions con confidence < 0.30 → "
                "fallback a defensive allocation (USE_DEFENSIVE_TRANSITION_MODE)."
            ),
        ))

    # Pattern 3: boundary case failures (max prob < 0.40)
    boundary_failures = []
    for f in failures:
        probs = json.loads(f.probabilities_json) if f.probabilities_json else {}
        if probs and max(probs.values()) < 0.40:
            boundary_failures.append(f)
    if len(boundary_failures) >= min_occurrences:
        alphas = [getattr(f, alpha_attr) for f in boundary_failures]
        patterns.append(FailurePattern(
            pattern_name="boundary_case_systematic",
            n_occurrences=len(boundary_failures),
            mean_alpha=round(sum(alphas) / len(alphas), 4),
            description=(
                f"{len(boundary_failures)} failures con max_prob < 0.40 "
                f"(regime boundary). Il modello sceglieva tra alternative simili."
            ),
            affected_regimes=list({f.regime for f in boundary_failures}),
            suggested_action=(
                "Lower calm_regime_gate threshold OR aumenta defensive_transition "
                "frequency quando entropy > 1.5 e max_prob < 0.40."
            ),
        ))

    return patterns


def generate_improvement_suggestions(
    db: Session,
    horizon: str = "3m",
) -> dict[str, Any]:
    """Meta-analysis: aggregate suggestions pronte per council review.

    Output:
        dict con:
        - n_failures
        - failure_rate
        - mean_failure_alpha
        - patterns: list of FailurePattern as dicts
        - top_3_actions: prioritized action list

    Use case: console quarterly review. L'utente legge questa lista e decide
    quali patterns indagare manualmente per migliorare il modello.
    """
    alpha_attr = f"alpha_{horizon}"
    all_evaluated = db.query(PredictionLog).filter(
        getattr(PredictionLog, alpha_attr).is_not(None)
    ).all()
    if not all_evaluated:
        return {
            "n_evaluated": 0, "n_failures": 0, "failure_rate": 0.0,
            "mean_failure_alpha": None,
            "patterns": [], "top_actions": [],
            "diagnostic": "Track record vuoto. Attendere maturazione predictions.",
        }

    failures = [r for r in all_evaluated if getattr(r, alpha_attr) < -0.05]
    n_eval = len(all_evaluated)
    n_fail = len(failures)
    mean_fail_alpha = (
        sum(getattr(f, alpha_attr) for f in failures) / n_fail if n_fail else None
    )

    patterns = find_failure_patterns(db, horizon=horizon)
    # Top actions: ordina per (n_occurrences × |mean_alpha|) descending
    sorted_patterns = sorted(
        patterns,
        key=lambda p: p.n_occurrences * abs(p.mean_alpha),
        reverse=True,
    )

    return {
        "horizon": horizon,
        "n_evaluated": n_eval,
        "n_failures": n_fail,
        "failure_rate": round(n_fail / n_eval, 4) if n_eval else 0.0,
        "mean_failure_alpha": round(mean_fail_alpha, 4) if mean_fail_alpha else None,
        "patterns": [
            {
                "name": p.pattern_name,
                "n": p.n_occurrences,
                "mean_alpha": p.mean_alpha,
                "description": p.description,
                "affected_regimes": p.affected_regimes,
                "suggested_action": p.suggested_action,
            }
            for p in patterns
        ],
        "top_actions": [p.suggested_action for p in sorted_patterns[:3]],
        "diagnostic": (
            f"Failure rate {n_fail}/{n_eval} ({100*n_fail/n_eval:.0f}%) "
            f"con mean alpha {(mean_fail_alpha or 0)*100:.1f}%. "
            f"{len(patterns)} patterns identificati."
        ),
    }


def post_mortem_recent_failures(
    db: Session,
    horizon: str = "3m",
    limit: int = 10,
    min_alpha_threshold: float = -0.05,
) -> list[dict]:
    """Lista recent failures con post-mortem ciascuna.

    Use case: frontend "Failures & Learnings" panel.
    """
    alpha_attr = f"alpha_{horizon}"
    rows = (
        db.query(PredictionLog)
        .filter(
            getattr(PredictionLog, alpha_attr).is_not(None),
            getattr(PredictionLog, alpha_attr) < min_alpha_threshold,
        )
        .order_by(PredictionLog.date_predicted.desc())
        .limit(limit)
        .all()
    )

    out = []
    for r in rows:
        pm = post_mortem(db, r.id, horizon=horizon)
        if pm is None:
            continue
        out.append({
            "id": pm.prediction_id,
            "date_predicted": pm.date_predicted.isoformat() if pm.date_predicted else None,
            "regime": pm.regime,
            "confidence": pm.confidence,
            "horizon": pm.horizon,
            "alpha": pm.alpha,
            "severity": pm.severity,
            "regime_was_boundary": pm.regime_was_boundary,
            "confidence_was_low": pm.confidence_was_low,
            "findings": pm.findings,
            "suggestions": pm.suggestions,
        })
    return out
