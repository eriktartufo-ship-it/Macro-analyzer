"""Ensemble regime: combina rule-based + HMM-Market + MS-VAR.

Strategia:
  1. Recupera la posterior corrente di ciascun modello
  2. Calcola weighted average con pesi configurabili (default: equal 1/3 ciascuno)
  3. Calcola **disagreement metric** via Jensen-Shannon divergence pairwise
  4. Determina **confidence**: alta se modelli concordano (low JS), bassa se discordano
  5. Flag esplicito quando JS > soglia → "alto disaccordo, prudenza"

Quando 3 modelli concordano → segnale forte (es. tutti dicono reflation > 0.5).
Quando discordano → output ensemble e' una media ma con confidence ridotto e
flag visibile all'utente. Nessun modello ha veto, tutti contribuiscono.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger
from sqlalchemy.orm import Session

from app.models import RegimeClassification
from app.services.regime.classifier import REGIMES

# Pesi default (equal). Esposti per future calibrazioni via backtest accuracy.
DEFAULT_WEIGHTS = {
    "rule_based": 1 / 3,
    "hmm_market": 1 / 3,
    "msvar": 1 / 3,
}

# Soglia oltre cui il flag disagreement scatta (JS divergence in [0, log(2)] ≈ 0.69)
DISAGREEMENT_THRESHOLD = 0.20


@dataclass
class ModelView:
    name: str
    probabilities: dict[str, float]
    error: str | None = None
    metadata: dict | None = None


@dataclass
class EnsembleResult:
    weights: dict[str, float]
    views: list[ModelView]
    ensemble_probabilities: dict[str, float]
    confidence: float                  # 0..1, 1 = pieno accordo
    disagreement_score: float          # JS-divergence media pairwise
    high_disagreement: bool
    dominant_regime: str
    notes: list[str]


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon: simmetrica, in [0, log 2]. 0 = identiche."""
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _to_array(probs: dict[str, float]) -> np.ndarray:
    return np.array([probs.get(r, 0.0) for r in REGIMES], dtype=float)


def _normalize(p: np.ndarray) -> np.ndarray:
    s = p.sum()
    if s <= 0:
        return np.full_like(p, 1.0 / len(p))
    return p / s


def _safe_rule_based(db: Session) -> ModelView:
    """Pesca l'ultima classificazione rule-based dal DB."""
    last = (
        db.query(RegimeClassification)
        .order_by(RegimeClassification.date.desc())
        .first()
    )
    if not last:
        return ModelView(name="rule_based", probabilities={r: 0.25 for r in REGIMES},
                         error="nessuna classificazione in DB")
    return ModelView(
        name="rule_based",
        probabilities={
            "reflation": last.probability_reflation,
            "stagflation": last.probability_stagflation,
            "deflation": last.probability_deflation,
            "goldilocks": last.probability_goldilocks,
        },
        metadata={"date": last.date.isoformat(), "regime": last.regime, "confidence": last.confidence},
    )


def _safe_hmm_market(db: Session) -> ModelView:
    try:
        from app.services.regime.hmm_market import fit_and_predict_hmm_market
        r = fit_and_predict_hmm_market(db, n_states=4)
        return ModelView(
            name="hmm_market",
            probabilities=r.probabilities,
            metadata={
                "n_training": r.n_training,
                "log_likelihood": r.log_likelihood,
                "state_to_regime": {str(k): v for k, v in r.state_to_regime.items()},
            },
        )
    except Exception as e:
        logger.warning(f"HMM-Market failed in ensemble: {e}")
        return ModelView(
            name="hmm_market", probabilities={r: 0.25 for r in REGIMES},
            error=str(e),
        )


def _safe_msvar(db: Session) -> ModelView:
    try:
        from app.services.regime.msvar import fit_and_predict_msvar
        r = fit_and_predict_msvar(db, n_states=2)
        return ModelView(
            name="msvar",
            probabilities=r.probabilities,
            metadata={
                "n_training": r.n_training,
                "log_likelihood": r.log_likelihood,
                "state_means": {str(k): v for k, v in r.state_means.items()},
                "state_vols": {str(k): v for k, v in r.state_vols.items()},
                "state_to_regime": {str(k): v for k, v in r.state_to_regime.items()},
            },
        )
    except Exception as e:
        logger.warning(f"MS-VAR failed in ensemble: {e}")
        return ModelView(
            name="msvar", probabilities={r: 0.25 for r in REGIMES},
            error=str(e),
        )


def compute_ensemble(
    db: Session,
    weights: dict[str, float] | None = None,
) -> EnsembleResult:
    w = weights or DEFAULT_WEIGHTS

    views = [
        _safe_rule_based(db),
        _safe_hmm_market(db),
        _safe_msvar(db),
    ]

    # Filtra modelli falliti (uniform 0.25 = no signal). Riallinea pesi.
    valid = [v for v in views if v.error is None]
    if len(valid) == 0:
        return EnsembleResult(
            weights=w,
            views=views,
            ensemble_probabilities={r: 0.25 for r in REGIMES},
            confidence=0.0,
            disagreement_score=0.0,
            high_disagreement=False,
            dominant_regime="reflation",
            notes=["tutti i modelli hanno errori, ensemble degradato a uniforme"],
        )

    # Weighted average
    sum_weights = sum(w.get(v.name, 0.0) for v in valid)
    if sum_weights <= 0:
        sum_weights = 1.0
    ens = np.zeros(len(REGIMES))
    for v in valid:
        wv = w.get(v.name, 0.0) / sum_weights
        ens += wv * _to_array(v.probabilities)
    ens = _normalize(ens)

    # Disagreement: JS divergence media pairwise tra modelli validi
    js_pairs: list[float] = []
    arrs = [_to_array(v.probabilities) for v in valid]
    for i in range(len(arrs)):
        for j in range(i + 1, len(arrs)):
            js_pairs.append(_js_divergence(arrs[i], arrs[j]))
    avg_js = float(np.mean(js_pairs)) if js_pairs else 0.0

    # Confidence: 1 - normalized JS. Max JS atteso ≈ log(2) ≈ 0.69
    confidence = max(0.0, min(1.0, 1.0 - (avg_js / 0.4)))
    high_disagreement = avg_js >= DISAGREEMENT_THRESHOLD

    ens_dict = {r: float(ens[i]) for i, r in enumerate(REGIMES)}
    dominant = max(ens_dict, key=ens_dict.get)

    notes: list[str] = []
    if high_disagreement:
        # Identifica chi e' l'outlier (max JS rispetto agli altri)
        if len(valid) >= 3:
            js_per_model = []
            for i, vi in enumerate(valid):
                others = np.mean([
                    _js_divergence(arrs[i], arrs[j]) for j in range(len(arrs)) if j != i
                ])
                js_per_model.append((vi.name, others))
            outlier = max(js_per_model, key=lambda kv: kv[1])
            notes.append(
                f"Alto disaccordo (avg JS {avg_js:.2f} ≥ {DISAGREEMENT_THRESHOLD}). "
                f"Outlier: {outlier[0]} (JS medio {outlier[1]:.2f} dagli altri)."
            )
        else:
            notes.append(f"Alto disaccordo (avg JS {avg_js:.2f}).")

    failed = [v.name for v in views if v.error]
    if failed:
        notes.append(f"Modelli falliti (esclusi dall'ensemble): {failed}")

    return EnsembleResult(
        weights=w,
        views=views,
        ensemble_probabilities=ens_dict,
        confidence=confidence,
        disagreement_score=avg_js,
        high_disagreement=high_disagreement,
        dominant_regime=dominant,
        notes=notes,
    )
