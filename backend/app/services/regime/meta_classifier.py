"""Meta-regime classifier: Brier-weighted blend di multiple regime models.

Tier 1.1 della roadmap Bridgewater. Sostituisce il regime "single rule-based"
con un ensemble pesato dove ogni modello (rule-based, HMM-Market, MS-VAR,
+ in futuro FOMC nudge, term-premium signal) contribuisce in proporzione al
suo Brier score storico.

Componenti basso livello (pure functions):
- `brier_score`: misura accuratezza calibrazione di una sequenza di previsioni.
- `compute_brier_weights`: pesi inversamente proporzionali al Brier (modello
  migliore = peso più alto), con floor per evitare modelli quasi-zero.
- `blend_probabilities`: combina probs di N modelli via weighted average,
  con floor finale per evitare zero categorici.

Componente alto livello:
- `compute_meta_regime(db)`: orchestratore che recupera le history dei 3
  modelli, calcola Brier vs ground truth, deriva pesi, fa blend dell'output
  corrente in un'unica probability finale.

Filosofia: ognuno dei modelli vede una dimensione diversa della verità.
- rule-based: macro indicators completi (14 condizioni)
- HMM-Market: feature di mercato disgiunte (yield curve, copper/gold, dollar)
- MS-VAR: volatility regimes su S&P returns
Combinati con pesi calibrati = "diversification of belief" (concetto core
Bridgewater).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from loguru import logger
from sqlalchemy.orm import Session

from app.models import RegimeClassification
from app.services.regime.classifier import REGIMES

_EPS = 1e-12


def brier_score(
    predictions: list[dict[str, float]],
    labels: list[str],
) -> float:
    """Brier score multi-class: media degli errori quadratici tra distribuzione
    predetta e indicator del label vero.

    BS = (1/N) Σ_n Σ_k (p_nk - y_nk)^2
    dove y_nk = 1 se label_n == k else 0.

    Range [0, 2]:
    - 0 = previsione perfetta (P(true)=1).
    - 0.75 = uniforme su 4 classi.
    - 2 = anti-correlazione perfetta (P(false)=1).

    Lower is better. Label fuori da REGIMES vengono saltati (no crash).
    Se nessuna obs valida, ritorna 0.0.
    """
    if not predictions or not labels:
        return 0.0
    if len(predictions) != len(labels):
        raise ValueError(
            f"brier_score: len mismatch predictions={len(predictions)} labels={len(labels)}"
        )

    total = 0.0
    n_valid = 0
    for pred, label in zip(predictions, labels):
        if label not in REGIMES:
            continue
        sq = 0.0
        for r in REGIMES:
            y = 1.0 if r == label else 0.0
            p = float(pred.get(r, 0.0))
            sq += (p - y) ** 2
        total += sq
        n_valid += 1

    if n_valid == 0:
        return 0.0
    return total / n_valid


def compute_brier_weights(
    model_briers: dict[str, float],
    min_weight: float = 0.05,
) -> dict[str, float]:
    """Pesi ∝ 1/Brier, con floor `min_weight` per modello.

    Logica:
    1. Se tutti hanno stesso Brier → pesi uguali.
    2. Modello con Brier più basso = peso più alto (inverso).
    3. Floor `min_weight` evita che un modello peggiore scenda a peso ~0
       (preserva diversificazione anche se Brier è alto).
    4. Output sommato a 1.

    Implementazione: w_i_raw = 1 / (Brier_i + eps), poi normalizza, poi applica
    floor + renormalize. Se tutti uguali, ritorna pesi uniformi (no floor needed).
    """
    if not model_briers:
        return {}

    # Caso single-model: peso 1.0
    if len(model_briers) == 1:
        name = next(iter(model_briers))
        return {name: 1.0}

    # Caso tutti uguali: pesi uniformi (no floor noise)
    briers_list = list(model_briers.values())
    if all(abs(b - briers_list[0]) < 1e-12 for b in briers_list):
        n = len(model_briers)
        return {name: 1.0 / n for name in model_briers}

    # Caso generale: w_raw = 1 / (Brier + eps)
    raw = {name: 1.0 / (b + _EPS) for name, b in model_briers.items()}
    total = sum(raw.values())
    weights = {name: w / total for name, w in raw.items()}

    # Apply floor + renormalize
    n = len(weights)
    if min_weight > 0 and min_weight * n < 1.0:
        # Mass minima totale = min_weight * n, mass libera = 1 - min_weight*n
        free_mass = 1.0 - min_weight * n
        # Distribuisci free_mass proporzionalmente ai weights raw normalizzati
        weights = {
            name: min_weight + free_mass * w
            for name, w in weights.items()
        }

    return weights


def blend_probabilities(
    model_probs: dict[str, dict[str, float]],
    weights: dict[str, float],
    prob_floor: float = 0.0,
) -> dict[str, float]:
    """Output P(regime) = Σ_i w_i · P_i(regime), normalizzato Σ=1.

    Args:
        model_probs: {model_name: {regime: prob}}.
        weights: {model_name: weight}. Modelli non in weights vengono saltati.
        prob_floor: floor minimo per ogni regime nell'output (default 0.0 = no
            floor). Utile se l'output va in una pipeline che richiede prob > 0
            (es. log-space transforms).

    Fallback: se weights vuoto → distribuzione uniforme su REGIMES.
    """
    if not weights:
        return {r: 1.0 / len(REGIMES) for r in REGIMES}

    blended = {r: 0.0 for r in REGIMES}
    for model_name, w in weights.items():
        probs = model_probs.get(model_name)
        if not probs:
            continue
        for r in REGIMES:
            blended[r] += w * float(probs.get(r, 0.0))

    # Renormalize prima del floor
    K = len(REGIMES)
    total = sum(blended.values())
    if total > 0:
        normalized = {r: v / total for r, v in blended.items()}
    else:
        normalized = {r: 1.0 / K for r in REGIMES}

    if prob_floor <= 0:
        return normalized

    # Floor via mix lineare con uniform: out = (1 - floor*K) * raw + floor.
    # Questo garantisce Σ=1 esatto + ogni regime >= floor (purche' floor*K < 1).
    if prob_floor * K >= 1.0:
        return {r: 1.0 / K for r in REGIMES}
    free_mass = 1.0 - prob_floor * K
    return {r: free_mass * p + prob_floor for r, p in normalized.items()}


# ============================================================================
# Alto livello: compute_meta_regime
# ============================================================================


@dataclass
class ModelContribution:
    """Contributo di un singolo modello nel meta-classifier."""
    name: str
    current_probabilities: dict[str, float]
    brier_score: float
    weight: float
    n_history_points: int
    error: str | None = None


@dataclass
class MetaRegimeResult:
    """Output del meta-regime classifier Brier-weighted."""
    date: str
    regime: str                              # argmax di final_probabilities
    final_probabilities: dict[str, float]    # blend di tutti i modelli validi
    contributions: list[ModelContribution]   # breakdown per modello
    ground_truth_window_months: int          # finestra usata per Brier
    notes: list[str] = field(default_factory=list)


def _align_predictions_to_labels(
    history: list[dict],
    label_dates: list[pd.Timestamp],
    label_lookup: dict[pd.Timestamp, str],
) -> tuple[list[dict[str, float]], list[str]]:
    """Allinea le predizioni di un modello (lista di {date, regime_probs}) ai
    label rule-based (ground truth). Ritorna le coppie (pred, label) allineate
    sulle date comuni.
    """
    pred_by_date: dict[pd.Timestamp, dict[str, float]] = {}
    for entry in history:
        date_str = entry.get("date")
        probs = entry.get("regime_probs")
        if not date_str or not isinstance(probs, dict):
            continue
        try:
            ts = pd.Timestamp(date_str)
        except Exception:
            continue
        # Allinea al month-end per match coi label mensili
        ts_month = ts.to_period("M").to_timestamp("M").normalize()
        pred_by_date[ts_month] = probs

    aligned_preds: list[dict[str, float]] = []
    aligned_labels: list[str] = []
    for ts in label_dates:
        ts_norm = ts.to_period("M").to_timestamp("M").normalize()
        pred = pred_by_date.get(ts_norm)
        if pred is None:
            continue
        label = label_lookup.get(ts_norm) or label_lookup.get(ts)
        if label is None:
            continue
        aligned_preds.append(pred)
        aligned_labels.append(label)
    return aligned_preds, aligned_labels


def compute_meta_regime(
    db: Session,
    lookback_months: int = 60,
    min_weight: float = 0.10,
    prob_floor: float = 0.02,
    apply_fomc: bool = True,
    fomc_half_life_days: int = 30,
) -> MetaRegimeResult:
    """Calcola la regime probability finale via Brier-weighted blend.

    Pipeline:
    1. Ground truth = rule-based history dal DB sull'ultimo `lookback_months`
       (label hard = argmax delle probability storiche).
    2. Per ogni modello (rule-based, HMM-Market, MS-VAR) recupera la sua
       predizione storica allineata alle stesse date e calcola Brier.
    3. compute_brier_weights → pesi inversamente proporzionali al Brier
       (con floor `min_weight` per preservare diversificazione).
    4. blend_probabilities su predizioni CORRENTI dei modelli + pesi.
    5. Ritorna probabilities + breakdown contributi.

    Caveat: rule-based è ground truth → suo Brier ~0 → peso dominante.
    Mitigation: `min_weight=0.10` garantisce HMM-Market/MS-VAR almeno 10%
    ognuno (preserva diversification of belief). Per ground truth davvero
    indipendente in v2: NBER recession labels (binario stag/defl vs refl/gold).
    """
    from app.models import RegimeClassification
    from app.services.regime.hmm_market import fit_and_predict_hmm_market
    from app.services.regime.msvar import fit_and_predict_msvar

    notes: list[str] = []

    # Step 1: rule-based history come ground truth
    rows: list[RegimeClassification] = (
        db.query(RegimeClassification)
        .order_by(RegimeClassification.date.asc())
        .all()
    )
    if not rows:
        notes.append("Nessuna RegimeClassification in DB: meta degraded a uniform")
        return MetaRegimeResult(
            date=pd.Timestamp.now().date().isoformat(),
            regime="reflation",
            final_probabilities={r: 0.25 for r in REGIMES},
            contributions=[],
            ground_truth_window_months=0,
            notes=notes,
        )

    rule_df = pd.DataFrame([
        {
            "date": pd.Timestamp(r.date),
            "reflation": r.probability_reflation,
            "stagflation": r.probability_stagflation,
            "deflation": r.probability_deflation,
            "goldilocks": r.probability_goldilocks,
        }
        for r in rows
    ]).set_index("date").sort_index().resample("ME").mean(numeric_only=True)
    rule_df = rule_df.dropna(subset=REGIMES).tail(lookback_months)
    if rule_df.empty:
        notes.append("Nessun mese rule-based valido nel lookback: meta degraded a uniform")
        return MetaRegimeResult(
            date=pd.Timestamp.now().date().isoformat(),
            regime="reflation",
            final_probabilities={r: 0.25 for r in REGIMES},
            contributions=[],
            ground_truth_window_months=lookback_months,
            notes=notes,
        )
    rule_df["regime"] = rule_df[REGIMES].idxmax(axis=1)

    label_dates = list(rule_df.index)
    label_lookup = {ts.normalize(): str(rule_df.loc[ts, "regime"]) for ts in label_dates}
    label_dates_norm = [ts.normalize() for ts in label_dates]

    # Step 2: predizioni di ogni modello + Brier
    contributions: list[ModelContribution] = []

    # 2a. rule-based: predizioni storiche dal DB stesso (tautologico ma simmetrico)
    rule_preds = [
        {r: float(rule_df.loc[ts, r]) for r in REGIMES}
        for ts in label_dates
    ]
    rule_labels = [label_lookup[ts.normalize()] for ts in label_dates]
    rule_brier = brier_score(rule_preds, rule_labels)
    rule_current = rule_preds[-1] if rule_preds else {r: 0.25 for r in REGIMES}
    contributions.append(ModelContribution(
        name="rule_based",
        current_probabilities=rule_current,
        brier_score=rule_brier,
        weight=0.0,  # placeholder, set sotto
        n_history_points=len(rule_preds),
    ))

    # 2b. HMM-Market
    try:
        hmm_market = fit_and_predict_hmm_market(db, n_states=4)
        preds_hm, labels_hm = _align_predictions_to_labels(
            hmm_market.regime_history, label_dates_norm, label_lookup,
        )
        hm_brier = brier_score(preds_hm, labels_hm) if preds_hm else 0.75
        contributions.append(ModelContribution(
            name="hmm_market",
            current_probabilities=hmm_market.probabilities,
            brier_score=hm_brier,
            weight=0.0,
            n_history_points=len(preds_hm),
        ))
    except Exception as e:
        logger.warning(f"meta_classifier: HMM-Market failed: {e}")
        contributions.append(ModelContribution(
            name="hmm_market",
            current_probabilities={r: 0.25 for r in REGIMES},
            brier_score=0.75,
            weight=0.0,
            n_history_points=0,
            error=str(e),
        ))

    # 2c. MS-VAR
    try:
        msvar = fit_and_predict_msvar(db, n_states=2)
        preds_ms, labels_ms = _align_predictions_to_labels(
            msvar.regime_history, label_dates_norm, label_lookup,
        )
        ms_brier = brier_score(preds_ms, labels_ms) if preds_ms else 0.75
        contributions.append(ModelContribution(
            name="msvar",
            current_probabilities=msvar.probabilities,
            brier_score=ms_brier,
            weight=0.0,
            n_history_points=len(preds_ms),
        ))
    except Exception as e:
        logger.warning(f"meta_classifier: MS-VAR failed: {e}")
        contributions.append(ModelContribution(
            name="msvar",
            current_probabilities={r: 0.25 for r in REGIMES},
            brier_score=0.75,
            weight=0.0,
            n_history_points=0,
            error=str(e),
        ))

    # Step 3: Brier weights (escludendo modelli failed)
    valid = [c for c in contributions if c.error is None and c.n_history_points > 0]
    if not valid:
        notes.append("Tutti i modelli falliti: meta degraded a uniform")
        return MetaRegimeResult(
            date=label_dates[-1].date().isoformat() if label_dates else pd.Timestamp.now().date().isoformat(),
            regime="reflation",
            final_probabilities={r: 0.25 for r in REGIMES},
            contributions=contributions,
            ground_truth_window_months=lookback_months,
            notes=notes,
        )

    briers_dict = {c.name: c.brier_score for c in valid}
    weights = compute_brier_weights(briers_dict, min_weight=min_weight)

    # Aggiorna contributions con i pesi calcolati
    for c in contributions:
        c.weight = weights.get(c.name, 0.0)

    # Step 4: blend dei current probs
    current_probs = {c.name: c.current_probabilities for c in valid}
    final = blend_probabilities(current_probs, weights, prob_floor=prob_floor)

    # Step 5 (Tier 1.2): FOMC nudge applicato come post-correzione.
    # Se il blocco fallisce per qualunque motivo, log + skip (non bloccante).
    if apply_fomc:
        try:
            from app.services.fomc.fomc_loader import latest_fomc_nudge
            from app.services.regime.fomc_nudge import apply_fomc_nudge

            nudge_data = latest_fomc_nudge()
            if nudge_data is not None:
                final = apply_fomc_nudge(
                    final,
                    nudge=nudge_data["nudge"],
                    confidence=nudge_data["confidence"],
                    doc_date=nudge_data["doc_date"],
                    half_life_days=fomc_half_life_days,
                )
                notes.append(
                    f"FOMC nudge applicato: doc {nudge_data['doc_date']}, "
                    f"confidence {nudge_data['confidence']:.2f}, "
                    f"half_life {fomc_half_life_days}d."
                )
        except Exception as e:
            logger.warning(f"meta_classifier: FOMC nudge skipped ({e})")

    dominant = max(final, key=final.get)

    failed = [c.name for c in contributions if c.error]
    if failed:
        notes.append(f"Modelli falliti (esclusi): {failed}")
    if rule_brier < 1e-6:
        notes.append(
            "rule_based brier ~ 0 (ground truth tautologico). HMM-Market/MS-VAR "
            "ricevono floor min_weight per preservare diversification of belief."
        )

    logger.info(
        f"compute_meta_regime: window={lookback_months}m, "
        f"weights={ {k: round(v, 3) for k, v in weights.items()} }, "
        f"dominant={dominant} ({final[dominant]:.2f})"
    )

    return MetaRegimeResult(
        date=label_dates[-1].date().isoformat() if label_dates else pd.Timestamp.now().date().isoformat(),
        regime=dominant,
        final_probabilities=final,
        contributions=contributions,
        ground_truth_window_months=lookback_months,
        notes=notes,
    )
