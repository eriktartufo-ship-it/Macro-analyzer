"""Tier 8 — ML-based regime classifier (RandomForest + GradientBoosting).

Approccio:
1. Feature engineering: indicators raw + z-score rolling 10y + deltas (gdp_roc - cpi_yoy, etc.)
2. Training: 18 episodi storici labeled (T8 historical_episodes.py).
3. Model: ensemble RandomForest (robusto, non overfit) + GradientBoosting (preciso).
4. Output: 4-regime probability distribution (compatibile con rule-based output).

NON sostituisce il rule-based: viene blendato in Tier 8.3 con weighted Brier score.

Pattern:
- `extract_features(indicators)` → array numpy (29 features)
- `train_classifier()` → (rf_model, gb_model, feature_names)
- `predict_regime(indicators, models)` → {regime: prob}
- `validate_oos()` → leave-one-out cross-validation con Brier score
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.services.regime.historical_episodes import HISTORICAL_EPISODES, HistoricalEpisode


# 17 base features + 12 derived = 29 feature vector
BASE_INDICATORS = (
    "gdp_roc", "cpi_yoy", "unrate", "unrate_roc",
    "vix", "fed_funds", "breakeven_10y", "baa_10y_spread",
    "pmi", "yield_curve_spread", "nfci", "core_pce_yoy",
    "consumer_sentiment", "claims_roc", "lei_roc", "indpro_yoy",
    "payrolls_yoy", "housing_starts_yoy", "acm_term_premium",
)

# T9-AUDIT-BUG-FIX: alias da nomi JSON live → nomi ML training set.
# Il fetcher live salva con nomi diversi dal training (es. `fed_funds_rate` vs
# `fed_funds`). Senza questo mapping, ML usa default per 8/19 features →
# predictions inaffidabili (bug 2026-05-16 deflation 73% con CPI 3.95%).
_INDICATOR_ALIASES: dict[str, tuple[str, ...]] = {
    "fed_funds": ("fed_funds_rate",),
    "baa_10y_spread": ("baa_spread",),
    "yield_curve_spread": ("yield_curve_10y2y",),
    "claims_roc": ("initial_claims_roc",),
    "indpro_yoy": ("indpro_roc_12m",),
    "payrolls_yoy": ("payrolls_roc_12m",),
    "housing_starts_yoy": ("housing_starts_roc_12m",),
    "acm_term_premium": ("term_premium_10y",),
}

# Anchor values per z-score normalizzato (storica long-run mean).
# Permette interpretation cross-time (1973 cpi 10 != 2024 cpi 2.5 ma z-score comparabile).
ANCHOR_MEAN: dict[str, float] = {
    "gdp_roc": 2.5, "cpi_yoy": 3.0, "unrate": 5.5, "unrate_roc": 0.0,
    "vix": 19.0, "fed_funds": 4.0, "breakeven_10y": 2.3, "baa_10y_spread": 2.2,
    "pmi": 50.0, "yield_curve_spread": 1.0, "nfci": 0.0, "core_pce_yoy": 2.5,
    "consumer_sentiment": 80.0, "claims_roc": 0.0, "lei_roc": 0.5,
    "indpro_yoy": 2.0, "payrolls_yoy": 1.5, "housing_starts_yoy": 2.0,
    "acm_term_premium": 0.5,
}

ANCHOR_STD: dict[str, float] = {
    "gdp_roc": 3.0, "cpi_yoy": 3.0, "unrate": 2.5, "unrate_roc": 1.0,
    "vix": 10.0, "fed_funds": 4.0, "breakeven_10y": 1.0, "baa_10y_spread": 1.5,
    "pmi": 6.0, "yield_curve_spread": 1.5, "nfci": 0.8, "core_pce_yoy": 2.5,
    "consumer_sentiment": 15.0, "claims_roc": 10.0, "lei_roc": 3.0,
    "indpro_yoy": 4.0, "payrolls_yoy": 2.0, "housing_starts_yoy": 15.0,
    "acm_term_premium": 0.8,
}

REGIMES = ["reflation", "stagflation", "deflation", "goldilocks"]


@dataclass
class MLModelBundle:
    """Pacchetto modelli + meta per inference."""
    rf_model: Any  # RandomForestClassifier
    gb_model: Any  # GradientBoostingClassifier
    feature_names: list[str]
    classes: list[str]


def _safe(d: dict[str, Any], key: str, default: float = 0.0) -> float:
    """NaN/Inf safe getter + alias fallback (T9-AUDIT-BUG-FIX).

    Cerca:
    1. `key` diretto nel dict
    2. Tutti gli alias in `_INDICATOR_ALIASES[key]`
    3. Default
    """
    import math

    def _try_value(v) -> float | None:
        if v is None:
            return None
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        if math.isnan(v) or math.isinf(v):
            return None
        return v

    # Try primary key first
    if key in d:
        result = _try_value(d[key])
        if result is not None:
            return result

    # Try aliases
    for alias in _INDICATOR_ALIASES.get(key, ()):
        if alias in d:
            result = _try_value(d[alias])
            if result is not None:
                return result

    return default


def extract_features_minimal(
    indicators: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    """T9-FIX-2: Feature set ridotto a 8 z-score core.

    Post-audit council (taleb + raydalio): 39 features × 32 samples = ratio 0.82
    è sub-determinato statisticamente. 8 z-score covarianti basse + decisive.

    Features (tutti z-score normalizzati):
    - gdp_z, cpi_z, unrate_z, vix_z, breakeven_z, baa_z, yield_curve_z, real_fed_funds_z

    Default per nuovo training. Mantiene `extract_features` legacy per back-compat.
    """
    feats: list[float] = []
    names: list[str] = []

    def z(key: str) -> float:
        v = _safe(indicators, key, ANCHOR_MEAN[key])
        return (v - ANCHOR_MEAN[key]) / ANCHOR_STD[key]

    for key in (
        "gdp_roc", "cpi_yoy", "unrate", "vix",
        "breakeven_10y", "baa_10y_spread",
    ):
        feats.append(z(key))
        names.append(f"{key}_z")

    # Yield curve z (proxy: yield_curve_spread)
    yc_v = _safe(indicators, "yield_curve_spread", 1.0)
    yc_anchor_mean, yc_anchor_std = 1.0, 1.5  # storico USA post-1970
    feats.append((yc_v - yc_anchor_mean) / yc_anchor_std)
    names.append("yield_curve_z")

    # Real Fed funds z (fed_funds - cpi_yoy normalizzato)
    fed = _safe(indicators, "fed_funds", ANCHOR_MEAN["fed_funds"])
    cpi = _safe(indicators, "cpi_yoy", ANCHOR_MEAN["cpi_yoy"])
    real_fed = fed - cpi
    real_fed_anchor_mean, real_fed_anchor_std = 1.0, 3.0
    feats.append((real_fed - real_fed_anchor_mean) / real_fed_anchor_std)
    names.append("real_fed_funds_z")

    return np.array(feats, dtype=np.float64), names


def extract_features(indicators: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    """Feature engineering: 19 raw + 10 z-score + 10 derived = 39 feature totali.

    LEGACY (T8): mantenuto per back-compat. Per nuovo training preferire
    `extract_features_minimal` (8 features post-audit T9).

    Derivati:
    - gdp_minus_cpi: indicatore crescita reale
    - real_fed_funds: fed_funds - cpi_yoy
    - growth_z: avg(gdp z, payrolls z, indpro z) — "growth dimension"
    - inflation_z: avg(cpi z, core_pce z, breakeven z) — "inflation dimension"
    - stress_z: avg(vix z, baa z, nfci z, claims z) — "stress dimension"
    - labor_tight: max(0, 5.0 - unrate) (1 if unrate << 5)
    - is_recession_signal: gdp<1 AND unrate_roc>0.2 (binary)
    - is_inflation_high: cpi_yoy > 4 (binary)
    - is_disinflation: breakeven < 1.8 AND cpi_yoy < 3 (binary, anticipatory)
    - is_inversion: yield_curve_spread < 0 (binary)
    """
    features: list[float] = []
    names: list[str] = []

    # AUDIT 2026-07-12 (CRITICAL): NAPM discontinuato → live non ha mai `pmi`.
    # Il proxy indpro (T9-AUDIT-BUG-4) era applicato solo al rule-based: il
    # blend ML vedeva sempre ANCHOR_MEAN 50.0 (manifatturiero cieco ogni
    # giorno). Stesso proxy qui: gli episodi storici hanno il PMI vero e
    # restano invariati, il live usa il proxy → train e serve coerenti.
    from app.services.common.indicator_aliases import derive_pmi_proxy
    indicators = {**indicators, "pmi": derive_pmi_proxy(indicators)}

    # Raw 19
    for k in BASE_INDICATORS:
        features.append(_safe(indicators, k, ANCHOR_MEAN.get(k, 0.0)))
        names.append(k)

    # Z-scores (10): subset core indicators
    for k in ("gdp_roc", "cpi_yoy", "unrate", "vix", "fed_funds",
              "breakeven_10y", "baa_10y_spread", "pmi",
              "core_pce_yoy", "payrolls_yoy"):
        v = _safe(indicators, k, ANCHOR_MEAN[k])
        z = (v - ANCHOR_MEAN[k]) / ANCHOR_STD[k]
        features.append(z)
        names.append(f"{k}_z")

    # Derived dimensions
    gdp_z = (features[0] - ANCHOR_MEAN["gdp_roc"]) / ANCHOR_STD["gdp_roc"]
    cpi_z = (features[1] - ANCHOR_MEAN["cpi_yoy"]) / ANCHOR_STD["cpi_yoy"]
    pay_z = (features[16] - ANCHOR_MEAN["payrolls_yoy"]) / ANCHOR_STD["payrolls_yoy"]
    indpro_z = (features[15] - ANCHOR_MEAN["indpro_yoy"]) / ANCHOR_STD["indpro_yoy"]
    pce_z = (features[11] - ANCHOR_MEAN["core_pce_yoy"]) / ANCHOR_STD["core_pce_yoy"]
    breakeven_z = (features[6] - ANCHOR_MEAN["breakeven_10y"]) / ANCHOR_STD["breakeven_10y"]
    vix_z = (features[4] - ANCHOR_MEAN["vix"]) / ANCHOR_STD["vix"]
    baa_z = (features[7] - ANCHOR_MEAN["baa_10y_spread"]) / ANCHOR_STD["baa_10y_spread"]
    nfci_z = (features[10] - ANCHOR_MEAN["nfci"]) / ANCHOR_STD["nfci"]
    claims_z = (features[13] - ANCHOR_MEAN["claims_roc"]) / ANCHOR_STD["claims_roc"]

    gdp_minus_cpi = features[0] - features[1]
    real_fed_funds = features[5] - features[1]
    growth_z = (gdp_z + pay_z + indpro_z) / 3.0
    inflation_z = (cpi_z + pce_z + breakeven_z) / 3.0
    stress_z = (vix_z + baa_z + nfci_z + claims_z) / 4.0
    labor_tight = max(0.0, 5.0 - features[2])
    is_recession_signal = 1.0 if (features[0] < 1.0 and features[3] > 0.2) else 0.0
    is_inflation_high = 1.0 if features[1] > 4.0 else 0.0
    is_disinflation = 1.0 if (features[6] < 1.8 and features[1] < 3.0) else 0.0
    is_inversion = 1.0 if features[9] < 0.0 else 0.0

    for name, val in (
        ("gdp_minus_cpi", gdp_minus_cpi),
        ("real_fed_funds", real_fed_funds),
        ("growth_z", growth_z),
        ("inflation_z", inflation_z),
        ("stress_z", stress_z),
        ("labor_tight", labor_tight),
        ("is_recession_signal", is_recession_signal),
        ("is_inflation_high", is_inflation_high),
        ("is_disinflation", is_disinflation),
        ("is_inversion", is_inversion),
    ):
        features.append(val)
        names.append(name)

    return np.array(features, dtype=np.float64), names


def build_training_data(
    episodes: list[HistoricalEpisode] | None = None,
    minimal: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Costruisce X (n_episodes, n_features), y (n_episodes,) + feature_names.

    Args:
        minimal: Se True usa 8 features (sperimentale T9, OOS peggiore in prove
            empiriche). Default False = 39 features legacy (LOO 71.9%, OOS 66.7%).
    """
    if episodes is None:
        episodes = HISTORICAL_EPISODES
    X_list = []
    y_list = []
    feat_names: list[str] = []
    extractor = extract_features_minimal if minimal else extract_features
    for ep in episodes:
        feat, names = extractor(ep.indicators)
        X_list.append(feat)
        y_list.append(ep.true_regime)
        if not feat_names:
            feat_names = names
    X = np.vstack(X_list)
    y = np.array(y_list)
    return X, y, feat_names


def train_classifier(
    episodes: list[HistoricalEpisode] | None = None,
    random_state: int = 42,
    minimal_features: bool = False,
) -> MLModelBundle:
    """Train RandomForest + GradientBoosting ensemble.

    Iperparametri default (39-feature legacy):
    - RF: n_estimators=200, max_depth=4, min_samples_leaf=2
    - GB: n_estimators=100, max_depth=3, learning_rate=0.05

    Args:
        minimal_features: Se True usa 8 z-score features (sperimentale T9,
            empiricamente OOS peggiore). Default False (39 features legacy).
            Empirical validation T9-FIX-2:
            - 39 feat: LOO 71.9%, OOS 66.7%
            - 8 feat:  LOO 56.2%, OOS 33.3% (counter-intuitive: meno feat peggio)
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    X, y, feat_names = build_training_data(episodes, minimal=minimal_features)
    # Anti-overfit hyperparams: depth più bassa per 8-feature minimal
    rf_depth = 3 if minimal_features else 4
    gb_depth = 2 if minimal_features else 3
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=rf_depth, min_samples_leaf=2,
        random_state=random_state, n_jobs=-1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=100, max_depth=gb_depth, learning_rate=0.05,
        random_state=random_state,
    )
    rf.fit(X, y)
    gb.fit(X, y)
    return MLModelBundle(
        rf_model=rf, gb_model=gb,
        feature_names=feat_names, classes=list(rf.classes_),
    )


def predict_regime(
    indicators: dict[str, Any],
    bundle: MLModelBundle,
    weights: tuple[float, float] = (0.5, 0.5),
) -> dict[str, float]:
    """Inference: ritorna {regime: probability} con somma=1.

    Args:
        weights: (w_rf, w_gb) per ensemble. Default 50/50.
    """
    # T9-FIX-2: usa extractor coerente con feature_names del bundle
    n_features_expected = len(bundle.feature_names)
    if n_features_expected <= 12:  # minimal 8-feature bundle
        feat, _ = extract_features_minimal(indicators)
    else:
        feat, _ = extract_features(indicators)
    feat_2d = feat.reshape(1, -1)
    proba_rf = bundle.rf_model.predict_proba(feat_2d)[0]
    proba_gb = bundle.gb_model.predict_proba(feat_2d)[0]
    w_rf, w_gb = weights
    blend = w_rf * proba_rf + w_gb * proba_gb
    blend = blend / blend.sum()  # safety renorm

    out = {}
    for cls, p in zip(bundle.classes, blend):
        out[cls] = float(p)
    # Garantisce 4 regimi presenti (anche se ML non li ha visti)
    for r in REGIMES:
        out.setdefault(r, 0.001)
    # Renormalize
    s = sum(out.values())
    return {k: v / s for k, v in out.items()}


def loo_cross_validation() -> dict[str, Any]:
    """Leave-one-out cross-validation: ogni episodio testato escludendolo dal training.

    Returns:
        {
            "accuracy": float,  # episodi predetti correttamente / totali
            "brier_score": float,  # multiclass Brier score (lower = better)
            "per_episode": [{name, true, predicted, correct, brier}],
        }
    """
    n = len(HISTORICAL_EPISODES)
    correct = 0
    brier_total = 0.0
    per_episode = []

    for i, test_ep in enumerate(HISTORICAL_EPISODES):
        train_eps = HISTORICAL_EPISODES[:i] + HISTORICAL_EPISODES[i + 1:]
        bundle = train_classifier(train_eps)
        pred = predict_regime(test_ep.indicators, bundle)
        predicted_regime = max(pred, key=pred.get)
        is_correct = predicted_regime == test_ep.true_regime
        if is_correct:
            correct += 1
        # Brier score: sum (p_r - y_r)^2 / n_classes
        brier_ep = sum(
            (pred.get(r, 0.0) - (1.0 if r == test_ep.true_regime else 0.0)) ** 2
            for r in REGIMES
        ) / len(REGIMES)
        brier_total += brier_ep
        per_episode.append({
            "name": test_ep.name,
            "true": test_ep.true_regime,
            "predicted": predicted_regime,
            "correct": is_correct,
            "brier": round(brier_ep, 4),
            "probs": {r: round(p, 3) for r, p in pred.items()},
        })

    return {
        "accuracy": round(correct / n, 3),
        "brier_score": round(brier_total / n, 4),
        "n_episodes": n,
        "per_episode": per_episode,
    }
