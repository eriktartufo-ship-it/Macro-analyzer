"""Diagnostic script: confronto HMM macro vs HMM-Market sullo stesso giorno.

Risponde a: perche' HMM macro vede goldilocks 94% mentre HMM-Market vede
stagflation 94% nello stesso giorno?

Output:
1. Per HMM macro: state means (de-standardizzati), posterior corrente,
   distanza dal punto corrente per ogni stato.
2. Per HMM-Market: feature means + values + state means standardizzati,
   posterior corrente.
3. Sintesi: quale feature/stato e' responsabile del disaccordo.

Uso (dal repo root del backend):
    "d:/Antigravity/.venv/Scripts/python.exe" scripts/diag_hmm_vs_market.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make 'app' importable
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))

import numpy as np

from app.database import SessionLocal
from app.services.regime.hmm_classifier import (
    _CORE_FEATURES,
    _EXTENDED_FEATURES,
    _baum_welch,
    _extract_feature_matrix,
    _map_states_to_regimes,
    _standardize,
)
from app.models import RegimeClassification
from app.services.regime.classifier import REGIMES
from app.services.regime.market_features import compute_market_features
from app.services.regime.hmm_market import (
    _map_states_via_soft_correlation,
    _standardize as _std_market,
)
from app.services.regime.hmm_classifier import _baum_welch as _bw_macro


_FEATURES = _CORE_FEATURES + _EXTENDED_FEATURES


def _diag_hmm_macro(db, n_states: int = 4) -> dict:
    rows: list[RegimeClassification] = (
        db.query(RegimeClassification).order_by(RegimeClassification.date.asc()).all()
    )
    X_raw, labels, _ = _extract_feature_matrix(rows)
    X_std, mu_raw, sd_raw = _standardize(X_raw)

    pi, A, mu_states, var_states, gamma, ll = _bw_macro(X_std, n_states=n_states)

    states = gamma.argmax(axis=1)
    state_to_regime = _map_states_to_regimes(states, labels, n_states)

    # Centroide de-standardizzato per ogni stato
    centroids_raw = mu_states * sd_raw + mu_raw  # (K, D)

    # Distribuzione regime corrente (last gamma)
    last_post = gamma[-1]
    last_x_std = X_std[-1]
    last_x_raw = X_raw[-1]

    # Distanza Mahalanobis dal punto corrente per ogni stato
    distances = np.zeros(n_states)
    for k in range(n_states):
        diff = last_x_std - mu_states[k]
        # diag covariance
        distances[k] = float(np.sqrt(np.sum((diff ** 2) / np.maximum(var_states[k], 1e-4))))

    # Conta osservazioni per stato
    state_counts = {int(s): int((states == s).sum()) for s in range(n_states)}

    # Per ogni stato, distribuzione dei label rule-based
    state_label_dist: dict[int, dict[str, float]] = {}
    for s in range(n_states):
        idx = (states == s)
        if idx.sum() == 0:
            state_label_dist[s] = {r: 0.0 for r in REGIMES}
            continue
        state_labs = [labels[i] for i in range(len(labels)) if idx[i]]
        total = len(state_labs)
        state_label_dist[s] = {
            r: state_labs.count(r) / total for r in REGIMES
        }

    return {
        "n_obs": X_raw.shape[0],
        "ll": ll,
        "feature_names": list(_FEATURES),
        "centroids_raw": centroids_raw,    # (K, D) feature units
        "feature_means_global": mu_raw,
        "feature_stds_global": sd_raw,
        "current_x_raw": last_x_raw,
        "current_x_std": last_x_std,
        "state_to_regime": state_to_regime,
        "state_counts": state_counts,
        "state_label_dist": state_label_dist,
        "current_post_state": last_post,
        "distances_to_states": distances,
    }


def _diag_hmm_market(db, n_states: int = 4) -> dict:
    df = compute_market_features().dropna()
    feature_names = list(df.columns)
    X = df.values
    mu_raw = X.mean(axis=0)
    sd_raw = X.std(axis=0, ddof=0)
    sd_raw[sd_raw == 0] = 1.0
    X_std = _std_market(X)

    pi, A, mu_states, var_states, gamma, ll = _baum_welch(X_std, n_states=n_states, max_iter=200)

    # Map via correlazione (replicando hmm_market.py)
    import pandas as pd
    rows = (
        db.query(RegimeClassification).order_by(RegimeClassification.date.asc()).all()
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
    ]).set_index("date").sort_index().resample("ME").mean().dropna()
    common = df.index.intersection(rule_df.index)
    gamma_aligned = pd.DataFrame(gamma, index=df.index).reindex(common).values
    rule_aligned = rule_df.reindex(common)

    state_to_regime, state_corr = _map_states_via_soft_correlation(gamma_aligned, rule_aligned)

    last_post = gamma[-1]
    last_x_raw = X[-1]
    last_x_std = X_std[-1]

    distances = np.zeros(n_states)
    for k in range(n_states):
        diff = last_x_std - mu_states[k]
        distances[k] = float(np.sqrt(np.sum((diff ** 2) / np.maximum(var_states[k], 1e-4))))

    centroids_raw = mu_states * sd_raw + mu_raw

    states = gamma.argmax(axis=1)
    state_counts = {int(s): int((states == s).sum()) for s in range(n_states)}

    return {
        "n_obs": X.shape[0],
        "ll": ll,
        "feature_names": feature_names,
        "centroids_raw": centroids_raw,
        "feature_means_global": mu_raw,
        "feature_stds_global": sd_raw,
        "current_x_raw": last_x_raw,
        "current_x_std": last_x_std,
        "state_to_regime": state_to_regime,
        "state_corr": state_corr,
        "state_counts": state_counts,
        "current_post_state": last_post,
        "distances_to_states": distances,
        "last_date": df.index[-1].date().isoformat(),
    }


def _print_macro(d: dict) -> None:
    print("=" * 78)
    print(f"HMM MACRO — n_obs={d['n_obs']}, ll={d['ll']:.2f}")
    print("=" * 78)
    fn = d["feature_names"]
    cur = d["current_x_raw"]
    print(f"\n-- Stato corrente (raw values) --")
    for i, name in enumerate(fn):
        print(f"  {name:25s} = {cur[i]:+.3f}  (mean {d['feature_means_global'][i]:+.3f}, std {d['feature_stds_global'][i]:.3f})")

    print(f"\n-- Mapping stati -> regimi --")
    for s, r in sorted(d["state_to_regime"].items()):
        cnt = d["state_counts"].get(s, 0)
        labels = d["state_label_dist"].get(s, {})
        lab_str = ", ".join(f"{lab[:4]} {p*100:.0f}%" for lab, p in labels.items() if p > 0.05)
        print(f"  state {s} -> {r:13s}  obs={cnt:4d}  labels: {lab_str}")

    print(f"\n-- Posterior corrente per stato --")
    for s, p in enumerate(d["current_post_state"]):
        r = d["state_to_regime"].get(s, "?")
        dist = d["distances_to_states"][s]
        print(f"  state {s} ({r:13s})  posterior={p*100:5.1f}%  mahal_dist={dist:.2f}")

    print(f"\n-- Centroide di ogni stato (raw feature units) --")
    print(f"  {'feature':25s} " + " ".join(f"state_{s}" for s in range(d['centroids_raw'].shape[0])) + "   current")
    for i, name in enumerate(fn):
        vals = " ".join(f"{d['centroids_raw'][s,i]:+7.2f}" for s in range(d['centroids_raw'].shape[0]))
        print(f"  {name:25s} {vals}   {cur[i]:+7.2f}")


def _print_market(d: dict) -> None:
    print("=" * 78)
    print(f"HMM MARKET — n_obs={d['n_obs']}, ll={d['ll']:.2f}, last={d['last_date']}")
    print("=" * 78)
    fn = d["feature_names"]
    cur = d["current_x_raw"]
    print(f"\n-- Stato corrente (raw values) --")
    for i, name in enumerate(fn):
        print(f"  {name:25s} = {cur[i]:+.3f}  (mean {d['feature_means_global'][i]:+.3f}, std {d['feature_stds_global'][i]:.3f})")

    print(f"\n-- Mapping stati -> regimi (via correlazione, hard greedy) --")
    for s, r in sorted(d["state_to_regime"].items()):
        cnt = d["state_counts"].get(s, 0)
        corrs = d["state_corr"].get(s, {})
        corr_str = ", ".join(f"{lab[:4]} {c:+.2f}" for lab, c in sorted(corrs.items(), key=lambda kv: -kv[1]))
        print(f"  state {s} -> {r:13s}  obs={cnt:4d}  corrs: {corr_str}")

    print(f"\n-- Posterior corrente per stato --")
    for s, p in enumerate(d["current_post_state"]):
        r = d["state_to_regime"].get(s, "?")
        dist = d["distances_to_states"][s]
        print(f"  state {s} ({r:13s})  posterior={p*100:5.1f}%  mahal_dist={dist:.2f}")

    print(f"\n-- Centroide di ogni stato (raw feature units) --")
    print(f"  {'feature':25s} " + " ".join(f"state_{s}" for s in range(d['centroids_raw'].shape[0])) + "   current")
    for i, name in enumerate(fn):
        vals = " ".join(f"{d['centroids_raw'][s,i]:+7.3f}" for s in range(d['centroids_raw'].shape[0]))
        print(f"  {name:25s} {vals}   {cur[i]:+7.3f}")


def main() -> None:
    db = SessionLocal()
    try:
        macro = _diag_hmm_macro(db, n_states=4)
        market = _diag_hmm_market(db, n_states=4)
    finally:
        db.close()

    _print_macro(macro)
    print()
    _print_market(market)

    print()
    print("=" * 78)
    print("SINTESI DISACCORDO")
    print("=" * 78)
    macro_post_state = int(np.argmax(macro["current_post_state"]))
    market_post_state = int(np.argmax(market["current_post_state"]))
    print(f"  HMM macro  -> state {macro_post_state} -> regime {macro['state_to_regime'][macro_post_state]}")
    print(f"  HMM market -> state {market_post_state} -> regime {market['state_to_regime'][market_post_state]}")


if __name__ == "__main__":
    main()
