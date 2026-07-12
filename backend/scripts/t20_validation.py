"""T20 VALIDATION GATE — livello-vs-accelerazione (council P2 2026-07-12).

A/B sui 32 episodi storici labeled: flag OFF vs USE_LEVEL_ACCEL_PILLARS=1.
Gli episodi vengono ARRICCHITI a runtime con i valori STORICI REALI (FRED)
di ulc_yoy / productivity_yoy / cpi_yoy_change_6m alla data medio-episodio —
il dataset in historical_episodes.py NON viene modificato (il training ML
resta invariato; qui si valuta solo il rule-based).

Metriche gate (diagnosi council):
- accuracy per-regime (target: goldilocks MIGLIORA, era 29.9% storico)
- prob. media deflation cross-episodi (target: SCENDE, era overfit 40.3%)
- nessuna regressione sugli episodi deflation veri (2008/2020/2001...)

Run:
    python scripts/t20_validation.py
"""
from __future__ import annotations

import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib

import pandas as pd
import requests

from app.services.regime.historical_episodes import HISTORICAL_EPISODES

_FRED_SERIES = {
    "ULCNFB": None,     # -> ulc_yoy (YoY su trimestrale)
    "OPHNFB": None,     # -> productivity_yoy
    "CPIAUCSL": None,   # -> cpi_yoy_change_6m
}


def _fred_key() -> str:
    for env_name in ("FRED_API_KEY",):
        v = os.environ.get(env_name)
        if v:
            return v
    # fallback: .env progetto
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), ".env")
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("FRED_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("FRED_API_KEY non trovata")


def _fetch(sid: str, key: str) -> pd.Series:
    r = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={"series_id": sid, "api_key": key, "file_type": "json",
                "sort_order": "asc", "limit": 100000},
        timeout=60,
    )
    r.raise_for_status()
    obs = r.json()["observations"]
    return pd.Series(
        {pd.Timestamp(o["date"]): float(o["value"]) for o in obs if o["value"] != "."}
    ).sort_index()


def _load_series() -> dict[str, pd.Series]:
    key = _fred_key()
    return {sid: _fetch(sid, key) for sid in _FRED_SERIES}


def _as_of(s: pd.Series, cutoff: pd.Timestamp) -> pd.Series:
    return s[s.index <= cutoff]


def enrich(ep, series: dict[str, pd.Series]) -> dict[str, float]:
    """Episodio + valori storici reali T20 alla data medio-episodio."""
    mid_year = (ep.start_year + ep.end_year) // 2
    cutoff = pd.Timestamp(date(mid_year, 6, 30))
    ind = dict(ep.indicators)

    ulc = _as_of(series["ULCNFB"].pct_change(4) * 100, cutoff).dropna()
    if len(ulc):
        ind["ulc_yoy"] = float(ulc.iloc[-1])
    prod = _as_of(series["OPHNFB"].pct_change(4) * 100, cutoff).dropna()
    if len(prod):
        ind["productivity_yoy"] = float(prod.iloc[-1])
    cpi_yoy = series["CPIAUCSL"].pct_change(12) * 100
    chg6 = _as_of((cpi_yoy - cpi_yoy.shift(6)), cutoff).dropna()
    if len(chg6):
        ind["cpi_yoy_change_6m"] = float(chg6.iloc[-1])
    return ind


def run_config(label: str, flag_on: bool, enriched: list[dict]):
    if flag_on:
        os.environ["USE_LEVEL_ACCEL_PILLARS"] = "1"
    else:
        os.environ.pop("USE_LEVEL_ACCEL_PILLARS", None)
    # ML blend OFF: qui si valuta SOLO il rule-based (il dataset ML non ha T20)
    os.environ["USE_ML_REGIME_BLEND"] = "0"
    import app.services.regime.classifier as clf
    importlib.reload(clf)

    correct: dict[str, list[int]] = {}
    defl_probs: list[float] = []
    rows = []
    for ep, ind in enriched:
        out = clf.classify_regime(ind)
        pred = out["regime"]
        ok = 1 if pred == ep.true_regime else 0
        correct.setdefault(ep.true_regime, []).append(ok)
        defl_probs.append(out["probabilities"]["deflation"])
        rows.append((ep.name, ep.true_regime, pred, out["probabilities"]))

    print(f"\n{'='*84}\nCONFIG: {label}\n{'='*84}")
    for name, true_r, pred, probs in rows:
        mark = "OK " if pred == true_r else "MISS"
        p_str = " ".join(f"{k[:4]}={v:.2f}" for k, v in sorted(probs.items()))
        print(f"  [{mark}] {name[:44]:44s} true={true_r[:4]} pred={pred[:4]}  {p_str}")

    print("\n  Accuracy per regime:")
    total_ok = total_n = 0
    acc = {}
    for regime in ("reflation", "stagflation", "deflation", "goldilocks"):
        oks = correct.get(regime, [])
        if oks:
            acc[regime] = sum(oks) / len(oks)
            total_ok += sum(oks)
            total_n += len(oks)
            print(f"    {regime:12s}: {sum(oks)}/{len(oks)} = {acc[regime]:.0%}")
    overall = total_ok / total_n if total_n else 0.0
    avg_defl = sum(defl_probs) / len(defl_probs)
    print(f"    {'OVERALL':12s}: {total_ok}/{total_n} = {overall:.0%}")
    print(f"  Prob. media deflation cross-episodi: {avg_defl:.1%}")
    return {"acc": acc, "overall": overall, "avg_defl": avg_defl}


def main():
    print("T20 VALIDATION GATE — livello-vs-accelerazione")
    print("Arricchimento episodi con dati FRED storici reali (ULC/prod/cpi_chg6)...")
    series = _load_series()
    enriched = [(ep, enrich(ep, series)) for ep in HISTORICAL_EPISODES]
    n_enriched = sum(1 for _, ind in enriched if "ulc_yoy" in ind)
    print(f"Episodi: {len(enriched)}, con ulc_yoy reale: {n_enriched}")

    off = run_config("T20 OFF (baseline)", False, enriched)
    on = run_config("T20 ON (USE_LEVEL_ACCEL_PILLARS=1)", True, enriched)

    print(f"\n{'='*84}\nVERDETTO GATE\n{'='*84}")
    for regime in ("reflation", "stagflation", "deflation", "goldilocks"):
        a, b = off["acc"].get(regime), on["acc"].get(regime)
        if a is not None and b is not None:
            delta = b - a
            print(f"  {regime:12s}: {a:.0%} -> {b:.0%}  ({delta:+.0%})")
    print(f"  {'OVERALL':12s}: {off['overall']:.0%} -> {on['overall']:.0%}")
    print(f"  avg deflation prob: {off['avg_defl']:.1%} -> {on['avg_defl']:.1%}")

    gate_gold = on["acc"].get("goldilocks", 0) >= off["acc"].get("goldilocks", 0)
    gate_defl = on["acc"].get("deflation", 0) >= off["acc"].get("deflation", 0) - 1e-9
    gate_overall = on["overall"] >= off["overall"]
    gate_defl_prob = on["avg_defl"] <= off["avg_defl"] + 1e-9
    print(f"\n  GATE goldilocks non peggiora:   {'PASS' if gate_gold else 'FAIL'}")
    print(f"  GATE deflation non regredisce:  {'PASS' if gate_defl else 'FAIL'}")
    print(f"  GATE overall non peggiora:      {'PASS' if gate_overall else 'FAIL'}")
    print(f"  GATE avg defl prob non sale:    {'PASS' if gate_defl_prob else 'FAIL'}")
    verdict = all([gate_gold, gate_defl, gate_overall, gate_defl_prob])
    print(f"\n  VERDETTO: {'PASS — candidabile a default-ON' if verdict else 'FAIL — resta OFF'}")

    os.environ.pop("USE_LEVEL_ACCEL_PILLARS", None)
    os.environ.pop("USE_ML_REGIME_BLEND", None)


if __name__ == "__main__":
    main()
