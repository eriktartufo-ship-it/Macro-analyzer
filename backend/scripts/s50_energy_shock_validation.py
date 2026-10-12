"""S50 VALIDATION GATE — pilastro shock energetico (USE_ENERGY_SHOCK_PILLAR).

Metodo T20: A/B sugli episodi storici etichettati, flag OFF contro ON, solo rule-based (ML blend
OFF: gli episodi sono il suo dataset di training). Ogni episodio e' arricchito a runtime con
`oil_yoy` STORICO REALE alla data medio-episodio (30 giugno dell'anno centrale), da FRED WTISPLC
(WTI mensile dal 1946; il live usa il Brent, correlazione a 12 mesi 0,934 dal 1988). Gli episodi
prima del 1947 restano senza dato (default = mediana 0%, neutro). historical_episodes.py NON
viene modificato.

Criteri scritti PRIMA della misura (2026-10-12, logbook S50), e il cancello si esegue UNA volta:
- accuratezza OVERALL, STAGFLAZIONE e DEFLAZIONE non peggiorano;
- probabilita' media di deflazione non sale;
- robustezza: PASS su almeno 7 delle 9 combinazioni center {p75, p90, p95} x peso {0,03 0,05 0,08}.

Run:  cd backend && PYTHONPATH=. python scripts/s50_energy_shock_validation.py
"""
from __future__ import annotations

import importlib
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from app.services.indicators.transforms import roc_between
from app.services.regime.historical_episodes import HISTORICAL_EPISODES

CENTRI = {"p75": 15.1, "p90": 46.6, "p95": 75.3}
PESI = (0.03, 0.05, 0.08)
SCALE = 15.7  # (p90 - p75) / 2, fissa


def _wti() -> pd.Series:
    from app.config import settings
    from fredapi import Fred
    return Fred(api_key=settings.fred_api_key).get_series("WTISPLC").dropna()


def enrich(ep, wti: pd.Series) -> dict:
    ind = dict(ep.indicators)
    mid = (ep.start_year + ep.end_year) // 2
    cutoff = pd.Timestamp(date(mid, 6, 30))
    s = wti[wti.index <= cutoff]
    if not s.empty:
        v = roc_between(s, s.index[-1], 12)
        if v is not None:
            ind["oil_yoy"] = v
    return ind


def run(flag_on: bool, enriched, center: float = 46.6, weight: float = 0.05, verbose: bool = False):
    if flag_on:
        os.environ["USE_ENERGY_SHOCK_PILLAR"] = "1"
    else:
        os.environ.pop("USE_ENERGY_SHOCK_PILLAR", None)
    os.environ["USE_ML_REGIME_BLEND"] = "0"
    import app.services.regime.classifier as clf
    importlib.reload(clf)
    clf._OIL_SHOCK_CENTER = center
    clf._OIL_SHOCK_SCALE = SCALE
    clf._ENERGY_SHOCK_PILLAR_ADD["stagflation"]["oil_supply_shock"]["weight"] = weight

    correct: dict[str, list[int]] = {}
    defl: list[float] = []
    rows = []
    for ep, ind in enriched:
        out = clf.classify_regime(dict(ind))
        ok = int(out["regime"] == ep.true_regime)
        correct.setdefault(ep.true_regime, []).append(ok)
        defl.append(out["probabilities"]["deflation"])
        rows.append((ep, ind.get("oil_yoy"), out["regime"], out["probabilities"]))
    acc = {r: sum(v) / len(v) for r, v in correct.items()}
    tot = sum(sum(v) for v in correct.values()) / sum(len(v) for v in correct.values())
    return {"acc": acc, "overall": tot, "avg_defl": sum(defl) / len(defl), "rows": rows,
            "n_ok": sum(sum(v) for v in correct.values()), "n": len(rows)}


def gate(off, on) -> tuple[bool, dict]:
    g = {
        "overall": on["overall"] >= off["overall"] - 1e-9,
        "stagflation": on["acc"].get("stagflation", 0) >= off["acc"].get("stagflation", 0) - 1e-9,
        "deflation": on["acc"].get("deflation", 0) >= off["acc"].get("deflation", 0) - 1e-9,
        "avg_defl": on["avg_defl"] <= off["avg_defl"] + 1e-9,
    }
    return all(g.values()), g


def main():
    wti = _wti()
    enriched = [(ep, enrich(ep, wti)) for ep in HISTORICAL_EPISODES]
    n_oil = sum(1 for _, i in enriched if "oil_yoy" in i)
    print(f"S50 — pilastro shock energetico · episodi {len(enriched)}, con oil_yoy reale {n_oil}")

    off = run(False, enriched)
    on = run(True, enriched)
    print(f"\nOFF (base, flag di produzione, ML spento): {off['n_ok']}/{off['n']} = {off['overall']:.1%}")
    print(f"ON  (center p90 46,6 · peso 0,05):        {on['n_ok']}/{on['n']} = {on['overall']:.1%}")
    for r in ("reflation", "stagflation", "deflation", "goldilocks"):
        print(f"  {r:12s} {off['acc'].get(r, 0):.0%} -> {on['acc'].get(r, 0):.0%}")
    print(f"  prob. media deflazione {off['avg_defl']:.1%} -> {on['avg_defl']:.1%}")

    print("\nEpisodi che CAMBIANO etichetta (OFF -> ON):")
    cambiati = 0
    for (ep, oil, p_off, _), (_, _, p_on, pr_on) in zip(off["rows"], on["rows"]):
        if p_off != p_on:
            cambiati += 1
            segno = "MEGLIO" if p_on == ep.true_regime else ("PEGGIO" if p_off == ep.true_regime else "altro")
            o = "—" if oil is None else f"{oil:+.0f}%"
            print(f"  [{segno}] {ep.name[:42]:42s} vero={ep.true_regime[:5]} {p_off[:5]}->{p_on[:5]} petrolio {o}")
    if not cambiati:
        print("  nessuno")

    ok, g = gate(off, on)
    print("\nCancello (combinazione centrale):", {k: "PASS" if v else "FAIL" for k, v in g.items()})

    print("\nPlateau 9 combinazioni:")
    passati = 0
    for nome, c in CENTRI.items():
        for w in PESI:
            r = run(True, enriched, center=c, weight=w)
            esito, _ = gate(off, r)
            passati += esito
            print(f"  center {nome} {c:5.1f} peso {w:.2f}: {r['n_ok']}/{r['n']} "
                  f"stag {r['acc'].get('stagflation', 0):.0%} defl {r['acc'].get('deflation', 0):.0%} "
                  f"avg_defl {r['avg_defl']:.1%} -> {'PASS' if esito else 'FAIL'}")
    robusto = passati >= 7
    print(f"\nVERDETTO: centrale {'PASS' if ok else 'FAIL'} · plateau {passati}/9 "
          f"⇒ {'PASS — candidabile a default-ON' if ok and robusto else 'FAIL — resta OFF'}")
    os.environ.pop("USE_ENERGY_SHOCK_PILLAR", None)
    os.environ.pop("USE_ML_REGIME_BLEND", None)


if __name__ == "__main__":
    main()
