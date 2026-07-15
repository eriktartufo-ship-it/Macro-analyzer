"""S41 probe — fire-rate REALE di assess_transition sulla storia mensile del classifier.

Spiega il Δ≈0 del transition-derisk: quante volte, sui regimi mensili walk-forward
1995-2024, il detector dichiara is_transition? Confronta 3/3 (default live, T9-audit)
vs 2/3 (legacy permissivo) vs il proxy grezzo (conf<0.45 OR max_prob<0.45).
Stampa anche la distribuzione di max_prob/entropy/confidence per capire PERCHE'.
"""
from __future__ import annotations

import os
import sys
from datetime import date

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("USE_MOMENTUM_PILLARS", "1")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    from app.services.backtest.walk_forward import WalkForwardCache, preload_walk_forward_series
    from app.services.regime.transition_detector import assess_transition, shannon_entropy

    print("Preloading FRED for walk-forward...")
    series = preload_walk_forward_series(start_date=date(1993, 1, 1), end_date=date(2025, 12, 31))
    wf = WalkForwardCache(series)

    fire_3of3 = fire_2of3 = fire_proxy = n = 0
    maxp_l, ent_l, conf_l = [], [], []
    fired_months = []
    for y in range(1995, 2025):
        for m in range(1, 13):
            cm = wf.get(date(y, m, 1))
            if cm is None:
                continue
            probs = cm["probs"]
            conf = cm.get("confidence", 0.5) or 0.5
            n += 1
            mp = max(probs.values()); ent = shannon_entropy(probs)
            maxp_l.append(mp); ent_l.append(ent); conf_l.append(conf)
            a3 = assess_transition(probs, conf, require_all_three=True)
            a2 = assess_transition(probs, conf, require_all_three=False)
            proxy = (conf < 0.45) or (mp < 0.45)
            fire_3of3 += a3.is_transition
            fire_2of3 += a2.is_transition
            fire_proxy += proxy
            if a3.is_transition:
                fired_months.append(f"{y}-{m:02d}(mp{mp:.2f} e{ent:.2f} c{conf:.2f})")

    print(f"\nMesi validi 1995-2024: {n}")
    print(f"  fire 3/3 (detector live):  {fire_3of3:3d}  ({fire_3of3/n*100:4.1f}%)")
    print(f"  fire 2/3 (legacy loose):   {fire_2of3:3d}  ({fire_2of3/n*100:4.1f}%)")
    print(f"  fire proxy (OR grezzo):    {fire_proxy:3d}  ({fire_proxy/n*100:4.1f}%)")
    print(f"\nDistribuzione (spiega il gate 3/3):")
    print(f"  max_prob:   min {min(maxp_l):.2f}  p10 {np.percentile(maxp_l,10):.2f}  median {np.median(maxp_l):.2f}  (soglia <0.38)")
    print(f"  entropy:    max {max(ent_l):.2f}  p90 {np.percentile(ent_l,90):.2f}  median {np.median(ent_l):.2f}  (soglia >1.55)")
    print(f"  confidence: min {min(conf_l):.2f}  p10 {np.percentile(conf_l,10):.2f}  median {np.median(conf_l):.2f}  (soglia <0.30)")
    if fired_months:
        print(f"\nMesi 3/3 fired ({len(fired_months)}): {', '.join(fired_months)}")
    else:
        print("\nMesi 3/3 fired: NESSUNO (il detector 3/3 non scatta mai sul mensile).")


if __name__ == "__main__":
    main()
