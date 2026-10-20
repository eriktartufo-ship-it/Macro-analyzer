"""S52 — il punteggio di crisi DILUISCE ogni trigger nuovo? (misura, criteri scritti prima: logbook S52)

Ipotesi nata dal cancello I2 (12/10): il trigger stress euro, spento nel 2008/2020/2022, aveva fatto
SCENDERE di livello 5 mesi di crisi vera. Lo score e' (critici accesi x2 + altri accesi) / (critici x2 +
altri) sui trigger PRESENTI: un trigger in piu' che resta spento aumenta solo il denominatore.
Qui si aggiunge un trigger FITTIZIO sempre spento, non critico, e si conta cosa cambia.

D1 provata se: almeno 1 punto scende di livello OPPURE lo score scende in almeno il 50% dei punti.

Run:  cd backend && PYTHONPATH=. python scripts/s52_diluizione_crisi.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text

logger.remove()

from app.config import settings
from app.services.regime import crisis_indicator as ci

LIVELLI = {"low": 0, "moderate": 1, "elevated": 2, "extreme": 3}


def carica():
    with create_engine(settings.database_url).connect() as c:
        rows = c.execute(text("select date, conditions_met from regime_classifications "
                              "where date >= '2004-09-06' order by date")).fetchall()
    out = []
    for d, cm in rows:
        try:
            ind = (json.loads(cm) or {}).get("indicators") or {}
        except (ValueError, TypeError):
            continue
        out.append((pd.Timestamp(d), {k: v for k, v in ind.items() if isinstance(v, (int, float))}))
    return out


def valuta(punti, con_fittizio: bool):
    originale = ci._evaluate_triggers

    def con_un_trigger_spento(ind, probs, correlation_assessment=None):
        t = originale(ind, probs, correlation_assessment=correlation_assessment)
        return t + [{"name": "fittizio_sempre_spento", "value": 0.0, "threshold": 1.0,
                     "operator": "ge", "fired": False, "description": "S52 misura"}]

    if con_fittizio:
        ci._evaluate_triggers = con_un_trigger_spento
    try:
        return {t: (a.risk_score, a.risk_level) for t, ind in punti
                for a in [ci.assess_crisis_risk(dict(ind), dedollar_combined=None, date=t.date().isoformat())]}
    finally:
        ci._evaluate_triggers = originale


def main():
    punti = carica()
    off, on = valuta(punti, False), valuta(punti, True)
    n = len(punti)
    scende_score = sum(on[t][0] < off[t][0] - 1e-12 for t, _ in punti)
    sale_score = sum(on[t][0] > off[t][0] + 1e-12 for t, _ in punti)
    giu = [t for t, _ in punti if LIVELLI[on[t][1]] < LIVELLI[off[t][1]]]
    print(f"S52 D1 — {n} punti dal {punti[0][0].date()} al {punti[-1][0].date()}")
    print(f"  score sceso in {scende_score}/{n} ({scende_score / n:.0%}) · salito in {sale_score}")
    print(f"  punti scesi di livello: {len(giu)}")
    for t in giu[:12]:
        print(f"    {t.date()}  {off[t][1]} {off[t][0]:.3f} -> {on[t][1]} {on[t][0]:.3f}")
    provata = len(giu) >= 1 or scende_score / n >= 0.5
    print(f"\nD1: diluizione {'PROVATA' if provata else 'SMENTITA'}")


if __name__ == "__main__":
    main()
