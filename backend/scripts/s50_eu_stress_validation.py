"""S50 VALIDATION GATE — trigger stress sovrano euro nel crisis indicator (USE_EU_STRESS_TRIGGER).

Ricostruisce il crisis indicator sui record RegimeClassification del DB (dal 2004-09, quando nasce
la curva BCE), arricchiti con lo spread sovrano euro ALLA DATA, flag OFF contro ON. Il resto e'
identico nei due bracci (stesso classificatore, stessi indicatori, dedollar non passato).

Criteri scritti PRIMA della misura (2026-10-12 15:42, logbook S50), una sola esecuzione:
(a) crisi euro 2011-07..2012-08: score medio ON − OFF >= +0,03 e almeno un punto sale di livello;
(b) periodi calmi 2005-06, 2017, 2019, 2024: punti elevated/extreme ON <= OFF, delta medio <= +0,01;
(c) crisi vere 2008-09..2009-03, 2020-03..05, 2022: zero punti che scendono di livello.

Run:  cd backend && PYTHONPATH=. python scripts/s50_eu_stress_validation.py
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
from app.services.indicators import europe_energy as ee
from app.services.regime.crisis_indicator import assess_crisis_risk

LIVELLI = {"low": 0, "moderate": 1, "elevated": 2, "extreme": 3}
FINESTRE = {
    "crisi_euro": [("2011-07-01", "2012-08-31")],
    "calma": [("2005-01-01", "2006-12-31"), ("2017-01-01", "2017-12-31"),
              ("2019-01-01", "2019-12-31"), ("2024-01-01", "2024-12-31")],
    "crisi_vere": [("2008-09-01", "2009-03-31"), ("2020-03-01", "2020-05-31"), ("2022-01-01", "2022-12-31")],
}


def carica() -> list[tuple[pd.Timestamp, dict]]:
    spread = (ee.fetch_ecb("ea_all_10y") - ee.fetch_ecb("ea_aaa_10y")).dropna()
    with create_engine(settings.database_url).connect() as c:
        rows = c.execute(text("select date, conditions_met from regime_classifications "
                              "where date >= '2004-09-06' order by date")).fetchall()
    out = []
    for d, cm in rows:
        try:
            ind = (json.loads(cm) or {}).get("indicators") or {}
        except (ValueError, TypeError):
            continue
        ind = {k: v for k, v in ind.items() if isinstance(v, (int, float))}
        t = pd.Timestamp(d)
        s = spread[spread.index <= t]
        if not s.empty and (t - s.index[-1]).days <= 7:
            ind["ea_sovereign_spread_10y"] = float(s.iloc[-1])
        out.append((t, ind))
    return out


def valuta(punti, flag_on: bool) -> dict:
    # esplicito nei due versi: dal 20/10 il default e' ON, togliere la variabile non spegne piu' il flag
    os.environ["USE_EU_STRESS_TRIGGER"] = "1" if flag_on else "0"
    res = {}
    for t, ind in punti:
        a = assess_crisis_risk(dict(ind), dedollar_combined=None, date=t.date().isoformat())
        res[t] = (a.risk_score, a.risk_level)
    return res


def nella(t, finestre):
    return any(pd.Timestamp(a) <= t <= pd.Timestamp(b) for a, b in finestre)


def main():
    punti = carica()
    print(f"S50 — stress euro · punti {len(punti)} dal {punti[0][0].date()} al {punti[-1][0].date()}, "
          f"con spread {sum('ea_sovereign_spread_10y' in i for _, i in punti)}")
    off, on = valuta(punti, False), valuta(punti, True)
    os.environ.pop("USE_EU_STRESS_TRIGGER", None)

    def stat(nome):
        ts = [t for t, _ in punti if nella(t, FINESTRE[nome])]
        d = [on[t][0] - off[t][0] for t in ts]
        su = sum(LIVELLI[on[t][1]] > LIVELLI[off[t][1]] for t in ts)
        giu = sum(LIVELLI[on[t][1]] < LIVELLI[off[t][1]] for t in ts)
        alti_off = sum(LIVELLI[off[t][1]] >= 2 for t in ts)
        alti_on = sum(LIVELLI[on[t][1]] >= 2 for t in ts)
        media = sum(d) / len(d) if d else 0.0
        print(f"  {nome:11s} n={len(ts):3d}  delta medio {media:+.3f}  su di livello {su}  giu' {giu}  "
              f"elevated+ {alti_off}->{alti_on}  score OFF {sum(off[t][0] for t in ts)/max(1,len(ts)):.3f}")
        return {"n": len(ts), "media": media, "su": su, "giu": giu, "alti_off": alti_off, "alti_on": alti_on}

    e, c, v = stat("crisi_euro"), stat("calma"), stat("crisi_vere")
    ga = e["n"] > 0 and e["media"] >= 0.03 and e["su"] >= 1
    gb = c["alti_on"] <= c["alti_off"] and c["media"] <= 0.01
    gc = v["giu"] == 0
    print(f"\n  (a) crisi euro alzata:      {'PASS' if ga else 'FAIL'}")
    print(f"  (b) nessun falso allarme:   {'PASS' if gb else 'FAIL'}")
    print(f"  (c) nessuna crisi abbassata: {'PASS' if gc else 'FAIL'}")
    print(f"\nVERDETTO: {'PASS — candidabile a default-ON' if ga and gb and gc else 'FAIL — resta OFF'}")
    oggi = punti[-1][0]
    print(f"Ultimo punto {oggi.date()}: OFF {off[oggi]} · ON {on[oggi]}")


if __name__ == "__main__":
    main()
