"""Il sito dice la verita'? Confronta i numeri PUBBLICATI di Macro con le fonti primarie.

Stessi 14 controlli del controllo automatico che gira dopo ogni refresh
(app/services/validation/source_crosscheck.py, esito in GET /data-quality): qui si leggono i numeri
dal sito pubblicato invece che dal refresh appena fatto. Nato il 2026-10-12 dal bug di S49.
    ✅ entro la tolleranza   ❌ fuori   ⚪ non giudicabile (fonte giu' o dato assente)
Esce con 1 se c'e' almeno un ❌.

Uso:  cd backend && PYTHONPATH=. ../.venv/Scripts/python.exe scripts/controllo_incrociato.py [--url https://macro.pieranlab.cloud]
"""

from __future__ import annotations

import json
import sys

from app.services.validation import source_crosscheck as sc

sys.stdout.reconfigure(encoding="utf-8")
URL = sys.argv[sys.argv.index("--url") + 1] if "--url" in sys.argv else "https://macro.pieranlab.cloud"
SEGNO = {"ok": "✅", "fail": "❌", "na": "⚪"}


def main() -> int:
    snap = json.loads(sc._get(f"{URL}/api/v1/data-snapshot", timeout=120))
    sito = {**snap.get("current_dedollar_indicators", {}), **snap.get("current_indicators", {})}
    r = sc.run_crosscheck(sito)
    print(f"Controllo incrociato {URL} — snapshot del {snap.get('date')}\n")
    print(f"   {'indicatore':30} {'sito':>9} {'fonte':>9}  fonte")
    for x in r["results"]:
        fs = "—" if x["site"] is None else f"{x['site']:9.2f}"
        fv = "—" if x["source_value"] is None else f"{x['source_value']:9.2f}"
        nota = f" — {x['note']}" if x["note"] else ""
        print(f"{SEGNO[x['status']]} {x['name']:30} {fs:>9} {fv:>9}  {x['source']} (toll. {x['tolerance']}){nota}")
    print(f"\n{r['n_fail']} fuori tolleranza" if r["n_fail"] else "\nTutto entro tolleranza")
    return 1 if r["n_fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
