"""ESPERIMENTO B1 — validazione su DATI REALI dei segnali di timing.

Conferma su storia vera ciò che l'esp. A (cadence_sim, sintetico) suggeriva:
 - segnale LIQUIDITÀ in ENTRATA (WALCL/M2 = QE) cattura le V-recovery;
 - segnale TASSI in USCITA (DGS10 in salita / curva invertita) taglia i bear da tassi;
 - la cadenza fine (daily) è ciò che permette di sfruttare i segnali.

Dati REALI: SPY daily (Yahoo, sleeve rischiosa vs cash=0) + segnali da FRED
(m2, fed_balance_sheet=WALCL, treasury_10y=DGS10, yield_curve_10y2y=T10Y2Y),
allineati ai giorni di borsa con forward-fill. Riusa la sim di cadence_sim
(trend MA50 + stop + phase-averaging). Il "regime costruttivo" è posto True ovunque:
qui si validano i SEGNALI, non il classifier di regime.

Finestre: 1994 (bond-shock), 2008 (GFC+QE), 2020 (pandemia+QE), 2022 (rate-hike bear).

⚠️ Soglie dei segnali scelte A PRIORI (sensate, NON tarate sull'esito): surge = WALCL
+3% su ~3m OR M2 YoY >7%; rate_stress = DGS10 +0.7pp su ~3m OR curva 10y2y < 0.
Cash=0 (conservativo). Direzionale, ma su prezzi/segnali VERI.
"""
from __future__ import annotations

import os
import sys
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cadence_sim import simulate  # noqa: E402

WINDOWS = {
    "1994 bond-shock": (date(1994, 1, 1), date(1995, 6, 30)),
    "2008 GFC (+QE)": (date(2007, 9, 1), date(2009, 6, 30)),
    "2020 pandemia (+QE)": (date(2020, 1, 1), date(2021, 3, 31)),
    "2022 rate-hike bear": (date(2022, 1, 1), date(2023, 6, 30)),
}


def _load():
    from app.services.prices.yahoo_fetcher import YahooFetcher
    from app.services.indicators.fetcher import FredFetcher
    yf, fred = YahooFetcher(), FredFetcher()
    spy = yf.fetch("SPY", start_date=date(1993, 1, 1), end_date=date(2023, 12, 31)).dropna()
    idx = spy.index
    def ff(name):
        s = fred.fetch_series(name, date(1990, 1, 1), date(2023, 12, 31))
        s.index = pd.to_datetime(s.index)
        return s.reindex(idx, method="ffill")
    walcl = ff("fed_balance_sheet")
    m2 = ff("m2")
    dgs10 = ff("treasury_10y")
    curve = ff("yield_curve_10y2y")
    # segnali (a priori, non tarati)
    surge = ((walcl.pct_change(63) > 0.03) | (m2.pct_change(252) > 0.07)).fillna(False)
    rate_stress = ((dgs10.diff(63) > 0.7) | (curve < 0.0)).fillna(False)
    return spy, surge.astype(bool), rate_stress.astype(bool)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("Loading SPY (Yahoo) + segnali FRED reali...")
    spy, surge, rate = _load()
    print("ESPERIMENTO B1 — DATI REALI. SPY sleeve vs cash. Segnali: QE(entrata) / rate-stress(uscita)")
    print("Combo: base=trend+stop · +QE=entrata liquidità · +RATE=uscita tassi · BOTH")
    print("=" * 116)
    for name, (d0, d1) in WINDOWS.items():
        mask = (spy.index >= pd.Timestamp(d0)) & (spy.index <= pd.Timestamp(d1))
        p = spy[mask].to_numpy()
        if len(p) < 60:
            print(f"\n▸ {name}: dati insufficienti ({len(p)}g)"); continue
        su = surge[mask].to_numpy(); ra = rate[mask].to_numpy()
        reg = np.ones(len(p), dtype=bool)
        bh = p[-1] / p[0] - 1.0
        qe_days = int(su.sum()); rt_days = int(ra.sum())
        print(f"\n▸ {name}  [{len(p)}g]  SPY buy&hold {bh*100:+.1f}%   (giorni segnale: QE {qe_days}, rate {rt_days})")
        for step, lbl in [(21, "mensile "), (5, "settim. "), (1, "giornal.")]:
            b = simulate(p, reg, su, ra, step, False, False)
            q = simulate(p, reg, su, ra, step, True, False)
            rr = simulate(p, reg, su, ra, step, False, True)
            bo = simulate(p, reg, su, ra, step, True, True)
            print(f"   {lbl}: base {b[0]*100:+5.1f}%/DD{b[1]*100:5.1f}  +QE {q[0]*100:+5.1f}%  "
                  f"+RATE {rr[0]*100:+5.1f}%/DD{rr[1]*100:5.1f}  BOTH {bo[0]*100:+5.1f}%/DD{bo[1]*100:5.1f}")


if __name__ == "__main__":
    main()
