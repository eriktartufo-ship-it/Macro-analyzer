"""ESPERIMENTO A+ — cadenza × SEGNALI (liquidità in ENTRATA, tassi/stress in USCITA).

Domande:
 1) monthly vs weekly vs daily — la cadenza fine cattura meglio i movimenti?
 2) INSIGHT ERIK #1: la V-shape è spesso QE-driven → liquidity-surge (M2/WALCL/real-rates,
    già in USE_LIQUIDITY_SURGE_OVERRIDE) come TRIGGER D'INGRESSO anticipato.
 3) INSIGHT ERIK #2: il ribasso è spesso guidato dal RIALZO TASSI → usa un segnale
    rate-stress (fed-funds ROC ↑ / curva invertita / spread HY-BAA ↑, cfr.
    USE_FINANCIAL_STRESS_VETO) come TRIGGER D'USCITA anticipato, PRIMA che scatti lo
    stop di prezzo (−10%). Cattura il ribasso in anticipo = meno drawdown.

Simmetria: entrata su liquidità (QE) ↔ uscita su tassi/stress. "Usare tutti i dati".

⚠️ STILIZZATO/DIREZIONALE (non il motore reale mensile; il daily storico = esp. B).
Path giornalieri sintetici + regola trend-MA50 + stop + gate regime, phase-averaged
(media su tutti gli sfasamenti della griglia → isola cadenza da fortuna-allineamento).
I segnali (surge/rate_stress) sono impostati per scenario coerentemente con l'analogo
storico, con lag di riconoscimento. Controlli: dove il segnale non c'entra → deve fare ~0.
"""
from __future__ import annotations

import numpy as np

STOP = -0.10
DEPLOY = 0.60
MA = 50


def _path(segments):
    """segments: (giorni, ret/giorno, regime_costruttivo, liquidity_surge, rate_stress)."""
    p, reg, sur, rat = [100.0], [], [], []
    for days, dr, constructive, surge, rate in segments:
        for _ in range(days):
            p.append(p[-1] * (1.0 + dr))
            reg.append(constructive); sur.append(surge); rat.append(rate)
    return np.array(p[1:]), np.array(reg), np.array(sur), np.array(rat)


def _scenarios():
    # constructive, surge(QE, entrata), rate_stress(tassi↑, uscita)
    return {
        "carry-unwind (no QE, no rate)": [(21, 0.0006, True, False, False), (5, -0.023, False, False, False),
                                          (40, 0.0046, True, False, False), (20, 0.0004, True, False, False)],
        "pandemia (V QE-driven)": [(20, 0.0005, True, False, False), (25, -0.0166, False, False, False),
                                   (80, 0.0043, True, True, False), (40, 0.0003, True, True, False)],
        "GFC-2008 (V QE-driven)": [(30, 0.0004, True, False, False), (90, -0.0055, False, False, False),
                                   (120, 0.0038, True, True, False), (40, 0.0003, True, True, False)],
        # NUOVO — bear da RIALZO TASSI (2022): Fed hikes, equity+bond giù insieme, rate_stress ON durante la discesa
        "rate-hike bear (2022)": [(21, 0.0004, True, False, False), (180, -0.0018, False, False, True),
                                  (60, 0.0006, True, False, False)],
        # NUOVO — shock obbligazionario 1994: spike tassi veloce, equity choppy-down, rate_stress ON early
        "bond-shock 1994 (rate spike)": [(15, 0.0004, True, False, False), (60, -0.0022, False, False, True),
                                         (45, 0.0008, True, False, False)],
        "bolla AI (no QE)": [(250, -0.002, False, False, False), (150, 0.0018, True, False, False)],
    }


def _sim_one(prices, regime, surge, rate, ma, step, offset, use_surge, use_rate, mom_lag=1):
    n = len(prices)
    nav, w, entry_peak = 1.0, 0.0, None
    peak_nav, maxdd, trips = 1.0, 0.0, 0
    for t in range(1, n):
        r = prices[t] / prices[t - 1] - 1.0
        nav *= (1.0 + w * r)
        peak_nav = max(peak_nav, nav)
        maxdd = min(maxdd, nav / peak_nav - 1.0)
        if w > 0:
            entry_peak = max(entry_peak, prices[t])
        if (t - offset) % step == 0 and t >= offset:
            j = max(0, t - mom_lag * step)
            trend_up = prices[t] > ma[t]
            surge_now = use_surge and bool(surge[j])
            rate_now = use_rate and bool(rate[j])
            if w > 0 and (rate_now or (prices[t] / entry_peak - 1.0) <= STOP):
                w, entry_peak = 0.0, None          # USCITA: rate-stress anticipato OPPURE stop di prezzo
            elif w == 0 and not rate_now and ((trend_up and regime[j]) or surge_now):
                w, entry_peak = DEPLOY, prices[t]; trips += 1   # ENTRATA: no se rate-stress attivo
    return nav - 1.0, maxdd, trips


def simulate(prices, regime, surge, rate, step, use_surge, use_rate):
    n = len(prices)
    ma = np.array([prices[max(0, i - MA + 1):i + 1].mean() for i in range(n)])
    rets, dds = [], []
    for off in range(step):
        rr, dd, _ = _sim_one(prices, regime, surge, rate, ma, step, off, use_surge, use_rate)
        rets.append(rr); dds.append(dd)
    return float(np.mean(rets)), float(np.mean(dds))


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("ESPERIMENTO A+ — cadenza × segnali (entrata=liquidità QE · uscita=rate-stress). Stilizzato, phase-averaged")
    print("Combo: [base]=trend+regime · [+QE]=+entrata su liquidità · [+RATE]=+uscita su tassi · [BOTH]")
    print("=" * 116)
    for name, segs in _scenarios().items():
        prices, regime, surge, rate = _path(segs)
        bh = prices[-1] / prices[0] - 1.0
        print(f"\n▸ {name}  [{len(prices)}g]  buy&hold {bh*100:+.1f}%")
        for step, lbl in [(21, "mensile "), (5, "settim. "), (1, "giornal.")]:
            b = simulate(prices, regime, surge, rate, step, False, False)
            q = simulate(prices, regime, surge, rate, step, True, False)
            rt = simulate(prices, regime, surge, rate, step, False, True)
            bo = simulate(prices, regime, surge, rate, step, True, True)
            print(f"   {lbl}: base {b[0]*100:+5.1f}%/DD{b[1]*100:5.1f}  +QE {q[0]*100:+5.1f}%  "
                  f"+RATE {rt[0]*100:+5.1f}%/DD{rt[1]*100:5.1f}  BOTH {bo[0]*100:+5.1f}%/DD{bo[1]*100:5.1f}")


if __name__ == "__main__":
    main()
