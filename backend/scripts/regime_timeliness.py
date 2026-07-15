"""S42 — Quanto è TEMPESTIVO il classifier ai turning point? (MISURA, non propone)

Il bersaglio di S41b: "prendere la V veloce (2020) senza over-tradare i grind lenti
(2008)". Prima di costruire una feature, si misura il ritardo REALE — è la legge del
progetto (12 volte: "misurato prima di costruire → NO").

LA TRAPPOLA CHE QUESTO SCRIPT EVITA
-----------------------------------
Misurare solo il "lead time" premia un segnale SEMPRE ACCESO: se P(stress) è alta il
40% del tempo, il classifier "anticipa" ogni recessione per costruzione, e la metrica
dice che è tempestivo mentre è solo rumoroso. È esattamente la trappola del segnale
liquidità di S40 (surge 314/314 giorni nel 2020 → "entra sul QE" sparava sempre).
Quindi ogni lead-time qui è riportato INSIEME al tasso di falsi allarmi, e i due
vanno letti solo in coppia (sensibilità vs specificità).

MISURE (walk-forward, no future leak: griglia settimanale, regime del venerdì ≤ day)
  1. base_rate      : % di tempo con segnale di stress acceso, 1995-2025.
  2. lead NBER      : mesi di anticipo dell'accensione sull'inizio recessione.
  3. falsi allarmi  : accensioni che NON precedono una recessione entro 12m.
  4. lag d'ingresso : giorni fra il PICCO di mercato e l'accensione (2008/2020/2022).
  5. lag d'uscita   : giorni fra il MINIMO di mercato e lo spegnimento — è qui che si
                      perde la V: restare in stress dopo il minimo = non ricomprare.

Uso:
    python scripts/regime_timeliness.py [--threshold 0.50] [--start-year 1995]

Prima esecuzione: ~7 min (costruisce la griglia settimanale, poi cache su disco).
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date

import pandas as pd

# La console Windows è cp1252: senza questo, una freccia "→" nell'output uccide
# lo script a metà misura (UnicodeEncodeError).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Il classifier va valutato com'è in PROD: i default del codice = i flag del live
# (verificato via GET /config/flags il 2026-07-15: unico runtime override è
# USE_MOMENTUM_PILLARS=True, che è già il default).

from app.services.backtest.daily_regime_grid import build_weekly_grid  # noqa: E402
from app.services.backtest.walk_forward import preload_walk_forward_series  # noqa: E402
from app.services.config_flags import classifier_flags_fingerprint  # noqa: E402
from app.services.indicators.fetcher import FredFetcher  # noqa: E402
from app.services.prices.yahoo_fetcher import YahooFetcher  # noqa: E402

STRESS_REGIMES = ("deflation", "stagflation")

# Finestre di mercato: (picco, minimo) reali dell'S&P, per il lag ingresso/uscita.
EPISODES = {
    "2008 (bear lento)": (date(2007, 10, 9), date(2009, 3, 9)),
    "2020 (V veloce)": (date(2020, 2, 19), date(2020, 3, 23)),
    "2022 (bear tassi)": (date(2022, 1, 3), date(2022, 10, 12)),
}


def stress_prob(frame: pd.DataFrame) -> pd.Series:
    return frame[[f"p_{r}" for r in STRESS_REGIMES]].sum(axis=1)


def nber_spans(start_year: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    usrec = FredFetcher().fetch_series(
        "nber_recession", start_date=date(start_year, 1, 1)
    )
    s = usrec.copy()
    s.index = pd.to_datetime(s.index)
    spans, in_rec, start = [], False, None
    for d, v in s.sort_index().items():
        if v == 1 and not in_rec:
            in_rec, start = True, d
        elif v == 0 and in_rec:
            in_rec = False
            spans.append((start, d - pd.Timedelta(days=1)))
    if in_rec and start is not None:
        spans.append((start, s.index[-1]))
    return spans


def on_episodes(sig: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Intervalli contigui in cui il segnale è acceso."""
    out, start, prev = [], None, None
    for d, v in sig.items():
        if v and start is None:
            start = d
        elif not v and start is not None:
            out.append((start, prev))
            start = None
        prev = d
    if start is not None:
        out.append((start, sig.index[-1]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.50,
                    help="P(deflation)+P(stagflation) sopra cui il segnale è acceso")
    ap.add_argument("--start-year", type=int, default=1995)
    args = ap.parse_args()

    S, E = date(args.start_year, 1, 1), date(2025, 12, 31)
    print(f"Config classifier (fingerprint): {classifier_flags_fingerprint()}")
    print(f"Finestra {S} → {E}, soglia stress = {args.threshold}\n")

    series = preload_walk_forward_series(S, E)
    grid = build_weekly_grid(series, S, E).frame
    stress = stress_prob(grid)
    sig = stress >= args.threshold

    # --- 1. base rate: quanto sta acceso? -----------------------------------
    print("=" * 74)
    print(f"1. BASE RATE — segnale acceso {sig.mean():.1%} delle settimane "
          f"({int(sig.sum())}/{len(sig)})")
    print(f"   regime top per frequenza: "
          f"{grid['regime'].value_counts(normalize=True).round(3).to_dict()}")
    print(f"   P(stress) mediana {stress.median():.3f} · "
          f"min {stress.min():.3f} · max {stress.max():.3f}")

    # --- 2/3. lead NBER + falsi allarmi -------------------------------------
    spans = [(s, e) for s, e in nber_spans(args.start_year) if s >= pd.Timestamp(S)]
    on_eps = on_episodes(sig)
    print("\n" + "=" * 74)
    print("2. LEAD TIME vs recessioni NBER (positivo = anticipa)")
    leads = []
    for rec_start, rec_end in spans:
        pre = sig.loc[(sig.index < rec_start) & (sig.index >= rec_start - pd.DateOffset(months=12))]
        fired = pre[pre]
        if fired.empty:
            print(f"   {rec_start.date()} → {rec_end.date()}: nessun segnale nei 12m prima  ❌")
            continue
        # inizio dell'accensione contigua che arriva alla recessione
        first = fired.index.min()
        lead_m = (rec_start - first).days / 30.4375
        leads.append(lead_m)
        print(f"   {rec_start.date()} → {rec_end.date()}: acceso dal {first.date()} "
              f"= {lead_m:+.1f} mesi di anticipo")
    if leads:
        print(f"   media {sum(leads)/len(leads):+.1f} mesi · hit {len(leads)}/{len(spans)}")

    print("\n3. FALSI ALLARMI — accensioni che NON precedono una recessione entro 12m")
    false_alarms, true_alarms = [], []
    for on_s, on_e in on_eps:
        precedes = any(
            on_s <= rec_s and (rec_s - on_s).days <= 365 * 1.0 or (on_s <= rec_e and on_e >= rec_s)
            for rec_s, rec_e in spans
        )
        (true_alarms if precedes else false_alarms).append((on_s, on_e))
    print(f"   accensioni totali {len(on_eps)} · utili {len(true_alarms)} · "
          f"FALSE {len(false_alarms)}")
    if on_eps:
        prec = len(true_alarms) / len(on_eps)
        print(f"   precisione = {prec:.0%}  (1.0 = mai un falso allarme)")
    for s_, e_ in false_alarms[:12]:
        wks = len(sig.loc[s_:e_])
        print(f"     falso allarme: {s_.date()} → {e_.date()} ({wks} sett.)")
    if len(false_alarms) > 12:
        print(f"     ... e altri {len(false_alarms)-12}")

    # --- 4/5. lag ingresso/uscita sugli episodi ------------------------------
    print("\n" + "=" * 74)
    print("4/5. LAG vs PICCO e MINIMO di mercato (negativo = in ritardo sul mercato)")
    spy = YahooFetcher().fetch("SPY", start_date=date(1995, 1, 1), end_date=E)
    for label, (peak, trough) in EPISODES.items():
        w = sig.loc[pd.Timestamp(peak) - pd.DateOffset(months=9):
                    pd.Timestamp(trough) + pd.DateOffset(months=12)]
        on_here = w[w]
        if on_here.empty:
            print(f"\n   {label}: MAI acceso in finestra  ❌")
            continue
        first_on = on_here.index.min()
        d_peak = (pd.Timestamp(peak) - first_on).days
        off_after = w.loc[w.index > pd.Timestamp(trough)]
        off_dates = off_after[~off_after]
        first_off = off_dates.index.min() if not off_dates.empty else None
        print(f"\n   {label}")
        print(f"     picco mercato {peak} · acceso il {first_on.date()} "
              f"→ {d_peak:+d} giorni ({'ANTICIPA' if d_peak > 0 else 'in ritardo'})")
        if first_off is not None:
            d_tr = (first_off - pd.Timestamp(trough)).days
            print(f"     minimo mercato {trough} · spento il {first_off.date()} "
                  f"→ {d_tr:+d} giorni dopo il minimo (quanta V si perde)")
        else:
            print(f"     minimo mercato {trough} · MAI spento entro +12m "
                  f"→ la ripresa NON viene ricomprata")
        # % di SPY dal minimo alla data di spegnimento = quanto rimbalzo si perde
        if first_off is not None and not spy.empty:
            try:
                px = spy.loc[pd.Timestamp(trough):first_off]
                if len(px) > 1:
                    miss = float(px.iloc[-1] / px.iloc[0] - 1.0)
                    print(f"     rimbalzo perso restando in stress: {miss:+.1%} di S&P")
            except Exception as ex:
                print(f"     (SPY non disponibile: {ex})")

    print("\n" + "=" * 74)
    print("Lettura: un lead alto con base rate alto NON è tempestività, è un segnale")
    print("sempre acceso. Guardare lead e falsi allarmi SOLO in coppia.")


if __name__ == "__main__":
    main()
