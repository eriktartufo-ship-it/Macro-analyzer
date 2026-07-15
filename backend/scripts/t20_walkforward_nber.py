"""S42 — Il gate che il codice di T20 chiede da S33: walk-forward NBER, non 32 episodi.

`classifier.py` (commento _LEVEL_ACCEL_PILLAR_ADD) dice testualmente:

    "FLAG RESTA OFF finché una validazione più pesante (walk-forward NBER su
     backfill mensile, o dataset episodi esteso) non scioglie il trade-off.
     NON iterare ancora su questi 32 episodi = overfitting del gate."

Questo script è quella validazione. NON tocca i 32 episodi: gira sulla griglia
settimanale walk-forward 1995-2025 (no future leak) e confronta T20 OFF vs ON.

⚠️ PERCHÉ FINO A IERI ERA IMPOSSIBILE: la griglia era cachata per SOLA DATA →
cambiando il flag si rileggeva la griglia dei flag vecchi e il confronto dava Δ=0.
Fixato in S42 (cache per data+fingerprint-flag).

IL PUNTO METODOLOGICO — CONFRONTO A PARITÀ DI BASE RATE
------------------------------------------------------
Confrontare le due config alla STESSA SOGLIA è un inganno: se T20 ON abbassa
P(deflation) di ~2pp, alla stessa soglia si accende meno e quindi sembra "più
preciso" — ma è solo una soglia più severa travestita da feature. La misura di
S42 mostra che precisione e lead si comprano l'un l'altro lungo una curva
(soglia 0.40→0.70: precisione 12%→80%, lead +11.8→+1.0 mesi).

Quindi qui si confronta A PARITÀ DI BASE RATE: per ogni livello di "quanto tempo
il segnale sta acceso", si trova la soglia che lo produce in CIASCUNA config e si
confrontano precisione, lead e ritardo d'uscita. Se T20 ON è meglio a parità di
base rate, allora è una feature vera e non una soglia mascherata.

VERDETTO = T20 ON deve, a parità di base rate:
  (a) NON peggiorare la precisione (falsi allarmi), e
  (b) migliorare almeno una fra: lead NBER, ritardo d'uscita.

Uso:
    python scripts/t20_walkforward_nber.py

Prima esecuzione: ~8 min (costruisce la griglia T20=ON; quella OFF è già in cache).
"""
from __future__ import annotations

import os
import sys
from datetime import date

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from regime_timeliness import (  # noqa: E402
    EPISODES, nber_spans, on_episodes, stress_prob,
)

from app.services.backtest.daily_regime_grid import build_weekly_grid  # noqa: E402
from app.services.backtest.walk_forward import preload_walk_forward_series  # noqa: E402
from app.services.config_flags import (  # noqa: E402
    classifier_flags_fingerprint, set_runtime_flag,
)

S, E = date(1995, 1, 1), date(2025, 12, 31)
BASE_RATES = (0.10, 0.15, 0.20, 0.30, 0.40)  # livelli di confronto
_THRESHOLDS = [round(x, 3) for x in [0.20 + 0.005 * i for i in range(140)]]


def _metrics(stress: pd.Series, thr: float, spans) -> dict:
    """Metriche a una soglia: base rate, precisione, lead NBER, ritardo d'uscita."""
    sig = stress >= thr
    eps = on_episodes(sig)

    true_a, false_a = 0, 0
    for on_s, on_e in eps:
        precedes = any(
            (on_s <= rec_s and (rec_s - on_s).days <= 365) or (on_s <= rec_e and on_e >= rec_s)
            for rec_s, rec_e in spans
        )
        if precedes:
            true_a += 1
        else:
            false_a += 1

    leads = []
    for rec_start, _ in spans:
        pre = sig.loc[(sig.index < rec_start)
                      & (sig.index >= rec_start - pd.DateOffset(months=12))]
        fired = pre[pre]
        if not fired.empty:
            leads.append((rec_start - fired.index.min()).days / 30.4375)

    # ritardo d'uscita: giorni dal minimo di mercato allo spegnimento
    exits = []
    for _, (peak, trough) in EPISODES.items():
        w = sig.loc[pd.Timestamp(peak) - pd.DateOffset(months=9):
                    pd.Timestamp(trough) + pd.DateOffset(months=18)]
        after = w.loc[w.index > pd.Timestamp(trough)]
        off = after[~after]
        if not off.empty:
            exits.append((off.index.min() - pd.Timestamp(trough)).days)
        else:
            exits.append(999)  # mai spento entro +18m: penalità esplicita

    return {
        "thr": thr,
        "base_rate": float(sig.mean()),
        "precision": (true_a / len(eps)) if eps else 0.0,
        "n_fires": len(eps),
        "lead": (sum(leads) / len(leads)) if leads else 0.0,
        "hit": len(leads),
        "exit_lag": sum(exits) / len(exits),
    }


def _curve(stress: pd.Series, spans) -> list[dict]:
    return [_metrics(stress, t, spans) for t in _THRESHOLDS]


def _at_base_rate(curve: list[dict], target: float) -> dict | None:
    """La soglia che produce il base rate più vicino al target."""
    ok = [c for c in curve if abs(c["base_rate"] - target) < 0.03]
    if not ok:
        return None
    return min(ok, key=lambda c: abs(c["base_rate"] - target))


def _build(flag_on: bool, series) -> pd.Series:
    set_runtime_flag("USE_LEVEL_ACCEL_PILLARS", flag_on, persist=False)
    fp = classifier_flags_fingerprint()
    print(f"  T20 {'ON ' if flag_on else 'OFF'} → fingerprint {fp}")
    grid = build_weekly_grid(series, S, E).frame
    return stress_prob(grid)


def main() -> None:
    print(f"Gate T20 walk-forward NBER · finestra {S} → {E}\n")
    series = preload_walk_forward_series(S, E)
    spans = [(s, e) for s, e in nber_spans(1995) if s >= pd.Timestamp(S)]
    print(f"Recessioni NBER nel campione: {len(spans)}")

    off_stress = _build(False, series)
    on_stress = _build(True, series)
    set_runtime_flag("USE_LEVEL_ACCEL_PILLARS", None, persist=False)

    if float((off_stress - on_stress).abs().max()) < 1e-9:
        print("\n⛔ Le due griglie sono IDENTICHE: il flag non ha effetto o la cache "
              "sta ancora collassando le config. Test non valido.")
        return

    off_c, on_c = _curve(off_stress, spans), _curve(on_stress, spans)

    print("\n" + "=" * 78)
    print("CONFRONTO A PARITÀ DI BASE RATE (il confronto onesto)")
    print("=" * 78)
    print(f"{'base rate':>10} | {'soglia OFF→ON':>14} | {'precisione':>18} | "
          f"{'lead NBER (m)':>16} | {'uscita (gg)':>16}")
    print("-" * 78)

    verdicts = []
    for br in BASE_RATES:
        a, b = _at_base_rate(off_c, br), _at_base_rate(on_c, br)
        if not a or not b:
            print(f"{br:>9.0%} | (base rate non raggiungibile in una delle due config)")
            continue
        d_prec = b["precision"] - a["precision"]
        d_lead = b["lead"] - a["lead"]
        d_exit = b["exit_lag"] - a["exit_lag"]
        verdicts.append((br, d_prec, d_lead, d_exit, a, b))
        print(f"{br:>9.0%} | {a['thr']:>6.3f}→{b['thr']:<6.3f} | "
              f"{a['precision']:>6.0%}→{b['precision']:<5.0%} ({d_prec:+.0%}) | "
              f"{a['lead']:>6.1f}→{b['lead']:<5.1f} ({d_lead:+.1f}) | "
              f"{a['exit_lag']:>5.0f}→{b['exit_lag']:<5.0f} ({d_exit:+.0f})")

    print("\n" + "=" * 78)
    print("VERDETTO")
    print("=" * 78)
    if not verdicts:
        print("Nessun livello di base rate confrontabile → test inconcludente.")
        return

    n = len(verdicts)
    prec_ok = sum(1 for v in verdicts if v[1] >= -0.01)
    lead_up = sum(1 for v in verdicts if v[2] > 0.1)
    exit_up = sum(1 for v in verdicts if v[3] < -1)
    avg_prec = sum(v[1] for v in verdicts) / n
    avg_lead = sum(v[2] for v in verdicts) / n
    avg_exit = sum(v[3] for v in verdicts) / n

    print(f"  precisione non peggiorata : {prec_ok}/{n} livelli (Δ medio {avg_prec:+.1%})")
    print(f"  lead NBER migliorato      : {lead_up}/{n} livelli (Δ medio {avg_lead:+.2f} mesi)")
    print(f"  uscita più rapida         : {exit_up}/{n} livelli (Δ medio {avg_exit:+.0f} giorni)")

    passed = prec_ok == n and (lead_up + exit_up) >= 1
    print()
    if passed:
        print("  ✅ PASS — T20 migliora a parità di base rate: NON è una soglia mascherata.")
        print("     Prossimo passo prima di promuovere: Δ multi-seed sull'alpha + OOS.")
    else:
        print("  ❌ FAIL — T20 non paga a parità di base rate. Il flag resta OFF.")
        print("     NON ritoccare i pesi per farlo passare: sarebbe hand-tuning del gate")
        print("     (regola .claude/rules/macro-no-hand-tuning.md).")


if __name__ == "__main__":
    main()
