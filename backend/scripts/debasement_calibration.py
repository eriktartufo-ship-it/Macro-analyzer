"""Calibrazione ONESTA delle soglie del meta-regime Debasement (DREAM, R4).

NON fit-su-outcome (overfitting: solo ~3 episodi 2008+, e sconfinerebbe nel market-timing
contro la regola S46 "vista informativa"). Invece:
  1. DATA-GROUNDING della sigmoid intensità sulla distribuzione reale del gap
     (center = mediana, scale = mezzo-IQR → p25/p75 cadono a ~0.27/0.73).
  2. SWEEP DI ROBUSTEZZA (plateau) sulle soglie on/off/conferma: si cercano i valori
     dove la struttura (n° episodi, on-ora) è STABILE, non un punto ottimo.

Uso: python -m scripts.debasement_calibration   (dalla dir backend/, venv attivo)
Read-only: stampa un report, NON modifica il codice.
"""
from __future__ import annotations

import numpy as np

from app.services.backtest.debasement_divergence import load_live_divergence
from app.services.backtest.debasement_regime import apply_hysteresis


def sigmoid(gap: float, center: float, scale: float) -> float:
    z = max(-30.0, min(30.0, (gap - center) / scale))
    return 1.0 / (1.0 + np.exp(-z))


def episodes(states: list[bool]) -> int:
    n, prev = 0, False
    for s in states:
        if s and not prev:
            n += 1
        prev = s
    return n


def main() -> None:
    d = load_live_divergence()
    gaps = np.array([h["gap"] for h in d["history"]])
    dates = [h["date"] for h in d["history"]]
    print(f"Storia fedele: {len(gaps)} mesi  {dates[0]} -> {dates[-1]}")

    # (1) data-grounding sigmoid
    med = float(np.median(gaps))
    p25, p75 = float(np.percentile(gaps, 25)), float(np.percentile(gaps, 75))
    half_iqr = (p75 - p25) / 2.0
    print("\n[1] DATA-GROUNDING SIGMOID")
    print(f"  mediana gap = {med*100:+.1f}%  -> center = {round(med, 2)}")
    print(f"  IQR = [{p25*100:+.1f}%, {p75*100:+.1f}%]  mezzo-IQR = {half_iqr:.3f}  -> scale = {round(half_iqr, 2)}")
    CENTER, SCALE = round(med, 2), round(half_iqr, 2)
    inten = [sigmoid(g, CENTER, SCALE) for g in gaps]
    today = inten[-1]
    print(f"  con (center={CENTER}, scale={SCALE}): intensità oggi = {today:.3f} "
          f"(gap {gaps[-1]*100:+.1f}%)")

    # (2) sweep di robustezza sulle soglie
    print("\n[2] SWEEP DI ROBUSTEZZA (plateau) — con la sigmoid data-grounded")
    print(f"{'on':>5}{'off':>6}{'c_on':>6}{'c_off':>7}{'#epis':>7}{'on_now':>8}{'mesi_now':>10}{'%on':>7}")
    stable = []
    for on in (0.50, 0.55, 0.60, 0.65, 0.70):
        for off in (0.30, 0.35, 0.40, 0.45):
            if off >= on:
                continue
            for c_on in (2, 3):
                for c_off in (4, 6, 8):
                    st = apply_hysteresis(inten, on=on, off=off, confirm_on=c_on, confirm_off=c_off)
                    ep = episodes(st)
                    on_now = st[-1]
                    # mesi nello stato corrente
                    m = 0
                    for s in reversed(st):
                        if s == on_now:
                            m += 1
                        else:
                            break
                    pct = sum(st) / len(st)
                    row = (on, off, c_on, c_off, ep, on_now, m, pct)
                    # plateau target: struttura ~3 episodi, acceso ORA
                    if ep in (2, 3, 4) and on_now:
                        stable.append(row)
    # stampa un sottoinsieme rappresentativo + sintesi plateau
    for on in (0.55, 0.60, 0.65):
        for c_off in (6,):
            st = apply_hysteresis(inten, on=on, off=0.40, confirm_on=2, confirm_off=c_off)
            ep = episodes(st); on_now = st[-1]
            m = 0
            for s in reversed(st):
                if s == on_now: m += 1
                else: break
            print(f"{on:>5}{0.40:>6}{2:>6}{c_off:>7}{ep:>7}{str(on_now):>8}{m:>10}{sum(st)/len(st):>7.2f}")

    print(f"\n  Config nel plateau (2-4 episodi & on-ora): {len(stable)}/"
          f"{5*4*2*3 - sum(1 for on in (0.5,0.55,0.6,0.65,0.7) for off in (0.3,0.35,0.4,0.45) if off>=on)*2*3}")
    if stable:
        eps_set = sorted({r[4] for r in stable})
        print(f"  #episodi osservati nel plateau: {eps_set}")
        # centro del plateau = mediana dei parametri stabili
        arr = np.array([[r[0], r[1], r[2], r[3]] for r in stable], dtype=float)
        c = arr.mean(axis=0)
        print(f"  centro del plateau ~ on={c[0]:.2f} off={c[1]:.2f} confirm_on={c[2]:.1f} confirm_off={c[3]:.1f}")
        onnow_all = all(r[5] for r in stable)
        print(f"  'acceso ORA' stabile su TUTTO il plateau: {onnow_all}")


if __name__ == "__main__":
    main()
