"""Stress-test SCENARI IPOTETICI — come si comporta il modello (deploy_boost ON)
in configurazioni NUOVE, non nel passato. Richiesta Erik (2026-07-13).

Metodo: per ogni scenario sintetizzo (a) una traiettoria di REGIME (probs+confidence
mese per mese, con un LAG di riconoscimento realistico) e (b) un path di rendimenti
mensili per le 24 asset class, poi ci faccio girare la logica REALE del modello
(`run_horserace_trade` con la stessa config deploy_boost del powerset via `_apply_arm`).
Confronto ON vs OFF (baseline) e vs 60/40, e misuro drawdown + mesi-sott'acqua (recupero).

⚠️ ONESTÀ: gli scenari sono STILIZZATI e a MANO (magnitudo/durata = mie assunzioni,
ancorate ad analoghi storici citati), NON previsioni. Servono a trovare i PUNTI DEBOLI,
non a dare un numero preciso. Il regime feed è impostato da me (con lag) — approssima
ciò che il classifier vedrebbe, incluso l'essere in ritardo.
"""
from __future__ import annotations

import os
import sys
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from horserace_backtest import run_horserace_trade  # noqa: E402
from horserace_powerset import _apply_arm  # noqa: E402

R = {"reflation": 0, "stagflation": 0, "deflation": 0, "goldilocks": 0}

# 24 asset → gruppo (per definire i profili di rendimento in modo compatto)
GROUP = {
    "us_equities_growth": "eqg", "factor_momentum": "eqg", "factor_quality": "eqg",
    "sector_technology": "tech",
    "us_equities_value": "eqv", "factor_value": "eqv",
    "international_dm_equities": "intl", "em_equities": "em",
    "us_bonds_long": "blong", "us_bonds_short": "bshort", "tips_inflation_bonds": "tips",
    "gold": "gold", "silver": "silver",
    "broad_commodities": "comm", "energy": "comm", "sector_energy": "comm",
    "real_estate_reits": "reits",
    "bitcoin": "crypto", "crypto_broad": "crypto",
    "sector_financials": "fin",
    "sector_utilities": "defs", "sector_healthcare": "defs", "sector_staples": "defs",
    "cash_money_market": "cash",
}
ALL_ASSETS = list(GROUP.keys())


def probs(**kw):
    p = dict(R); p.update(kw)
    s = sum(p.values()) or 1.0
    return {k: v / s for k, v in p.items()}


# --- scenari: lista di fasi (mesi, probs, confidence, profilo rendimenti per gruppo) ---
# groups non citati in un profilo ereditano `base`; 'cash' default +0.003/mese.
def scen_pandemic():
    return ("Pandemia (crash 1-2m + V-recovery)", "analogo 2020", [
        (1, probs(deflation=0.75), 0.70, {"base": -0.16, "blong": 0.05, "gold": 0.03, "tips": 0.01,
                                          "comm": -0.22, "reits": -0.25, "crypto": -0.30, "fin": -0.22,
                                          "tech": -0.12, "defs": -0.06, "bshort": 0.01}),
        (1, probs(deflation=0.60, reflation=0.2), 0.55, {"base": -0.05, "blong": 0.02, "gold": 0.01, "comm": -0.08}),
        (4, probs(reflation=0.65), 0.60, {"base": 0.08, "eqg": 0.10, "tech": 0.12, "comm": 0.05,
                                          "blong": -0.01, "gold": 0.005, "crypto": 0.15}),
        (6, probs(goldilocks=0.62), 0.60, {"base": 0.015, "blong": 0.0}),
    ])


def scen_ai_bubble():
    return ("Bolla AI che scoppia (growth/tech -40%)", "analogo dot-com 2000-02", [
        (6, probs(deflation=0.45, stagflation=0.25), 0.50, {"base": -0.01, "eqg": -0.08, "tech": -0.11,
                                                            "em": -0.04, "eqv": 0.0, "defs": 0.004,
                                                            "blong": 0.008, "gold": 0.006}),
        (3, probs(deflation=0.6), 0.60, {"base": -0.05, "eqg": -0.10, "tech": -0.13, "crypto": -0.20,
                                         "blong": 0.02, "gold": 0.01, "defs": -0.01}),
        (9, probs(goldilocks=0.5, reflation=0.2), 0.50, {"base": 0.02, "eqv": 0.03, "defs": 0.02,
                                                         "eqg": 0.015, "tech": 0.02}),
    ])


def scen_credit():
    return ("Bolla credito privato (GFC-like)", "analogo 2008", [
        (3, probs(deflation=0.5, stagflation=0.15), 0.50, {"base": -0.04, "fin": -0.10, "reits": -0.12,
                                                          "eqg": -0.06, "blong": 0.02, "gold": 0.01}),
        (3, probs(deflation=0.72), 0.72, {"base": -0.12, "fin": -0.16, "reits": -0.15, "crypto": -0.25,
                                          "blong": 0.04, "gold": 0.02, "bshort": 0.005}),
        (6, probs(deflation=0.4, reflation=0.2), 0.40, {"base": 0.0, "blong": 0.01}),
        (12, probs(reflation=0.55), 0.55, {"base": 0.02, "eqg": 0.025, "fin": 0.02}),
    ])


def scen_carry():
    return ("Unwind carry-trade JPY (vol-shock veloce)", "analogo ago-2024 / 1998", [
        (1, probs(goldilocks=0.4, deflation=0.2), 0.35, {"base": -0.11, "em": -0.14, "eqg": -0.12,
                                                        "crypto": -0.20, "blong": 0.02, "gold": 0.01}),
        (2, probs(reflation=0.55), 0.55, {"base": 0.07, "eqg": 0.08, "em": 0.06}),
        (3, probs(goldilocks=0.6), 0.58, {"base": 0.012}),
    ])


def scen_dedollar():
    return ("Dedollarizzazione (tilt secolare, no crash)", "analogo anni '70 debasement", [
        (24, probs(stagflation=0.5, reflation=0.2), 0.55, {"base": 0.002, "eqg": 0.002, "blong": -0.004,
                                                          "eqv": 0.004, "gold": 0.010, "silver": 0.012,
                                                          "comm": 0.008, "em": 0.007, "intl": 0.006,
                                                          "crypto": 0.015, "tips": 0.003, "reits": 0.001,
                                                          "cash": 0.004}),
    ])


def scen_stagflation():
    return ("Stagflazione radicata (equities+bonds insieme giù)", "analogo 1973-79", [
        (18, probs(stagflation=0.65), 0.60, {"base": -0.004, "eqg": -0.005, "blong": -0.006, "eqv": 0.001,
                                             "gold": 0.012, "silver": 0.014, "comm": 0.010, "tips": 0.004,
                                             "reits": -0.003, "em": -0.002, "cash": 0.005}),
    ])


def scen_fiscal():
    return ("Crisi fiscale/sovrana (rate spike, bond crash)", "analogo 2022 UK gilt / '80s", [
        (6, probs(stagflation=0.5, deflation=0.15), 0.50, {"base": -0.03, "blong": -0.035, "eqg": -0.03,
                                                          "eqv": -0.01, "gold": 0.008, "cash": 0.005, "tips": -0.005}),
        (6, probs(goldilocks=0.45, reflation=0.2), 0.45, {"base": 0.01, "blong": 0.005, "cash": 0.005}),
    ])


SCENARIOS = [scen_pandemic, scen_ai_bubble, scen_credit, scen_carry,
             scen_dedollar, scen_stagflation, scen_fiscal]


class SynthWF:
    def __init__(self, d):
        self._d = d

    def get(self, dt):
        return self._d.get(dt)


def _row_from_profile(prof):
    base = prof.get("base", 0.0)
    out = {}
    for a in ALL_ASSETS:
        g = GROUP[a]
        if g == "cash":
            out[a] = prof.get("cash", 0.003)
        else:
            out[a] = prof.get(g, base)
    return out


def build_scenario(phases, start=date(2012, 1, 1), prehist=18, lag=1):
    """Ritorna (matrix, wf, start, horizon). `lag`=mesi di ritardo del regime feed."""
    horizon = sum(m for m, *_ in phases)
    # sequenza mesi: prehist calmi + fasi
    idx_ms = pd.date_range(end=start, periods=prehist + 1, freq="MS")[:-1]  # prehistory month-starts
    scen_ms = pd.date_range(start=start, periods=horizon, freq="MS")
    all_ms = idx_ms.append(scen_ms)

    # regime + returns per mese di scenario
    regime_seq, ret_seq = [], []
    for m, p, c, prof in phases:
        for _ in range(m):
            regime_seq.append((p, c))
            ret_seq.append(_row_from_profile(prof))
    # lag: il modello "vede" il regime di lag mesi fa (all'inizio di ogni scenario resta il calmo)
    calm_reg = (probs(goldilocks=0.6), 0.6)
    lagged = [calm_reg] * lag + regime_seq[:-lag] if lag > 0 else regime_seq

    # matrice rendimenti (month-END index) — prehistory calma
    calm_row = _row_from_profile({"base": 0.008, "blong": 0.002, "gold": 0.003, "cash": 0.003})
    rows, index = [], []
    for ms in idx_ms:
        rows.append(calm_row); index.append(pd.Timestamp(ms) + pd.offsets.MonthEnd(0))
    for ms, r in zip(scen_ms, ret_seq):
        rows.append(r); index.append(pd.Timestamp(ms) + pd.offsets.MonthEnd(0))
    matrix = pd.DataFrame(rows, index=pd.DatetimeIndex(index), columns=ALL_ASSETS)

    # wf_cache: prehistory calma + scenario laggato
    wf = {}
    for ms in idx_ms:
        wf[ms.date().replace(day=1)] = {"probs": calm_reg[0], "confidence": calm_reg[1]}
    for ms, (p, c) in zip(scen_ms, lagged):
        wf[date(ms.year, ms.month, 1)] = {"probs": p, "confidence": c}
    return matrix, SynthWF(wf), start, horizon


def _underwater(path):
    """Max mesi consecutivi sotto un picco precedente (proxy tempo-di-recupero)."""
    peak, longest, cur = -1e9, 0, 0
    for v in path:
        if v >= peak:
            peak, cur = v, 0
        else:
            cur += 1; longest = max(longest, cur)
    return longest


def _run(matrix, wf, start, horizon, mode):
    """mode: 'on'|'off'|'gated'(A)|'reentry'(B)|'gated+B'."""
    if mode in ("gated", "reentry", "gated+B"):
        t = run_horserace_trade(start, matrix, wf, horizon_months=horizon, mom_threshold=0.0,
                                regime_gate=(mode != "reentry"), floor_pct=0.30,
                                fast_reentry=(mode != "gated"))
        return t
    restore, kw = _apply_arm(("deployboost",) if mode == "on" else ())
    try:
        t = run_horserace_trade(start, matrix, wf, horizon_months=horizon, mom_threshold=0.0, **kw)
    finally:
        restore()
    return t


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.environ.setdefault("USE_MOMENTUM_PILLARS", "1")
    os.environ.setdefault("USE_AI_PORTFOLIO_DEPLOY_BOOST", "0")  # config via _apply_arm, non via flag

    print("STRESS-TEST SCENARI IPOTETICI (deploy_boost ON vs OFF, lag regime = 1 mese)")
    print("ret=return modello · a60=alpha vs 60/40 · DD=max drawdown · UW=mesi sott'acqua · dep=deployment medio")
    print("=" * 118)
    for fn in SCENARIOS:
        name, analog, phases = fn()
        matrix, wf, start, horizon = build_scenario(phases, lag=1)
        arms = [("boost ON   ", _run(matrix, wf, start, horizon, "on")),
                ("baseline   ", _run(matrix, wf, start, horizon, "off")),
                ("GATED (A)  ", _run(matrix, wf, start, horizon, "gated")),
                ("REENTRY (B)", _run(matrix, wf, start, horizon, "reentry")),
                ("A+B        ", _run(matrix, wf, start, horizon, "gated+B"))]
        b6040 = arms[0][1].benchmark_60_40
        print(f"\n▸ {name}  [{analog}, {horizon}m]")
        print(f"   60/40 buy&hold: ret {b6040*100:+6.1f}%")
        for tag, t in arms:
            uw = _underwater(t.nav_path)
            print(f"   {tag}: ret {t.realized_return*100:+6.1f}%  a60 {t.alpha_vs_60_40*100:+6.1f}pp  "
                  f"DD {t.max_drawdown*100:6.1f}%  UW {uw:2d}m  dep {t.avg_deployed*100:3.0f}% "
                  f"(min {t.min_deployed*100:2.0f}%)")


if __name__ == "__main__":
    main()
