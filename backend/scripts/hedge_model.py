"""Tail-hedge REALE (put OTM 1M rollata, prezzata Black-Scholes) vs il proxy.

Follow-up #1 del powerset 2026-07-13: nel powerset l'hedge era un payoff convesso
SINTETICO (carry −0.4%/mese, moltiplicatore k=3 hand-tuned) → direzione giusta,
MAGNITUDO non affidabile. Qui la sostituiamo con un modello ancorato ai prezzi:

  ogni mese compriamo put europee 1-mese OTM sull'indice equity, spendendo un
  BUDGET-premio fisso `prem` (frac di NAV). Il numero di contratti esce dal PREZZO
  Black-Scholes (non da un k arbitrario); il payoff è `max(K − S_fine_mese, 0)` sui
  rendimenti equity REALI della matrice. Contributo netto al mese = payoff − prem.

Fedeltà: REAL-ish. Approssimazioni DICHIARATE:
  - implied vol = costante `sigma` (default 0.18, tipico indice equity). NON time-varying
    → sweep di sigma per mostrare la sensibilità. r mensile ≈ risk-free costante.
  - 1 solo strike/scadenza (no smile, no term structure). Standard per un first-cut.

Non tocca i sorgenti sotto audit: monkeypatcha `horserace_backtest.hedge_month_contribution`
a runtime, così il flag `hedge_pct>0` instrada al modello put invece che al proxy.

Uso: python scripts/hedge_model.py [--prem 0.005] [--otm 0.10] [--sigma 0.18]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import horserace_backtest as H  # noqa: E402
from horserace_powerset import _apply_arm, _batch, _agg, _crisis_pool  # noqa: E402


# --- Black-Scholes put (no scipy: N via math.erf) --------------------------
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Prezzo put europea Black-Scholes. Guard su input degeneri."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def make_put_contribution(prem: float, otm: float, sigma: float,
                          skew: float = 0.0, r_ann: float = 0.03):
    """Factory: ritorna una fn (hedge_pct, equity_ret, cash_ret) compatibile con
    il call-site dell'harness, che modella il rolled-put invece del proxy.

    hedge_pct funge solo da interruttore (>0 = attivo); il dimensionamento è dato
    dal budget-premio `prem`, non da hedge_pct.

    VOL SKEW (fix 2026-07-13): le put OTM NON si prezzano a vol ATM — hanno uno
    skew (implied cresce quanto più sono OTM). σ_eff = sigma + skew·otm. Con skew=0
    (default) resta flat-vol. Prezzare 10%-OTM a vol ATM dava 15x di leva → payoff
    assurdi (+171pp = artefatto). Empirico equity: 1M 10%-OTM ~30-40% implied.
    """
    T = 1.0 / 12.0
    K = 1.0 - otm
    sigma_eff = sigma + skew * otm
    p = bs_put(1.0, K, T, r_ann, sigma_eff)  # premio per unità di nozionale (S=1)

    def _contrib(hedge_pct, equity_ret, cash_ret, **_kw):
        if hedge_pct <= 0 or p <= 0:
            return 0.0
        notional = prem / p                       # contratti dal prezzo reale
        payoff = notional * max(K - (1.0 + equity_ret), 0.0)
        return payoff - prem                       # netto del premio speso

    _contrib.premium = p
    return _contrib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prem", type=float, default=0.005, help="budget-premio mensile (frac NAV)")
    ap.add_argument("--otm", type=float, default=0.10, help="moneyness put (0.10 = 10%% OTM)")
    ap.add_argument("--sigma", type=float, default=0.18, help="implied vol annua costante")
    ap.add_argument("--broad-sims", type=int, default=50)
    ap.add_argument("--crisis-sims", type=int, default=45)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.environ.setdefault("USE_MOMENTUM_PILLARS", "1")
    os.environ.setdefault("USE_AI_PORTFOLIO_DEPLOY_BOOST", "0")  # baseline pulito

    from app.services.scoring.engine import ASSET_REGIME_DATA
    from app.services.backtest.real_returns_matrix import build_real_returns_matrix
    from app.services.backtest.walk_forward import WalkForwardCache, preload_walk_forward_series

    print("Building matrix + FRED cache...")
    real_matrix = build_real_returns_matrix(list(ASSET_REGIME_DATA.keys()))
    series = preload_walk_forward_series(start_date=date(1993, 1, 1), end_date=date(2025, 12, 31))
    wf_cache = WalkForwardCache(series)

    # Sweep con VOL SKEW realistico: σ_eff = atm_vol + skew·otm.
    # skew 1.8 → 10%OTM ~0.34, 5%OTM ~0.25 (coerente con lo skew equity reale).
    # (label, prem, otm, atm_vol, skew)
    put_configs = [
        ("put(0.5%,10%OTM,skew)", 0.005, 0.10, 0.16, 1.8),
        ("put(0.3%,10%OTM,skew)", 0.003, 0.10, 0.16, 1.8),
        ("put(0.5%, 5%OTM,skew)", 0.005, 0.05, 0.16, 1.8),
        ("put(1.0%,10%OTM,skew)", 0.010, 0.10, 0.16, 1.8),
        ("put(0.5%,10%OTM,skew-lite)", 0.005, 0.10, 0.16, 1.0),  # skew più basso = più ottimista
    ]

    orig_contrib = H.hedge_month_contribution

    def run_arm(names, crisis_window="all_crisis"):
        restore, kwargs = _apply_arm(names)
        try:
            broad = _batch(None, args.broad_sims, 7, real_matrix, wf_cache, kwargs)
            crisis = _batch(_crisis_pool(crisis_window), args.crisis_sims, 42, real_matrix, wf_cache, kwargs)
        finally:
            restore()
        return {"broad": _agg(broad), "crisis": _agg(crisis)}

    rows = []
    # riferimenti: baseline, deployboost puro, deployboost+hedge(PROXY)
    rows.append(("baseline", run_arm(())))
    rows.append(("deployboost", run_arm(("deployboost",))))
    rows.append(("deployboost+hedge[PROXY]", run_arm(("deployboost", "hedge"))))
    # deployboost+hedge con la put REALE (sweep)
    for label, prem, otm, atm_vol, skew in put_configs:
        H.hedge_month_contribution = make_put_contribution(prem, otm, atm_vol, skew)
        premq = H.hedge_month_contribution.premium
        rows.append((f"deployboost+hedge[{label} p={premq*100:.2f}% carry~{prem*12*100:.0f}%/yr]",
                     run_arm(("deployboost", "hedge"))))
    H.hedge_month_contribution = orig_contrib

    base = rows[0][1]

    def _fmt(a):
        return (f"dep{a['deployed']*100:3.0f}% a60{a['a60']*100:+5.1f} DD{a['dd']*100:5.1f} "
                f"sev{a['sev']*100:3.0f}%") if a else "n/a"

    print("\n" + "=" * 118)
    print("TAIL-HEDGE REALE (rolled put BS) vs PROXY — BROAD (1995-2023) | CRISI all_crisis")
    print("=" * 118)
    for label, r in rows:
        d = ""
        if r["broad"] and base["broad"]:
            db = (r["broad"]["a60"] - base["broad"]["a60"]) * 100
            dc = (r["crisis"]["a60"] - base["crisis"]["a60"]) * 100 if r["crisis"] and base["crisis"] else 0.0
            d = f"  Δbull {db:+.1f} Δcrisi {dc:+.1f}pp"
        print(f"  {label[:52]:52s} | B {_fmt(r['broad'])} | C {_fmt(r['crisis'])}{d}")


if __name__ == "__main__":
    main()
