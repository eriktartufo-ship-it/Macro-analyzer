"""Horserace strategist backtest — rigioca la logica REALE di `strategist._decide`
(5 gate + tranche DCA + cash-overnight) su dati storici, walk-forward senza
future-leak, per capire se lo STRATO strategist preserva o distrugge l'edge del
motore macro (validato separatamente da paper_trading_simulation).

Differenza vs paper_trading_simulation.py:
- paper_trading_sim: allocazione top-N quasi-full-invested (testa il MOTORE macro).
- questo: replica i gate del model_driven horserace (score≥55, momentum BULLISH,
  top-12 regime, tranche 1/N, uscite stop/TP) → testa la STRATEGIA dell'AI portfolio,
  quella che tiene molta liquidità.

FEDELTÀ (cosa è reale vs approssimato):
- REALE: `strategist._decide`, `sizer.raw_target_weight/normalize_and_cap/dynamic_n_tranches`,
  `strategist._regime_top_n_assets`, regime+confidence as-of (wf_cache, no leak),
  rendimenti mensili reali (real_returns_matrix), cash idle remunerato al return
  reale di `cash_money_market` (BIL) → coerente col fix cash-overnight 2026-07-13.
- APPROSSIMATO (dichiarato):
  1. Momentum: il live usa MA50+RSI su prezzi DAILY; qui proxy = segno del
     rendimento cumulato trailing-3m dalla matrice MENSILE (soglia configurabile).
  2. Learning ratchet DISATTIVATO (entry_threshold_shift=0): isola i gate base
     dal ratchet adattivo. v1 volutamente.
  3. PnL per-posizione compounded mensilmente; su tranche aggiuntiva si diluisce
     pro-quota (nuovo capitale entra a prezzo corrente).

Uso:
    python scripts/horserace_backtest.py [--n-sims 30] [--seed 42]
        [--crisis-window 2008|2020|2022|all_crisis] [--horizon-months 12]
        [--start-min-year 1995] [--start-max-year 2023]
        [--mom-threshold 0.0]
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Funzioni pure di stato (testabili senza rete/DB)
# ---------------------------------------------------------------------------

def dilute_pnl_on_add(pnl_pct: float, cost_before: float, size_added: float) -> float:
    """PnL diluito quando entra una tranche a prezzo corrente (pnl=0 sul nuovo lotto).

    Nuovo pnl = pnl_vecchio × cost_before / (cost_before + size_added).
    """
    total = cost_before + size_added
    if total <= 0:
        return 0.0
    return pnl_pct * (cost_before / total)


def portfolio_month_return(
    positions: dict, cash_weight: float, asset_returns: dict, cash_return: float,
) -> float:
    """Return mensile del portafoglio = Σ(peso_i × ret_i) + cash × cash_return.

    Asset senza return nel mese → contributo 0 (peso resta, prezzo invariato).
    """
    total = 0.0
    for asset, pos in positions.items():
        r = asset_returns.get(asset)
        if r is not None and not (isinstance(r, float) and np.isnan(r)):
            total += pos["weight"] * r
    total += cash_weight * cash_return
    return total


def drift_weights(positions: dict, cash_weight: float, asset_returns: dict,
                  cash_return: float, port_return: float) -> float:
    """Aggiorna i pesi dopo i rendimenti del mese (rinormalizza a Σ=1 via NAV).

    Ritorna il nuovo cash_weight; muta positions in place.
    """
    denom = 1.0 + port_return
    if denom <= 0:
        denom = 1e-9
    for asset, pos in positions.items():
        r = asset_returns.get(asset)
        r = r if (r is not None and not (isinstance(r, float) and np.isnan(r))) else 0.0
        pos["weight"] = pos["weight"] * (1.0 + r) / denom
        pos["pnl_pct"] = (1.0 + pos["pnl_pct"]) * (1.0 + r) - 1.0
    return cash_weight * (1.0 + cash_return) / denom


# ---------------------------------------------------------------------------
# Backtest core
# ---------------------------------------------------------------------------

@dataclass
class HorseraceTrade:
    start_date: date
    end_date: date
    realized_return: float | None = None
    benchmark_60_40: float | None = None
    benchmark_sp500: float | None = None
    alpha_vs_60_40: float | None = None
    alpha_vs_sp500: float | None = None
    max_drawdown: float = 0.0
    avg_deployed: float = 0.0       # frazione media schierata (1 - cash)
    min_deployed: float = 1.0
    n_opens: int = 0
    n_closes: int = 0
    severity: str = "n/a"
    nav_path: list = field(default_factory=list)  # NAV mensile (per recovery-time negli scenari)


BENCHMARK_60_40 = {"us_equities_growth": 0.60, "us_bonds_long": 0.40}
BENCHMARK_SP500 = {"us_equities_growth": 1.0}
CASH_ASSET = "cash_money_market"


def _bench_return_month(matrix, target_ts, weights) -> float | None:
    if target_ts not in matrix.index:
        return None
    row = matrix.loc[target_ts]
    total, wsum = 0.0, 0.0
    for a, w in weights.items():
        if a in row.index and not pd.isna(row[a]):
            total += w * float(row[a]); wsum += w
    return total / wsum if wsum > 0.5 else None


def _momentum_proxy(matrix, asset, target_ts, threshold: float) -> str:
    """Proxy mensile del momentum: segno del cumulato trailing-3m."""
    if asset not in matrix.columns:
        return "NEUTRAL"
    hist = matrix.loc[matrix.index < target_ts, asset].dropna().tail(3)
    if len(hist) < 3:
        return "NEUTRAL"
    cum = float((1.0 + hist).prod() - 1.0)
    if cum > threshold:
        return "BULLISH"
    if cum < -threshold:
        return "BEARISH"
    return "NEUTRAL"


def _trailing_return(matrix, asset, target_ts, months: int = 2) -> float:
    """Rendimento cumulato degli ultimi `months` mesi PRIMA di target_ts (0 se dati insuff.)."""
    if asset not in matrix.columns:
        return 0.0
    hist = matrix.loc[matrix.index < target_ts, asset].dropna().tail(months)
    if len(hist) < months:
        return 0.0
    return float((1.0 + hist).prod() - 1.0)


def _vol_trailing(matrix, asset, target_ts) -> float:
    if asset not in matrix.columns:
        return 0.20
    hist = matrix.loc[matrix.index <= target_ts, asset].dropna().tail(12)
    return float(hist.std(ddof=1)) * np.sqrt(12) if len(hist) >= 6 else 0.20


# ---------------------------------------------------------------------------
# Leve candidate (batch 2026-07-13) — funzioni PURE, testabili senza rete/DB.
# Ognuna è opt-in: default = no-op → i risultati baseline restano identici.
# Fedeltà dichiarata: REAL = rigioca meccanica esistente · PROXY = approssima.
# ---------------------------------------------------------------------------

def vol_target_scale(
    weights: dict, vols: dict, target_vol: float, rho: float = 0.3, max_scale: float = 1.0,
) -> float:
    """voltarget (REAL·#2): scale per portare la vol annualizzata stimata a `target_vol`.

    Stima vol di portafoglio = sqrt(Σ(w_i σ_i)² + 2ρ Σ_{i<j} w_i σ_i w_j σ_j),
    con ρ=0.3 (empirico Bridgewater, coerente con use_vol_targeting).
    Cap a `max_scale` (default 1.0 = solo cash-buffer, NIENTE leva/margin: conservativo).
    Ritorna 1.0 se il portafoglio è vuoto o la vol stimata è ~0.
    """
    contribs = [w * vols.get(a, 0.20) for a, w in weights.items() if w > 0]
    if not contribs:
        return 1.0
    var = sum(c * c for c in contribs)
    for i in range(len(contribs)):
        for j in range(i + 1, len(contribs)):
            var += 2 * rho * contribs[i] * contribs[j]
    port_vol = var ** 0.5
    if port_vol <= 1e-6:
        return 1.0
    return min(max_scale, target_vol / port_vol)


def hedge_month_contribution(
    hedge_pct: float, equity_ret: float, cash_ret: float,
    carry: float = -0.004, k: float = 3.0, thresh: float = -0.08,
) -> float:
    """hedge (PROXY·#4): overlay convesso di tail-hedge finanziato dalla liquidità.

    NON è una put reale (non è nella matrice rendimenti): approssima il payoff
    convesso di una protezione OTM. Contributo al return mensile del portafoglio:
      - carry negativo costante (default −0.4%/mese ≈ −4.7%/anno) sulla sleeve;
      - se il mese equity crolla sotto `thresh` (−8%): payoff = k × (|ret| − |thresh|).
    Il contributo è al NETTO del cash_return che quella fetta avrebbe reso comunque
    (la sleeve è carvata dalla liquidità idle). hedge_pct=0 → 0.0 (no-op).
    """
    if hedge_pct <= 0:
        return 0.0
    sleeve_ret = carry
    if equity_ret < thresh:
        sleeve_ret = k * (-equity_ret - (-thresh))
    return hedge_pct * (sleeve_ret - cash_ret)


def run_horserace_trade(
    start_date: date, real_matrix, wf_cache, horizon_months: int = 12,
    mom_threshold: float = 0.0, levers: set | None = None,
    top_n_regime: int | None = None, floor_pct: float = 0.0,
    vol_target: float | None = None, cost_bps: float = 0.0,
    cadence_months: int = 1, hedge_pct: float = 0.0, derisk: bool = False,
    regime_gate: bool = False, fast_reentry: bool = False,
    transition_derisk: bool = False, stress_derisk: bool = False,
) -> HorseraceTrade | None:
    """Rigioca la strategia AI su una finestra storica.

    Leve backtestabili (opt-in; default = comportamento originale):
      - vol_target (REAL·#2): scala i target alla vol-budget annualizzata (cap 1.0).
      - cost_bps (REAL·#6): costo di transazione sul turnover mensile (bps).
      - cadence_months (REAL·#6): ribilancia ogni K mesi (fra un rebalance e l'altro
        le posizioni driftano; nessuna decisione, incluse le uscite → dichiarato).
      - hedge_pct (PROXY·#4): overlay convesso di tail-hedge (vedi hedge_month_contribution).
      - derisk (PROXY·#3): dimezza il deployment nei mesi di transizione/incertezza
        (confidence<0.45 o max prob<0.45) — proxy della qualità-segnale senza
        ricostruire il classifier.
      - regime_gate (EXP·A 2026-07-13): deploy_boost CONDIZIONATO — le meccaniche
        boost (deadband ENTRY + fastramp≤2 + floor) si attivano SOLO quando il regime
        dominante è costruttivo (reflation/goldilocks); nei regimi avversi
        (stagflation/deflation) resta il deployment conservativo baseline. Nato dallo
        stress-test scenari: deploy_boost always-on danneggia i bear lenti. Gate
        OSSERVABILE (nessun foresight). Ignora i patch di modulo di deployboost.
      - fast_reentry (EXP·B 2026-07-13): dopo un crollo brusco recente (rendimento 2m
        < −8%) in regime favorevole, consente il rientro col momentum NEUTRAL (non
        aspetta il proxy-momentum lento a 3m). Mira alle V-recovery post vol-shock
        (tallone d'Achille del carry-unwind).
    Le leve d'ingresso/uscita "a costante" (deadband, fastramp, tightstop, entry,
    top_n, momneutral) restano patch di modulo applicate dal runner.
    """
    from app.services.ai_portfolio import strategist, sizer
    from app.services.scoring.engine import calculate_final_scores

    levers = levers or set()
    top_n_regime = top_n_regime or strategist.TOP_N_REGIME
    _orig_incr = strategist.INCREMENTAL_SCORE_THRESHOLD  # per regime_gate (set/restore)
    boost_active = True

    trade = HorseraceTrade(
        start_date=start_date,
        end_date=start_date + timedelta(days=30 * horizon_months),
    )

    positions: dict = {}          # asset -> {weight, pnl_pct, tranches_filled, tranches_total, avg_entry_score, took_partial_tp, entry_regime}
    cash_weight = 1.0
    nav = 1.0
    peak = 1.0
    deployed_samples = []

    hedge_pct = max(0.0, min(hedge_pct, 0.30))  # sleeve carvata dalla liquidità
    current = date(start_date.year, start_date.month, 1)
    for _m_idx in range(horizon_months):
        month_end = pd.Timestamp(current) + pd.offsets.MonthEnd(0)
        cm = wf_cache.get(current)
        if cm is None:
            current = (month_end + pd.offsets.MonthBegin(1)).date()
            continue
        probs = cm["probs"]
        confidence = cm.get("confidence", 0.5) or 0.5

        # cadence (REAL·#6): decidi solo ogni K mesi; fra un rebalance e l'altro
        # le posizioni driftano soltanto (nessuna decisione, incluse le uscite).
        decide_now = (_m_idx % max(1, cadence_months) == 0)
        month_turnover = 0.0

        if decide_now:
            try:
                scores = calculate_final_scores(probs)
            except Exception:
                scores = {}
            regime_top = strategist._regime_top_n_assets(probs, top_n=top_n_regime)
            # regime_gate (EXP·A): boost attivo SOLO se regime dominante costruttivo.
            boost_active = True
            if regime_gate:
                dom = max(probs, key=probs.get)
                boost_active = dom in ("reflation", "goldilocks")
                strategist.INCREMENTAL_SCORE_THRESHOLD = (
                    strategist.ENTRY_SCORE_THRESHOLD if boost_active else _orig_incr)
            n_tranches = sizer.dynamic_n_tranches(confidence)
            if regime_gate and boost_active and n_tranches > 0:
                n_tranches = min(2, n_tranches)  # fastramp solo se boost attivo

            # Target weights (stesso path del live: raw solo per top-12, poi normalize+cap)
            raw = {}
            asset_vols = {}
            for a, sc in scores.items():
                vol = _vol_trailing(real_matrix, a, month_end)
                asset_vols[a] = vol
                raw[a] = sizer.raw_target_weight(sc, vol) if a in regime_top else 0.0
            targets = sizer.normalize_and_cap(raw)

            # derisk (PROXY·#3): nei mesi confusi (transizione) dimezza il deployment.
            if derisk and (confidence < 0.45 or (probs and max(probs.values()) < 0.45)):
                targets = {a: w * 0.5 for a, w in targets.items()}

            # transition_derisk (S41 2026-07-14): come derisk ma col DETECTOR VERO
            # (transition_detector.assess_transition: entropy + max_prob + confidence,
            # require_all_three) invece del proxy OR grezzo. Agisce SOLO sulle
            # transizioni GENUINE (confusione vera), non sul rumore → dimezza il
            # deployment quel mese. Nota: il financial_stress_veto (che nel live FORZA
            # transizione su 2018Q4/SVB) NON è nel wf_cache → qui resta off → il
            # backtest è un LOWER BOUND della protezione (il live sarebbe ≥ protettivo).
            if transition_derisk:
                from app.services.regime.transition_detector import assess_transition
                ta = assess_transition(probs=probs, confidence=confidence)
                if ta.is_transition:
                    targets = {a: w * 0.5 for a, w in targets.items()}

            # stress_derisk (S41): de-risk sul segnale di STRESS FINANZIARIO GREZZO
            # (VIX+BAA combo OR HY-OAS), a PRESCINDERE dal regime — analogo del
            # rate/stress veto che B1 (S40) ha validato REGGE sul reale. A differenza
            # di assess_financial_stress_veto (vincolato ai regimi benigni per l'override
            # del classifier), qui serve il segnale puro: scatta anche in deflation
            # 2008/2020 dove il confusione-detector è cieco. hy_oas non nel backfill →
            # default 3.5 → di fatto conta il combo VIX>=24 AND BAA>=2.0.
            if stress_derisk:
                from app.services.regime.financial_stress_veto import _safe_float
                from app.services.common.indicator_aliases import safe_indicator
                ind = cm.get("indicators", {})
                _vix = _safe_float(ind.get("vix"), 18.0)
                _baa = safe_indicator(ind, "baa_10y_spread", default=1.8)
                _hy = _safe_float(ind.get("hy_oas"), 3.5)
                if (_vix >= 24.0 and _baa >= 2.0) or _hy >= 5.0:
                    targets = {a: w * 0.5 for a, w in targets.items()}

            # voltarget (REAL·#2): scala i target alla vol-budget annualizzata (cap 1.0).
            if vol_target is not None and vol_target > 0:
                scale = vol_target_scale(targets, asset_vols, vol_target)
                targets = {a: w * scale for a, w in targets.items()}

            # Decidi per ogni asset con la logica REALE
            for asset, sc in scores.items():
                target = targets.get(asset, 0.0)
                mom = _momentum_proxy(real_matrix, asset, month_end, mom_threshold)
                # Leva momneutral: basta non-BEARISH → NEUTRAL diventa BULLISH per passare il gate
                if "momneutral" in levers and mom == "NEUTRAL":
                    mom = "BULLISH"
                in_favor = asset in regime_top
                # fast_reentry (EXP·B): dopo un crollo BRUSCO recente (2m < −8%) in regime
                # favorevole rientra sul rimbalzo, scavalcando anche il momentum BEARISH
                # (il proxy 3m è lento). SICUREZZA: solo se il regime dominante è
                # COSTRUTTIVO (reflation/goldilocks) → non compra il coltello nei bear
                # lenti (stagflation/deflation). Mira alle V-recovery post vol-shock.
                if fast_reentry and in_favor and mom != "BULLISH":
                    if (max(probs, key=probs.get) in ("reflation", "goldilocks")
                            and _trailing_return(real_matrix, asset, month_end, 2) < -0.08):
                        mom = "BULLISH"
                pos = positions.get(asset)
                mock = None
                if pos is not None:
                    mock = SimpleNamespace(
                        estimated_pnl_pct=pos["pnl_pct"],
                        tranches_filled=pos["tranches_filled"],
                        tranches_total=pos["tranches_total"],
                        took_partial_tp=pos["took_partial_tp"],
                    )
                action, _reason, size = strategist._decide(
                    asset=asset, position=mock, score=sc, target=target,
                    confidence=confidence, momentum_sig=mom, in_favor=in_favor,
                    n_tranches_default=n_tranches, vol=asset_vols.get(asset, 0.20),
                    entry_threshold_shift=0.0,
                )
                month_turnover += _apply_sim(positions, asset, action, size, sc, n_tranches,
                                             str(max(probs, key=probs.get)), trade)

            # Ricalcola cash come 1 - somma pesi (robusto a chiusure)
            cash_weight = max(0.0, 1.0 - sum(p["weight"] for p in positions.values()))

            # Leva floor: deployment minimo forzato sui top del regime (overlay, non in _decide).
            # Con regime_gate, il floor si attiva SOLO nei mesi con boost attivo.
            floor_on = ("floor" in levers) or (regime_gate and boost_active)
            if floor_on and floor_pct > 0 and (1.0 - cash_weight) < floor_pct:
                deployed_before = 1.0 - cash_weight
                _apply_floor(positions, scores, regime_top, floor_pct, cash_weight,
                             str(max(probs, key=probs.get)))
                cash_weight = max(0.0, 1.0 - sum(p["weight"] for p in positions.values()))
                month_turnover += max(0.0, (1.0 - cash_weight) - deployed_before)

        deployed_samples.append(1.0 - cash_weight)

        # Applica rendimenti reali del mese
        if month_end in real_matrix.index:
            row = real_matrix.loc[month_end]
            asset_returns = {a: (float(row[a]) if a in row.index and not pd.isna(row[a]) else None)
                             for a in positions}
            cash_return = float(row[CASH_ASSET]) if CASH_ASSET in row.index and not pd.isna(row[CASH_ASSET]) else 0.0
            pr = portfolio_month_return(positions, cash_weight, asset_returns, cash_return)
            # hedge (PROXY·#4): overlay convesso finanziato dalla liquidità idle
            if hedge_pct > 0:
                eq = row.get("us_equities_growth")
                eq = float(eq) if (eq is not None and not pd.isna(eq)) else 0.0
                pr += hedge_month_contribution(hedge_pct, eq, cash_return)
            # costs (REAL·#6): costo di transazione sul turnover del mese
            if cost_bps > 0 and month_turnover > 0:
                pr -= (cost_bps / 1e4) * month_turnover
            cash_weight = drift_weights(positions, cash_weight, asset_returns, cash_return, pr)
            nav *= (1.0 + pr)
            peak = max(peak, nav)
            dd = nav / peak - 1.0
            trade.max_drawdown = min(trade.max_drawdown, dd)

        trade.nav_path.append(nav)
        current = (month_end + pd.offsets.MonthBegin(1)).date()

    if regime_gate:
        strategist.INCREMENTAL_SCORE_THRESHOLD = _orig_incr  # ripristina la soglia mutata per-mese

    trade.realized_return = nav - 1.0
    trade.avg_deployed = float(np.mean(deployed_samples)) if deployed_samples else 0.0
    trade.min_deployed = float(np.min(deployed_samples)) if deployed_samples else 0.0

    # Benchmarks (buy&hold sul periodo)
    def _bh(weights):
        idx = [t for t in real_matrix.index if start_date <= t.date() <= trade.end_date]
        acc = 1.0
        for t in idx:
            r = _bench_return_month(real_matrix, t, weights)
            if r is not None:
                acc *= (1.0 + r)
        return acc - 1.0

    trade.benchmark_60_40 = _bh(BENCHMARK_60_40)
    trade.benchmark_sp500 = _bh(BENCHMARK_SP500)
    if trade.realized_return is not None:
        trade.alpha_vs_60_40 = trade.realized_return - trade.benchmark_60_40
        trade.alpha_vs_sp500 = trade.realized_return - trade.benchmark_sp500
        a = trade.alpha_vs_60_40
        trade.severity = ("severe" if a < -0.10 else "moderate" if a < -0.05
                          else "minor" if a < 0 else "winning")
    return trade


def _apply_sim(positions, asset, action, size, score, n_tranches, regime, trade) -> float:
    """Muta lo stato simulato (pesi + tranche + pnl) come `strategist._apply`.

    Ritorna il notional scambiato (frazione di NAV) → alimenta il costo di
    transazione della leva `costs`.
    """
    pos = positions.get(asset)
    if action == "OPEN_TRANCHE":
        if pos is None:
            positions[asset] = {
                "weight": size, "pnl_pct": 0.0, "tranches_filled": 1,
                "tranches_total": n_tranches, "avg_entry_score": score,
                "took_partial_tp": False, "entry_regime": regime,
            }
            trade.n_opens += 1
        else:
            pos["pnl_pct"] = dilute_pnl_on_add(pos["pnl_pct"], pos["weight"], size)
            pos["weight"] += size
            pos["tranches_filled"] += 1
            trade.n_opens += 1
        return abs(size)
    elif action == "CLOSE_TRANCHE":
        if pos is None:
            return 0.0
        per = pos["weight"] / max(pos["tranches_filled"], 1)
        pos["weight"] -= per
        pos["tranches_filled"] = max(0, pos["tranches_filled"] - 1)
        pos["took_partial_tp"] = True
        trade.n_closes += 1
        if pos["weight"] <= 0.001:
            positions.pop(asset, None)
        return abs(per)
    elif action in ("FULL_CLOSE", "STOP_LOSS", "TAKE_PROFIT"):
        if pos is not None:
            traded = abs(pos["weight"])
            positions.pop(asset, None)
            trade.n_closes += 1
            return traded
    return 0.0


def _apply_floor(positions, scores, regime_top, floor_pct, cash_weight, regime,
                 cap: float = 0.25, spread_k: int = 5):
    """Overlay leva-floor: schiera il deficit fino a floor_pct sui migliori
    per score del regime (cap per asset). Muta positions in place.
    """
    deficit = floor_pct - (1.0 - cash_weight)
    if deficit <= 0:
        return
    cands = sorted(
        [(a, scores.get(a, 0.0)) for a in regime_top if scores.get(a, 0.0) > 0],
        key=lambda x: -x[1],
    )[:spread_k]
    if not cands:
        return
    per = deficit / len(cands)
    for asset, sc in cands:
        pos = positions.get(asset)
        room = cap - (pos["weight"] if pos else 0.0)
        add = min(per, max(0.0, room))
        if add <= 0:
            continue
        if pos is None:
            positions[asset] = {
                "weight": add, "pnl_pct": 0.0, "tranches_filled": 1,
                "tranches_total": 1, "avg_entry_score": sc,
                "took_partial_tp": False, "entry_regime": regime,
            }
        else:
            pos["pnl_pct"] = dilute_pnl_on_add(pos["pnl_pct"], pos["weight"], add)
            pos["weight"] += add


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-sims", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--horizon-months", type=int, default=12)
    p.add_argument("--start-min-year", type=int, default=1995)
    p.add_argument("--start-max-year", type=int, default=2023)
    p.add_argument("--crisis-window", choices=["2008", "2020", "2022", "all_crisis"], default=None)
    p.add_argument("--mom-threshold", type=float, default=0.0)
    p.add_argument("--fix-deadband", action="store_true",
                   help="Alias di --lever deadband (retrocompat).")
    p.add_argument("--lever", default="",
                   help="Leve A/B separate da virgola: deadband,fastramp,momneutral,"
                        "entry50,top18,floor. Override a runtime, non toccano i file live.")
    p.add_argument("--floor-pct", type=float, default=0.30,
                   help="Deployment minimo per la leva 'floor' (default 0.30).")
    args = p.parse_args()

    levers = {x.strip() for x in args.lever.split(",") if x.strip()}
    if args.fix_deadband:
        levers.add("deadband")

    # Allinea i flag al modello live (momentum + uncertainty gate default-on)
    os.environ.setdefault("USE_MOMENTUM_PILLARS", "1")
    # Igiene A/B: il baseline dev'essere PULITO. deploy_boost è ora default-ON nel
    # live, ma qui è una LEVA esplicita (--lever deadband,fastramp,floor) → forza OFF
    # il flag così il baseline non lo eredita e il Δ misura l'effetto marginale.
    os.environ.setdefault("USE_AI_PORTFOLIO_DEPLOY_BOOST", "0")

    from app.services.ai_portfolio import strategist as _strat, sizer as _sizer
    top_n_regime = _strat.TOP_N_REGIME
    applied = []
    if "deadband" in levers:
        _strat.INCREMENTAL_SCORE_THRESHOLD = _strat.ENTRY_SCORE_THRESHOLD
        applied.append(f"deadband(INCR->{_strat.ENTRY_SCORE_THRESHOLD})")
    if "entry50" in levers:
        _strat.ENTRY_SCORE_THRESHOLD = 50.0
        applied.append("entry50")
    if "top18" in levers:
        top_n_regime = 18
        applied.append("top18")
    if "fastramp" in levers:
        _orig_ntr = _sizer.dynamic_n_tranches
        _sizer.dynamic_n_tranches = lambda conf, _o=_orig_ntr: (min(2, _o(conf)) or 1) if _o(conf) > 0 else 0
        applied.append("fastramp(<=2)")
    if "momneutral" in levers:
        applied.append("momneutral")
    if "floor" in levers:
        applied.append(f"floor({args.floor_pct:.0%})")
    # Leve booleane passate a run_horserace_trade (default off). derisk=proxy grezzo,
    # transition=detector vero (assess_transition). A/B: baseline vs derisk vs transition.
    _derisk = "derisk" in levers
    _transition = "transition" in levers
    _regime_gate = "regime_gate" in levers
    _fast_reentry = "fast_reentry" in levers
    if _derisk:
        applied.append("derisk(proxy)")
    if _transition:
        applied.append("transition(detector)")
    if _regime_gate:
        applied.append("regime_gate")
    if _fast_reentry:
        applied.append("fast_reentry")

    print("=" * 70)
    print(f"HORSERACE STRATEGIST BACKTEST — {args.n_sims} sims, {args.horizon_months}m")
    print(f"Window: {args.start_min_year}-{args.start_max_year}, seed {args.seed}, "
          f"mom_thr={args.mom_threshold}")
    print(f"Levers: {', '.join(applied) if applied else 'OFF (baseline)'}")
    print("=" * 70)

    from app.services.scoring.engine import ASSET_REGIME_DATA
    from app.services.backtest.real_returns_matrix import build_real_returns_matrix
    from app.services.backtest.walk_forward import WalkForwardCache, preload_walk_forward_series

    print("\nLoading real returns matrix...")
    real_matrix = build_real_returns_matrix(list(ASSET_REGIME_DATA.keys()))
    print(f"  Matrix shape: {real_matrix.shape}")

    print("Pre-loading FRED series...")
    series = preload_walk_forward_series(
        start_date=date(args.start_min_year - 2, 1, 1),
        end_date=date(args.start_max_year + 2, 12, 31),
    )
    wf_cache = WalkForwardCache(series)

    CRISIS_WINDOWS = {
        "2008": [(2007, 10), (2008, 12)],
        "2020": [(2019, 12), (2020, 8)],
        "2022": [(2021, 12), (2022, 12)],
    }
    pool = None
    if args.crisis_window:
        pool = []
        wins = ([CRISIS_WINDOWS[args.crisis_window]] if args.crisis_window != "all_crisis"
                else list(CRISIS_WINDOWS.values()))
        for (y0, m0), (y1, m1) in wins:
            y, m = y0, m0
            while (y, m) <= (y1, m1):
                pool.append(date(y, m, 1))
                y, m = (y + 1, 1) if m == 12 else (y, m + 1)

    rng = random.Random(args.seed)
    trades = []
    for i in range(args.n_sims):
        if pool:
            d = rng.choice(pool)
        else:
            d = date(rng.randint(args.start_min_year, args.start_max_year), rng.randint(1, 12), 1)
        t = run_horserace_trade(d, real_matrix, wf_cache, args.horizon_months, args.mom_threshold,
                                levers=levers, top_n_regime=top_n_regime, floor_pct=args.floor_pct,
                                derisk=_derisk, transition_derisk=_transition,
                                regime_gate=_regime_gate, fast_reentry=_fast_reentry)
        if t is None or t.realized_return is None:
            continue
        trades.append(t)
        mark = {"winning": "[+]", "minor": "[-]", "moderate": "[?]", "severe": "[X]"}.get(t.severity, "[ ]")
        print(f"  {mark} Sim {i+1}: {t.start_date} ret={t.realized_return*100:+6.1f}%  "
              f"a60/40={t.alpha_vs_60_40*100:+6.1f}pp  aSPX={t.alpha_vs_sp500*100:+6.1f}pp  "
              f"deployed avg={t.avg_deployed*100:.0f}% min={t.min_deployed*100:.0f}%  "
              f"opens={t.n_opens} closes={t.n_closes}  DD={t.max_drawdown*100:.1f}%")

    if not trades:
        print("\nNessun trade valido.")
        return

    def _m(attr):
        vals = [getattr(t, attr) for t in trades if getattr(t, attr) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    n = len(trades)
    win60 = sum(1 for t in trades if t.alpha_vs_60_40 and t.alpha_vs_60_40 > 0)
    winspx = sum(1 for t in trades if t.alpha_vs_sp500 and t.alpha_vs_sp500 > 0)
    print("\n" + "=" * 70)
    print("AGGREGATE — HORSERACE STRATEGIST")
    print("=" * 70)
    print(f"  Trades: {n}")
    print(f"  Mean return {args.horizon_months}m: {_m('realized_return')*100:+.2f}%")
    print(f"  Mean deployed: {_m('avg_deployed')*100:.0f}%  (min medio {_m('min_deployed')*100:.0f}%)")
    print(f"  Win vs 60/40: {win60}/{n} ({win60/n*100:.0f}%)  · mean alpha {_m('alpha_vs_60_40')*100:+.2f}pp")
    print(f"  Win vs S&P500: {winspx}/{n} ({winspx/n*100:.0f}%)  · mean alpha {_m('alpha_vs_sp500')*100:+.2f}pp")
    print(f"  Mean max drawdown: {_m('max_drawdown')*100:.2f}%")
    print(f"  Mean opens/closes: {_m('n_opens'):.1f}/{_m('n_closes'):.1f}")
    sev = {}
    for t in trades:
        sev[t.severity] = sev.get(t.severity, 0) + 1
    print("  Severity: " + " · ".join(f"{k} {v}/{n}" for k, v in sorted(sev.items())))


if __name__ == "__main__":
    main()
