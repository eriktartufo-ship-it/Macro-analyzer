"""T19 Trigger-based rebalance — Council action #1 (sessione 26).

Council brutal verdict: WEEKLY trade = 520bps/yr drag morte alpha.
WEEKLY monitor + MONTHLY trade UNLESS trigger fires = ottimo trade-off.

Triggers (council sessione 26):
1. regime_prob_shift > 0.20 (cambio significativo classifier)
2. VIX jump > 50% week-over-week (panic shift)
3. NFCI > +1σ (credit conditions deteriorate)

Output: rebalance_now boolean + reason string per audit.

Pattern: usato in backtest paper_trading_simulation.py.
Wired anche in scheduler.py per live trading futuro.

Council expected: +0.5 a +1pp alpha vs pure calendar monthly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# Council thresholds (sessione 26)
REGIME_PROB_SHIFT_THRESHOLD: float = 0.20
VIX_JUMP_PCT_THRESHOLD: float = 0.50      # +50% week-over-week
NFCI_Z_THRESHOLD: float = 1.0             # +1σ deviation
HY_OAS_JUMP_PCT_THRESHOLD: float = 0.30   # +30% week-over-week (extreme stress)


@dataclass
class TriggerCheckResult:
    """Risultato check trigger + reason audit."""
    rebalance_now: bool
    reason: str
    regime_prob_delta: float = 0.0
    vix_jump_pct: float = 0.0
    nfci_z: float = 0.0
    hy_oas_jump_pct: float = 0.0


def check_rebalance_triggers(
    current_probs: dict[str, float] | None,
    previous_probs: dict[str, float] | None,
    current_vix: float | None,
    previous_vix: float | None,
    current_nfci: float | None,
    nfci_mean: float = 0.0,
    nfci_std: float = 0.5,
    current_hy_oas: float | None = None,
    previous_hy_oas: float | None = None,
) -> TriggerCheckResult:
    """Valuta trigger rebalance off-calendar.

    Args:
        current_probs / previous_probs: dict regime probabilities {refl/stag/defl/gold}
        current_vix / previous_vix: VIX level
        current_nfci: NFCI level
        nfci_mean / nfci_std: long-run statistics per z-score
        current_hy_oas / previous_hy_oas: High-Yield OAS spread

    Returns:
        TriggerCheckResult con rebalance_now + reason audit string.
    """
    triggers_fired = []

    # Trigger 1: regime probability shift
    regime_delta = 0.0
    if current_probs and previous_probs:
        max_shift = 0.0
        for regime in ("reflation", "stagflation", "deflation", "goldilocks"):
            shift = abs(current_probs.get(regime, 0) - previous_probs.get(regime, 0))
            max_shift = max(max_shift, shift)
        regime_delta = max_shift
        if max_shift >= REGIME_PROB_SHIFT_THRESHOLD:
            triggers_fired.append(f"regime_shift({max_shift:.2f}>={REGIME_PROB_SHIFT_THRESHOLD})")

    # Trigger 2: VIX jump week-over-week
    vix_jump = 0.0
    if current_vix and previous_vix and previous_vix > 0:
        vix_jump = (current_vix - previous_vix) / previous_vix
        if vix_jump >= VIX_JUMP_PCT_THRESHOLD:
            triggers_fired.append(f"vix_jump({vix_jump*100:.0f}%>={VIX_JUMP_PCT_THRESHOLD*100:.0f}%)")

    # Trigger 3: NFCI z-score above threshold
    nfci_z = 0.0
    if current_nfci is not None:
        nfci_z = (current_nfci - nfci_mean) / nfci_std if nfci_std > 0 else 0
        if nfci_z >= NFCI_Z_THRESHOLD:
            triggers_fired.append(f"nfci_z({nfci_z:.2f}>={NFCI_Z_THRESHOLD})")

    # Trigger 4: HY OAS jump (extreme credit stress)
    hy_jump = 0.0
    if current_hy_oas and previous_hy_oas and previous_hy_oas > 0:
        hy_jump = (current_hy_oas - previous_hy_oas) / previous_hy_oas
        if hy_jump >= HY_OAS_JUMP_PCT_THRESHOLD:
            triggers_fired.append(f"hy_oas_jump({hy_jump*100:.0f}%>={HY_OAS_JUMP_PCT_THRESHOLD*100:.0f}%)")

    fired = bool(triggers_fired)
    reason = " | ".join(triggers_fired) if fired else "no triggers"

    return TriggerCheckResult(
        rebalance_now=fired,
        reason=reason,
        regime_prob_delta=regime_delta,
        vix_jump_pct=vix_jump,
        nfci_z=nfci_z,
        hy_oas_jump_pct=hy_jump,
    )


def any_trigger_fired_simple(
    current_probs: dict[str, float] | None,
    previous_probs: dict[str, float] | None,
    indicators_curr: dict[str, float] | None,
    indicators_prev: dict[str, float] | None,
) -> bool:
    """Helper semplificato per paper trading: legge da indicators dict.

    Returns: True se qualunque trigger scatta.
    """
    if not indicators_curr:
        return False
    result = check_rebalance_triggers(
        current_probs=current_probs,
        previous_probs=previous_probs,
        current_vix=indicators_curr.get("vix"),
        previous_vix=(indicators_prev or {}).get("vix"),
        current_nfci=indicators_curr.get("nfci"),
        current_hy_oas=indicators_curr.get("hy_credit_spread"),
        previous_hy_oas=(indicators_prev or {}).get("hy_credit_spread"),
    )
    return result.rebalance_now
