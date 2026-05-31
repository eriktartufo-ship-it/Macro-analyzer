"""T15 Tail hedge allocator — Council action #3 (sessione 18).

Razionale: il portfolio attuale è "fragile" in tail events (-15%+ SPY drawdowns).
Aggiunge piccolo budget allocation (1-2%) a synthetic tail hedge che paga in
crash scenarios. Costa 1-2%/anno in calm, ma cattura 20-50pp in tail year.

Trigger: Crisis Risk Indicator score > 0.60 (council threshold conservativo).
Hedge size: scales con risk_score, cap 2% del portfolio.

Synthetic hedge asset = OTM SPY put 6m -15% strike (approssimato):
- SPY monthly return < -10%: hedge +50% (deep ITM)
- SPY monthly return -10% a -5%: hedge +20% (ATM)
- SPY monthly return -5% a 0%: hedge -5% (theta decay near strike)
- SPY monthly return > 0%: hedge -8% (theta + delta loss)

Council T15 2026-05-31. Ackman + Taleb consensus.

Usage:
    from app.services.portfolio.tail_hedge import (
        should_activate_hedge, compute_hedge_weight, simulate_hedge_return,
    )

    if should_activate_hedge(crisis_risk_score):
        hedge_w = compute_hedge_weight(crisis_risk_score)
        # Subtract hedge_w from top-N alloc, add to hedge_asset
"""
from __future__ import annotations

# Council T15 thresholds (default conservativi)
HEDGE_TRIGGER_THRESHOLD: float = 0.60
HEDGE_MAX_BUDGET: float = 0.02       # max 2% portfolio
HEDGE_MIN_BUDGET: float = 0.005      # min 0.5% se attivato

# Synthetic OTM SPY put 6m -15% strike — payoff approssimato per scenario
HEDGE_PAYOFFS = {
    "deep_crash":  {"threshold": -0.10, "return": 0.50},   # SPY < -10% month
    "moderate_crash": {"threshold": -0.05, "return": 0.20}, # SPY -10% to -5%
    "minor_dip": {"threshold": 0.0, "return": -0.05},       # SPY -5% to 0%
    "calm_or_up": {"threshold": float("inf"), "return": -0.08},  # SPY >= 0%
}


def should_activate_hedge(
    crisis_risk_score: float,
    threshold: float = HEDGE_TRIGGER_THRESHOLD,
) -> bool:
    """True se Crisis Risk Indicator score >= threshold (default 0.60).

    Args:
        crisis_risk_score: output di crisis_indicator (0-1).
        threshold: soglia trigger (default 0.60 = conservative).

    Returns:
        bool. True se hedge va attivato.
    """
    return crisis_risk_score >= threshold


def compute_hedge_weight(
    crisis_risk_score: float,
    max_budget: float = HEDGE_MAX_BUDGET,
    min_budget: float = HEDGE_MIN_BUDGET,
    threshold: float = HEDGE_TRIGGER_THRESHOLD,
) -> float:
    """Computa weight tail hedge proporzionale al crisis risk score.

    Scaling lineare:
        score = threshold → min_budget (0.5%)
        score = 1.0 → max_budget (2%)
        score < threshold → 0 (no hedge)

    Args:
        crisis_risk_score: 0-1.
        max_budget: cap budget hedge (default 2% portfolio).
        min_budget: floor budget se attivato (default 0.5%).
        threshold: soglia attivazione (default 0.60).

    Returns:
        float weight 0..max_budget.
    """
    if crisis_risk_score < threshold:
        return 0.0
    # Lineare da min al threshold → max a score=1.0
    range_score = 1.0 - threshold
    if range_score <= 0:
        return max_budget
    range_budget = max_budget - min_budget
    excess_score = crisis_risk_score - threshold
    return min_budget + (excess_score / range_score) * range_budget


def simulate_hedge_return(spy_monthly_return: float) -> float:
    """Synthetic OTM SPY put 6m -15% payoff approximation.

    Buckets:
    - SPY < -10%: hedge +50% (deep crash, put deep ITM)
    - SPY -10% to -5%: hedge +20% (moderate crash, put ATM)
    - SPY -5% to 0%: hedge -5% (minor dip, theta near strike)
    - SPY >= 0%: hedge -8% (calm, theta + delta loss)

    Args:
        spy_monthly_return: SPY return nel mese (decimal: -0.10 = -10%).

    Returns:
        Hedge return nel mese (decimal).
    """
    if spy_monthly_return < -0.10:
        return HEDGE_PAYOFFS["deep_crash"]["return"]
    if spy_monthly_return < -0.05:
        return HEDGE_PAYOFFS["moderate_crash"]["return"]
    if spy_monthly_return < 0.0:
        return HEDGE_PAYOFFS["minor_dip"]["return"]
    return HEDGE_PAYOFFS["calm_or_up"]["return"]


def apply_hedge_to_allocation(
    base_weights: dict[str, float],
    hedge_weight: float,
    hedge_asset_key: str = "tail_hedge_spy_put",
) -> dict[str, float]:
    """Modifica una allocation aggiungendo tail hedge come asset class separato.

    Strategia: scale-down proporzionale tutti gli asset esistenti per `1 - hedge_weight`,
    poi aggiunge hedge_weight al hedge_asset_key. Mantiene sum=1.

    Args:
        base_weights: dict {asset: weight}, sum=1 atteso.
        hedge_weight: weight da allocare al hedge (0..max_budget).
        hedge_asset_key: chiave per identificare l'hedge asset (default
                         "tail_hedge_spy_put").

    Returns:
        new dict con hedge inserito, sum=1.
    """
    if hedge_weight <= 0:
        return dict(base_weights)
    if hedge_weight >= 1.0:
        # Edge: tutto hedge (non realistico, ma robustness)
        return {hedge_asset_key: 1.0}

    scale = 1.0 - hedge_weight
    new_weights = {a: w * scale for a, w in base_weights.items()}
    new_weights[hedge_asset_key] = hedge_weight
    return new_weights
