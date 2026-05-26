"""Tier 6.4 — Position sizing layer.

Converte asset scores (0-100) in pesi % portfolio allocation via 4
algoritmi modulari. Filosofia: separazione clean tra scoring (regime
classifier produce score) e portfolio construction (questo modulo).

**Algoritmi disponibili**:

1. **equal_weight** — top-N equipesati. Baseline trivial. weight = 1/N
   per top-N, 0 per altri. Pattern usato come benchmark per misurare
   miglioramento degli altri metodi.

2. **score_weighted** — weight ∝ score / sum(scores_top_N). Asset con
   score più alto ricevono peso maggiore proporzionalmente.

3. **risk_parity** — weight ∝ 1/vol_asset normalizzato. Pattern
   Bridgewater All-Weather: tutti gli asset contribuiscono ugualmente
   al rischio di portafoglio. Asset volatili (equities, crypto) pesati
   meno di asset stabili (bonds, cash).

4. **kelly_fractional** — weight ∝ (score-50)² / vol² (semplificato,
   no full Markowitz). Fractional 0.25× per ridurre rischio bancarotta
   (Kelly puro tende a aggressive sizing).

**Output uniforme**: `PortfolioAllocation` dataclass con `weights` dict
{asset: weight}, sum=1.0, method, top_n, metadata.

**Note**: NON modifica asset scoring. Layer ortogonale che aggrega
scoring + vol stats per produrre allocation deployment-ready.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PortfolioAllocation:
    """Output uniforme per qualsiasi algoritmo position sizing."""
    method: str
    top_n: int
    weights: dict[str, float] = field(default_factory=dict)
    total_weight: float = 0.0
    excluded_assets: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    description: str = ""


def _sort_top_n(scores: dict[str, float], top_n: int) -> list[tuple[str, float]]:
    """Restituisce top-N asset ordinati per score discendente.

    Args:
        scores: dict {asset: score 0-100}.
        top_n: numero asset da selezionare.

    Returns:
        Lista di tuple (asset, score) ordinata score desc, lunghezza ≤ top_n.
    """
    if not scores or top_n <= 0:
        return []
    sorted_items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return sorted_items[:top_n]


def equal_weight(scores: dict[str, float], top_n: int = 5) -> PortfolioAllocation:
    """Equal-weight top-N. Baseline trivial: weight = 1/N per top, 0 altri."""
    top = _sort_top_n(scores, top_n)
    excluded = [a for a in scores if a not in {t[0] for t in top}]
    if not top:
        return PortfolioAllocation(
            method="equal_weight",
            top_n=top_n,
            weights={},
            total_weight=0.0,
            excluded_assets=excluded,
            description="Nessun asset disponibile",
        )
    w = 1.0 / len(top)
    weights = {asset: w for asset, _ in top}
    return PortfolioAllocation(
        method="equal_weight",
        top_n=top_n,
        weights=weights,
        total_weight=sum(weights.values()),
        excluded_assets=excluded,
        description=f"Equal-weight top-{len(top)} (1/{len(top)} cad)",
    )


def score_weighted(scores: dict[str, float], top_n: int = 5) -> PortfolioAllocation:
    """Score-weighted top-N. weight = score / sum(scores_top_N)."""
    top = _sort_top_n(scores, top_n)
    excluded = [a for a in scores if a not in {t[0] for t in top}]
    if not top:
        return PortfolioAllocation(
            method="score_weighted",
            top_n=top_n,
            weights={},
            total_weight=0.0,
            excluded_assets=excluded,
            description="Nessun asset disponibile",
        )
    total_score = sum(s for _, s in top)
    if total_score <= 0:
        # Edge: tutti score 0 → fallback equal_weight
        return equal_weight(scores, top_n)
    weights = {asset: score / total_score for asset, score in top}
    return PortfolioAllocation(
        method="score_weighted",
        top_n=top_n,
        weights=weights,
        total_weight=sum(weights.values()),
        excluded_assets=excluded,
        description=f"Score-weighted top-{len(top)}, total_score={total_score:.1f}",
    )


def risk_parity(
    scores: dict[str, float],
    vols: dict[str, float],
    top_n: int = 5,
    min_vol: float = 0.01,
) -> PortfolioAllocation:
    """Risk-parity vol-based: weight ∝ 1/vol normalizzato.

    Pattern Bridgewater All-Weather. Asset volatili pesati meno.
    Selezione: top-N per score, poi pesatura inversamente proporzionale a vol.

    Args:
        scores: dict {asset: score 0-100}.
        vols: dict {asset: volatility annualizzata}. Required per top-N.
        top_n: numero asset.
        min_vol: floor per evitare division by zero.
    """
    top = _sort_top_n(scores, top_n)
    excluded = [a for a in scores if a not in {t[0] for t in top}]
    if not top:
        return PortfolioAllocation(
            method="risk_parity", top_n=top_n,
            weights={}, total_weight=0.0,
            excluded_assets=excluded,
            description="Nessun asset disponibile",
        )

    inv_vols = {}
    for asset, _ in top:
        v = max(vols.get(asset, min_vol), min_vol)
        inv_vols[asset] = 1.0 / v
    total_inv = sum(inv_vols.values())
    if total_inv <= 0:
        return equal_weight(scores, top_n)
    weights = {asset: inv / total_inv for asset, inv in inv_vols.items()}
    return PortfolioAllocation(
        method="risk_parity",
        top_n=top_n,
        weights=weights,
        total_weight=sum(weights.values()),
        excluded_assets=excluded,
        metadata={"vols": {a: vols.get(a) for a, _ in top}},
        description=f"Risk-parity top-{len(top)} (inverse vol weighting)",
    )


def kelly_fractional(
    scores: dict[str, float],
    vols: dict[str, float],
    top_n: int = 5,
    fraction: float = 0.25,
    score_baseline: float = 50.0,
    min_vol: float = 0.01,
) -> PortfolioAllocation:
    """Kelly fractional semplificato: weight ∝ (score - baseline)² / vol²
    × fraction.

    Approssimazione di Markowitz: edge proxy = (score - 50)/100, variance
    da vol². Solo asset con score > baseline (positive edge) ricevono peso.

    Fractional 0.25× per ridurre rischio bancarotta (Kelly puro = aggressive).
    """
    top = _sort_top_n(scores, top_n)
    excluded = [a for a in scores if a not in {t[0] for t in top}]
    if not top:
        return PortfolioAllocation(
            method="kelly_fractional", top_n=top_n,
            weights={}, total_weight=0.0,
            excluded_assets=excluded,
            description="Nessun asset disponibile",
        )

    raw_weights = {}
    for asset, score in top:
        edge = max(0.0, (score - score_baseline) / 100.0)  # 0..0.5 range
        v = max(vols.get(asset, min_vol), min_vol)
        # Kelly: edge / variance; clamped
        raw = (edge ** 2) / (v ** 2)
        raw_weights[asset] = raw * fraction

    total_raw = sum(raw_weights.values())
    if total_raw <= 0:
        # Tutti score sotto baseline → no positive edge → fallback equal
        return equal_weight(scores, top_n)
    weights = {a: w / total_raw for a, w in raw_weights.items()}
    return PortfolioAllocation(
        method="kelly_fractional",
        top_n=top_n,
        weights=weights,
        total_weight=sum(weights.values()),
        excluded_assets=excluded,
        metadata={"fraction": fraction, "score_baseline": score_baseline},
        description=f"Kelly fractional top-{len(top)} (fraction={fraction})",
    )


def compute_allocation(
    method: str,
    scores: dict[str, float],
    vols: dict[str, float] | None = None,
    top_n: int = 5,
    **kwargs,
) -> PortfolioAllocation:
    """Dispatch a algoritmo via `method` string."""
    method = method.lower()
    if method == "equal_weight":
        return equal_weight(scores, top_n=top_n)
    if method == "score_weighted":
        return score_weighted(scores, top_n=top_n)
    if method == "risk_parity":
        if vols is None:
            raise ValueError("risk_parity richiede `vols` dict")
        return risk_parity(scores, vols, top_n=top_n)
    if method == "kelly_fractional":
        if vols is None:
            raise ValueError("kelly_fractional richiede `vols` dict")
        return kelly_fractional(
            scores, vols, top_n=top_n,
            fraction=kwargs.get("fraction", 0.25),
            score_baseline=kwargs.get("score_baseline", 50.0),
        )
    raise ValueError(f"Method '{method}' non supportato. "
                     f"Disponibili: equal_weight, score_weighted, risk_parity, kelly_fractional")


def list_methods() -> list[str]:
    """Lista algoritmi disponibili."""
    return ["equal_weight", "score_weighted", "risk_parity", "kelly_fractional"]


# ============================================================================
# Tier 6.10 — Vol targeting + leverage
# ============================================================================


@dataclass
class VolTargetedAllocation:
    """Output post vol-targeting: weights scalati + leverage/cash buffer."""
    base_allocation: PortfolioAllocation
    target_vol: float
    realized_vol_pre: float           # vol portfolio pre-targeting
    realized_vol_post: float          # vol portfolio post-targeting (~target)
    scale_factor: float               # multiplier applicato
    weights: dict[str, float]         # weights post-scaling
    leverage_used: float              # 1.0 = no margin, >1 = margin, <1 = cash buffer
    cash_buffer: float                # 1 - sum(weights) se scale<1, else 0
    description: str = ""


def compute_portfolio_vol(
    weights: dict[str, float],
    vols: dict[str, float],
    avg_correlation: float = 0.3,
) -> float:
    """Stima portfolio vol annualizzato.

    Math:
        σ_p² = Σ_i w_i² σ_i² + Σ_{i≠j} w_i w_j σ_i σ_j ρ_ij

    Semplificazione: usa `avg_correlation` uniforme per tutte le coppie
    (vs full correlation matrix). Bridgewater empirico avg ~0.3 cross-asset.

    Args:
        weights: dict {asset: weight}, sum tipicamente 1.0.
        vols: dict {asset: annualized vol}.
        avg_correlation: correlazione media cross-asset (default 0.3).

    Returns:
        portfolio vol annualizzato (float, sempre >= 0).
    """
    if not weights or not vols:
        return 0.0
    # Variance solo per asset presenti in entrambi
    common = [a for a in weights if a in vols]
    if not common:
        return 0.0
    # Σ w_i² σ_i²
    var_diag = sum((weights[a] ** 2) * (vols[a] ** 2) for a in common)
    # Σ_{i<j} 2 · w_i w_j σ_i σ_j ρ (matrix symmetric)
    var_cross = 0.0
    n = len(common)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = common[i], common[j]
            var_cross += 2.0 * weights[a] * weights[b] * vols[a] * vols[b] * avg_correlation
    total_var = max(0.0, var_diag + var_cross)
    return float(total_var ** 0.5)


def apply_vol_targeting(
    allocation: PortfolioAllocation,
    vols: dict[str, float],
    target_vol: float = 0.10,
    avg_correlation: float = 0.3,
    max_leverage: float = 2.0,
) -> VolTargetedAllocation:
    """Scala weights per matchare `target_vol` annualizzato.

    Args:
        allocation: output di un algoritmo position_sizing.
        vols: dict {asset: annualized vol} (richiesto).
        target_vol: target portfolio vol (default 10% = 0.10).
        avg_correlation: correlation media assumed cross-asset.
        max_leverage: cap su scale factor se vol portfolio < target.

    Returns:
        VolTargetedAllocation con weights scalati, leverage_used, cash_buffer.
    """
    if not allocation.weights:
        return VolTargetedAllocation(
            base_allocation=allocation,
            target_vol=target_vol,
            realized_vol_pre=0.0,
            realized_vol_post=0.0,
            scale_factor=0.0,
            weights={},
            leverage_used=0.0,
            cash_buffer=0.0,
            description="Empty allocation",
        )

    realized_vol_pre = compute_portfolio_vol(
        allocation.weights, vols, avg_correlation=avg_correlation
    )

    if realized_vol_pre <= 0:
        # No vol info → no scaling (return identity)
        return VolTargetedAllocation(
            base_allocation=allocation,
            target_vol=target_vol,
            realized_vol_pre=0.0,
            realized_vol_post=0.0,
            scale_factor=1.0,
            weights=dict(allocation.weights),
            leverage_used=1.0,
            cash_buffer=0.0,
            description="No vol data, identity scaling",
        )

    # Scale factor
    raw_scale = target_vol / realized_vol_pre
    # Cap a max_leverage per safety
    scale = min(raw_scale, max_leverage)
    scale = max(scale, 0.0)

    weights_scaled = {a: w * scale for a, w in allocation.weights.items()}
    total_weight_scaled = sum(weights_scaled.values())

    if scale >= 1.0:
        leverage_used = scale
        cash_buffer = 0.0
        desc = f"Vol target {target_vol*100:.1f}% via leverage {scale:.2f}× (realized vol {realized_vol_pre*100:.1f}%)"
    else:
        leverage_used = 1.0
        cash_buffer = 1.0 - total_weight_scaled
        desc = f"Vol target {target_vol*100:.1f}% via cash buffer {cash_buffer*100:.1f}% (realized vol {realized_vol_pre*100:.1f}%)"

    # Post-targeting vol (should match target_vol ± numerical drift)
    realized_vol_post = compute_portfolio_vol(weights_scaled, vols, avg_correlation=avg_correlation)

    return VolTargetedAllocation(
        base_allocation=allocation,
        target_vol=target_vol,
        realized_vol_pre=realized_vol_pre,
        realized_vol_post=realized_vol_post,
        scale_factor=scale,
        weights=weights_scaled,
        leverage_used=leverage_used,
        cash_buffer=cash_buffer,
        description=desc,
    )
