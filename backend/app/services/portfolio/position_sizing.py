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


def _apply_caps(
    weights: dict[str, float],
    cap_max: float,
    cap_min: float,
) -> dict[str, float]:
    """Applica cap [cap_min, cap_max] via waterfill (no re-normalize naive).

    Algoritmo waterfill (converge guaranteed):
    1. Edge: se cap_min*N > 1 → impossibile, fallback equal-weight.
    2. Edge: se cap_max*N < 1 → impossibile, fallback cap_max ovunque + drop.
    3. Itera (max 20):
       a. Identifica asset "high" (>cap_max), "low" (<cap_min), "free" (in range).
       b. Set high = cap_max, low = cap_min. "Excess" = (sum_high_pre - cap_max*n_high)
          + (cap_min*n_low - sum_low_pre).
       c. Distribuisci excess agli asset "free" proporzionalmente al loro weight.
       d. Se nessun nuovo violation → done.

    Args:
        weights: dict {asset: weight}, sum=1 atteso.
        cap_max: max weight per singolo asset (es. 0.25).
        cap_min: min weight per singolo asset selected (es. 0.03).

    Returns:
        dict weights con cap applicati, sum=1.
    """
    if not weights:
        return {}
    n = len(weights)
    # Edge case 1: floor*N > 1 → impossibile, equal-weight forzato
    if cap_min * n > 1.0:
        return {a: 1.0 / n for a in weights}
    # Edge case 2: cap_max*N < 1 → impossibile coprire 100%, clip top a cap_max
    # e accetta sum < 1 (= cash buffer implicit). Distribuisci proporzionalmente.
    if cap_max * n < 1.0:
        total = sum(weights.values()) or 1.0
        scaled = {a: (w / total) * (cap_max * n) for a, w in weights.items()}
        return {a: min(cap_max, w) for a, w in scaled.items()}

    out = {a: max(0.0, w) for a, w in weights.items()}
    total = sum(out.values()) or 1.0
    out = {a: w / total for a, w in out.items()}

    # Iterative waterfill: pin violators a caps, redistribuisci budget rimanente
    # ai free assets proporzionalmente. Converge in O(N) iter perché ogni iter
    # pinna almeno 1 asset, e n_free decresce monotonamente.
    for _ in range(n + 2):
        too_high = [a for a, w in out.items() if w > cap_max + 1e-9]
        too_low = [a for a, w in out.items() if w < cap_min - 1e-9]
        if not too_high and not too_low:
            return out

        pinned = {a: cap_max for a in too_high}
        pinned.update({a: cap_min for a in too_low})

        free_assets = [a for a in out if a not in pinned]
        free_budget = 1.0 - sum(pinned.values())

        if not free_assets:
            return pinned  # tutto pinned, accetta sum eventualmente != 1

        if free_budget <= 0:
            # Budget esaurito: forza free a 0 (clip al floor cap_min)
            for a in free_assets:
                pinned[a] = cap_min
            return pinned

        free_sum_current = sum(out[a] for a in free_assets) or 1e-9
        new_out = dict(pinned)
        for a in free_assets:
            new_out[a] = (out[a] / free_sum_current) * free_budget
        out = new_out

    return out


def confidence_kelly(
    scores: dict[str, float],
    vols: dict[str, float],
    confidence: float,
    top_n: int = 7,
    cap_max: float = 0.25,
    cap_min: float = 0.03,
    conf_low: float = 0.60,
    conf_high: float = 0.80,
    fraction: float = 0.50,
    score_baseline: float = 50.0,
    min_vol: float = 0.01,
) -> PortfolioAllocation:
    """T12 Council action #2 — Kelly fractional risk-weighted con confidence gating.

    Logica regime-aware sizing:
    - confidence < conf_low (0.60): full equal-weight (safe fallback)
    - conf_low ≤ confidence ≤ conf_high: smooth blend λ=(conf-low)/(high-low)
      → weights = (1-λ)·equal + λ·kelly
    - confidence > conf_high (0.80): full Kelly con caps

    Caps applicati post-blend:
    - cap_max (default 25%): no asset > 1/4 portfolio (single-name risk)
    - cap_min (default 3%): asset selezionato deve avere meaningful weight

    Council 2026-05-28 rationale: equal-weight top-N è troppo neutro quando
    il modello ha alta convinzione. Kelly fractional concentra capital sui
    top-edge assets ma con risk-budget cap che evita bancarotta.

    Args:
        scores: dict {asset: score 0-100}.
        vols: dict {asset: annualized vol}.
        confidence: regime confidence [0, 1].
        top_n: numero asset.
        cap_max: max weight singolo (default 25%).
        cap_min: floor weight (default 3%).
        conf_low: soglia sotto cui usa solo equal.
        conf_high: soglia sopra cui usa solo Kelly.
        fraction: Kelly fractional multiplier (default 0.50 = half-Kelly,
                  più aggressivo del 0.25 default di kelly_fractional).
        score_baseline: edge zero (default 50).

    Returns:
        PortfolioAllocation con blend equal + Kelly + caps applicati.
    """
    top = _sort_top_n(scores, top_n)
    excluded = [a for a in scores if a not in {t[0] for t in top}]
    if not top:
        return PortfolioAllocation(
            method="confidence_kelly", top_n=top_n,
            weights={}, total_weight=0.0,
            excluded_assets=excluded,
            description="Nessun asset disponibile",
        )

    # Componente equal-weight (baseline)
    equal_w = equal_weight(scores, top_n=top_n).weights
    # Componente Kelly (full risk-weighted)
    kelly_w = kelly_fractional(
        scores, vols, top_n=top_n,
        fraction=fraction, score_baseline=score_baseline, min_vol=min_vol,
    ).weights

    # Blend lambda: 0 sotto conf_low, 1 sopra conf_high, lineare in mezzo
    if confidence <= conf_low:
        lam = 0.0
    elif confidence >= conf_high:
        lam = 1.0
    else:
        lam = (confidence - conf_low) / (conf_high - conf_low)

    # Blend (mantieni TUTTI i top-N asset: anche se Kelly assegna 0 ad alcuni,
    # equal-weight component li tiene nel portfolio. Caps post-blend garantirà
    # cap_min per ognuno).
    blended = {}
    top_assets = {a for a, _ in top}  # universo top-N selezionato
    for a in top_assets:
        e_w = equal_w.get(a, 0.0)
        k_w = kelly_w.get(a, 0.0)
        blended[a] = (1.0 - lam) * e_w + lam * k_w

    # Re-normalize pre-cap (Kelly può aver dropped, equal compensa parzialmente)
    total = sum(blended.values())
    if total <= 0:
        return equal_weight(scores, top_n=top_n)
    blended = {a: w / total for a, w in blended.items()}

    # Apply caps (max 25%, floor 3%) + re-normalize
    capped = _apply_caps(blended, cap_max=cap_max, cap_min=cap_min)

    return PortfolioAllocation(
        method="confidence_kelly",
        top_n=top_n,
        weights=capped,
        total_weight=sum(capped.values()),
        excluded_assets=excluded,
        metadata={
            "confidence": confidence,
            "blend_lambda": lam,
            "cap_max": cap_max,
            "cap_min": cap_min,
            "fraction": fraction,
        },
        description=(
            f"Confidence-Kelly top-{len(capped)} (conf={confidence:.2f}, "
            f"λ={lam:.2f}, caps {cap_min*100:.0f}-{cap_max*100:.0f}%)"
        ),
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
    if method == "confidence_kelly":
        if vols is None:
            raise ValueError("confidence_kelly richiede `vols` dict")
        if "confidence" not in kwargs:
            raise ValueError("confidence_kelly richiede `confidence` (0-1)")
        return confidence_kelly(
            scores, vols, confidence=kwargs["confidence"], top_n=top_n,
            cap_max=kwargs.get("cap_max", 0.25),
            cap_min=kwargs.get("cap_min", 0.03),
            conf_low=kwargs.get("conf_low", 0.60),
            conf_high=kwargs.get("conf_high", 0.80),
            fraction=kwargs.get("fraction", 0.50),
            score_baseline=kwargs.get("score_baseline", 50.0),
        )
    raise ValueError(f"Method '{method}' non supportato. "
                     f"Disponibili: equal_weight, score_weighted, risk_parity, "
                     f"kelly_fractional, confidence_kelly")


def list_methods() -> list[str]:
    """Lista algoritmi disponibili."""
    return [
        "equal_weight", "score_weighted", "risk_parity",
        "kelly_fractional", "confidence_kelly",
    ]


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
