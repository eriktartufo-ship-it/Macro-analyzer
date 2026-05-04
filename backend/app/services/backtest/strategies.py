"""Strategie di allocation per il backtest.

Generano `target_weights` (DataFrame mensile) DA shiftare di 1 mese prima dell'uso
(per evitare lookahead bias). Il chiamante (runner) si occupa dello shift.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy.orm import Session

from app.models import RegimeClassification
from app.services.scoring.engine import ASSET_CLASSES, calculate_final_scores


def regime_probs_monthly(db: Session) -> pd.DataFrame:
    """Carica posteriori regime mensili dal DB (aggregando se piu' record per mese)."""
    rows = (
        db.query(RegimeClassification)
        .order_by(RegimeClassification.date.asc())
        .all()
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([
        {
            "date": pd.Timestamp(r.date),
            "reflation": r.probability_reflation,
            "stagflation": r.probability_stagflation,
            "deflation": r.probability_deflation,
            "goldilocks": r.probability_goldilocks,
        }
        for r in rows
    ]).set_index("date").sort_index()
    return df.resample("ME").mean().dropna()


def score_weighted_strategy(
    db: Session,
    top_n: int = 5,
    score_threshold: float = 30.0,
    asset_classes: list[str] | None = None,
    force_include_dedollar: bool | None = None,
) -> pd.DataFrame:
    """Per ogni mese: calcola asset scores dai regime probs, alloca proporzionalmente
    ai top N asset con score >= threshold. Resto = cash (0% peso).

    Returns: DataFrame index=month-end, cols=asset, valori in [0,1] con sum<=1.
    """
    assets = asset_classes or list(ASSET_CLASSES)
    rp = regime_probs_monthly(db)
    if rp.empty:
        return pd.DataFrame()

    rows = []
    for ts, probs in rp.iterrows():
        prob_dict = {k: float(v) for k, v in probs.items()}
        scores = calculate_final_scores(prob_dict, force_include_dedollar=force_include_dedollar)
        # Filtra asset richiesti
        scores = {a: s for a, s in scores.items() if a in assets}
        # Sopra threshold, top-N
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:top_n]
        top = [(a, s) for a, s in top if s >= score_threshold]
        if not top:
            # Nessun asset sufficiente -> tutto cash
            row = {a: 0.0 for a in assets}
        else:
            total = sum(s for _, s in top)
            row = {a: 0.0 for a in assets}
            for a, s in top:
                row[a] = s / total
        row["__date__"] = ts
        rows.append(row)

    df = pd.DataFrame(rows).set_index("__date__")
    df.index.name = None
    return df


def buy_and_hold_strategy(
    asset_classes: list[str], dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Equal-weight buy-and-hold (rebilanciato mensile per semplicita)."""
    n = len(asset_classes)
    if n == 0:
        return pd.DataFrame(index=dates)
    weight = 1.0 / n
    df = pd.DataFrame(weight, index=dates, columns=asset_classes)
    return df


def sixty_forty_strategy(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """60% us_equities_growth + 40% us_bonds_long (proxy classico 60/40)."""
    return pd.DataFrame({
        "us_equities_growth": 0.60,
        "us_bonds_long": 0.40,
    }, index=dates)


def spy_only_strategy(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Buy-and-hold SPY (us_equities_growth proxy)."""
    return pd.DataFrame({"us_equities_growth": 1.0}, index=dates)


# ============================================================================
# Tier 2.4 roadmap Bridgewater: All-Weather risk parity allocation
# ============================================================================

# 4 quadrant macro (= REGIMES dal classifier). Ogni quadrant prende 25% del rischio.
_QUADRANT_REGIMES = ("reflation", "stagflation", "deflation", "goldilocks")
_QUADRANT_WEIGHT = 0.25  # 1/4 per quadrant (Bridgewater All-Weather)


def all_weather_risk_parity_strategy(
    asset_classes: list[str],
    dates: pd.DatetimeIndex,
    top_n_per_quadrant: int = 3,
) -> pd.DataFrame:
    """Bridgewater All-Weather: 25% rischio per ogni quadrant macro, asset
    within quadrant pesati per **vol parity** (1/vol).

    Filosofia: invece di "inseguire" il regime corrente, diversifica per QUALSIASI
    regime arrivi. Pesi statici nel tempo (selection da ASSET_REGIME_DATA hardcoded
    + vol parity → peso costante per asset, indipendente dal regime live).

    Algoritmo:
    1. Per ogni regime r:
       a. Trova top-N asset con `sharpe[r]` storico migliore (esclude vol=0).
       b. Vol parity within: `w_a = (1/vol_a) / Σ_b∈top(1/vol_b)`.
       c. Contributo asset a al peso finale: `0.25 × w_a`.
    2. Asset presente in piu' top-N (cross-regime overlap): pesi sommati.
    3. Output DataFrame `dates × asset_classes` con righe identiche (time-invariant).

    Args:
        asset_classes: universe asset disponibili (filtrato su Yahoo prices).
        dates: indice mensile per il backtest (le righe sono identiche).
        top_n_per_quadrant: quanti asset selezionare per ogni regime (default 3).

    Returns:
        DataFrame `index=dates, cols=asset_classes` con pesi sommati a 1
        (full investment). Vuoto se asset_classes vuoto.
    """
    from app.services.scoring.engine import ASSET_REGIME_DATA

    if not asset_classes:
        return pd.DataFrame(index=dates)

    weights: dict[str, float] = {a: 0.0 for a in asset_classes}

    for regime in _QUADRANT_REGIMES:
        # Candidati asset con stat regime + vol > 0
        candidates: list[tuple[str, float, float]] = []
        for a in asset_classes:
            stats_by_regime = ASSET_REGIME_DATA.get(a, {})
            stats = stats_by_regime.get(regime)
            if not stats:
                continue
            vol = float(stats.get("vol", 0.0))
            sharpe = float(stats.get("sharpe", 0.0))
            if vol > 0:
                candidates.append((a, sharpe, vol))

        if not candidates:
            continue

        # Top-N per Sharpe regime-specific (asset che fanno bene in quel regime)
        candidates.sort(key=lambda x: -x[1])
        top = candidates[:top_n_per_quadrant]

        # Vol parity within quadrant
        inv_vol_sum = sum(1.0 / vol for _, _, vol in top)
        if inv_vol_sum <= 0:
            continue
        for asset, _, vol in top:
            within_weight = (1.0 / vol) / inv_vol_sum
            weights[asset] += _QUADRANT_WEIGHT * within_weight

    # Renormalize (alcuni quadrant potrebbero non aver candidati → peso totale < 1)
    total = sum(weights.values())
    if total > 0 and abs(total - 1.0) > 1e-9:
        weights = {a: w / total for a, w in weights.items()}

    # DataFrame time-invariant: stessa riga per ogni data
    return pd.DataFrame([weights] * len(dates), index=dates)
