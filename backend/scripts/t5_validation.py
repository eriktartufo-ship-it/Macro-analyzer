"""Tier 5 LOOP CLOSURE — Validation comparativa.

Esegue:
1. Ricalcolo regime su tutti i record storici per N config Tier 5 flags.
2. Lead-time NBER per ogni config.
3. Brier score per ogni pillar T5 vs NBER recession (ground truth).
4. Tabella comparativa salvata in obsidian/.../T5_Validation_Report.md.

Pattern flag-isolato via env var pre-classify, no subprocess (più veloce).

Run:
    python scripts/t5_validation.py
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import date

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib

import pandas as pd

# Configs: subset di flag che influenzano il regime corrente.
# T5.4 (DFM asset bonus) e T5.5 (insider trajectory) e T5.7 (vol regime info)
# NON modificano probs regime → esclusi da lead-time analysis.
CONFIGS = {
    "baseline":            {},
    "T5.2_dedollar":       {"USE_DEDOLLAR_PILLAR": "1"},
    "T5.1_crisis_mod":     {"USE_CRISIS_MODULATION": "1"},
    "T5.3_news":           {"USE_NEWS_PILLAR": "1"},
    "T5_safe_3_pillars":   {"USE_DEDOLLAR_PILLAR": "1", "USE_CRISIS_MODULATION": "1", "USE_NEWS_PILLAR": "1"},
    "T5_all_regime_flags": {
        "USE_DEDOLLAR_PILLAR": "1",
        "USE_CRISIS_MODULATION": "1",
        "USE_NEWS_PILLAR": "1",
    },
    "T6.1_adaptive":       {"USE_ADAPTIVE_THRESHOLDS": "1"},
    "T6.1+T5_all":         {
        "USE_ADAPTIVE_THRESHOLDS": "1",
        "USE_DEDOLLAR_PILLAR": "1",
        "USE_CRISIS_MODULATION": "1",
        "USE_NEWS_PILLAR": "1",
    },
}

# Indicatori target T6.1 da preloadare per adaptive recipes
_T6_INDICATOR_KEYS = ("vix", "baa_spread", "cpi_yoy", "yield_curve_10y2y", "nfci")

LEAD_TIME_THRESHOLD = 0.35
LEAD_TIME_LOOKBACK_MONTHS = 12  # baseline storico documentato (9.2m avg)


@dataclass
class ConfigResult:
    config_name: str
    hit_rate: float
    n_recessions: int
    avg_lead_months: float | None
    median_lead_months: float | None
    avg_deflation_prob: float
    avg_stagflation_prob: float
    brier_deflation: float | None       # Brier P(defl) vs NBER recession
    brier_stagflation: float | None     # Brier P(stag) vs NBER recession
    brier_stress: float | None          # Brier max(P_defl, P_stag) vs NBER
    delta_brier_stress: float | None    # Delta vs baseline (sign: <0=better)


def _reload_modules() -> None:
    """Re-importa moduli che leggono env vars al runtime."""
    import app.services.config_flags
    import app.services.regime.classifier
    import app.services.regime.crisis_modulation
    import app.services.indicators.adaptive_thresholds
    importlib.reload(app.services.config_flags)
    importlib.reload(app.services.indicators.adaptive_thresholds)
    importlib.reload(app.services.regime.classifier)
    importlib.reload(app.services.regime.crisis_modulation)


def _build_adaptive_recipes_history(records: list[dict]) -> dict[str, dict]:
    """Precompute adaptive recipes per ogni mese, da serie storica dei records.

    Returns:
        {month_iso: {condition_name: AdaptiveRecipe}}.
        Lookup veloce a runtime: month_of(record_date) → recipes.

    Pattern: per ogni mese, calcola quantili rolling 10y backward.
    """
    from app.services.indicators.adaptive_thresholds import (
        get_recipe_for_condition,
        list_adaptive_conditions,
    )

    # Costruisci pd.Series per ogni indicator target dai records
    series_by_key: dict[str, pd.Series] = {}
    for key in _T6_INDICATOR_KEYS:
        data = {}
        for rec in records:
            ind = rec["indicators"]
            val = ind.get(key)
            if val is None:
                continue
            try:
                data[rec["date"]] = float(val)
            except (ValueError, TypeError):
                continue
        if data:
            series_by_key[key] = pd.Series(data).sort_index()
        else:
            series_by_key[key] = pd.Series(dtype=float)

    # Per ogni mese, calcola recipes con end_date=month_end, window 10y
    months = sorted({d["date"].to_period("M").to_timestamp("M") for d in records})
    recipes_by_month: dict[str, dict] = {}
    conditions = list_adaptive_conditions()

    # Mapping condition → indicator key (read directly)
    from app.services.indicators.adaptive_thresholds import _CONDITION_RECIPES

    for month_end in months:
        month_recipes = {}
        for cond in conditions:
            ind_key, _ttype = _CONDITION_RECIPES[cond]
            series = series_by_key.get(ind_key)
            if series is None or series.empty:
                continue
            recipe = get_recipe_for_condition(cond, series, end_date=month_end.date(), window_years=10)
            if recipe is not None:
                month_recipes[cond] = recipe
        recipes_by_month[str(month_end.date())] = month_recipes
    return recipes_by_month


def _month_key(d: pd.Timestamp) -> str:
    return str(d.to_period("M").to_timestamp("M").date())


def _load_records() -> list[dict]:
    """Carica records storici da DB. Estrae indicators + date."""
    from app.database import SessionLocal
    from app.models import RegimeClassification, SecularTrend

    with SessionLocal() as session:
        recs = (
            session.query(RegimeClassification)
            .order_by(RegimeClassification.date.asc())
            .all()
        )
        # Dedollar lookup per data (per T5.2 + T5.1)
        dedollar_records = (
            session.query(SecularTrend)
            .filter(SecularTrend.trend_name == "dedollarization")
            .all()
        )
        dedollar_by_date: dict[str, float] = {}
        for d in dedollar_records:
            try:
                payload = json.loads(d.components) if d.components else {}
                score = payload.get("combined_score")
                if score is None and d.score is not None:
                    score = d.score
                if score is not None:
                    dedollar_by_date[str(d.date)] = float(score)
            except (ValueError, TypeError):
                continue

        out = []
        for r in recs:
            if not r.conditions_met:
                continue
            try:
                meta = json.loads(r.conditions_met)
            except (ValueError, TypeError):
                continue
            indicators = meta.get("indicators", {}) or {}
            if not indicators:
                continue
            # Inject dedollar_combined da lookup
            ded = dedollar_by_date.get(str(r.date))
            if ded is not None:
                indicators["dedollar_combined"] = ded
            out.append({
                "date": pd.Timestamp(r.date),
                "indicators": indicators,
                "dedollar_combined": ded,
            })
        return out


def _recompute_regime_for_config(
    records: list[dict],
    flag_env: dict[str, str],
    recipes_by_month: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Ricalcola regime probs per ogni record con flag attivi.

    Args:
        recipes_by_month: precompute T6.1 adaptive recipes per mese.
            Solo usato se USE_ADAPTIVE_THRESHOLDS=1 in flag_env.
    """
    # Set env vars
    for k in (
        "USE_DEDOLLAR_PILLAR", "USE_CRISIS_MODULATION", "USE_NEWS_PILLAR",
        "USE_DFM_ASSET_BONUS", "USE_ASSET_FEEDBACK", "USE_ADAPTIVE_THRESHOLDS",
    ):
        os.environ.pop(k, None)
    for k, v in flag_env.items():
        os.environ[k] = v
    _reload_modules()

    from app.services.regime.classifier import classify_regime
    from app.services.regime.crisis_modulation import apply_crisis_modulation

    apply_modulation = flag_env.get("USE_CRISIS_MODULATION") == "1"
    use_adaptive = flag_env.get("USE_ADAPTIVE_THRESHOLDS") == "1"

    rows = []
    for rec in records:
        ind = rec["indicators"]
        ded = rec["dedollar_combined"]
        recipes = None
        if use_adaptive and recipes_by_month is not None:
            recipes = recipes_by_month.get(_month_key(rec["date"]))
        try:
            result = classify_regime(ind, adaptive_recipes=recipes)
            probs = result["probabilities"]
            if apply_modulation:
                probs = apply_crisis_modulation(probs, ind, dedollar_combined=ded)
        except Exception:
            continue
        rows.append({
            "date": rec["date"],
            "reflation": probs.get("reflation", 0.0),
            "stagflation": probs.get("stagflation", 0.0),
            "deflation": probs.get("deflation", 0.0),
            "goldilocks": probs.get("goldilocks", 0.0),
        })
    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df


def _list_nber_recessions(min_year: int = 1970) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """NBER recession spans (start, end)."""
    from app.services.indicators.fetcher import FredFetcher

    fred = FredFetcher()
    usrec = fred.fetch_series("nber_recession", start_date=date(min_year, 1, 1))
    if usrec is None or usrec.empty:
        return []
    # USREC = 1 during recession, 0 otherwise. Find spans.
    spans = []
    in_rec = False
    start = None
    for d, v in usrec.items():
        if v == 1 and not in_rec:
            in_rec = True
            start = pd.Timestamp(d)
        elif v == 0 and in_rec:
            in_rec = False
            spans.append((start, pd.Timestamp(d)))
    if in_rec:
        spans.append((start, pd.Timestamp(usrec.index[-1])))
    return spans


def _compute_lead_time(
    df: pd.DataFrame,
    nber_spans: list,
    threshold: float = LEAD_TIME_THRESHOLD,
    lookback_months: int = LEAD_TIME_LOOKBACK_MONTHS,
) -> tuple[float, int, float | None, float | None]:
    """Lead-time NBER: hit_rate, n_recessions, avg/median lead_months.

    Stress = max(deflation, stagflation). Trigger se stress >= threshold
    nei `lookback_months` precedenti l'inizio recession.
    """
    if df.empty or not nber_spans:
        return 0.0, 0, None, None

    monthly = df.resample("ME").mean().dropna()
    hits = 0
    leads = []
    n_analyzed = 0
    for start, _end in nber_spans:
        if start.year < 1970:
            continue
        start_m = start.to_period("M").to_timestamp("M")
        if start_m < monthly.index.min() or start_m > monthly.index.max():
            continue
        n_analyzed += 1
        pre_start = start_m - pd.DateOffset(months=lookback_months)
        pre_window = monthly.loc[(monthly.index >= pre_start) & (monthly.index < start_m)]
        if pre_window.empty:
            continue
        stress = pre_window[["deflation", "stagflation"]].max(axis=1)
        triggered = stress[stress >= threshold]
        if triggered.empty:
            continue
        hits += 1
        lead_months = (start_m - triggered.index.min()).days / 30.4375
        leads.append(lead_months)

    hit_rate = hits / n_analyzed if n_analyzed > 0 else 0.0
    avg_lead = sum(leads) / len(leads) if leads else None
    median_lead = sorted(leads)[len(leads) // 2] if leads else None
    return hit_rate, n_analyzed, avg_lead, median_lead


def _compute_brier(df: pd.DataFrame, nber_spans: list) -> dict[str, float | None]:
    """Brier scores: prob_deflation, prob_stagflation, max(defl, stag) vs NBER.

    Lower = better. Range [0, 1].
    """
    if df.empty or not nber_spans:
        return {"defl": None, "stag": None, "stress": None}
    monthly = df.resample("ME").mean().dropna()
    nber = pd.Series(0.0, index=monthly.index)
    for start, end in nber_spans:
        mask = (nber.index >= start) & (nber.index <= end)
        nber.loc[mask] = 1.0
    brier_defl = float(((monthly["deflation"] - nber) ** 2).mean())
    brier_stag = float(((monthly["stagflation"] - nber) ** 2).mean())
    stress = monthly[["deflation", "stagflation"]].max(axis=1)
    brier_stress = float(((stress - nber) ** 2).mean())
    return {"defl": brier_defl, "stag": brier_stag, "stress": brier_stress}


def main():
    print("Tier 5/6 VALIDATION — Loading records + NBER spans...")
    records = _load_records()
    print(f"  Records: {len(records)}")
    nber_spans = _list_nber_recessions(min_year=1970)
    print(f"  NBER recessions: {len(nber_spans)}")
    print("Precomputing T6.1 adaptive recipes by month...")
    recipes_by_month = _build_adaptive_recipes_history(records)
    print(f"  Months with recipes: {len(recipes_by_month)}")
    print()

    results: list[ConfigResult] = []
    baseline_brier_stress: float | None = None
    for name, flag_env in CONFIGS.items():
        print(f"[{name}] flags={flag_env}")
        df = _recompute_regime_for_config(records, flag_env, recipes_by_month)
        if df.empty:
            print("  → empty df, skip")
            continue
        hit_rate, n_rec, avg_lead, median_lead = _compute_lead_time(df, nber_spans)
        briers = _compute_brier(df, nber_spans)
        if name == "baseline":
            baseline_brier_stress = briers["stress"]
        delta_stress = None
        if baseline_brier_stress is not None and briers["stress"] is not None:
            delta_stress = briers["stress"] - baseline_brier_stress
        avg_defl = float(df["deflation"].mean())
        avg_stag = float(df["stagflation"].mean())
        results.append(ConfigResult(
            config_name=name,
            hit_rate=hit_rate,
            n_recessions=n_rec,
            avg_lead_months=avg_lead,
            median_lead_months=median_lead,
            avg_deflation_prob=avg_defl,
            avg_stagflation_prob=avg_stag,
            brier_deflation=briers["defl"],
            brier_stagflation=briers["stag"],
            brier_stress=briers["stress"],
            delta_brier_stress=delta_stress,
        ))
        print(f"  hit_rate={hit_rate*100:.1f}% (n={n_rec}) "
              f"avg_lead={avg_lead:.1f}m" if avg_lead else "  hit_rate=0.0% n=0")
        print(f"  Brier defl={briers['defl']:.4f} stag={briers['stag']:.4f} stress={briers['stress']:.4f}")
        print()

    # Print tabella riepilogo
    print("=" * 110)
    print("TIER 5 VALIDATION SUMMARY (lookback NBER 12m, threshold 0.35)")
    print("=" * 110)
    print(f"{'Config':<24} {'Hit%':>5} {'N':>3} {'Lead':>6} "
          f"{'Brier_D':>8} {'Brier_S':>8} {'Brier_X':>8} {'D_X':>9}")
    print("-" * 110)
    for r in results:
        al = f"{r.avg_lead_months:.1f}m" if r.avg_lead_months else "—"
        bd = f"{r.brier_deflation:.4f}" if r.brier_deflation is not None else "—"
        bs = f"{r.brier_stagflation:.4f}" if r.brier_stagflation is not None else "—"
        bx = f"{r.brier_stress:.4f}" if r.brier_stress is not None else "—"
        dx = f"{r.delta_brier_stress:+.5f}" if r.delta_brier_stress is not None else "—"
        print(f"{r.config_name:<24} {r.hit_rate*100:>4.1f}% "
              f"{r.n_recessions:>3} {al:>6} {bd:>8} {bs:>8} {bx:>8} {dx:>9}")
    print("=" * 110)
    print()
    print("Interpretation:")
    print("  Brier_D = Brier P(deflation) vs NBER recession binary, lower=better")
    print("  Brier_S = Brier P(stagflation) vs NBER")
    print("  Brier_X = Brier max(P_defl, P_stag) vs NBER (= stress signal)")
    print("  D_X = delta Brier_X vs baseline. Negativo = pillar migliora signal")
    print()
    # Salva JSON results
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "obsidian", "Erik", "02_Progetti", "Macro_Analyzer",
        "T5_Validation_Report.json",
    )
    out_path = os.path.abspath(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            [asdict(r) for r in results],
            f, indent=2, default=str,
        )
    print(f"Report JSON: {out_path}")
    return results


if __name__ == "__main__":
    main()
