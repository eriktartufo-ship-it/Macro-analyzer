"""T10 Honest Validation Batch — Council 2026-05-17 action item AI-3.

Esegue il random_backtest con real prices per:
- baseline (no flags)
- ognuno dei 6 flag prioritari council

Misura Sortino delta vs benchmark per ogni configurazione.
Gate council: Sortino delta > +0.05 = keep flag, altrimenti KILL.

Output:
- T10_Honest_Validation_Report.md (markdown human)
- T10_Honest_Validation_Report.json (machine-readable)

Uso:
    python scripts/t10_validation_batch.py [--n-simulations 15] [--horizon-years 10]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Council priority order (D2 verdict 2026-05-17)
# scoring_side = flag impatta calculate_final_scores (testabile da backtest)
# classifier_side = flag impatta probs/regime (richiede re-classification DB)
# allocation_side = flag impatta allocation engine separato (non testabile da backtest)
FLAGS_TO_TEST = [
    ("USE_CALIBRATED_SCORING", "Bayesian shrinkage.", "scoring_side"),
    ("USE_RANK_PERCENTILE_SCORING", "Within-regime rank.", "scoring_side"),
    ("USE_GDP_COLLAPSE_OVERRIDE", "Extreme regime trigger.", "classifier_side"),
    ("USE_ML_REGIME_BLEND", "RF+GB ensemble blend.", "classifier_side"),
    ("USE_DEDOLLAR_BONUS", "Secular tilt asset bonus.", "scoring_side_needs_inputs"),
    ("USE_DEFENSIVE_TRANSITION_MODE", "Mass redistribution to cash.", "allocation_side"),
]

KEEP_THRESHOLD = 0.05  # Sortino delta required to keep flag (council gate)
MIN_SORTINO_DELTA_FOR_REAL_EFFECT = 0.001  # Below = considered no-op


def reset_all_flags():
    """T10c: forza OFF esplicito su TUTTI i flag (parte da baseline puro).

    Necessario dopo T10b che ha promosso CALIBRATED/GDP_COLLAPSE/ML_BLEND a
    default-ON. Senza explicit OFF, il baseline mostrerebbe gli effetti dei
    default e i singoli flag testati sembrerebbero no-op.
    """
    from app.services.config_flags import get_all_flags_state, set_runtime_flag
    for name in get_all_flags_state():
        set_runtime_flag(name, None)
        os.environ[name] = "0"  # explicit disable


def set_flag(name: str):
    """Attiva un singolo flag via env var."""
    os.environ[name] = "1"


def run_batch(
    flag_name: str | None,
    description: str,
    cm_by_month: dict,
    asset_regime_data: dict,
    real_matrix,
    n_simulations: int,
    horizon_years: int,
    seed_base: int,
    start_min_year: int,
    start_max_year: int,
    walk_forward_cache=None,
):
    """Esegue il batch random per una configurazione (baseline o singolo flag)."""
    from scripts.random_backtest import run_simulation
    from app.services.scoring.engine import reload_calibration

    reset_all_flags()
    if flag_name:
        set_flag(flag_name)

    # Reload caches scoring che dipendono dai flag (calibration JSON + percentili).
    reload_calibration()

    # T10b: clear walk-forward cache (flag change can change classify_regime output).
    if walk_forward_cache is not None:
        walk_forward_cache.clear()

    config_name = flag_name or "BASELINE"
    print(f"\n{'='*60}")
    print(f"  CONFIG: {config_name}")
    print(f"  {description}")
    # Debug: verifica flag state — TUTTI gli active
    from app.services.config_flags import get_all_flags_state
    active = [n for n, v in get_all_flags_state().items() if v["value"]]
    print(f"  Active flags: {active}")
    print('='*60)

    rng_starts = random.Random(seed_base)
    results = []
    for i in range(n_simulations):
        start_year = rng_starts.randint(start_min_year, start_max_year)
        start_month = rng_starts.randint(1, 12)
        start_d = date(start_year, start_month, 1)
        res = run_simulation(
            start_date=start_d,
            horizon_years=horizon_years,
            classifications_by_month=cm_by_month,
            regime_data=asset_regime_data,
            seed=seed_base + i,
            use_calm_gate=False,
            real_returns_matrix=real_matrix,
            walk_forward_cache=walk_forward_cache,
        )
        if res is not None:
            results.append(res)

    if not results:
        return None

    alphas = [r.alpha_total for r in results]
    model_dds = [r.model_max_drawdown for r in results]
    bench_dds = [r.benchmark_max_drawdown for r in results]
    model_sortinos = [r.model_sortino for r in results]
    bench_sortinos = [r.benchmark_sortino for r in results]
    win_rate = sum(1 for a in alphas if a > 0) / len(alphas)
    dd_delta = float(np.mean(model_dds) - np.mean(bench_dds))
    sortino_delta = float(np.mean(model_sortinos) - np.mean(bench_sortinos))

    summary = {
        "config": config_name,
        "description": description,
        "n_simulations": len(results),
        "win_rate": float(win_rate),
        "alpha_mean_pct": float(np.mean(alphas) * 100),
        "alpha_std_pct": float(np.std(alphas) * 100),
        "model_dd_mean": float(np.mean(model_dds) * 100),
        "bench_dd_mean": float(np.mean(bench_dds) * 100),
        "dd_delta_pp": float(dd_delta * 100),
        "model_sortino_mean": float(np.mean(model_sortinos)),
        "bench_sortino_mean": float(np.mean(bench_sortinos)),
        "sortino_delta": float(sortino_delta),
    }

    print(f"  Win {sum(1 for a in alphas if a > 0)}/{len(alphas)}  "
          f"alpha={np.mean(alphas)*100:+.1f}%  "
          f"DD delta={dd_delta*100:+.2f}pp  "
          f"Sortino delta={sortino_delta:+.3f}")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-simulations", type=int, default=15)
    p.add_argument("--horizon-years", type=int, default=10)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--start-min-year", type=int, default=1990)
    p.add_argument("--start-max-year", type=int, default=2014)
    p.add_argument("--walk-forward", action="store_true",
                   help="T10b: classifica on-the-fly per testare flag classifier-side.")
    p.add_argument("--report-suffix", default="",
                   help="Suffisso filename report (es. '_walk_forward').")
    args = p.parse_args()

    t0 = time.time()
    print("=" * 60)
    print("T10 HONEST VALIDATION BATCH (Council 2026-05-17 AI-3)")
    print("=" * 60)
    print(f"Simulations per config: {args.n_simulations}")
    print(f"Horizon: {args.horizon_years} years")
    print(f"Window: {args.start_min_year}-{args.start_max_year}")
    print(f"Configs: baseline + {len(FLAGS_TO_TEST)} flags = "
          f"{(1 + len(FLAGS_TO_TEST)) * args.n_simulations} total simulations")

    # Pre-load shared state
    print("\nLoading classifications + ASSET_REGIME_DATA + real returns matrix...")
    from scripts.random_backtest import load_classifications_by_month
    from app.services.scoring.engine import ASSET_REGIME_DATA
    from app.services.backtest.real_returns_matrix import build_real_returns_matrix

    cm_by_month = load_classifications_by_month()
    real_matrix = build_real_returns_matrix(list(ASSET_REGIME_DATA.keys()))
    print(f"  Loaded {len(cm_by_month)} classifications + "
          f"matrix {real_matrix.shape}")

    wf_cache = None
    if args.walk_forward:
        from datetime import date as _date
        from app.services.backtest.walk_forward import (
            WalkForwardCache, preload_walk_forward_series,
        )
        print("\nT10b: pre-loading FRED series for walk-forward classifier...")
        series = preload_walk_forward_series(
            start_date=_date(args.start_min_year, 1, 1),
            end_date=_date(args.start_max_year + args.horizon_years, 12, 31),
        )
        wf_cache = WalkForwardCache(series)
        print(f"  Pre-loaded {len(series)} FRED series.")

    # Baseline + 6 flags
    summaries = []
    baseline = run_batch(
        flag_name=None,
        description="No flags. Pure rule-based classifier + base scoring.",
        cm_by_month=cm_by_month,
        asset_regime_data=ASSET_REGIME_DATA,
        real_matrix=real_matrix,
        n_simulations=args.n_simulations,
        horizon_years=args.horizon_years,
        seed_base=args.seed_base,
        start_min_year=args.start_min_year,
        start_max_year=args.start_max_year,
        walk_forward_cache=wf_cache,
    )
    summaries.append(baseline)
    baseline_sortino = baseline["model_sortino_mean"] if baseline else 0.0

    for flag_name, description, scope in FLAGS_TO_TEST:
        s = run_batch(
            flag_name=flag_name,
            description=description,
            cm_by_month=cm_by_month,
            asset_regime_data=ASSET_REGIME_DATA,
            real_matrix=real_matrix,
            n_simulations=args.n_simulations,
            horizon_years=args.horizon_years,
            seed_base=args.seed_base,
            start_min_year=args.start_min_year,
            start_max_year=args.start_max_year,
            walk_forward_cache=wf_cache,
        )
        if s is not None:
            s["scope"] = scope
            s["sortino_vs_baseline"] = s["model_sortino_mean"] - baseline_sortino
            # Verdict logic:
            # - se delta < MIN: no effect (flag non testabile da scoring layer)
            # - se delta > KEEP_THRESHOLD: KEEP
            # - altrimenti: KILL
            if abs(s["sortino_vs_baseline"]) < MIN_SORTINO_DELTA_FOR_REAL_EFFECT:
                s["verdict"] = "UNTESTABLE"
            elif s["sortino_vs_baseline"] > KEEP_THRESHOLD:
                s["verdict"] = "KEEP"
            else:
                s["verdict"] = "KILL"
            summaries.append(s)

    # Save JSON
    out_dir = Path(__file__).resolve().parent.parent.parent / "obsidian" / "Erik" / "02_Progetti" / "Macro_Analyzer"
    if not out_dir.exists():
        # Fallback: backend root
        out_dir = Path(__file__).resolve().parent.parent
    json_path = out_dir / f"T10_Honest_Validation_Report{args.report_suffix}.json"
    json_path.write_text(json.dumps({
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_simulations": args.n_simulations,
            "horizon_years": args.horizon_years,
            "seed_base": args.seed_base,
            "window": f"{args.start_min_year}-{args.start_max_year}",
            "keep_threshold": KEEP_THRESHOLD,
            "real_prices": True,
        },
        "summaries": summaries,
    }, indent=2), encoding="utf-8")
    print(f"\nJSON saved: {json_path}")

    # Markdown report
    md_lines = [
        "# T10 Honest Validation Report",
        "",
        f"> Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"> Council AI-3 action item — post-leakage validation con real prices yfinance + FRED CPI.",
        f"> {args.n_simulations} sim x {args.horizon_years}y per config, "
        f"window {args.start_min_year}-{args.start_max_year}, "
        f"seed_base {args.seed_base}.",
        f"> Gate: Sortino delta vs baseline > +{KEEP_THRESHOLD:.2f} = KEEP, < -{KEEP_THRESHOLD:.2f} = KILL.",
        "",
        "## Scope limitations",
        "",
        "Il random backtest legge classifications PRE-COMPUTED dal DB (`RegimeClassification`)",
        "e calcola top-5 via `calculate_final_scores(probs)`. Pertanto:",
        "",
        "- **scoring_side** (CALIBRATED, RANK_PERCENTILE): TESTABILE — il flag",
        "  modifica `calculate_final_scores` direttamente. Effetto immediato sul backtest.",
        "- **classifier_side** (GDP_COLLAPSE, ML_BLEND): NO-OP nel backtest. Il flag",
        "  modificherebbe `probs` se il classifier fosse ri-eseguito, ma le probs nel DB sono",
        "  già fissate. Per testare serve re-classify storico.",
        "- **scoring_side_needs_inputs** (DEDOLLAR_BONUS): NO-OP perché il backtest non",
        "  passa il dict `secular_bonus` a `calculate_final_scores`. Per testare serve",
        "  integrare un dedollar_signal store o ricalcolare per ogni mese.",
        "- **allocation_side** (DEFENSIVE_TRANSITION_MODE): NO-OP perché il backtest non",
        "  invoca l'allocation engine separato (`portfolio_construction.py`). Per testare",
        "  serve emulare il flow allocation con `apply_defensive_transition`.",
        "",
        "## Risultati",
        "",
        "| Config | Scope | Win | Alpha | DD delta | Sortino M | Sortino B | Sortino vs Baseline | Verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        win_str = f"{int(s['win_rate'] * s['n_simulations'])}/{s['n_simulations']}"
        sortino_v_baseline = s.get("sortino_vs_baseline", 0.0)
        verdict = s.get("verdict", "BASELINE")
        scope = s.get("scope", "—")
        if verdict == "BASELINE":
            verdict_str = "—"
        elif verdict == "KEEP":
            verdict_str = "✅ KEEP"
        elif verdict == "UNTESTABLE":
            verdict_str = "⚠️ UNTESTABLE"
        else:
            verdict_str = "❌ KILL"
        md_lines.append(
            f"| {s['config']} | {scope} | {win_str} ({s['win_rate']*100:.0f}%) | "
            f"{s['alpha_mean_pct']:+.1f}% | {s['dd_delta_pp']:+.2f}pp | "
            f"{s['model_sortino_mean']:.3f} | {s['bench_sortino_mean']:.3f} | "
            f"{sortino_v_baseline:+.3f} | {verdict_str} |"
        )
    md_lines.append("")
    md_lines.append("## Verdict per flag")
    md_lines.append("")
    for s in summaries[1:]:  # skip baseline
        md_lines.append(f"### {s['config']} — {s['verdict']}")
        md_lines.append("")
        md_lines.append(f"- **Scope**: {s.get('scope', '—')}")
        md_lines.append(f"- **Description**: {s['description']}")
        md_lines.append(f"- **Sortino vs baseline**: {s['sortino_vs_baseline']:+.3f}")
        md_lines.append(f"- **Win rate**: {int(s['win_rate']*s['n_simulations'])}/{s['n_simulations']}")
        md_lines.append(f"- **Alpha**: {s['alpha_mean_pct']:+.1f}%")
        md_lines.append(f"- **DD delta**: {s['dd_delta_pp']:+.2f}pp")
        if s["verdict"] == "UNTESTABLE":
            md_lines.append(f"- **Note**: flag non testabile in questo batch "
                            f"(vedi Scope limitations). Verdict deferred a future re-validation.")
        md_lines.append("")
    md_lines.append("## Conclusioni")
    md_lines.append("")
    kills = sum(1 for s in summaries[1:] if s.get("verdict") == "KILL")
    keeps = sum(1 for s in summaries[1:] if s.get("verdict") == "KEEP")
    untestables = sum(1 for s in summaries[1:] if s.get("verdict") == "UNTESTABLE")
    md_lines.append(f"- **KEEP**: {keeps}/{len(summaries)-1} flags")
    md_lines.append(f"- **KILL**: {kills}/{len(summaries)-1} flags")
    md_lines.append(f"- **UNTESTABLE**: {untestables}/{len(summaries)-1} flags (deferred)")
    md_lines.append("")
    md_lines.append("## Council expectation vs actual")
    md_lines.append("")
    md_lines.append(f"Council aspettava 3-4 di 6 flag killed. Actual sui flag testabili: "
                    f"{kills}/{keeps+kills} killed.")
    md_lines.append("")
    md_lines.append("## TODO follow-up")
    md_lines.append("")
    md_lines.append("- Re-classify DB storico con singolo flag classifier_side ON, "
                    "salva snapshot, ri-runna backtest su snapshot per testare "
                    "GDP_COLLAPSE + ML_BLEND realmente.")
    md_lines.append("- Implementa dedollar_signal store storico (FRED-based) per "
                    "testare DEDOLLAR_BONUS.")
    md_lines.append("- Estendi backtest per usare allocation engine completo (con "
                    "DEFENSIVE_TRANSITION) invece di top-5 equal weight.")
    md_lines.append("")

    md_path = out_dir / f"T10_Honest_Validation_Report{args.report_suffix}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Markdown saved: {md_path}")
    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print(f"Kills: {kills}, Keeps: {keeps}")


if __name__ == "__main__":
    main()
