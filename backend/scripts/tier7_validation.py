"""Tier 7+8+9 VALIDATION GATE — episodi storici estesi.

Confronta predicted top-N asset vs actual top-N reali storici, con e senza
i flag Tier 7/8/9. Supporta top_3 e top_5 audit modes.

Target Ray Dalio (per top_5):
- Hit rate >=4/5: ≥80% degli episodi
- Hit rate 5/5: ≥30% degli episodi
- 2008 GFC: deflation deve dominare con prob > 0.6
- 2022: cash deve apparire nei top-5

Run:
    python scripts/tier7_validation.py            # top_3 + top_5 default
    python scripts/tier7_validation.py --top 5    # solo top_5
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib

# Episodi storici: import diretto dal modulo (32 episodi labeled, actual_top3 + actual_top5)
from app.services.regime.historical_episodes import HISTORICAL_EPISODES as _HE

EPISODES = [
    {
        "name": ep.name,
        "indicators": ep.indicators,
        "actual_top3": ep.actual_top3,
        "actual_top5": ep.actual_top5 if ep.actual_top5 else ep.actual_top3,
    }
    for ep in _HE
]


ALL_T_FLAGS = (
    "USE_GDP_COLLAPSE_OVERRIDE", "USE_RANK_PERCENTILE_SCORING",
    "USE_DOWNSIDE_PROTECTION_BONUS", "USE_ML_REGIME_BLEND",
    "USE_DEFENSIVE_TRANSITION_MODE",
)


def _episode_actual(ep: dict, top_n: int) -> set[str]:
    """Get actual top-N asset set per ep dict (fallback top-3 expansion)."""
    if top_n == 5 and "actual_top5" in ep:
        return ep["actual_top5"]
    return ep["actual_top3"]


def run_config(name: str, flags: dict[str, str], top_n: int = 3):
    # Cleanup all then set
    for k in ALL_T_FLAGS:
        os.environ.pop(k, None)
    for k, v in flags.items():
        os.environ[k] = v
    # Forza reload
    import app.services.regime.classifier as classifier_mod
    import app.services.scoring.engine as engine_mod
    importlib.reload(engine_mod)
    importlib.reload(classifier_mod)

    print(f"\n{'='*80}")
    print(f"CONFIG: {name}  flags: {flags}  top_n={top_n}")
    print(f"{'='*80}")

    hits_by_level = {i: 0 for i in range(top_n + 1)}
    for ep in EPISODES:
        actual = _episode_actual(ep, top_n)
        out = classifier_mod.classify_regime(ep["indicators"])
        scores = engine_mod.calculate_final_scores(out["probabilities"])
        top = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
        top_names = {n for n, _ in top}
        overlap = len(top_names & actual)
        hits_by_level[overlap] += 1

        # Regime accuracy (top vs ep regime — derivato da true_regime se mappato)
        # Per ora skip — solo asset hit rate

        probs = out["probabilities"]
        regime_str = " ".join(f"{k[:4]}={v:.2f}" for k, v in sorted(probs.items()))
        top_str = ", ".join(f"{n}({s:.0f})" for n, s in top)
        actual_str = ", ".join(sorted(actual))
        gdp_collapse = "[GDP_COLLAPSE]" if out.get("gdp_collapse_triggered") else ""
        ml_blend = "[ML]" if out.get("ml_blend") else ""
        verdict_thresh = top_n - 1 if top_n <= 3 else top_n - 2
        verdict = "STRONG" if overlap >= verdict_thresh else ("WIN" if overlap >= 1 else "MISS")
        print(f"\n  {ep['name']}  {gdp_collapse}{ml_blend}")
        print(f"    probs:   {regime_str}  regime={out['regime']}")
        print(f"    top-{top_n}: {top_str}")
        print(f"    actual:  {actual_str}")
        print(f"    overlap: {overlap}/{top_n} [{verdict}]")

    total = sum(hits_by_level.values())
    perfect = hits_by_level[top_n]
    strong_thresh = top_n - 1 if top_n <= 3 else top_n - 2
    strong = sum(v for k, v in hits_by_level.items() if k >= strong_thresh)
    any_hit = sum(v for k, v in hits_by_level.items() if k >= 1)
    print(f"\n  SUMMARY: perfect_{top_n}/{top_n}={perfect}/{total}  >={strong_thresh}/{top_n}={strong}/{total}  any={any_hit}/{total}")
    return hits_by_level


def main():
    print("TIER 7+8+9 VALIDATION GATE")
    args = sys.argv[1:]
    top_modes = []
    if "--top" in args:
        idx = args.index("--top")
        top_modes = [int(args[idx + 1])]
    else:
        top_modes = [3, 5]

    for top_n in top_modes:
        print(f"\n{'#'*80}\n# TOP-{top_n} VALIDATION\n{'#'*80}")

        # Baseline
        base = run_config(f"baseline (T7-T9 OFF)", {}, top_n=top_n)

        # Tier 7 only
        t7 = run_config("Tier 7 ON (T7.1+T7.2+T7.3)", {
            "USE_GDP_COLLAPSE_OVERRIDE": "1",
            "USE_RANK_PERCENTILE_SCORING": "1",
            "USE_DOWNSIDE_PROTECTION_BONUS": "1",
        }, top_n=top_n)

        # Tier 7 + Tier 8 ML
        t78 = run_config("Tier 7+8 ON (+ ML BLEND)", {
            "USE_GDP_COLLAPSE_OVERRIDE": "1",
            "USE_RANK_PERCENTILE_SCORING": "1",
            "USE_DOWNSIDE_PROTECTION_BONUS": "1",
            "USE_ML_REGIME_BLEND": "1",
        }, top_n=top_n)

        # Final diff
        print(f"\n{'='*80}\nDIFF top-{top_n}: baseline -> T7 -> T7+T8\n{'='*80}")
        for level in sorted(hits_by_level := base, reverse=True):
            b = base.get(level, 0)
            seven = t7.get(level, 0)
            seven_eight = t78.get(level, 0)
            print(f"  hits=={level}/{top_n}:  baseline={b}  T7={seven}  T7+T8={seven_eight}")

    # Cleanup
    for k in ALL_T_FLAGS:
        os.environ.pop(k, None)


if __name__ == "__main__":
    main()
