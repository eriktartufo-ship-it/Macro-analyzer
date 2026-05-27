"""T11 Validation: misura accuracy + Crisis Brier rule-based classifier
con USE_MOMENTUM_PILLARS ON vs OFF sui 49 historical episodes.

Acceptance criteria (council 2026-05-27 spec T11):
- Accuracy aggregate &gt;= 73.5% baseline (LOO CV equivalent on labeled set)
- Crisis Brier &lt;= 0.045 (oggi 0.036 baseline)
- 1973-75 stagflation: stagflation prob &gt;= 0.60
- 2018 Q4 / 2019 NOT mis-classified as stagflation

Output: confronto side-by-side OFF vs ON con verdict GO/NO-GO.

Uso: `python scripts/t11_validation.py`
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _run_with_flag(flag_value: bool):
    """Esegue accuracy + brier rule-based con USE_MOMENTUM_PILLARS=flag."""
    from app.services.config_flags import set_runtime_flag
    set_runtime_flag("USE_MOMENTUM_PILLARS", flag_value)
    # Disable ML blend e GDP_COLLAPSE per isolare effetto momentum sul rule-based puro
    set_runtime_flag("USE_ML_REGIME_BLEND", False)
    set_runtime_flag("USE_GDP_COLLAPSE_OVERRIDE", False)
    # Clear any env overrides
    for f in ("USE_MOMENTUM_PILLARS", "USE_ML_REGIME_BLEND", "USE_GDP_COLLAPSE_OVERRIDE"):
        if f in os.environ:
            del os.environ[f]
    if flag_value:
        os.environ["USE_MOMENTUM_PILLARS"] = "1"

    from app.services.regime.classifier import classify_regime
    from app.services.regime.historical_episodes import HISTORICAL_EPISODES
    from app.services.validation.brier_crisis import assess_crisis_brier

    correct = 0
    per_regime_correct: dict[str, list[int]] = {}
    misclassifications = []
    briers = []
    for ep in HISTORICAL_EPISODES:
        probs = classify_regime(ep.indicators)["probabilities"]
        predicted = max(probs, key=probs.get)
        is_correct = predicted == ep.true_regime
        if is_correct:
            correct += 1
        per_regime_correct.setdefault(ep.true_regime, []).append(int(is_correct))
        if not is_correct:
            misclassifications.append({
                "ep": ep.name, "true": ep.true_regime, "pred": predicted,
                "stag_prob": probs.get("stagflation", 0.0),
            })
        # Brier on full episode (not just crisis)
        brier = sum(
            (probs.get(r, 0.0) - (1.0 if r == ep.true_regime else 0.0)) ** 2
            for r in ("reflation", "stagflation", "deflation", "goldilocks")
        )
        briers.append(brier)

    accuracy = correct / len(HISTORICAL_EPISODES)
    mean_brier = sum(briers) / len(briers)

    # Crisis Brier (subset)
    def predict_fn(ind):
        return classify_regime(ind)["probabilities"]
    crisis_result = assess_crisis_brier(predict_fn)

    # Specific episodes verification
    target_1973 = None
    target_2018 = None
    for ep in HISTORICAL_EPISODES:
        probs = classify_regime(ep.indicators)["probabilities"]
        if "1973" in ep.name and target_1973 is None:
            target_1973 = (ep.name, probs.get("stagflation", 0.0))
        if "2018" in ep.name and target_2018 is None:
            predicted = max(probs, key=probs.get)
            target_2018 = (ep.name, predicted, probs.get("stagflation", 0.0))

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(HISTORICAL_EPISODES),
        "mean_brier": mean_brier,
        "crisis_brier": crisis_result.mean_brier,
        "crisis_accuracy": crisis_result.accuracy,
        "per_regime_accuracy": {r: sum(v) / len(v) for r, v in per_regime_correct.items()},
        "misclassifications": misclassifications,
        "target_1973": target_1973,
        "target_2018": target_2018,
    }


def main():
    print("=" * 70)
    print("T11 VALIDATION — Momentum pillars vs baseline (rule-based, ML OFF)")
    print("=" * 70)

    off = _run_with_flag(False)
    on = _run_with_flag(True)

    print(f"\n{'Metric':<25} {'OFF (baseline)':<20} {'ON (T11)':<20}")
    print("-" * 65)
    print(f"{'Accuracy':<25} {off['accuracy']*100:>5.1f}% ({off['correct']}/{off['total']})    "
          f"{on['accuracy']*100:>5.1f}% ({on['correct']}/{on['total']})")
    print(f"{'Mean Brier':<25} {off['mean_brier']:>12.4f}        {on['mean_brier']:>12.4f}")
    print(f"{'Crisis Brier':<25} {off['crisis_brier']:>12.4f}        {on['crisis_brier']:>12.4f}")
    print(f"{'Crisis accuracy':<25} {off['crisis_accuracy']*100:>5.1f}%              "
          f"{on['crisis_accuracy']*100:>5.1f}%")

    print("\n--- Per-regime accuracy ---")
    regimes = sorted(set(list(off['per_regime_accuracy'].keys()) + list(on['per_regime_accuracy'].keys())))
    for r in regimes:
        o = off['per_regime_accuracy'].get(r, 0.0)
        n = on['per_regime_accuracy'].get(r, 0.0)
        delta = (n - o) * 100
        print(f"  {r:<14} OFF={o*100:>5.1f}%  ON={n*100:>5.1f}%  delta={delta:+.1f}pp")

    print(f"\n--- Specific episodes ---")
    if off['target_1973']:
        name, stag = off['target_1973']
        print(f"  1973 ({name}) OFF: stag_prob={stag:.3f}")
    if on['target_1973']:
        name, stag = on['target_1973']
        print(f"  1973 ({name}) ON:  stag_prob={stag:.3f}  (target &gt;=0.60)")
    if off['target_2018']:
        name, pred, stag = off['target_2018']
        print(f"  2018 ({name}) OFF: predicted={pred} stag_prob={stag:.3f}")
    if on['target_2018']:
        name, pred, stag = on['target_2018']
        print(f"  2018 ({name}) ON:  predicted={pred} stag_prob={stag:.3f}  (target NOT stagflation)")

    # Verdict
    print(f"\n--- Acceptance criteria ---")
    target_accuracy = 0.735  # 73.5% baseline da CLAUDE.md
    target_crisis_brier = 0.045
    target_1973_stag = 0.60

    checks = []
    accuracy_ok = on['accuracy'] >= target_accuracy
    checks.append(("Accuracy &gt;= 73.5%", accuracy_ok, f"{on['accuracy']*100:.1f}%"))

    accuracy_no_regress = on['accuracy'] >= off['accuracy'] - 0.02  # tolleranza -2pp
    checks.append(("Accuracy non regredisce (tolleranza -2pp)", accuracy_no_regress,
                   f"OFF {off['accuracy']*100:.1f}% -&gt; ON {on['accuracy']*100:.1f}%"))

    crisis_ok = on['crisis_brier'] <= target_crisis_brier
    checks.append(("Crisis Brier &lt;= 0.045", crisis_ok, f"{on['crisis_brier']:.4f}"))

    target_1973_ok = on['target_1973'] and on['target_1973'][1] >= target_1973_stag
    checks.append(("1973 stag &gt;= 0.60", target_1973_ok,
                   f"{on['target_1973'][1]:.3f}" if on['target_1973'] else "missing"))

    target_2018_ok = on['target_2018'] and on['target_2018'][1] != "stagflation"
    checks.append(("2018 NOT stagflation", target_2018_ok,
                   on['target_2018'][1] if on['target_2018'] else "missing"))

    print(f"  {'Check':<45} {'Pass':<8} Value")
    print("  " + "-" * 70)
    for label, ok, value in checks:
        status = "OK" if ok else "X"
        print(f"  {label:<45} {status:<8} {value}")

    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    print(f"\n  Verdict: {passed}/{total} acceptance criteria passed")
    if passed == total:
        print("  OK GO: T11 ready for default-ON promotion")
        return 0
    elif passed >= total - 1:
        print("  ? CONDITIONAL: 1 criterion miss, review case-by-case")
        return 0
    else:
        print("  X NO-GO: &gt;=2 criteria failed, keep opt-in")
        return 1


if __name__ == "__main__":
    sys.exit(main())
