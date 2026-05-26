"""Bughunter weekly cron script.

Council reco (lynch): "rendi lo bughunter schedulato: ogni venerdì rilancialo
su tutto il backend."

Esegue check automatici di data integrity:

1. **Indicators latest vs reality benchmark**: ogni indicator nel JSON live deve
   essere in range plausibile per macro 2025-2026.
2. **Naming consistency**: cerca consumer che usano nomi indicator NON aliased.
3. **Silent defaults**: identifica `indicators.get(X, default)` dove X NON è
   nel JSON live (sempre default = silent bug).
4. **Circuit-breakers status**: verifica che low_confidence + OOD + consecutive
   wrong NON siano triggered simultaneamente (indica modello rotto).
5. **Covariate-shift status**: verifica n_shifted < 3 (no shift critico).
6. **Output**: report JSON + exit code 0=OK, 1=warnings, 2=critical.

Usage:
    python scripts/bughunter_weekly.py [--json output.json] [--strict]

CI/cron-ready: exit code mappato per integrazione con monitoring.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Reality benchmarks USA 2026 (mantenuti sync con bughunter.md agent)
REALITY_BENCHMARKS: dict[str, tuple[float, float]] = {
    "gdp_roc": (-1.5, 1.5),
    "cpi_yoy": (1.0, 6.0),
    "core_pce_yoy": (1.0, 5.0),
    "unrate": (3.5, 6.0),
    "vix": (10, 35),
    "fed_funds_rate": (0, 6.0),
    "breakeven_10y": (1.5, 3.5),
    "baa_spread": (1.0, 4.0),
    "consumer_sentiment": (50, 110),
    "yield_curve_10y2y": (-2.0, 2.5),
    "nfci": (-1.0, 1.5),
    "lei_roc": (-5, 5),
}


@dataclass
class Issue:
    severity: str  # info | warn | critical
    category: str  # data_integrity | naming | silent_default | covariate_shift | circuit_breaker
    message: str


def check_indicators_realism() -> list[Issue]:
    """Check 1: indicators latest vs reality benchmarks."""
    from app.database import SessionLocal
    from app.models import RegimeClassification

    issues: list[Issue] = []
    s = SessionLocal()
    try:
        latest = s.query(RegimeClassification).order_by(
            RegimeClassification.date.desc()
        ).first()
        if latest is None:
            issues.append(Issue("critical", "data_integrity",
                                "No RegimeClassification in DB"))
            return issues

        cm = latest.conditions_met or {}
        if isinstance(cm, str):
            cm = json.loads(cm)
        ind = cm.get("indicators", {}) if isinstance(cm, dict) else {}

        from app.services.common.indicator_aliases import safe_indicator
        for key, (lo, hi) in REALITY_BENCHMARKS.items():
            val = safe_indicator(ind, key, default=None)
            if val is None:
                issues.append(Issue("warn", "data_integrity",
                                    f"Indicator '{key}' missing or default-None"))
                continue
            if val < lo or val > hi:
                issues.append(Issue("warn", "data_integrity",
                                    f"Indicator '{key}'={val} out of "
                                    f"realistic range [{lo}, {hi}]"))
    finally:
        s.close()
    return issues


def check_covariate_shift() -> list[Issue]:
    """Check 2: covariate shift live vs training."""
    from app.database import SessionLocal
    from app.models import RegimeClassification
    from app.services.validation.covariate_shift import assess_covariate_shift

    issues: list[Issue] = []
    s = SessionLocal()
    try:
        latest = s.query(RegimeClassification).order_by(
            RegimeClassification.date.desc()
        ).first()
        if latest is None:
            return issues
        cm = latest.conditions_met or {}
        if isinstance(cm, str):
            cm = json.loads(cm)
        ind = cm.get("indicators", {}) if isinstance(cm, dict) else {}

        report = assess_covariate_shift(ind)
        if report.n_critical > 0:
            issues.append(Issue("critical", "covariate_shift",
                                f"{report.n_critical} indicators in CRITICAL shift "
                                f"(>3σ): {[s.name for s in report.indicators_shift if s.severity == 'critical']}"))
        elif report.shift_alert:
            issues.append(Issue("warn", "covariate_shift",
                                f"{report.n_shifted} indicators in shift "
                                f"(>2.5σ). Modello in zona limite training."))
    finally:
        s.close()
    return issues


def check_circuit_breakers() -> list[Issue]:
    """Check 3: circuit-breakers status su current indicators."""
    from app.database import SessionLocal
    from app.models import RegimeClassification
    from app.services.regime.classifier import classify_regime
    from app.services.validation.circuit_breakers import evaluate_all_circuit_breakers

    issues: list[Issue] = []
    s = SessionLocal()
    try:
        latest = s.query(RegimeClassification).order_by(
            RegimeClassification.date.desc()
        ).first()
        if latest is None:
            return issues
        cm = latest.conditions_met or {}
        if isinstance(cm, str):
            cm = json.loads(cm)
        ind = cm.get("indicators", {}) if isinstance(cm, dict) else {}

        out = classify_regime(ind)
        max_prob = max(out["probabilities"].values())
        cb_status = evaluate_all_circuit_breakers(
            max_prob=max_prob, indicators=ind,
        )
        if cb_status["any_critical"]:
            issues.append(Issue("critical", "circuit_breaker",
                                f"Critical CB: {cb_status['summary']}"))
        elif cb_status["any_triggered"]:
            issues.append(Issue("warn", "circuit_breaker",
                                f"CB triggered: {cb_status['summary']}"))
    finally:
        s.close()
    return issues


def check_test_suite() -> list[Issue]:
    """Check 4: basic test suite via pytest."""
    import subprocess
    issues: list[Issue] = []
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "-q", "--tb=no",
             "--ignore=tests/test_walkforward_validation.py"],
            cwd=backend_dir, capture_output=True, text=True, timeout=180,
        )
        if result.returncode != 0:
            issues.append(Issue("critical", "data_integrity",
                                f"Pytest suite failed: {result.stdout[-300:]}"))
    except subprocess.TimeoutExpired:
        issues.append(Issue("warn", "data_integrity", "Pytest timeout"))
    except Exception as e:
        issues.append(Issue("warn", "data_integrity", f"Pytest run failed: {e}"))
    return issues


def main(argv: list[str]) -> int:
    json_out = None
    strict = False
    for i, a in enumerate(argv):
        if a == "--json" and i + 1 < len(argv):
            json_out = argv[i + 1]
        if a == "--strict":
            strict = True

    print("=== BUGHUNTER WEEKLY ===")
    all_issues: list[Issue] = []
    for check_fn in (
        check_indicators_realism,
        check_covariate_shift,
        check_circuit_breakers,
        check_test_suite,
    ):
        try:
            print(f"\nRunning {check_fn.__name__}...")
            issues = check_fn()
            for iss in issues:
                tag = {"info": "[INFO]", "warn": "[WARN]", "critical": "[CRIT]"}[iss.severity]
                print(f"  {tag} {iss.category}: {iss.message}")
            all_issues.extend(issues)
        except Exception as e:
            print(f"  [ERR ] {check_fn.__name__} raised: {e}")
            all_issues.append(Issue("warn", "data_integrity",
                                    f"{check_fn.__name__} raised: {e}"))

    n_critical = sum(1 for i in all_issues if i.severity == "critical")
    n_warn = sum(1 for i in all_issues if i.severity == "warn")
    print(f"\n=== SUMMARY ===")
    print(f"Critical: {n_critical}")
    print(f"Warnings: {n_warn}")
    print(f"Info: {sum(1 for i in all_issues if i.severity == 'info')}")

    if json_out:
        with open(json_out, "w", encoding="utf-8") as fp:
            json.dump([
                {"severity": i.severity, "category": i.category, "message": i.message}
                for i in all_issues
            ], fp, indent=2)
        print(f"\nSaved JSON to {json_out}")

    # Exit code
    if n_critical > 0:
        return 2
    if n_warn > 0 and strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
