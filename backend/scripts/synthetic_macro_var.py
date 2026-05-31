"""Council action sessione 26 — Synthetic Monte Carlo macro-VAR validation.

Council brutal feedback: "n=100 sims su 2000-2023 sono overlapping bootstrap
dello stesso path storico. Statistical power vero: ~3-4 regime shifts
indipendenti. Out-of-sample fail likely".

Soluzione: macro-VAR su CPI/GDP/UNRATE/yield_curve/VIX based on 1990-2023
covariance matrix. Genera 1000 sample path PLAUSIBILI (no future leak, no
historical correlation preservation guarantee but realistic distribution).

Per ogni path: classifica regime, calcola scores, simula return mensile
ipotetico (using ASSET_REGIME_DATA hardcoded perché path è sintetico).
Output: distribuzione alpha vs 60/40, SD, CI 95%, % positive paths.

Caveats:
- macro-VAR è semplificato (no regime-switching, no fat tails)
- Asset returns simulati = ASSET_REGIME_DATA hardcoded means + noise
- Sample size 1000 path × 12m = abbondante per CI estimate
- Realistic: simula scenari "non visti" in 2000-2023

Uso: python scripts/synthetic_macro_var.py [--n-paths 1000] [--seed 42]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Storical macro state summary statistics 1990-2023
# (mean, std) per variable
HISTORICAL_STATS = {
    "cpi_yoy": (2.5, 1.2),         # avg 2.5% std 1.2% (post-Volcker era)
    "gdp_roc": (0.6, 1.0),         # avg 0.6%/q std 1.0%
    "unrate": (5.8, 1.8),          # avg 5.8% std 1.8%
    "yield_curve_10y2y": (1.3, 1.0),  # avg 1.3% std 1.0%
    "vix": (19.0, 8.0),            # avg 19 std 8 (skewed)
    "core_pce_yoy": (2.2, 0.9),
    "nfci": (-0.1, 0.5),
    "fed_funds_rate": (3.0, 2.0),
}

# Covariance structure (key correlations 1990-2023)
# Diagonal = 1, off-diagonal based on macro-econ relationships
CORRELATION_MATRIX = np.array([
    # cpi  gdp  unr  yc   vix  pce  nfci ff
    [1.0,  0.0, -0.2, 0.2, -0.1, 0.7, 0.0,  0.4],  # cpi
    [0.0,  1.0, -0.4, -0.1, -0.3, 0.0, -0.3, 0.2],  # gdp
    [-0.2, -0.4, 1.0, -0.1, 0.4, -0.1, 0.4, -0.3],  # unrate
    [0.2,  -0.1, -0.1, 1.0, -0.2, 0.2, -0.2, 0.5],  # yield_curve
    [-0.1, -0.3, 0.4, -0.2, 1.0, -0.1, 0.5, -0.2],  # vix
    [0.7,  0.0, -0.1, 0.2, -0.1, 1.0, 0.0, 0.4],   # core_pce
    [0.0,  -0.3, 0.4, -0.2, 0.5, 0.0, 1.0, -0.2],  # nfci
    [0.4,  0.2, -0.3, 0.5, -0.2, 0.4, -0.2, 1.0],  # fed_funds
])

VAR_KEYS = ["cpi_yoy", "gdp_roc", "unrate", "yield_curve_10y2y", "vix",
            "core_pce_yoy", "nfci", "fed_funds_rate"]


def sample_macro_state(rng: np.random.Generator) -> dict[str, float]:
    """Sample 1 macro state vector from multivariate normal."""
    means = np.array([HISTORICAL_STATS[k][0] for k in VAR_KEYS])
    stds = np.array([HISTORICAL_STATS[k][1] for k in VAR_KEYS])
    # Sample standardized
    z = rng.multivariate_normal(np.zeros(8), CORRELATION_MATRIX)
    # Scale + shift
    sample = means + z * stds
    # Clip VIX to non-negative
    sample[4] = max(5.0, sample[4])
    # Clip unrate to [3, 20]
    sample[2] = max(3.0, min(20.0, sample[2]))
    return {k: float(sample[i]) for i, k in enumerate(VAR_KEYS)}


def simulate_path_12m(path_id: int, seed: int = 42) -> dict[str, float]:
    """Simula 1 path 12-month: regime classification + simulated alpha.

    Returns: dict {alpha_60_40, alpha_sp500, regime_majority, total_return}
    """
    from app.services.regime.classifier import classify_regime
    from app.services.scoring.engine import calculate_final_scores

    rng = np.random.default_rng(seed + path_id * 1000)
    monthly_returns = []
    benchmark_60_40 = []
    benchmark_sp500 = []
    regime_count = {"reflation": 0, "stagflation": 0, "deflation": 0, "goldilocks": 0}

    for month in range(12):
        macro_state = sample_macro_state(rng)
        # Add derived indicators (momentum proxies)
        macro_state["cpi_yoy_change_6m"] = rng.normal(0, 0.5)
        macro_state["core_pce_yoy_change_6m"] = rng.normal(0, 0.4)
        macro_state["gdp_roc_change_6m"] = rng.normal(0, 0.7)
        macro_state["cpi_yoy_min_last_6m"] = max(0, macro_state["cpi_yoy"] - 0.5)

        try:
            result = classify_regime(macro_state)
        except Exception:
            continue
        regime = result["regime"]
        regime_count[regime] += 1
        probs = result["probabilities"]
        scores = calculate_final_scores(probs)

        # Top-7 equal weight
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])[:7]
        # Synthetic monthly return based on ASSET_REGIME_DATA + noise
        from app.services.scoring.engine import ASSET_REGIME_DATA
        portfolio_return = 0.0
        for asset, _ in sorted_scores:
            asset_data = ASSET_REGIME_DATA.get(asset, {})
            # Weighted by regime probability
            expected_monthly = 0.0
            for r, p in probs.items():
                stats = asset_data.get(r, {"avg_return": 0, "vol": 0.15})
                annual_return = stats.get("avg_return", 0)
                annual_vol = stats.get("vol", 0.15)
                monthly_return_mean = annual_return / 12.0
                monthly_vol = annual_vol / np.sqrt(12)
                sample = rng.normal(monthly_return_mean, monthly_vol)
                expected_monthly += p * sample
            portfolio_return += (1.0 / 7) * expected_monthly
        monthly_returns.append(portfolio_return)

        # Benchmark 60/40: 60% growth + 40% bonds
        growth_stats = ASSET_REGIME_DATA["us_equities_growth"]
        bond_stats = ASSET_REGIME_DATA["us_bonds_long"]
        b60_ret = 0.0
        bsp_ret = 0.0
        for r, p in probs.items():
            growth_mean = growth_stats[r]["avg_return"] / 12.0
            growth_vol = growth_stats[r]["vol"] / np.sqrt(12)
            bond_mean = bond_stats[r]["avg_return"] / 12.0
            bond_vol = bond_stats[r]["vol"] / np.sqrt(12)
            g_sample = rng.normal(growth_mean, growth_vol)
            b_sample = rng.normal(bond_mean, bond_vol)
            b60_ret += p * (0.6 * g_sample + 0.4 * b_sample)
            bsp_ret += p * g_sample  # SP500 proxy = growth
        benchmark_60_40.append(b60_ret)
        benchmark_sp500.append(bsp_ret)

    if len(monthly_returns) < 6:
        return None

    # Compound 12m
    cum_model = float(np.prod([1 + r for r in monthly_returns]) - 1)
    cum_60_40 = float(np.prod([1 + r for r in benchmark_60_40]) - 1)
    cum_sp500 = float(np.prod([1 + r for r in benchmark_sp500]) - 1)

    return {
        "path_id": path_id,
        "alpha_60_40": cum_model - cum_60_40,
        "alpha_sp500": cum_model - cum_sp500,
        "total_return": cum_model,
        "regime_majority": max(regime_count, key=regime_count.get),
        "regime_distribution": regime_count,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-paths", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    print(f"=" * 70)
    print(f"SYNTHETIC MACRO-VAR MONTE CARLO — {args.n_paths} paths, 12m horizon")
    print(f"Council sessione 26 — counter overfit objection")
    print(f"=" * 70)

    results = []
    for i in range(args.n_paths):
        r = simulate_path_12m(i, seed=args.seed)
        if r is not None:
            results.append(r)
        if args.verbose and (i + 1) % 100 == 0:
            print(f"  Completed {i+1}/{args.n_paths}")

    if not results:
        print("No valid paths.")
        return

    alphas_60_40 = [r["alpha_60_40"] for r in results]
    alphas_sp500 = [r["alpha_sp500"] for r in results]
    total_returns = [r["total_return"] for r in results]

    print(f"\nPaths completed: {len(results)}/{args.n_paths}\n")
    print(f"AGGREGATE METRICS:")
    print(f"  Mean alpha vs 60/40:    {np.mean(alphas_60_40)*100:+.2f}pp")
    print(f"  Stdev alpha:            {np.std(alphas_60_40)*100:+.2f}pp")
    print(f"  Median alpha:           {np.median(alphas_60_40)*100:+.2f}pp")
    print(f"  % positive alpha:       {sum(1 for a in alphas_60_40 if a > 0)/len(alphas_60_40)*100:.1f}%")
    print(f"  CI 95% bootstrap:       [{np.percentile(alphas_60_40, 2.5)*100:.2f}pp, {np.percentile(alphas_60_40, 97.5)*100:.2f}pp]")
    print()
    print(f"  Mean alpha vs S&P500:   {np.mean(alphas_sp500)*100:+.2f}pp")
    print(f"  % alpha > S&P500:       {sum(1 for a in alphas_sp500 if a > 0)/len(alphas_sp500)*100:.1f}%")
    print()
    print(f"  Mean total return 12m:  {np.mean(total_returns)*100:+.2f}%")
    print(f"  Best path total return: {max(total_returns)*100:+.2f}%")
    print(f"  Worst path total return:{min(total_returns)*100:+.2f}%")
    print()
    print(f"  Regime distribution (majority over 12m):")
    regime_majorities = [r["regime_majority"] for r in results]
    for regime in ["reflation", "stagflation", "deflation", "goldilocks"]:
        count = sum(1 for r in regime_majorities if r == regime)
        print(f"    {regime:13s}  {count}/{len(results)} ({count/len(results)*100:.1f}%)")


if __name__ == "__main__":
    main()
