"""
Statistical Significance Tests — LinkRank vs Baselines
=======================================================
Performs Wilcoxon signed-rank tests (one-sided: LinkRank > Baseline)
across 5-fold CV results, plus Cliff's delta effect sizes.

With n=5 folds, the minimum achievable one-sided p-value is 0.03125
(all 5 folds favoring LinkRank). We use α=0.05.

Usage:
  python statistical_significance.py

Outputs:
  - results/significance_tests.csv
  - results/significance_tests_summary.txt
"""

import json, os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# ══════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════

RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results"
DATASETS = ["pytorch", "beam", "dubbo", "iceberg", "datafusion", "mxnet"]
METRICS = ["known_k_f1", "abs_mm_f1", "rel_f1", "mrr", "ndcg_at_k", "p_at_k"]

# Map metric names between LinkRank CSVs and baseline JSONs
LINKRANK_METRIC_MAP = {
    "known_k_f1": "KnownK_F1",
    "abs_mm_f1": "ABSmm_F1",
    "rel_f1": "REL_F1",
    "mrr": "MRR",
    "ndcg_at_k": "NDCG_at_K",
    "p_at_k": "P_at_K",
}

ALPHA = 0.05  # Significance level


def load_linkrank_folds(dataset: str) -> dict:
    """Load LinkRank per-fold metrics from all_folds_results.csv."""
    path = RESULTS_ROOT / f"linkrank_v6_5fold_{dataset}_k7_no_gemma" / "all_folds_results.csv"
    if not path.exists():
        print(f"  ⚠ LinkRank results not found: {path}")
        return None
    df = pd.read_csv(path)
    result = {}
    for metric in METRICS:
        col = LINKRANK_METRIC_MAP[metric]
        if col in df.columns:
            result[metric] = df[col].values
        else:
            result[metric] = None
    return result


def load_baseline_folds(baseline: str, dataset: str) -> dict:
    """Load baseline per-fold metrics from aggregated_results.json."""
    path = RESULTS_ROOT / f"{baseline}_k7_5fold_{dataset}" / "aggregated_results.json"
    if not path.exists():
        print(f"  ⚠ Baseline results not found: {path}")
        return None
    with open(path) as f:
        data = json.load(f)

    result = {}

    # Format 1: metrics_by_fold dict (EasyLink, MPLinker)
    if "metrics_by_fold" in data and data["metrics_by_fold"]:
        for metric in METRICS:
            vals = data["metrics_by_fold"].get(metric)
            if vals:
                result[metric] = np.array(vals, dtype=float)
            else:
                result[metric] = None
        return result

    # Format 2: per_fold list of dicts (EALink)
    per_fold = data.get("per_fold", [])
    if per_fold:
        for metric in METRICS:
            vals = [fold_dict.get(metric) for fold_dict in per_fold if metric in fold_dict]
            if vals:
                result[metric] = np.array(vals, dtype=float)
            else:
                result[metric] = None
        return result

    print(f"  ⚠ Unknown JSON format in: {path}")
    return None


def cliffs_delta(a: np.ndarray, b: np.ndarray):
    """
    Compute Cliff's delta effect size (non-parametric).
    Interpretation: |d|<0.147 negligible, <0.33 small, <0.474 medium, else large.
    """
    n1, n2 = len(a), len(b)
    count = 0
    for x in a:
        for y in b:
            if x > y:
                count += 1
            elif x < y:
                count -= 1
    delta = count / (n1 * n2)
    if abs(delta) < 0.147:
        mag = "negligible"
    elif abs(delta) < 0.33:
        mag = "small"
    elif abs(delta) < 0.474:
        mag = "medium"
    else:
        mag = "large"
    return delta, mag


def wilcoxon_test(a: np.ndarray, b: np.ndarray):
    """
    Perform Wilcoxon signed-rank test (one-sided: H1: a > b).
    Returns (statistic, p_value, cliff_delta, magnitude).

    With n=5, if all differences are positive, exact p = 1/2^5 = 0.03125.
    """
    n = len(a)
    diffs = a - b
    delta, mag = cliffs_delta(a, b)

    # Remove zeros
    nonzero_diffs = diffs[diffs != 0]
    n_nonzero = len(nonzero_diffs)

    if n_nonzero == 0:
        return 0, 1.0, delta, mag

    if n_nonzero < 5:
        # Sign test fallback (one-sided)
        n_pos = np.sum(nonzero_diffs > 0)
        # P(X >= n_pos) under Binomial(n_nonzero, 0.5)
        p_val = stats.binom.sf(n_pos - 1, n_nonzero, 0.5)
        return 0, p_val, delta, mag

    try:
        stat, p_val = stats.wilcoxon(nonzero_diffs, alternative='greater')
        return stat, p_val, delta, mag
    except ValueError:
        return 0, 1.0, delta, mag


def main():
    print("=" * 70)
    print("  Statistical Significance Tests — LinkRank vs Baselines")
    print("  (One-sided Wilcoxon signed-rank: H1: LinkRank > Baseline)")
    print("=" * 70)

    baselines = ["ealink", "easylink", "mplinker"]
    all_rows = []

    for dataset in DATASETS:
        print(f"\n{'─' * 60}")
        print(f"  Dataset: {dataset}")
        print(f"{'─' * 60}")

        lr_folds = load_linkrank_folds(dataset)
        if lr_folds is None:
            continue

        for bl_name in baselines:
            bl_folds = load_baseline_folds(bl_name, dataset)
            if bl_folds is None:
                continue

            for metric in METRICS:
                lr_vals = lr_folds.get(metric)
                bl_vals = bl_folds.get(metric)

                if lr_vals is None or bl_vals is None:
                    continue

                # Ensure same number of folds
                n_folds = min(len(lr_vals), len(bl_vals))
                lr_v = lr_vals[:n_folds]
                bl_v = bl_vals[:n_folds]

                lr_mean = lr_v.mean()
                bl_mean = bl_v.mean()
                diff_mean = lr_mean - bl_mean

                stat, p_val, delta, mag = wilcoxon_test(lr_v, bl_v)
                significant = "YES" if p_val < ALPHA else "no"

                row = {
                    "dataset": dataset,
                    "baseline": bl_name,
                    "metric": metric,
                    "linkrank_mean": round(lr_mean, 2),
                    "baseline_mean": round(bl_mean, 2),
                    "difference": round(diff_mean, 2),
                    "n_folds": n_folds,
                    "statistic": round(stat, 4),
                    "p_value": round(p_val, 6),
                    "cliffs_delta": round(delta, 4),
                    "effect_magnitude": mag,
                    "significant": significant,
                }
                all_rows.append(row)

                if metric == "known_k_f1":
                    sym = "✓" if p_val < ALPHA else "✗"
                    print(f"  {bl_name:10s} | Known-K F1: LR={lr_mean:.1f} vs BL={bl_mean:.1f} | "
                          f"Δ={diff_mean:+.1f} | p={p_val:.4f} | δ={delta:.2f} ({mag}) {sym}")

    # Save results
    df = pd.DataFrame(all_rows)
    out_path = RESULTS_ROOT / "significance_tests.csv"
    df.to_csv(out_path, index=False)
    print(f"\n\nResults saved to: {out_path}")

    # Summary
    summary_path = RESULTS_ROOT / "significance_tests_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Statistical Significance Tests — LinkRank vs Baselines\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Test: Wilcoxon signed-rank (one-sided: H1: LinkRank > Baseline), α={ALPHA}\n")
        f.write(f"Effect size: Cliff's delta (|d|<0.147=negligible, <0.33=small, <0.474=medium, >=0.474=large)\n")
        f.write(f"Datasets: {', '.join(DATASETS)}\n")
        f.write(f"Metrics: {', '.join(METRICS)}\n")
        f.write(f"Note: With n=5 folds, minimum achievable p = 0.03125 (all folds consistent)\n\n")

        # Count significant results
        n_total = len(df)
        n_sig = (df["significant"] == "YES").sum()
        f.write(f"Total comparisons: {n_total}\n")
        f.write(f"Significant (p<{ALPHA}): {n_sig} ({100*n_sig/max(n_total,1):.0f}%)\n\n")

        # Per-baseline summary
        for bl in baselines:
            bl_df = df[df["baseline"] == bl]
            n_bl = len(bl_df)
            if n_bl == 0:
                continue
            n_bl_sig = (bl_df["significant"] == "YES").sum()
            avg_diff = bl_df["difference"].mean()
            f.write(f"\n{bl.upper()} ({n_bl} comparisons):\n")
            f.write(f"  Significant: {n_bl_sig}/{n_bl}\n")
            f.write(f"  Avg improvement: {avg_diff:+.2f}\n")
            f.write(f"  Effect sizes: {bl_df['effect_magnitude'].value_counts().to_dict()}\n")

            # Per-metric
            for metric in METRICS:
                m_df = bl_df[bl_df["metric"] == metric]
                if len(m_df) == 0:
                    continue
                n_m_sig = (m_df["significant"] == "YES").sum()
                avg_m_diff = m_df["difference"].mean()
                avg_delta = m_df["cliffs_delta"].mean()
                f.write(f"    {metric:14s}: {n_m_sig}/{len(m_df)} sig, "
                        f"avg Δ={avg_m_diff:+.2f}, avg δ={avg_delta:.2f}\n")

        # Detailed table
        f.write(f"\n\n{'='*70}\n")
        f.write("Detailed Results\n")
        f.write(f"{'='*70}\n\n")
        f.write(df.to_string(index=False))

    print(f"Summary saved to: {summary_path}")

    # Print key results table
    print(f"\n{'='*70}")
    print(f"  SUMMARY: LinkRank vs Baselines (Known-K F1, one-sided Wilcoxon)")
    print(f"{'='*70}")
    kf1 = df[df["metric"] == "known_k_f1"]
    if len(kf1) > 0:
        print(f"\n  {'Dataset':<12} | {'Baseline':<10} | {'LR':>5} | {'BL':>5} | {'Δ':>6} | {'p':>7} | {'δ':>5} | {'Effect':>8} | Sig?")
        print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*5}-+-{'-'*5}-+-{'-'*6}-+-{'-'*7}-+-{'-'*5}-+-{'-'*8}-+-----")
        for _, row in kf1.iterrows():
            sym = "✓" if row["significant"] == "YES" else ""
            print(f"  {row['dataset']:<12} | {row['baseline']:<10} | "
                  f"{row['linkrank_mean']:>5.1f} | {row['baseline_mean']:>5.1f} | "
                  f"{row['difference']:>+6.1f} | {row['p_value']:>7.4f} | "
                  f"{row['cliffs_delta']:>5.2f} | {row['effect_magnitude']:>8} | {sym}")

    # Also show per-metric summary
    print(f"\n{'='*70}")
    print(f"  EFFECT SIZES across all datasets (Cliff's delta)")
    print(f"{'='*70}")
    for metric in METRICS:
        m_df = df[df["metric"] == metric]
        if len(m_df) == 0:
            continue
        n_sig = (m_df["significant"] == "YES").sum()
        n_large = (m_df["effect_magnitude"] == "large").sum()
        avg_delta = m_df["cliffs_delta"].mean()
        print(f"  {metric:14s}: {n_sig}/{len(m_df)} significant, "
              f"{n_large}/{len(m_df)} large effect, avg δ={avg_delta:.3f}")


if __name__ == "__main__":
    main()
