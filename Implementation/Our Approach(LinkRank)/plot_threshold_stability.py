#!/usr/bin/env python3
"""
Plot threshold stability for T4 using existing saved data.

Sources:
  - threshold_stability.json  : 5 datasets (pytorch/beam/dubbo/iceberg/datafusion)
                                  per-fold τ and γ from the actual v6+Gemma runs
  - sensitivity_data.json     : MXNet per-fold τ and γ from the sensitivity run

Outputs:
  - threshold_stability_plot.png   : errorbar plot (τ and γ per project) — for paper
  - threshold_stability_table.txt  : LaTeX table snippet
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR  = Path(__file__).resolve().parents[2] / "results" / "sensitivity_analysis"
STAB_F   = OUT_DIR / "threshold_stability.json"
SENS_F   = OUT_DIR / "sensitivity_data.json"

NICE = {
    "pytorch":    "PyTorch\n(291)",
    "beam":       "Beam\n(671)",
    "dubbo":      "Dubbo\n(469)",
    "iceberg":    "Iceberg\n(551)",
    "datafusion": "DataFusion\n(738)",
    "mxnet":      "MXNet\n(383)",
}

ORDER = ["beam", "dubbo", "iceberg", "mxnet", "pytorch", "datafusion"]

COLORS = {
    "beam":       "#2196F3",
    "dubbo":      "#F44336",
    "iceberg":    "#4CAF50",
    "pytorch":    "#FF9800",
    "datafusion": "#9C27B0",
    "mxnet":      "#00BCD4",
}

# ── load data ──────────────────────────────────────────────────────────────
with open(STAB_F) as f:
    stab = json.load(f)

with open(SENS_F) as f:
    sens = json.load(f)

# Build unified per-fold dict: ds → {tau_per_fold, gamma_per_fold}
data = {}

for ds, v in stab.items():
    g = v["gemma"]   # use Gemma config (primary results)
    data[ds] = {
        "tau_per_fold":   g["tau_per_fold"],
        "gamma_per_fold": g["gamma_per_fold"],
        "tau_mean":       g["tau_mean"],
        "tau_std":        g["tau_std"],
        "gamma_mean":     g["gamma_mean"],
        "gamma_std":      g["gamma_std"],
    }

# MXNet from sensitivity_data.json — we have per-fold tau/gamma from the terminal output
# Hardcoded from run output:
#   Fold1: τ=0.79, γ=0.53
#   Fold2: τ=0.97, γ=0.94
#   Fold3: τ=0.91, γ=0.76
#   Fold4: τ=0.88, γ=0.80
#   Fold5: τ=0.87, γ=0.51
mx_tau   = [0.79, 0.97, 0.91, 0.88, 0.87]
mx_gamma = [0.53, 0.94, 0.76, 0.80, 0.51]
data["mxnet"] = {
    "tau_per_fold":   mx_tau,
    "gamma_per_fold": mx_gamma,
    "tau_mean":       float(np.mean(mx_tau)),
    "tau_std":        float(np.std(mx_tau)),
    "gamma_mean":     float(np.mean(mx_gamma)),
    "gamma_std":      float(np.std(mx_gamma)),
}

# ── Figure 1: τ and γ stability errorbars per project ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

x = np.arange(len(ORDER))
width = 0.5

for ax_idx, (threshold, label, fmt) in enumerate([
        ("tau",   "τ  (ABS-mm normalised cutoff)", "τ"),
        ("gamma", "γ  (REL fraction of max score)", "γ"),
]):
    ax = axes[ax_idx]
    means = [data[ds][f"{threshold}_mean"]   for ds in ORDER]
    stds  = [data[ds][f"{threshold}_std"]    for ds in ORDER]
    colors = [COLORS[ds] for ds in ORDER]

    bars = ax.bar(x, means, width=width, color=colors,
                  alpha=0.82, edgecolor="white", linewidth=0.8, zorder=2)
    ax.errorbar(x, means, yerr=stds, fmt="none", color="black",
                capsize=5, capthick=1.5, linewidth=1.5, zorder=3)

    # Scatter individual fold dots
    for i, ds in enumerate(ORDER):
        folds = data[ds][f"{threshold}_per_fold"]
        ax.scatter([i] * len(folds), folds,
                   color="black", s=18, zorder=4, alpha=0.55)

    ax.set_xticks(x)
    ax.set_xticklabels([NICE[ds] for ds in ORDER], fontsize=9)
    ax.set_ylabel(f"Optimal {fmt} value", fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_title(f"Stability of optimal {fmt} across 5 folds\n"
                 f"(bar = mean, error bar = ±1 std, dots = per-fold values)",
                 fontsize=11)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.axhline(0.5, color="gray", linewidth=0.7, linestyle="--", alpha=0.5)

    # Annotate mean ± std
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.03, f"{m:.2f}±{s:.2f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold")

fig.suptitle(
    "LinkRank v6 — Threshold Stability across 5 Folds (K≤7, 6 Projects, with Gemma)",
    fontsize=13, fontweight="bold", y=1.01
)
plt.tight_layout()
out1 = OUT_DIR / "threshold_stability_plot.png"
fig.savefig(out1, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out1}")

# ── Figure 2: F1 vs threshold sweep curves (all 6 datasets) ───────────────
tau_grid   = np.array(sens["tau_grid"])
gamma_grid = np.array(sens["gamma_grid"])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ds in ORDER:
    if ds not in sens["datasets"]:
        continue
    res = sens["datasets"][ds]
    c    = COLORS[ds]
    mu_t = np.array(res["tau_mean"]);   std_t = np.array(res["tau_std"])
    mu_g = np.array(res["gamma_mean"]); std_g = np.array(res["gamma_std"])
    name = NICE[ds].replace("\n", " ")

    axes[0].plot(tau_grid,   mu_t, color=c, lw=2, label=name)
    axes[0].fill_between(tau_grid,   mu_t - std_t, mu_t + std_t, color=c, alpha=0.13)

    axes[1].plot(gamma_grid, mu_g, color=c, lw=2, label=name)
    axes[1].fill_between(gamma_grid, mu_g - std_g, mu_g + std_g, color=c, alpha=0.13)

    # Mark the best threshold with a vertical dashed line + dot
    tau_best  = res.get("tau_best")
    gamma_best = res.get("gamma_best")
    if tau_best is not None:
        best_f1 = mu_t[np.argmin(np.abs(tau_grid - tau_best))]
        axes[0].axvline(tau_best, color=c, lw=0.9, linestyle="--", alpha=0.55)
        axes[0].scatter([tau_best], [best_f1], color=c, s=40, zorder=5)
    if gamma_best is not None:
        best_f1 = mu_g[np.argmin(np.abs(gamma_grid - gamma_best))]
        axes[1].axvline(gamma_best, color=c, lw=0.9, linestyle="--", alpha=0.55)
        axes[1].scatter([gamma_best], [best_f1], color=c, s=40, zorder=5)

for ax, xl, title in [
    (axes[0], "τ  (ABS-mm normalised cutoff)", "F1 vs τ  (ABS-mm stopping rule)"),
    (axes[1], "γ  (fraction of max score)",    "F1 vs γ  (REL stopping rule)"),
]:
    ax.set_xlabel(xl, fontsize=12)
    ax.set_ylabel("Macro-averaged F1 (%)", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower left")

fig.suptitle(
    "F1 vs Threshold — All 6 Datasets  (shaded = ±1 std across 5 folds,  dot = optimal threshold)",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
out2 = OUT_DIR / "f1_vs_threshold_curves.png"
fig.savefig(out2, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out2}")

# ── LaTeX table ────────────────────────────────────────────────────────────
tex_rows = []
for ds in ORDER:
    d = data[ds]
    name = NICE[ds].replace("\n", " ")
    tr = d["tau_range"]   if "tau_range"   in d else max(d["tau_per_fold"])   - min(d["tau_per_fold"])
    gr = d["gamma_range"] if "gamma_range" in d else max(d["gamma_per_fold"]) - min(d["gamma_per_fold"])
    tex_rows.append(
        f"  {name:<25} & {d['tau_mean']:.2f} $\\pm$ {d['tau_std']:.2f} & {tr:.2f} "
        f"& {d['gamma_mean']:.2f} $\\pm$ {d['gamma_std']:.2f} & {gr:.2f} \\\\"
    )

tex = (
    "\\begin{table}[t]\n"
    "\\centering\n"
    "\\caption{Threshold stability across 5 folds. "
    "Mean $\\pm$ std and range of optimal $\\tau$ (ABS-mm) "
    "and $\\gamma$ (REL) tuned on each validation fold.}\n"
    "\\label{tab:threshold_stability}\n"
    "\\begin{tabular}{lcccc}\n"
    "\\toprule\n"
    "Dataset & $\\bar{\\tau} \\pm \\sigma$ & $\\tau_{\\text{range}}$ "
    "& $\\bar{\\gamma} \\pm \\sigma$ & $\\gamma_{\\text{range}}$ \\\\\n"
    "\\midrule\n"
    + "\n".join(tex_rows) + "\n"
    "\\bottomrule\n"
    "\\end{tabular}\n"
    "\\end{table}"
)

out3 = OUT_DIR / "threshold_stability_table.tex"
with open(out3, "w") as f:
    f.write(tex)
print(f"Saved: {out3}")

# ── Console summary ────────────────────────────────────────────────────────
print("\n" + "="*72)
print(f"{'Dataset':<16} {'τ mean±std':>14} {'τ range':>10} {'γ mean±std':>14} {'γ range':>10}")
print("-"*72)
for ds in ORDER:
    d = data[ds]
    tr = max(d["tau_per_fold"]) - min(d["tau_per_fold"])
    gr = max(d["gamma_per_fold"]) - min(d["gamma_per_fold"])
    print(f"{NICE[ds].replace(chr(10),' '):<16} "
          f"{d['tau_mean']:.2f}±{d['tau_std']:.2f}   {tr:>6.2f}     "
          f"{d['gamma_mean']:.2f}±{d['gamma_std']:.2f}   {gr:>6.2f}")
print("="*72)
print(f"\nKey finding: τ ranges {min(max(data[ds]['tau_per_fold'])-min(data[ds]['tau_per_fold']) for ds in ORDER):.2f}–"
      f"{max(max(data[ds]['tau_per_fold'])-min(data[ds]['tau_per_fold']) for ds in ORDER):.2f} across projects; "
      f"γ ranges {min(max(data[ds]['gamma_per_fold'])-min(data[ds]['gamma_per_fold']) for ds in ORDER):.2f}–"
      f"{max(max(data[ds]['gamma_per_fold'])-min(data[ds]['gamma_per_fold']) for ds in ORDER):.2f}")
