# LinkRank: A Learning-to-Rank Framework for One-to-Many Issue–Commit Traceability

LinkRank recovers the **complete set of commits** that resolve a software issue.
Instead of classifying issue–commit pairs independently, it treats each issue as a
query over a pool of candidate commits and recovers linked commits through
**iterative selection**: rank all candidates jointly, pick the highest-ranked commit,
remove it, renormalize the remaining scores, and repeat until a stopping criterion
is met.

Across six open-source projects (3,103 issues, 7,688 commits), LinkRank achieves
**73.17% F1** under the Known-K setting and **65.94% F1** (ABS) / **62.19% F1** (REL)
under the realistic Unknown-K setting, outperforming all baselines — including
recent LLM-based approaches — by a substantial margin.

---

## Repository structure

```
LinkRank/
├── Dataset/                          # Dataset (hosted on Zenodo — see Dataset/README.md)
├── Pre-processing/                   # Data collection & preprocessing scripts
├── Implementation/
│   └── Our Approach(LinkRank)/
│       ├── LinkRank_code.py          # ★ Main LinkRank pipeline (5-fold stratified CV)
│       ├── ablation_study.py         # Feature-group ablation (RQ3)
│       ├── sensitivity_analysis.py   # Threshold sensitivity curves (Fig. 4)
│       ├── threshold_sensitivity.py  # τ / γ stability analysis across folds
│       ├── plot_threshold_stability.py
│       ├── statistical_significance.py
│       └── wilcoxon_significance.py  # Wilcoxon signed-rank tests + Cliff's delta
├── baseline/                         # Baseline references (see baseline/README.md)
└── results/                          # All experiment outputs (summary CSV/JSON)
```

## Dataset

The dataset covers **six open-source GitHub projects** with full commit multiplicity
(K = 1..7 commits per issue) and realistic RDS candidate pools (±365-day window):

| Project | Issues | Commits | Avg. Candidate Pool | Avg K |
|---|---|---|---|---|
| Apache Beam | 671 | 1,625 | ~590 | 2.42 |
| Apache DataFusion | 738 | 2,270 | ~1,264 | 3.08 |
| Apache Dubbo | 469 | 938 | ~133 | 2.00 |
| Apache Iceberg | 551 | 1,357 | ~193 | 2.46 |
| Apache MXNet | 383 | 903 | ~427 | 2.36 |
| PyTorch | 291 | 595 | ~439 | 2.04 |
| **Total** | **3,103** | **7,688** | | **2.48** |

The dataset is hosted on **Zenodo** — download link, file schemas, and setup
instructions are in [`Dataset/README.md`](Dataset/README.md).

## Approach

LinkRank has four phases (see Section 3 of the paper):

1. **Candidate pool construction** — Relative Date Span (RDS): all repository
   commits within the ground-truth time span ±365 days form each issue's pool.
2. **Feature representation** — 17 features across four groups:
   TF-IDF+SVD (4), BM25 (4), SBERT `all-mpnet-base-v2` (3), and metadata
   (temporal proximity/direction, description length, diff size, files changed,
   file-path Jaccard) (6).
3. **Learning-to-rank** — LambdaMART (LightGBM `lambdarank`), tuned with Optuna
   (50 trials) on a dev split within each fold.
4. **Iterative selection** — pick–remove–renormalize with three stopping rules:
   **Known-K** (oracle), **ABS** (absolute threshold τ), **REL** (relative threshold γ);
   τ and γ are tuned per fold on the development set.

## Requirements

- Python 3.10+
- `lightgbm`, `sentence-transformers`, `rank-bm25`, `optuna`, `scikit-learn`,
  `pandas`, `numpy`, `scipy`, `torch`, `tqdm`

```bash
pip install lightgbm sentence-transformers rank-bm25 optuna scikit-learn pandas numpy scipy torch tqdm
```

## Usage

1. Download the dataset from Zenodo and extract it into `Dataset/`
   (see [`Dataset/README.md`](Dataset/README.md)).

2. Run the main pipeline (5-fold stratified cross-validation):

```bash
cd "Implementation/Our Approach(LinkRank)"
python LinkRank_code.py <dataset>        # dataset ∈ {pytorch, beam, datafusion, dubbo, iceberg, mxnet}
```

Results are written to `results/linkrank_v6_5fold_<dataset>_k7_no_gemma/`:
per-fold metrics, `aggregated_results.csv` (mean ± std across folds), Optuna
parameters, and fold assignments.

3. Analysis scripts (run after the main pipeline):

```bash
python wilcoxon_significance.py      # Wilcoxon signed-rank tests vs. baselines
python sensitivity_analysis.py       # τ / γ threshold-sensitivity curves
python threshold_sensitivity.py      # threshold stability across folds
python ablation_study.py             # feature-group ablation (PyTorch)
```

## Results

Known-K F1 (%) — 5-fold stratified cross-validation, mean ± std:

| Method | PyTorch | Dubbo | Iceberg | Beam | DataFusion | MXNet | Avg |
|---|---|---|---|---|---|---|---|
| EALink | 13.58 | 13.90 | 28.37 | 20.68 | 22.86 | 11.75 | 18.52 |
| MPLinker | 16.83 | 23.54 | 34.77 | 25.91 | 21.77 | 20.96 | 23.96 |
| EasyLink | 33.57 | 41.02 | 46.80 | 34.51 | 26.73 | 27.18 | 34.97 |
| LinkAnchor | 54.30 | 53.79 | 52.67 | 46.98 | 35.86 | 46.73 | 48.39 |
| **LinkRank (ours)** | **76.38** | **76.96** | **80.68** | **69.69** | **65.62** | **69.69** | **73.17** |

Unknown-K (automatic stopping): **65.94% F1** (ABS) / **62.19% F1** (REL) on average.
Ranking quality: **87.19% MRR**, **74.87% NDCG@K**, **72.74% P@K** on average.
Full per-project results for all settings and baselines are in `results/`.

## Baselines

Baseline implementations are not redistributed here; see
[`baseline/README.md`](baseline/README.md) for references and links to their
original replication packages. Baseline result summaries are included under
`results/`.


