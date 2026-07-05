#!/usr/bin/env python3
"""
Sensitivity Analysis for τ (ABS-mm) and γ (REL) thresholds — LinkRank T4
=========================================================================
Sweeps τ ∈ [0, 1] and γ ∈ [0, 1] in 100 steps over all 5 folds for each
project, records macro-averaged F1 at every threshold, and produces
publication-quality plots.

Usage:
    python sensitivity_analysis.py                  # all 5 projects
    python sensitivity_analysis.py --dataset pytorch # single project
    python sensitivity_analysis.py --help

Outputs (in results/sensitivity_analysis/):
    tau_sensitivity.png   — F1 vs τ  (ABS-mm), one line per project
    gamma_sensitivity.png — F1 vs γ  (REL),    one line per project
    sensitivity_data.json — raw data for LaTeX re-plotting
"""

import os, sys, json, logging, warnings, argparse, random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Optional SBERT / BM25 — no Gemma, no Optuna for speed
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED)

BASE       = Path(__file__).resolve().parents[2]
K7_DIR     = BASE / "Dataset"
OUT_DIR    = BASE / "results" / "sensitivity_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["beam", "dubbo", "iceberg", "pytorch", "datafusion"]

# LambdaMART — fixed (no Optuna) for speed
LGBM_PARAMS = dict(
    objective        = "lambdarank",
    metric           = "ndcg",
    ndcg_eval_at     = [5, 10],
    learning_rate    = 0.05,
    num_leaves       = 63,
    min_data_in_leaf = 10,
    n_estimators     = 300,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    random_state     = SEED,
    n_jobs           = -1,
    verbosity        = -1,
)

# Threshold sweep grid
TAU_GRID   = np.linspace(0.0, 1.0, 101)   # 0.00, 0.01, … 1.00
GAMMA_GRID = np.linspace(0.01, 1.0, 100)  # 0.01 … 1.00

SBERT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# ── text helpers ─────────────────────────────────────────────────────

def clean(t) -> str:
    import re
    return re.sub(r"\s+", " ", str(t) if pd.notna(t) else "").strip()

def issue_text(r):
    return f"{clean(r.get('Title',''))} {clean(r.get('Description',''))}".strip()

def commit_full(r):
    parts = [clean(r.get("Message", "")),
             clean(r.get("Diff Summary", "")),
             clean(r.get("Full Diff", ""))[:2000]]
    return " ".join(p for p in parts if p)

def commit_nl(r):
    return f"{clean(r.get('Message',''))} {clean(r.get('Diff Summary',''))}".strip()

def commit_code(r):
    return clean(r.get("Full Diff", ""))[:2000]

# ── feature extraction ───────────────────────────────────────────────

def build_features(issues_df, commits_df, links_df,
                   train_issue_ids, eval_issue_ids,
                   sbert_model):
    """
    Returns (X_train, y_train, qids_train, X_eval, issue_rank_data)
    where issue_rank_data[iid] = list of (commit_id, y_true) in scoring order.
    """
    log.info("    Building text caches …")
    i_text  = {r["Issue ID"]: issue_text(r)  for _, r in issues_df.iterrows()}
    c_full  = {r["Commit ID"]: commit_full(r) for _, r in commits_df.iterrows()}
    c_nl    = {r["Commit ID"]: commit_nl(r)   for _, r in commits_df.iterrows()}
    c_code  = {r["Commit ID"]: commit_code(r) for _, r in commits_df.iterrows()}

    all_cids = commits_df["Commit ID"].tolist()
    pos_map  = defaultdict(set)
    for _, r in links_df[links_df["Output"] == 1].iterrows():
        pos_map[r["Issue ID"]].add(r["Commit ID"])
    cand_map = defaultdict(list)      # issue → [(cid, label)]
    for _, r in links_df.iterrows():
        cand_map[r["Issue ID"]].append((r["Commit ID"], int(r["Output"])))

    # ── SBERT ───────────────────────────────────────────────────────
    log.info("    Encoding with SBERT …")
    all_i_texts = [i_text.get(iid, "") for iid in
                   list(train_issue_ids) + list(eval_issue_ids)]
    all_c_full  = [c_full.get(cid, "") for cid in all_cids]
    all_c_nl    = [c_nl.get(cid, "")   for cid in all_cids]

    sbert_i  = sbert_model.encode(all_i_texts,   batch_size=128, show_progress_bar=False, normalize_embeddings=True)
    sbert_cf = sbert_model.encode(all_c_full,     batch_size=128, show_progress_bar=False, normalize_embeddings=True)
    sbert_cn = sbert_model.encode(all_c_nl,       batch_size=128, show_progress_bar=False, normalize_embeddings=True)

    n_train = len(train_issue_ids)
    sbert_i_train = sbert_i[:n_train]
    sbert_i_eval  = sbert_i[n_train:]
    sbert_cf_map  = dict(zip(all_cids, sbert_cf))
    sbert_cn_map  = dict(zip(all_cids, sbert_cn))

    # ── TF-IDF + SVD ────────────────────────────────────────────────
    log.info("    Fitting TF-IDF + SVD on train corpus …")
    train_corpus = ([i_text.get(iid,"") for iid in train_issue_ids] +
                    [c_full.get(cid,"") for cid, _ in
                     [p for iid in train_issue_ids for p in cand_map[iid]]])
    tfidf = TfidfVectorizer(max_features=30000, sublinear_tf=True)
    tfidf.fit(train_corpus)

    def svd_sim(a_vec, b_vec):
        svd = TruncatedSVD(n_components=50, random_state=SEED)
        svd.fit(np.vstack([a_vec, b_vec]))
        a2 = svd.transform(a_vec);  b2 = svd.transform(b_vec)
        a2 /= (np.linalg.norm(a2, axis=1, keepdims=True) + 1e-9)
        b2 /= (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-9)
        return (a2 * b2).sum(axis=1)

    # ── BM25 ────────────────────────────────────────────────────────
    log.info("    Building BM25 corpus on train commits …")
    train_cids = list({cid for iid in train_issue_ids
                       for cid, _ in cand_map[iid]})
    bm25_docs   = [c_nl.get(cid, "").split() for cid in train_cids]
    bm25_idx    = BM25Okapi(bm25_docs)
    bm25_cid_pos = {cid: idx for idx, cid in enumerate(train_cids)}

    # ── build row vectors ────────────────────────────────────────────
    def make_rows(issue_ids, sbert_i_vecs):
        rows, labels, qids = [], [], []
        issue_rank_data = {}   # used only for eval
        for qidx, (iid, si) in enumerate(zip(issue_ids, sbert_i_vecs)):
            pairs = cand_map.get(iid, [])
            if not pairs:
                continue
            cids_here   = [cid for cid, _ in pairs]
            labels_here = [lbl for _, lbl in pairs]
            it = i_text.get(iid, "")

            # SBERT features
            sf  = np.array([sbert_cf_map.get(cid, np.zeros(768)) for cid in cids_here])
            sn  = np.array([sbert_cn_map.get(cid, np.zeros(768)) for cid in cids_here])
            sbert_full_sim  = (si * sf).sum(axis=1)
            sbert_nl_sim    = (si * sn).sum(axis=1)

            # SVD features (fit on train corpus — reuse tfidf)
            i_tfidf = tfidf.transform([it])
            c_tfidf = tfidf.transform([c_full.get(c,"") for c in cids_here])
            c_nl_tfidf = tfidf.transform([c_nl.get(c,"") for c in cids_here])
            c_code_tfidf = tfidf.transform([c_code.get(c,"") for c in cids_here])

            def cosine_sparse(a, B):
                """a: (1,V) sparse, B: (n,V) sparse → (n,) float64. Uses sklearn for correctness."""
                return sk_cosine(a, B).ravel()

            svd_full = cosine_sparse(i_tfidf, c_tfidf)
            svd_nl   = cosine_sparse(i_tfidf, c_nl_tfidf)
            svd_code = cosine_sparse(i_tfidf, c_code_tfidf)

            # BM25 features
            query_tok = it.split()
            scores_bm25 = bm25_idx.get_scores(query_tok)
            bm25_full = np.array([scores_bm25[bm25_cid_pos[c]]
                                  if c in bm25_cid_pos else 0.0
                                  for c in cids_here])
            bm25_max = bm25_full.max() if bm25_full.max() > 0 else 1.0
            bm25_full = bm25_full / bm25_max

            # Meta features
            i_date = issues_df.loc[issues_df["Issue ID"]==iid, "Issue Date"]
            i_ts   = pd.to_datetime(i_date.values[0], errors="coerce") if len(i_date) else None
            feat_time, feat_time_dir = [], []
            for cid in cids_here:
                c_date = commits_df.loc[commits_df["Commit ID"]==cid, "Commit Date"]
                c_ts   = pd.to_datetime(c_date.values[0], errors="coerce") if len(c_date) else None
                if i_ts and c_ts and pd.notna(i_ts) and pd.notna(c_ts):
                    delta = (c_ts - i_ts).total_seconds() / 86400
                    feat_time.append(abs(delta))
                    feat_time_dir.append(1.0 if delta >= 0 else -1.0)
                else:
                    feat_time.append(0.0)
                    feat_time_dir.append(0.0)
            feat_time     = np.array(feat_time)
            feat_time_dir = np.array(feat_time_dir)

            desc_len = len(it.split())
            feat_desc_len = np.full(len(cids_here), np.log1p(desc_len))

            feat_diff_size, feat_nfiles, feat_file_bn_jaccard = [], [], []
            i_words = set(it.lower().split())
            for cid in cids_here:
                diff = c_code.get(cid, "")
                feat_diff_size.append(np.log1p(len(diff.split())))
                feat_nfiles.append(np.log1p(diff.count("\n--- a/")))
                files_in_diff = {w for w in diff.lower().split() if "." in w and "/" in w}
                jac = len(i_words & files_in_diff) / (len(i_words | files_in_diff) + 1e-9)
                feat_file_bn_jaccard.append(jac)
            feat_diff_size         = np.array(feat_diff_size)
            feat_nfiles            = np.array(feat_nfiles)
            feat_file_bn_jaccard   = np.array(feat_file_bn_jaccard)

            n = len(cids_here)
            X_block = np.column_stack([
                sbert_full_sim, sbert_nl_sim,
                svd_full, svd_nl, svd_code,
                bm25_full,
                feat_time, feat_time_dir,
                feat_desc_len,
                feat_diff_size, feat_nfiles, feat_file_bn_jaccard,
            ])  # 12 features (fast version, no Gemma)

            rows.extend(X_block)
            labels.extend(labels_here)
            qids.extend([qidx] * n)

            if issue_ids is not issue_ids:  # never — just placeholder
                pass
            issue_rank_data[iid] = list(zip(cids_here, labels_here))

        return (np.array(rows), np.array(labels), np.array(qids),
                issue_rank_data)

    log.info("    Building training rows …")
    train_iids = list(train_issue_ids)
    eval_iids  = list(eval_issue_ids)
    X_tr, y_tr, q_tr, _ = make_rows(train_iids, sbert_i_train)
    log.info("    Building eval rows …")
    X_ev, y_ev, q_ev, eval_rank = make_rows(eval_iids, sbert_i_eval)

    return X_tr, y_tr, q_tr, X_ev, y_ev, q_ev, eval_rank, eval_iids


# ── metrics at threshold ──────────────────────────────────────────────

def f1_at_tau(ranked_per_issue, tau_values):
    """ranked_per_issue: {iid: [(score, label), ...] sorted desc}"""
    f1s = np.zeros(len(tau_values))
    for ranked in ranked_per_issue.values():
        if not ranked:
            continue
        scores = np.array([s for s, _ in ranked])
        labels = np.array([l for _, l in ranked])
        K = labels.sum()
        if K == 0:
            continue
        s_min, s_max = scores.min(), scores.max()
        if s_max == s_min:
            continue
        norm_scores = (scores - s_min) / (s_max - s_min)
        for ti, tau in enumerate(tau_values):
            sel = norm_scores >= tau
            tp = (sel & (labels == 1)).sum()
            fp = (sel & (labels == 0)).sum()
            fn = K - tp
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1s[ti] += (2 * p * r / (p + r)) if (p + r) > 0 else 0
    return f1s / max(len(ranked_per_issue), 1) * 100


def f1_at_gamma(ranked_per_issue, gamma_values):
    """ranked_per_issue: {iid: [(score, label), ...] sorted desc}"""
    f1s = np.zeros(len(gamma_values))
    for ranked in ranked_per_issue.values():
        if not ranked:
            continue
        scores = np.array([s for s, _ in ranked])
        labels = np.array([l for _, l in ranked])
        K = labels.sum()
        if K == 0 or scores.max() == 0:
            continue
        for gi, gamma in enumerate(gamma_values):
            thr = gamma * scores.max()
            sel = scores >= thr
            tp = (sel & (labels == 1)).sum()
            fp = (sel & (labels == 0)).sum()
            fn = K - tp
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1s[gi] += (2 * p * r / (p + r)) if (p + r) > 0 else 0
    return f1s / max(len(ranked_per_issue), 1) * 100


# ── 5-fold split ─────────────────────────────────────────────────────

def make_5fold_splits(issues_df, links_df):
    pos = links_df[links_df["Output"] == 1]
    k_map = pos.groupby("Issue ID").size()
    buckets = defaultdict(list)
    for _, row in issues_df.iterrows():
        buckets[k_map.get(row["Issue ID"], 0)].append(row["Issue ID"])
    fold_issues = [[] for _ in range(5)]
    for k in sorted(buckets):
        iids = buckets[k]; random.shuffle(iids)
        for idx, iid in enumerate(iids):
            fold_issues[idx % 5].append(iid)
    splits = []
    for test_fold in range(5):
        test_ids  = set(fold_issues[test_fold])
        train_ids = set()
        for fi in range(5):
            if fi != test_fold:
                train_ids.update(fold_issues[fi])
        splits.append((train_ids, test_ids))
    return splits


# ── main per-dataset routine ──────────────────────────────────────────

def run_dataset(dataset: str, sbert_model) -> dict:
    log.info(f"\n{'='*60}\n  Dataset: {dataset}\n{'='*60}")
    data_dir = K7_DIR / dataset
    issues_df  = pd.read_csv(data_dir / "rds_issues.csv")
    commits_df = pd.read_csv(data_dir / "rds_commits.csv")
    links_df   = pd.read_csv(data_dir / "rds_links.csv")
    log.info(f"  {len(issues_df)} issues, {len(commits_df)} commits, {len(links_df)} links")

    splits = make_5fold_splits(issues_df, links_df)
    tau_curves   = []
    gamma_curves = []

    for fold_idx, (train_ids, test_ids) in enumerate(splits):
        log.info(f"\n  Fold {fold_idx+1}/5 — train {len(train_ids)}, test {len(test_ids)}")
        (X_tr, y_tr, q_tr,
         X_ev, y_ev, q_ev,
         eval_rank, eval_iids) = build_features(
            issues_df, commits_df, links_df,
            train_ids, test_ids, sbert_model)

        if len(X_tr) == 0 or len(X_ev) == 0:
            log.warning("  Empty split — skipping fold")
            continue

        # Train LambdaMART
        log.info("    Training LambdaMART …")
        model = lgb.LGBMRanker(**LGBM_PARAMS)
        model.fit(X_tr, y_tr, group=np.bincount(q_tr),
                  eval_set=[(X_ev, y_ev)],
                  eval_group=[np.bincount(q_ev)],
                  callbacks=[lgb.early_stopping(30, verbose=False),
                              lgb.log_evaluation(-1)])

        # Score test issues
        scores_flat = model.predict(X_ev)

        # Reassemble per-issue ranked lists
        ranked_issues = {}
        ptr = 0
        for iid in eval_iids:
            pairs = eval_rank.get(iid, [])
            n = len(pairs)
            if n == 0:
                continue
            s = scores_flat[ptr:ptr+n]
            labeled = sorted(zip(s.tolist(),
                                 [lbl for _, lbl in pairs]),
                             key=lambda x: -x[0])
            ranked_issues[iid] = labeled
            ptr += n

        tau_f1   = f1_at_tau(ranked_issues,   TAU_GRID)
        gamma_f1 = f1_at_gamma(ranked_issues, GAMMA_GRID)
        tau_curves.append(tau_f1)
        gamma_curves.append(gamma_f1)
        log.info(f"    Best τ-F1: {tau_f1.max():.2f} @ τ={TAU_GRID[tau_f1.argmax()]:.2f} | "
                 f"Best γ-F1: {gamma_f1.max():.2f} @ γ={GAMMA_GRID[gamma_f1.argmax()]:.2f}")

    mean_tau   = np.mean(tau_curves,   axis=0) if tau_curves   else np.zeros(len(TAU_GRID))
    std_tau    = np.std(tau_curves,    axis=0) if tau_curves    else np.zeros(len(TAU_GRID))
    mean_gamma = np.mean(gamma_curves, axis=0) if gamma_curves else np.zeros(len(GAMMA_GRID))
    std_gamma  = np.std(gamma_curves,  axis=0) if gamma_curves else np.zeros(len(GAMMA_GRID))

    return {
        "tau_mean":   mean_tau.tolist(),
        "tau_std":    std_tau.tolist(),
        "gamma_mean": mean_gamma.tolist(),
        "gamma_std":  std_gamma.tolist(),
        "tau_best":   float(TAU_GRID[mean_tau.argmax()]),
        "gamma_best": float(GAMMA_GRID[mean_gamma.argmax()]),
    }


# ── plotting ──────────────────────────────────────────────────────────

COLORS = {
    "beam":       "#2196F3",   # blue
    "dubbo":      "#F44336",   # red
    "iceberg":    "#4CAF50",   # green
    "pytorch":    "#FF9800",   # orange
    "datafusion": "#9C27B0",   # purple
    "mxnet":      "#00BCD4",   # cyan
}

NICE_NAMES = {
    "beam": "Apache Beam", "dubbo": "Apache Dubbo",
    "iceberg": "Apache Iceberg", "pytorch": "PyTorch",
    "datafusion": "Apache DataFusion", "mxnet": "MXNet",
}

def make_plots(all_results: dict):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── left: τ sensitivity (ABS-mm) ────────────────────────────────
    ax = axes[0]
    for ds, res in all_results.items():
        mu  = np.array(res["tau_mean"])
        std = np.array(res["tau_std"])
        c   = COLORS.get(ds, "#555555")
        ax.plot(TAU_GRID, mu, color=c, linewidth=2,
                label=f"{NICE_NAMES.get(ds, ds)} (peak {mu.max():.1f}%)")
        ax.fill_between(TAU_GRID, mu - std, mu + std, color=c, alpha=0.12)
        ax.axvline(res["tau_best"], color=c, linewidth=0.8, linestyle="--", alpha=0.7)

    ax.set_xlabel("Threshold τ (ABS-mm normalised score cutoff)", fontsize=12)
    ax.set_ylabel("Macro-averaged F1 (%)", fontsize=12)
    ax.set_title("Sensitivity of ABS-mm F1 to τ\n(shaded = ±1 std across 5 folds)", fontsize=12)
    ax.set_xlim(0, 1); ax.set_ylim(0, 100)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── right: γ sensitivity (REL) ───────────────────────────────────
    ax = axes[1]
    for ds, res in all_results.items():
        mu  = np.array(res["gamma_mean"])
        std = np.array(res["gamma_std"])
        c   = COLORS.get(ds, "#555555")
        ax.plot(GAMMA_GRID, mu, color=c, linewidth=2,
                label=f"{NICE_NAMES.get(ds, ds)} (peak {mu.max():.1f}%)")
        ax.fill_between(GAMMA_GRID, mu - std, mu + std, color=c, alpha=0.12)
        ax.axvline(res["gamma_best"], color=c, linewidth=0.8, linestyle="--", alpha=0.7)

    ax.set_xlabel("Threshold γ (fraction of max score)", fontsize=12)
    ax.set_ylabel("Macro-averaged F1 (%)", fontsize=12)
    ax.set_title("Sensitivity of REL F1 to γ\n(shaded = ±1 std across 5 folds)", fontsize=12)
    ax.set_xlim(0, 1); ax.set_ylim(0, 100)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.suptitle("LinkRank v6 — Threshold Sensitivity Analysis (K≤7, 5-Fold CV, No Gemma)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = OUT_DIR / "threshold_sensitivity.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    log.info(f"  Saved plot: {out}")
    plt.close(fig)

    # ── per-project plots ────────────────────────────────────────────
    fig, axes = plt.subplots(len(all_results), 2,
                              figsize=(13, 4 * len(all_results)))
    if len(all_results) == 1:
        axes = [axes]

    for row_idx, (ds, res) in enumerate(all_results.items()):
        c     = COLORS.get(ds, "#555555")
        name  = NICE_NAMES.get(ds, ds)
        mu_t  = np.array(res["tau_mean"]);   std_t  = np.array(res["tau_std"])
        mu_g  = np.array(res["gamma_mean"]); std_g  = np.array(res["gamma_std"])

        ax = axes[row_idx][0]
        ax.plot(TAU_GRID, mu_t, color=c, linewidth=2)
        ax.fill_between(TAU_GRID, mu_t - std_t, mu_t + std_t, color=c, alpha=0.15)
        ax.axvline(res["tau_best"], color="black", linewidth=1.2, linestyle="--",
                   label=f"optimal τ = {res['tau_best']:.2f}")
        ax.set_title(f"{name} — ABS-mm F1 vs τ", fontsize=11)
        ax.set_xlabel("τ"); ax.set_ylabel("F1 (%)"); ax.legend(fontsize=9)
        ax.set_xlim(0, 1); ax.grid(True, alpha=0.3)

        ax = axes[row_idx][1]
        ax.plot(GAMMA_GRID, mu_g, color=c, linewidth=2)
        ax.fill_between(GAMMA_GRID, mu_g - std_g, mu_g + std_g, color=c, alpha=0.15)
        ax.axvline(res["gamma_best"], color="black", linewidth=1.2, linestyle="--",
                   label=f"optimal γ = {res['gamma_best']:.2f}")
        ax.set_title(f"{name} — REL F1 vs γ", fontsize=11)
        ax.set_xlabel("γ"); ax.set_ylabel("F1 (%)"); ax.legend(fontsize=9)
        ax.set_xlim(0, 1); ax.grid(True, alpha=0.3)

    fig.suptitle("Per-Project Threshold Sensitivity (LinkRank v6, K≤7)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out2 = OUT_DIR / "threshold_sensitivity_per_project.png"
    fig.savefig(out2, dpi=200, bbox_inches="tight")
    log.info(f"  Saved per-project plot: {out2}")
    plt.close(fig)


# ── entry point ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sensitivity analysis of τ/γ thresholds for LinkRank")
    parser.add_argument("--dataset", nargs="+", default=DATASETS,
                        choices=DATASETS + ["mxnet"],
                        help="Which datasets to run (default: all 5 main datasets)")
    args = parser.parse_args()

    log.info(f"Loading SBERT ({SBERT_MODEL_NAME}) …")
    sbert = SentenceTransformer(SBERT_MODEL_NAME)

    all_results = {}
    for ds in args.dataset:
        result = run_dataset(ds, sbert)
        all_results[ds] = result
        log.info(f"  {ds}: optimal τ={result['tau_best']:.2f} "
                 f"(F1={max(result['tau_mean']):.2f}%), "
                 f"optimal γ={result['gamma_best']:.2f} "
                 f"(F1={max(result['gamma_mean']):.2f}%)")

    # Save raw data — merge with any existing results so we never lose prior runs
    out_json = OUT_DIR / "sensitivity_data.json"
    if out_json.exists():
        with open(out_json) as f:
            existing = json.load(f)
        existing["datasets"].update(all_results)
        all_results_merged = existing["datasets"]
    else:
        all_results_merged = all_results

    with open(out_json, "w") as f:
        json.dump({
            "tau_grid":   TAU_GRID.tolist(),
            "gamma_grid": GAMMA_GRID.tolist(),
            "datasets":   all_results_merged,
        }, f, indent=2)
    log.info(f"  Saved data: {out_json}")
    all_results = all_results_merged   # use merged set for plotting

    # Generate plots
    make_plots(all_results)

    # Print summary table
    print("\n" + "="*70)
    print(f"{'Dataset':<15} {'Opt τ':>7} {'ABS-mm F1':>12} {'Opt γ':>7} {'REL F1':>10}")
    print("-"*70)
    for ds, res in all_results.items():
        tau_f1   = max(res["tau_mean"])
        gamma_f1 = max(res["gamma_mean"])
        print(f"{NICE_NAMES.get(ds,ds):<15} {res['tau_best']:>7.2f} "
              f"{tau_f1:>12.2f}% {res['gamma_best']:>7.2f} {gamma_f1:>10.2f}%")
    print("="*70)
    print(f"\nOutputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
