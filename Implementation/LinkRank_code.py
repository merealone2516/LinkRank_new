"""
LinkRank v6 — 5-Fold Stratified Cross-Validation
==================================================
Same pipeline as Linkrank_v6.py (all 22 features, Optuna-tuned LambdaMART),
but with proper 5-fold CV stratified by K (number of true commits per issue).

Each fold:
  - Train on 4/5 of issues, test on 1/5
  - SBERT/BM25 embeddings computed on ALL data (unsupervised)
  - TF-IDF/SVD fitted on TRAIN only, transformed on TEST
  - Optuna tunes LambdaMART on train-dev split (20% of train)
  - Evaluate Known-K, ABS-mm, REL with iterative refinement

Reports: per-fold results + mean ± std across 5 folds.
"""

import os, re, time, logging, math, json, sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import timedelta
from rank_bm25 import BM25Okapi
import optuna

try:
    import psutil
    def mem_gb(): return psutil.Process(os.getpid()).memory_info().rss / (1024**3)
except Exception:
    psutil = None
    def mem_gb(): return float('nan')

import torch
TORCH_HAS_CUDA = torch.cuda.is_available()
torch_device = torch.device("cuda" if TORCH_HAS_CUDA else "cpu")

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD as SkTruncatedSVD

# ══════════════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════════════
# Repo root (this file lives in Implementation/Our Approach(LinkRank)/)
REPO_ROOT = Path(__file__).resolve().parents[2]

# Legacy flat-CSV inputs (only used when USE_NORMALIZED = False; not shipped)
ENRICHED_CSV     = str(REPO_ROOT / "Dataset/pytorch_enriched.csv")
RDS_ENRICHED_CSV = str(REPO_ROOT / "Dataset/pytorch_rds_enriched.csv")

# ── CLI: python LinkRank_code.py [dataset] [no_gemma|gemma] ──
_CLI_DATASET = sys.argv[1] if len(sys.argv) > 1 else "pytorch"
_CLI_GEMMA   = sys.argv[2] if len(sys.argv) > 2 else "no_gemma"

# ── Dataset directory (Dataset/{project}: rds_issues, rds_commits, rds_links) ──
NORMALIZED_DIR   = str(REPO_ROOT / "Dataset" / _CLI_DATASET)
USE_NORMALIZED   = True   # Load from normalized files instead of flat CSV
USE_GEMMA        = (_CLI_GEMMA.lower() == "gemma")
_gemma_tag = "gemma" if USE_GEMMA else "no_gemma"
OUT_ROOT = REPO_ROOT / "results" / f"linkrank_v6_5fold_{_CLI_DATASET}_k7_{_gemma_tag}"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

RDS_WINDOW_DAYS  = 365
TRUE_LINKS_ONLY  = True
REUSE_RDS_CSV    = True
RANDOM_SEED      = 42
N_FOLDS          = 5

# SBERT
SBERT_MODEL_NAME = "all-mpnet-base-v2"
SBERT_MAX_CHARS  = 2048

# TF-IDF + SVD
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TFIDF_NGRAMS = (1, 2)
SVD_DIM_FULL  = 128
SVD_DIM_CODE  = 80
SVD_DIM_NL    = 80
SVD_DIM_TITLE = 64
SVD_DIM_GEMMA = 100

# Time
TIME_TAU_DAYS = 7

# Iterative refinement
USE_ITERATION_KNOWNK  = True
USE_ITERATION_NOK_REL = True
ALPHA = 0.7
BETA  = 0.3

LOG_EVERY_N_ITERS = 10

# Optuna
OPTUNA_N_TRIALS = 50
OPTUNA_TIMEOUT  = 300

# ══════════════════════════════════════════════════════════════════════
#  Logging
# ══════════════════════════════════════════════════════════════════════
log_path = OUT_ROOT / "run_5fold.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
)
log = logging.getLogger("linkrank_v6_5fold")

def stage(name):
    class _Stage:
        def __enter__(self):
            self.name = name; self.t0 = time.perf_counter()
            m = f"{mem_gb():.2f} GB" if psutil else "n/a"
            log.info(f"▶ START: {self.name} | mem={m}")
            return self
        def __exit__(self, exc_type, exc, tb):
            dt = time.perf_counter() - self.t0
            m = f"{mem_gb():.2f} GB" if psutil else "n/a"
            if exc_type is None:
                log.info(f"✔ END:   {self.name} | {dt:.2f}s | mem={m}")
            else:
                log.error(f"✖ FAIL:  {self.name} | {dt:.2f}s | mem={m} | {exc_type.__name__}: {exc}")
            return False
    return _Stage()

rng = np.random.default_rng(RANDOM_SEED)
log.info(f"Device: {torch_device}, CUDA: {TORCH_HAS_CUDA}")

# ══════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
PATH_RE  = re.compile(r"[\w/\\]+\.[\w]+")

def tokenize(s):
    return [t.lower() for t in TOKEN_RE.findall(s or "")]

def cos_sim_np(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def time_prox(issue_date, commit_date, tau_days=7.0):
    if pd.isna(issue_date) or pd.isna(commit_date): return 0.0
    days = abs((commit_date - issue_date).total_seconds()) / 86400.0
    return float(np.exp(-days / tau_days))

def time_direction(issue_date, commit_date):
    if pd.isna(issue_date) or pd.isna(commit_date): return 0.0
    days = (commit_date - issue_date).total_seconds() / 86400.0
    return float(np.tanh(days / 30.0))

def extract_file_paths(text):
    if pd.isna(text) or not text: return set()
    return set(m.lower().strip() for m in PATH_RE.findall(str(text)))

def extract_file_basenames(text):
    return set(p.rsplit('/', 1)[-1].rsplit('\\', 1)[-1] for p in extract_file_paths(text))

def jaccard(s1, s2):
    if not s1 or not s2: return 0.0
    return len(s1 & s2) / max(len(s1 | s2), 1)

def parse_diff_summary(ds):
    if pd.isna(ds) or not ds: return 0, 0, 0
    ds = str(ds)
    m_a = re.search(r'\+(\d+)', ds)
    m_d = re.search(r'-(\d+)', ds)
    m_f = re.search(r'(\d+)\s+file', ds)
    return (int(m_a.group(1)) if m_a else 0,
            int(m_d.group(1)) if m_d else 0,
            int(m_f.group(1)) if m_f else 0)

def trunc(text, max_chars=SBERT_MAX_CHARS):
    s = str(text) if not pd.isna(text) else ""
    return s[:max_chars] if len(s) > max_chars else s

def macro_percent(df):
    p = round(100*df["Precision"].mean(), 2) if len(df) else 0.0
    r = round(100*df["Recall"].mean(), 2) if len(df) else 0.0
    f = round(100*df["F1"].mean(), 2) if len(df) else 0.0
    return p, r, f

# ══════════════════════════════════════════════════════════════════════
#  Text builders
# ══════════════════════════════════════════════════════════════════════
# When loading from normalized files, Full Diff and Gemma_Summary are
# kept in lookup dicts (_fulldiff_lookup, _gemma_lookup) rather than
# in the flat DataFrame.  The text builders below check for these.
_fulldiff_lookup = None   # set in main if USE_NORMALIZED
_gemma_lookup    = None

def _agg_text(d, group_col, agg_dict, concat_cols):
    g = d.groupby(group_col).agg(**agg_dict)
    parts = [g[c].fillna("") for c in concat_cols]
    out = parts[0]
    for p in parts[1:]:
        out = out + " " + p
    return out.reset_index(name="text")

def issue_text_full(d):
    return _agg_text(d, "Issue ID",
        {"title": ("Title","first"), "desc": ("Description","first"), "comm": ("Comments","first")},
        ["title","desc","comm"])

def issue_text_nl(d):
    return _agg_text(d, "Issue ID",
        {"title": ("Title","first"), "desc": ("Description","first")},
        ["title","desc"])

def issue_text_title(d):
    return _agg_text(d, "Issue ID", {"title": ("Title","first")}, ["title"])

def issue_text_code(d):
    return _agg_text(d, "Issue ID",
        {"desc": ("Description","first"), "comm": ("Comments","first")},
        ["desc","comm"])

def commit_text_full(d):
    result = _agg_text(d, "Commit ID",
        {"msg": ("Message","first"), "dif": ("Diff Summary","first"),
         "files": ("File Changes","first"), "full": ("Full Diff","first")},
        ["msg","dif","files","full"])
    # If using normalized loading, Full Diff in df is empty — patch from lookup
    if _fulldiff_lookup is not None:
        fulldiffs = result["Commit ID"].map(_fulldiff_lookup).fillna("")
        # Rebuild: text already has msg+dif+files+"", just append real Full Diff
        result["text"] = result["text"].str.strip() + " " + fulldiffs
    return result

def commit_text_code(d):
    result = _agg_text(d, "Commit ID",
        {"dif": ("Diff Summary","first"), "files": ("File Changes","first"),
         "full": ("Full Diff","first")},
        ["dif","files","full"])
    if _fulldiff_lookup is not None:
        fulldiffs = result["Commit ID"].map(_fulldiff_lookup).fillna("")
        result["text"] = result["text"].str.strip() + " " + fulldiffs
    return result

def commit_text_msg(d):
    return _agg_text(d, "Commit ID", {"msg": ("Message","first")}, ["msg"])

def commit_text_gemma(d):
    result = _agg_text(d, "Commit ID", {"gemma": ("Gemma_Summary","first")}, ["gemma"])
    if _gemma_lookup is not None:
        result["text"] = result["Commit ID"].map(_gemma_lookup).fillna("")
    return result

# ══════════════════════════════════════════════════════════════════════
#  BM25 Index
# ══════════════════════════════════════════════════════════════════════
class BM25Index:
    def __init__(self, commit_ids, commit_texts):
        self.commit_ids = list(commit_ids)
        self.cid_to_idx = {cid: i for i, cid in enumerate(self.commit_ids)}
        tokenized = [tokenize(t) for t in commit_texts]
        # Check if all documents are empty — BM25 will crash on empty corpus
        non_empty = sum(1 for t in tokenized if len(t) > 0)
        if non_empty == 0:
            self.bm25 = None  # sentinel: all docs empty
        else:
            self.bm25 = BM25Okapi(tokenized)

    def score_batch(self, issue_text):
        if self.bm25 is None:
            return {cid: 0.0 for cid in self.commit_ids}
        query_tokens = tokenize(issue_text)
        if not query_tokens:
            return {cid: 0.0 for cid in self.commit_ids}
        scores = self.bm25.get_scores(query_tokens)
        return {cid: float(scores[i]) for i, cid in enumerate(self.commit_ids)}

# ══════════════════════════════════════════════════════════════════════
#  RDS builder
# ══════════════════════════════════════════════════════════════════════
def build_rds_dataset(df_true_links, window_days=RDS_WINDOW_DAYS):
    log.info(f"Building RDS dataset: window={window_days} days")
    df = df_true_links.copy()
    df["Issue Date"] = pd.to_datetime(df["Issue Date"], errors="coerce")
    df["Commit Date"] = pd.to_datetime(df["Commit Date"], errors="coerce")
    commit_meta = df.drop_duplicates(subset="Commit ID").set_index("Commit ID")
    commit_dates = commit_meta["Commit Date"]
    true_links = df.groupby("Issue ID")["Commit ID"].apply(set).to_dict()
    issue_meta = df.drop_duplicates(subset="Issue ID").set_index("Issue ID")
    rds_rows = []
    for iid in tqdm(true_links, desc="RDS generation"):
        idate = issue_meta.loc[iid, "Issue Date"]
        if pd.isna(idate): continue
        ws = idate - timedelta(days=window_days)
        we = idate + timedelta(days=window_days)
        true_set = true_links[iid]
        i_row = issue_meta.loc[iid]
        for cid in commit_dates[(commit_dates >= ws) & (commit_dates <= we)].index:
            c_row = commit_meta.loc[cid]
            rds_rows.append({
                "Repository": i_row.get("Repository",""), "Issue ID": iid,
                "Issue Date": idate, "Title": i_row.get("Title",""),
                "Description": i_row.get("Description",""),
                "Labels": i_row.get("Labels",""), "Comments": i_row.get("Comments",""),
                "Commit ID": cid, "Commit Date": c_row.get("Commit Date", pd.NaT),
                "Message": c_row.get("Message",""), "Diff Summary": c_row.get("Diff Summary",""),
                "File Changes": c_row.get("File Changes",""),
                "Full Diff": c_row.get("Full Diff",""),
                "Gemma_Summary": c_row.get("Gemma_Summary",""),
                "Output": 1 if cid in true_set else 0,
            })
    rds_df = pd.DataFrame(rds_rows)
    n_pos = (rds_df["Output"]==1).sum(); n_neg = (rds_df["Output"]==0).sum()
    log.info(f"RDS: {rds_df['Issue ID'].nunique()} issues, {len(rds_df)} rows, "
             f"pos={n_pos}, neg={n_neg}, ratio=1:{n_neg//max(n_pos,1)}")
    return rds_df

# ══════════════════════════════════════════════════════════════════════
#  Precompute metadata
# ══════════════════════════════════════════════════════════════════════
def precompute_metadata(df):
    meta = {}
    issue_files, issue_basenames, issue_labels, issue_desc_len = {}, {}, {}, {}
    for iid, grp in df.groupby("Issue ID"):
        desc = str(grp["Description"].iloc[0]) if pd.notna(grp["Description"].iloc[0]) else ""
        comm = str(grp["Comments"].iloc[0]) if pd.notna(grp["Comments"].iloc[0]) else ""
        title = str(grp["Title"].iloc[0]) if pd.notna(grp["Title"].iloc[0]) else ""
        full = title + " " + desc + " " + comm
        issue_files[iid] = extract_file_paths(full)
        issue_basenames[iid] = extract_file_basenames(full)
        issue_desc_len[iid] = len(desc)
        lbl = grp["Labels"].iloc[0]
        issue_labels[iid] = set(t.strip().lower() for t in str(lbl).split(',')) if pd.notna(lbl) else set()

    commit_files, commit_basenames, commit_diff_stats, commit_msg_tokens = {}, {}, {}, {}
    for cid, grp in df.groupby("Commit ID"):
        fc = grp["File Changes"].iloc[0]
        if pd.notna(fc):
            paths = set(f.strip().lower() for f in str(fc).split(','))
            bns = set(p.rsplit('/',1)[-1].rsplit('\\',1)[-1] for p in paths)
        else:
            paths, bns = set(), set()
        commit_files[cid] = paths
        commit_basenames[cid] = bns
        commit_diff_stats[cid] = parse_diff_summary(grp["Diff Summary"].iloc[0])
        msg = grp["Message"].iloc[0]
        commit_msg_tokens[cid] = set(tokenize(str(msg))) if pd.notna(msg) else set()

    meta.update({"issue_files": issue_files, "issue_basenames": issue_basenames,
                 "issue_labels": issue_labels, "issue_desc_len": issue_desc_len,
                 "commit_files": commit_files, "commit_basenames": commit_basenames,
                 "commit_diff_stats": commit_diff_stats, "commit_msg_tokens": commit_msg_tokens})
    return meta


# ══════════════════════════════════════════════════════════════════════
#  5-Fold Stratified Split by K
# ══════════════════════════════════════════════════════════════════════
def make_stratified_kfold_splits(df, n_folds=N_FOLDS, seed=RANDOM_SEED):
    """
    Create n_folds splits stratified by K (number of true linked commits per issue).
    Within each K-bucket, issues are shuffled and distributed round-robin across folds.
    This ensures each fold has a similar distribution of easy (K=1) and hard (K>1) issues.
    """
    rng_split = np.random.default_rng(seed)

    # Get K per issue
    k_per_issue = df[df["Output"] == 1].groupby("Issue ID")["Commit ID"].nunique()
    issue_k = k_per_issue.reset_index()
    issue_k.columns = ["Issue ID", "K"]
    k_groups = issue_k.groupby("K")["Issue ID"].apply(list).to_dict()

    # Assign folds round-robin within each K-bucket
    fold_assignment = {}  # issue_id -> fold_idx
    for k in sorted(k_groups.keys()):
        issues = sorted(k_groups[k])
        rng_split.shuffle(issues)
        for i, iid in enumerate(issues):
            fold_assignment[iid] = i % n_folds

    # Build fold lists
    folds = [[] for _ in range(n_folds)]
    for iid, fold_idx in fold_assignment.items():
        folds[fold_idx].append(iid)

    # Sort for reproducibility
    folds = [sorted(f) for f in folds]

    # Log distribution
    all_issues = set(fold_assignment.keys())
    log.info(f"5-Fold split: {len(all_issues)} issues across {n_folds} folds")
    for fi in range(n_folds):
        fold_issues = set(folds[fi])
        fold_k = issue_k[issue_k["Issue ID"].isin(fold_issues)]
        k_dist = fold_k["K"].value_counts().sort_index().to_dict()
        log.info(f"  Fold {fi+1}: {len(fold_issues)} issues, K-dist: {k_dist}")

    return folds


# ══════════════════════════════════════════════════════════════════════
#  Run One Fold (same as v6 run_one_split)
# ══════════════════════════════════════════════════════════════════════
def run_one_fold(fold_idx, ds_name, df, train_ids, test_ids, out_dir, meta, sbert_model):
    """Run full LinkRank v6 pipeline for one fold. Returns summary DataFrame."""
    fold_label = f"Fold{fold_idx+1}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for c in ["Issue Date", "Commit Date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    with stage(f"[{fold_label}] Build train/test"):
        train_df = df[df["Issue ID"].isin(train_ids)].copy()
        test_df  = df[df["Issue ID"].isin(test_ids)].copy()
        log.info(f"Train issues={len(train_ids)}, rows={len(train_df):,}")
        log.info(f"Test  issues={len(test_ids)}, rows={len(test_df):,}")

    # ══════════════════════════════════════════════════════════════
    #  SBERT EMBEDDINGS (encode all data — unsupervised)
    # ══════════════════════════════════════════════════════════════
    def sbert_encode_unique(text_df, id_col, label, max_chars=SBERT_MAX_CHARS):
        with stage(f"[{fold_label}] SBERT encode [{label}] ({len(text_df)} texts)"):
            ids = text_df[id_col].tolist()
            texts = [trunc(t, max_chars) for t in text_df["text"].tolist()]
            embs = sbert_model.encode(texts, batch_size=64, show_progress_bar=True,
                                       normalize_embeddings=True)
            return {iid: embs[i] for i, iid in enumerate(ids)}

    all_issue_full   = issue_text_full(df)
    all_issue_nl     = issue_text_nl(df)
    all_issue_title  = issue_text_title(df)
    all_commit_full  = commit_text_full(df)
    all_commit_msg   = commit_text_msg(df)
    all_commit_gemma = commit_text_gemma(df) if USE_GEMMA else None

    sbert_issue_full   = sbert_encode_unique(all_issue_full, "Issue ID", "issue_full")
    sbert_issue_nl     = sbert_encode_unique(all_issue_nl, "Issue ID", "issue_nl")
    sbert_issue_title  = sbert_encode_unique(all_issue_title, "Issue ID", "issue_title")
    sbert_commit_full  = sbert_encode_unique(all_commit_full, "Commit ID", "commit_full")
    sbert_commit_msg   = sbert_encode_unique(all_commit_msg, "Commit ID", "commit_msg")
    sbert_commit_gemma = sbert_encode_unique(all_commit_gemma, "Commit ID", "commit_gemma") if USE_GEMMA else None

    SBERT_DIM = sbert_model.get_sentence_embedding_dimension()

    # ══════════════════════════════════════════════════════════════
    #  BM25 INDICES (unsupervised, built on ALL commits)
    # ══════════════════════════════════════════════════════════════
    with stage(f"[{fold_label}] Build BM25 indices"):
        bm25_full_idx = BM25Index(all_commit_full["Commit ID"].tolist(),
                                   all_commit_full["text"].fillna("").tolist())
        all_commit_code = commit_text_code(df)
        bm25_code_idx = BM25Index(all_commit_code["Commit ID"].tolist(),
                                   all_commit_code["text"].fillna("").tolist())
        bm25_nl_idx = BM25Index(all_commit_msg["Commit ID"].tolist(),
                                 all_commit_msg["text"].fillna("").tolist())
        bm25_title_idx = BM25Index(all_commit_msg["Commit ID"].tolist(),
                                    all_commit_msg["text"].fillna("").tolist())
        bm25_gemma_idx = BM25Index(all_commit_gemma["Commit ID"].tolist(),
                                    all_commit_gemma["text"].fillna("").tolist()) if USE_GEMMA else None

        log.info("Pre-computing BM25 scores for all issues...")
        bm25_cache = {}
        all_issues = df["Issue ID"].unique()
        issue_full_map  = dict(zip(all_issue_full["Issue ID"], all_issue_full["text"].fillna("")))
        issue_nl_map    = dict(zip(all_issue_nl["Issue ID"], all_issue_nl["text"].fillna("")))
        issue_title_map = dict(zip(all_issue_title["Issue ID"], all_issue_title["text"].fillna("")))
        _issue_code_df  = issue_text_code(df)
        issue_code_map  = dict(zip(_issue_code_df["Issue ID"], _issue_code_df["text"].fillna("")))

        for iid in tqdm(all_issues, desc=f"BM25[{fold_label}]"):
            bm25_cache[iid] = {
                "full":  bm25_full_idx.score_batch(issue_full_map.get(iid, "")),
                "code":  bm25_code_idx.score_batch(issue_code_map.get(iid, "")),
                "nl":    bm25_nl_idx.score_batch(issue_nl_map.get(iid, "")),
                "title": bm25_title_idx.score_batch(issue_title_map.get(iid, "")),
                "gemma": bm25_gemma_idx.score_batch(issue_nl_map.get(iid, "")) if USE_GEMMA else None,
            }

    # ══════════════════════════════════════════════════════════════
    #  TF-IDF/SVD (fitted on TRAIN only)
    # ══════════════════════════════════════════════════════════════
    def fit_tfidf_svd(issue_texts, commit_texts, svd_dim, label):
        with stage(f"[{fold_label}] TF-IDF+SVD [{label}] (dim={svd_dim})"):
            tfidf = TfidfVectorizer(min_df=TFIDF_MIN_DF, max_df=TFIDF_MAX_DF,
                                    ngram_range=TFIDF_NGRAMS, sublinear_tf=True)
            combined = pd.concat([issue_texts["text"], commit_texts["text"]], axis=0).fillna("")
            X = tfidf.fit_transform(combined)
            svd = SkTruncatedSVD(n_components=min(svd_dim, X.shape[1]-1), random_state=RANDOM_SEED)
            Xr = svd.fit_transform(X)
            Ei = Xr[:len(issue_texts)]
            Ec = Xr[len(issue_texts):]
        return tfidf, svd, Ei, Ec

    i_full_tr = issue_text_full(train_df); c_full_tr = commit_text_full(train_df)
    i_code_tr = issue_text_code(train_df); c_code_tr = commit_text_code(train_df)
    i_nl_tr   = issue_text_nl(train_df);   c_nl_tr   = commit_text_msg(train_df)
    i_ttl_tr  = issue_text_title(train_df); c_msg_tr  = commit_text_msg(train_df)
    i_nl_gemma_tr = issue_text_nl(train_df) if USE_GEMMA else None
    c_gemma_tr = commit_text_gemma(train_df) if USE_GEMMA else None

    tfidf_full, svd_full, Ei_full_tr, Ec_full_tr = fit_tfidf_svd(i_full_tr, c_full_tr, SVD_DIM_FULL, "full")
    tfidf_code, svd_code, Ei_code_tr, Ec_code_tr = fit_tfidf_svd(i_code_tr, c_code_tr, SVD_DIM_CODE, "code")
    tfidf_nl,   svd_nl,   Ei_nl_tr,   Ec_nl_tr   = fit_tfidf_svd(i_nl_tr, c_nl_tr, SVD_DIM_NL, "nl")
    tfidf_ttl,  svd_ttl,  Ei_ttl_tr,  Ec_ttl_tr  = fit_tfidf_svd(i_ttl_tr, c_msg_tr, SVD_DIM_TITLE, "title")
    if USE_GEMMA:
        tfidf_gemma, svd_gemma, Ei_gemma_tr, Ec_gemma_tr = fit_tfidf_svd(i_nl_gemma_tr, c_gemma_tr, SVD_DIM_GEMMA, "gemma")
    else:
        tfidf_gemma, svd_gemma, Ei_gemma_tr, Ec_gemma_tr = None, None, None, None

    with stage(f"[{fold_label}] Transform test SVD"):
        i_full_te = issue_text_full(test_df); c_full_te = commit_text_full(test_df)
        Ei_full_te = svd_full.transform(tfidf_full.transform(i_full_te["text"].fillna("")))
        Ec_full_te = svd_full.transform(tfidf_full.transform(c_full_te["text"].fillna("")))

        i_code_te = issue_text_code(test_df); c_code_te = commit_text_code(test_df)
        Ei_code_te = svd_code.transform(tfidf_code.transform(i_code_te["text"].fillna("")))
        Ec_code_te = svd_code.transform(tfidf_code.transform(c_code_te["text"].fillna("")))

        i_nl_te = issue_text_nl(test_df); c_nl_te = commit_text_msg(test_df)
        Ei_nl_te = svd_nl.transform(tfidf_nl.transform(i_nl_te["text"].fillna("")))
        Ec_nl_te = svd_nl.transform(tfidf_nl.transform(c_nl_te["text"].fillna("")))

        i_ttl_te = issue_text_title(test_df); c_msg_te = commit_text_msg(test_df)
        Ei_ttl_te = svd_ttl.transform(tfidf_ttl.transform(i_ttl_te["text"].fillna("")))
        Ec_ttl_te = svd_ttl.transform(tfidf_ttl.transform(c_msg_te["text"].fillna("")))

        if USE_GEMMA:
            i_nl_gemma_te = issue_text_nl(test_df); c_gemma_te = commit_text_gemma(test_df)
            Ei_gemma_te = svd_gemma.transform(tfidf_gemma.transform(i_nl_gemma_te["text"].fillna("")))
            Ec_gemma_te = svd_gemma.transform(tfidf_gemma.transform(c_gemma_te["text"].fillna("")))
        else:
            Ei_gemma_te, Ec_gemma_te = None, None

    def build_idx(idf, cdf):
        if idf is None or cdf is None:
            return None
        return ({iid: i for i, iid in enumerate(idf["Issue ID"].tolist())},
                {cid: i for i, cid in enumerate(cdf["Commit ID"].tolist())})

    idx_full_tr  = build_idx(i_full_tr, c_full_tr)
    idx_code_tr  = build_idx(i_code_tr, c_code_tr)
    idx_nl_tr    = build_idx(i_nl_tr, c_nl_tr)
    idx_ttl_tr   = build_idx(i_ttl_tr, c_msg_tr)
    idx_gemma_tr = build_idx(i_nl_gemma_tr, c_gemma_tr) if USE_GEMMA else None
    idx_full_te  = build_idx(i_full_te, c_full_te)
    idx_code_te  = build_idx(i_code_te, c_code_te)
    idx_nl_te    = build_idx(i_nl_te, c_nl_te)
    idx_ttl_te   = build_idx(i_ttl_te, c_msg_te)
    idx_gemma_te = build_idx(i_nl_gemma_te, c_gemma_te) if USE_GEMMA else None

    # ══════════════════════════════════════════════════════════════
    #  FEATURE EXTRACTION (22 features with Gemma, 17 without)
    # ══════════════════════════════════════════════════════════════
    FEATURES = [
        "svd_full", "svd_code", "svd_nl", "svd_title",
        "sbert_full", "sbert_nl", "sbert_title",
        "bm25_full", "bm25_code", "bm25_nl", "bm25_title",
        "feat_time", "feat_time_dir",
        "feat_file_bn_jaccard",
        "feat_diff_size", "feat_nfiles",
        "feat_desc_len",
    ]
    if USE_GEMMA:
        FEATURES.extend(["svd_gemma", "sbert_gemma_full", "sbert_gemma_nl", "sbert_gemma_title", "bm25_gemma"])

    def _svd_sim(iid, cid, idx_pair, Ei, Ec):
        i_map, c_map = idx_pair
        if iid in i_map and cid in c_map:
            return cos_sim_np(Ei[i_map[iid]], Ec[c_map[cid]])
        return 0.0

    def _sbert_sim(iid, cid, issue_dict, commit_dict):
        ei = issue_dict.get(iid)
        ec = commit_dict.get(cid)
        if ei is not None and ec is not None:
            return cos_sim_np(ei, ec)
        return 0.0

    def _bm25_score(iid, cid, channel):
        cache = bm25_cache.get(iid)
        if cache is None: return 0.0
        channel_scores = cache.get(channel)
        if channel_scores is None: return 0.0
        return channel_scores.get(cid, 0.0)

    def build_features(sub_df, split="train"):
        if split == "train":
            idx_f, idx_c, idx_n, idx_t, idx_g = idx_full_tr, idx_code_tr, idx_nl_tr, idx_ttl_tr, idx_gemma_tr
            Ef, Ecf = Ei_full_tr, Ec_full_tr
            Eco, Ecco = Ei_code_tr, Ec_code_tr
            En, Ecn = Ei_nl_tr, Ec_nl_tr
            Et, Ect = Ei_ttl_tr, Ec_ttl_tr
            Eg, Ecg = Ei_gemma_tr, Ec_gemma_tr
        else:
            idx_f, idx_c, idx_n, idx_t, idx_g = idx_full_te, idx_code_te, idx_nl_te, idx_ttl_te, idx_gemma_te
            Ef, Ecf = Ei_full_te, Ec_full_te
            Eco, Ecco = Ei_code_te, Ec_code_te
            En, Ecn = Ei_nl_te, Ec_nl_te
            Et, Ect = Ei_ttl_te, Ec_ttl_te
            Eg, Ecg = (Ei_gemma_te, Ec_gemma_te) if USE_GEMMA else (None, None)

        rows = []
        for _, row in tqdm(sub_df.iterrows(), total=len(sub_df), desc=f"features[{split}]"):
            iid, cid = row["Issue ID"], row["Commit ID"]

            s_svd_full  = _svd_sim(iid, cid, idx_f, Ef, Ecf)
            s_svd_code  = _svd_sim(iid, cid, idx_c, Eco, Ecco)
            s_svd_nl    = _svd_sim(iid, cid, idx_n, En, Ecn)
            s_svd_title = _svd_sim(iid, cid, idx_t, Et, Ect)
            if USE_GEMMA:
                s_svd_gemma = _svd_sim(iid, cid, idx_g, Eg, Ecg)
            else:
                s_svd_gemma = None

            s_sbert_full  = _sbert_sim(iid, cid, sbert_issue_full, sbert_commit_full)
            s_sbert_nl    = _sbert_sim(iid, cid, sbert_issue_nl, sbert_commit_msg)
            s_sbert_title = _sbert_sim(iid, cid, sbert_issue_title, sbert_commit_msg)
            if USE_GEMMA:
                s_sbert_gemma_full  = _sbert_sim(iid, cid, sbert_issue_full, sbert_commit_gemma)
                s_sbert_gemma_nl    = _sbert_sim(iid, cid, sbert_issue_nl, sbert_commit_gemma)
                s_sbert_gemma_title = _sbert_sim(iid, cid, sbert_issue_title, sbert_commit_gemma)
            else:
                s_sbert_gemma_full = s_sbert_gemma_nl = s_sbert_gemma_title = None

            b_full  = _bm25_score(iid, cid, "full")
            b_code  = _bm25_score(iid, cid, "code")
            b_nl    = _bm25_score(iid, cid, "nl")
            b_title = _bm25_score(iid, cid, "title")
            b_gemma = _bm25_score(iid, cid, "gemma") if USE_GEMMA else None

            f_time     = time_prox(row.get("Issue Date"), row.get("Commit Date"), TIME_TAU_DAYS)
            f_time_dir = time_direction(row.get("Issue Date"), row.get("Commit Date"))

            i_bn = meta["issue_basenames"].get(iid, set())
            c_bn = meta["commit_basenames"].get(cid, set())
            f_fbj = jaccard(i_bn, c_bn)

            added, deleted, nfiles = meta["commit_diff_stats"].get(cid, (0,0,0))
            f_ds = math.log1p(added + deleted)
            f_nf = math.log1p(nfiles)

            f_dl = math.log1p(meta["issue_desc_len"].get(iid, 0))

            row_data = [
                iid, cid,
                s_svd_full, s_svd_code, s_svd_nl, s_svd_title,
                s_sbert_full, s_sbert_nl, s_sbert_title,
                b_full, b_code, b_nl, b_title,
                f_time, f_time_dir,
                f_fbj,
                f_ds, f_nf,
                f_dl,
            ]
            if USE_GEMMA:
                row_data.extend([s_svd_gemma, s_sbert_gemma_full, s_sbert_gemma_nl, s_sbert_gemma_title, b_gemma])
            row_data.append(row["Output"])
            rows.append(row_data)

        return pd.DataFrame(rows, columns=["Issue ID","Commit ID"] + FEATURES + ["Output"])

    with stage(f"[{fold_label}] Build features: TRAIN"):
        train_feat = build_features(train_df, split="train")
    with stage(f"[{fold_label}] Build features: TEST"):
        test_feat = build_features(test_df, split="test")

    # ══════════════════════════════════════════════════════════════
    #  PREPARE RANK DATASETS
    # ══════════════════════════════════════════════════════════════
    def prep_rank(df_feat):
        df_s = df_feat.sort_values("Issue ID").reset_index(drop=True)
        X = df_s[FEATURES].values
        y = df_s["Output"].astype(int).values
        groups = df_s.groupby("Issue ID").size().tolist()
        return df_s, X, y, groups

    with stage(f"[{fold_label}] Prepare rank data"):
        uids = train_feat["Issue ID"].drop_duplicates().tolist()
        rng_fold = np.random.default_rng(RANDOM_SEED + fold_idx)
        rng_fold.shuffle(uids)
        n_dev = max(1, int(0.20 * len(uids)))
        dev_ids = set(uids[:n_dev])

        trn_feat = train_feat[~train_feat["Issue ID"].isin(dev_ids)].copy()
        dev_feat = train_feat[train_feat["Issue ID"].isin(dev_ids)].copy()

        def _filter(df_f, need_neg=True):
            grp = df_f.groupby("Issue ID")["Output"]
            pos, cnt = grp.sum(), grp.count()
            ok = (pos >= 1) & ((cnt - pos) >= 1) if need_neg else (pos >= 1)
            return df_f[df_f["Issue ID"].isin(pos[ok].index)].copy()

        trn_feat = _filter(trn_feat, True)
        dev_feat = _filter(dev_feat, True)
        test_feat_f = _filter(test_feat, False)

        trn_s, Xtr, ytr, gtr = prep_rank(trn_feat)
        dev_s, Xdv, ydv, gdv = prep_rank(dev_feat)
        tst_s, Xte, yte, gte = prep_rank(test_feat_f)
        log.info(f"Train groups={len(gtr)}, Dev={len(gdv)}, Test={len(gte)}")

    # ══════════════════════════════════════════════════════════════
    #  OPTUNA HYPERPARAMETER TUNING
    # ══════════════════════════════════════════════════════════════
    with stage(f"[{fold_label}] Optuna hyperparameter tuning"):
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        dtrain_opt = lgb.Dataset(Xtr, label=ytr, group=gtr, feature_name=FEATURES)
        dvalid_opt = lgb.Dataset(Xdv, label=ydv, group=gdv, reference=dtrain_opt, feature_name=FEATURES)

        def objective(trial):
            p = dict(
                objective="lambdarank", metric=["ndcg"], ndcg_eval_at=[1,3,5],
                feature_pre_filter=False,
                learning_rate=trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
                num_leaves=trial.suggest_int("num_leaves", 15, 127),
                min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 5, 100),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
                bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
                bagging_freq=1,
                lambda_l1=trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
                lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
                min_gain_to_split=trial.suggest_float("min_gain_to_split", 1e-4, 0.1, log=True),
                verbose=-1, num_threads=os.cpu_count() or 1, seed=RANDOM_SEED,
            )
            if TORCH_HAS_CUDA:
                p.update(device_type="gpu", gpu_platform_id=0, gpu_device_id=0)
            model = lgb.train(
                p, dtrain_opt, valid_sets=[dvalid_opt], valid_names=["valid"],
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False),
                           lgb.log_evaluation(0)]
            )
            return model.best_score["valid"]["ndcg@1"]

        study = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
        study.optimize(objective, n_trials=OPTUNA_N_TRIALS, timeout=OPTUNA_TIMEOUT)

        best_params = study.best_params
        log.info(f"Optuna best trial: {study.best_trial.number}, NDCG@1={study.best_value:.4f}")
        log.info(f"Best params: {json.dumps(best_params, indent=2)}")

        with open(out_dir / "optuna_results.json", "w") as f:
            json.dump({
                "best_trial": study.best_trial.number,
                "best_ndcg1": study.best_value,
                "best_params": best_params,
                "n_trials": len(study.trials),
            }, f, indent=2)

    # ══════════════════════════════════════════════════════════════
    #  TRAIN LAMBDAMART
    # ══════════════════════════════════════════════════════════════
    with stage(f"[{fold_label}] Train LambdaMART (tuned)"):
        dtrain = lgb.Dataset(Xtr, label=ytr, group=gtr, feature_name=FEATURES)
        dvalid = lgb.Dataset(Xdv, label=ydv, group=gdv, reference=dtrain, feature_name=FEATURES)

        params = dict(
            objective="lambdarank", metric=["ndcg"], ndcg_eval_at=[1,3,5],
            feature_pre_filter=False,
            bagging_freq=1,
            verbose=-1, num_threads=os.cpu_count() or 1, seed=RANDOM_SEED,
            **best_params
        )
        if TORCH_HAS_CUDA:
            params.update(device_type="gpu", gpu_platform_id=0, gpu_device_id=0)

        model = lgb.train(
            params, dtrain, valid_sets=[dtrain, dvalid], valid_names=["train","valid"],
            num_boost_round=3000,
            callbacks=[lgb.log_evaluation(LOG_EVERY_N_ITERS),
                       lgb.early_stopping(stopping_rounds=200, verbose=True)]
        )
        log.info(f"Best iteration: {model.best_iteration}")

        imp = model.feature_importance(importance_type='gain')
        feat_imp = sorted(zip(FEATURES, imp), key=lambda x: -x[1])
        log.info("Feature importance (gain):")
        for fn, iv in feat_imp:
            log.info(f"  {fn}: {iv:.2f}")

    # ══════════════════════════════════════════════════════════════
    #  SCORING + ITERATIVE REFINEMENT
    # ══════════════════════════════════════════════════════════════
    def score_pool(pool_df):
        if len(pool_df) == 0:
            return pool_df.assign(score=[])
        X = pool_df[FEATURES].values
        s = model.predict(X, num_iteration=model.best_iteration)
        out = pool_df.copy(); out["score"] = s
        if len(out) <= 1:
            out["score_mm"] = 1.0; out["score_zn"] = 0.0
        else:
            mn, mx = float(out["score"].min()), float(out["score"].max())
            out["score_mm"] = 1.0 if mx == mn else (out["score"] - mn) / (mx - mn)
            mu = float(out["score"].mean()); sd = float(out["score"].std(ddof=0) + 1e-9)
            out["score_zn"] = (out["score"] - mu) / sd
        return out.sort_values("score", ascending=False).reset_index(drop=True)

    issue_vec_state = {}

    def init_issue_state(iid):
        v = sbert_issue_full.get(iid)
        if v is not None:
            issue_vec_state[iid] = v.copy() / (np.linalg.norm(v) + 1e-12)
        else:
            issue_vec_state[iid] = np.zeros(SBERT_DIM)

    def update_issue_state(iid, cid):
        v_i = issue_vec_state[iid]
        if USE_GEMMA:
            v_c = sbert_commit_gemma.get(cid)
            if v_c is None:
                v_c = sbert_commit_full.get(cid)
        else:
            v_c = sbert_commit_full.get(cid)
        if v_c is not None:
            v_new = ALPHA * v_i + BETA * v_c
            issue_vec_state[iid] = v_new / (np.linalg.norm(v_new) + 1e-12)

    def refresh_features_for_issue_pool(iid, pool_df):
        if len(pool_df) == 0: return pool_df
        v_i = issue_vec_state[iid]
        new_sbert = []
        new_sbert_gemma = []
        for _, r in pool_df.iterrows():
            cid = r["Commit ID"]
            vc = sbert_commit_full.get(cid)
            new_sbert.append(cos_sim_np(v_i, vc) if vc is not None else 0.0)
            if USE_GEMMA:
                vc_g = sbert_commit_gemma.get(cid)
                new_sbert_gemma.append(cos_sim_np(v_i, vc_g) if vc_g is not None else 0.0)
            else:
                new_sbert_gemma.append(0.0)
        result = pool_df.copy()
        result["sbert_full"] = new_sbert
        if USE_GEMMA:
            result["sbert_gemma_full"] = new_sbert_gemma
        return result

    # ══════════════════════════════════════════════════════════════
    #  EVALUATE Known-K
    # ══════════════════════════════════════════════════════════════
    with stage(f"[{fold_label}] Evaluate Known-K (iterative)"):
        test_rows_by_issue = {iid: tst_s[tst_s["Issue ID"]==iid].reset_index(drop=True)
                              for iid in tst_s["Issue ID"].unique()}
        true_by_issue = tst_s[tst_s["Output"]==1].groupby("Issue ID")["Commit ID"].apply(set).to_dict()
        issues_test = sorted(true_by_issue.keys())

        rowsK = []
        for iid in tqdm(issues_test, desc=f"eval[Known-K][{fold_label}]"):
            true_set = true_by_issue[iid]; K = len(true_set)
            pool = test_rows_by_issue[iid].copy()
            init_issue_state(iid)
            picks = []
            for _ in range(K):
                ranked = score_pool(pool)
                if len(ranked) == 0: break
                cid = ranked.iloc[0]["Commit ID"]
                picks.append(cid)
                if USE_ITERATION_KNOWNK:
                    update_issue_state(iid, cid)
                    pool = pool[pool["Commit ID"] != cid].reset_index(drop=True)
                    if len(pool):
                        pool = refresh_features_for_issue_pool(iid, pool)
                else:
                    pool = pool[pool["Commit ID"] != cid].reset_index(drop=True)
            pred_set = set(picks)
            inter = len(pred_set & true_set)
            p = inter / max(len(pred_set), 1); r = inter / max(K, 1)
            f1 = (2*inter) / max(len(pred_set)+K, 1)
            rowsK.append(dict(Issue=iid, K=K, Predicted=len(pred_set), Inter=inter,
                              AllCorrect=int(inter==K),
                              HalfCorrect=int(inter >= int(np.ceil(K/2))),
                              Precision=p, Recall=r, F1=f1))
        res_knownK = pd.DataFrame(rowsK)
        res_knownK.to_csv(out_dir / "results_KnownK_iter.csv", index=False)

    # ══════════════════════════════════════════════════════════════
    #  EVALUATE No-K (ABS-mm + REL)
    # ══════════════════════════════════════════════════════════════
    with stage(f"[{fold_label}] No-K: tune + evaluate"):
        dev_rows_by_issue = {iid: dev_feat[dev_feat["Issue ID"]==iid].reset_index(drop=True)
                             for iid in dev_feat["Issue ID"].unique()}
        true_dev = dev_feat[dev_feat["Output"]==1].groupby("Issue ID")["Commit ID"].apply(set).to_dict()
        issues_dev = sorted(true_dev.keys())

        tst_rbi = {iid: tst_s[tst_s["Issue ID"]==iid].reset_index(drop=True)
                   for iid in tst_s["Issue ID"].unique()}
        true_tst = tst_s[tst_s["Output"]==1].groupby("Issue ID")["Commit ID"].apply(set).to_dict()
        issues_tst = sorted(true_tst.keys())

        def mfi(pred_set, true_set):
            inter = len(pred_set & true_set)
            p = inter/max(len(pred_set),1); r = inter/max(len(true_set),1)
            f1 = (2*inter)/max(len(pred_set)+len(true_set),1)
            return dict(Precision=p, Recall=r, F1=f1,
                       AllCorrect=int(pred_set==true_set),
                       HalfCorrect=int(inter>=int(np.ceil(len(true_set)/2))))

        ranked_dev = {iid: score_pool(d) for iid,d in dev_rows_by_issue.items()}
        ranked_tst = {iid: score_pool(d) for iid,d in tst_rbi.items()}

        # ABS-mm threshold search
        taus = [x/100 for x in range(10,96,2)]
        best_abs = (-1, None)
        for t in taus:
            rows = [mfi(set(ranked_dev[iid][ranked_dev[iid]["score_mm"]>=t]["Commit ID"]),
                        true_dev[iid]) for iid in issues_dev]
            sc = pd.DataFrame(rows)["F1"].mean()
            if sc > best_abs[0]: best_abs = (sc, t)

        # REL gamma search
        gammas = [x/100 for x in range(30,96,2)]
        best_rel = (-1, None)
        for g in gammas:
            rows = []
            for iid in issues_dev:
                rd = ranked_dev[iid]
                if len(rd)==0: rows.append(mfi(set(), true_dev[iid])); continue
                best = float(rd["score"].iloc[0])
                chosen = set(rd[rd["score"]>=g*best]["Commit ID"])
                rows.append(mfi(chosen, true_dev[iid]))
            sc = pd.DataFrame(rows)["F1"].mean()
            if sc > best_rel[0]: best_rel = (sc, g)

        log.info(f"Best ABS-mm: tau={best_abs[1]:.2f} (dev F1={best_abs[0]:.4f})")
        log.info(f"Best REL: gamma={best_rel[1]:.2f} (dev F1={best_rel[0]:.4f})")

        # Test ABS-mm
        rows_abs = []
        for iid in issues_tst:
            rd = ranked_tst[iid]
            pred = set(rd[rd["score_mm"]>=best_abs[1]]["Commit ID"])
            m = mfi(pred, true_tst[iid])
            rows_abs.append(dict(Issue=iid, **m, Predicted=len(pred), TrueCount=len(true_tst[iid])))
        res_absmm = pd.DataFrame(rows_abs)
        res_absmm.to_csv(out_dir / "results_NoK_ABSmm.csv", index=False)

        # Test REL iterative
        rows_rel = []
        for iid in issues_tst:
            init_issue_state(iid)
            pool = ranked_tst[iid].copy()
            if len(pool)==0:
                m = mfi(set(), true_tst[iid])
                rows_rel.append(dict(Issue=iid, **m, Predicted=0, TrueCount=len(true_tst[iid])))
                continue
            best0 = float(pool["score"].iloc[0])
            accepted = []
            while len(pool):
                top = pool.iloc[0]
                if float(top["score"]) >= best_rel[1] * best0:
                    cid = top["Commit ID"]; accepted.append(cid)
                    if USE_ITERATION_NOK_REL:
                        update_issue_state(iid, cid)
                        pool = pool[pool["Commit ID"]!=cid].reset_index(drop=True)
                        if len(pool):
                            pool = refresh_features_for_issue_pool(iid, pool)
                            pool = score_pool(pool)
                    else:
                        pool = pool[pool["Commit ID"]!=cid].reset_index(drop=True)
                else:
                    break
            pred = set(accepted); m = mfi(pred, true_tst[iid])
            rows_rel.append(dict(Issue=iid, **m, Predicted=len(pred), TrueCount=len(true_tst[iid])))
        res_rel = pd.DataFrame(rows_rel)
        res_rel.to_csv(out_dir / "results_NoK_REL_ITER.csv", index=False)

    # ══════════════════════════════════════════════════════════════
    #  RANKING METRICS: MRR, NDCG@K, P@K (from initial ranking)
    # ══════════════════════════════════════════════════════════════
    with stage(f"[{fold_label}] Compute MRR, NDCG@K, P@K"):
        ranking_rows = []
        for iid in issues_tst:
            rd = ranked_tst[iid]
            K = len(true_tst[iid])
            if len(rd) == 0:
                ranking_rows.append(dict(Issue=iid, K=K, MRR=0.0, NDCG_at_K=0.0, P_at_K=0.0))
                continue
            # Build ranked list with relevance labels
            relevances = [1 if cid in true_tst[iid] else 0
                          for cid in rd["Commit ID"].values]
            # MRR: 1 / rank of first relevant
            mrr = 0.0
            for rank_idx, rel in enumerate(relevances):
                if rel == 1:
                    mrr = 1.0 / (rank_idx + 1)
                    break
            # NDCG@K
            top_k_rels = relevances[:K]
            dcg = sum(r / np.log2(i + 2) for i, r in enumerate(top_k_rels))
            ideal_k = min(K, sum(relevances))
            idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_k))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            # P@K
            p_at_k = sum(top_k_rels) / K if K > 0 else 0.0

            ranking_rows.append(dict(Issue=iid, K=K, MRR=mrr, NDCG_at_K=ndcg, P_at_K=p_at_k))

        res_ranking = pd.DataFrame(ranking_rows)
        res_ranking.to_csv(out_dir / "results_ranking_metrics.csv", index=False)

        avg_mrr = res_ranking["MRR"].mean() * 100
        avg_ndcg = res_ranking["NDCG_at_K"].mean() * 100
        avg_pak = res_ranking["P_at_K"].mean() * 100
        log.info(f"  MRR={avg_mrr:.2f}  NDCG@K={avg_ndcg:.2f}  P@K={avg_pak:.2f}")

    # ══════════════════════════════════════════════════════════════
    #  FOLD SUMMARY
    # ══════════════════════════════════════════════════════════════
    pK, rK, fK = macro_percent(res_knownK)
    pA, rA, fA = macro_percent(res_absmm)
    pR, rR, fR = macro_percent(res_rel)

    sum_df = pd.DataFrame([
        ["Known-K", pK, rK, fK],
        ["No-K (ABS-mm)", pA, rA, fA],
        ["No-K (REL)", pR, rR, fR],
    ], columns=["Setting","Precision","Recall","F1"])
    sum_df.to_csv(out_dir / "summary.csv", index=False)

    log.info(f"\n{'='*60}\n  RESULTS: {fold_label}\n{'='*60}")
    log.info(f"\n{sum_df.to_string(index=False)}")

    return {
        "fold": fold_idx + 1,
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "KnownK_P": pK, "KnownK_R": rK, "KnownK_F1": fK,
        "ABSmm_P": pA, "ABSmm_R": rA, "ABSmm_F1": fA,
        "REL_P": pR, "REL_R": rR, "REL_F1": fR,
        "MRR": avg_mrr, "NDCG_at_K": avg_ndcg, "P_at_K": avg_pak,
    }


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    total_t0 = time.perf_counter()

    with stage("Load SBERT model"):
        sbert_model = SentenceTransformer(SBERT_MODEL_NAME,
                                          device="cuda" if TORCH_HAS_CUDA else "cpu")
        SBERT_DIM = sbert_model.get_sentence_embedding_dimension()
        log.info(f"SBERT model: {SBERT_MODEL_NAME}, dim={SBERT_DIM}")

    ds_name = Path(NORMALIZED_DIR).stem if USE_NORMALIZED else Path(ENRICHED_CSV).stem

    print(f"\n{'='*70}")
    print(f"  LinkRank v6 — 5-Fold Stratified Cross-Validation")
    print(f"  Dataset: {ds_name}")
    print(f"{'='*70}\n")

    # ──────────────────────────────────────────────────────────
    #  Load data
    # ──────────────────────────────────────────────────────────
    with stage(f"[{ds_name}] Load data"):
        if USE_NORMALIZED:
            norm_dir = Path(NORMALIZED_DIR)
            log.info(f"Loading from normalized dir: {norm_dir}")

            rds_issues  = pd.read_csv(norm_dir / "rds_issues.csv")
            rds_commits = pd.read_csv(norm_dir / "rds_commits.csv")
            rds_links   = pd.read_csv(norm_dir / "rds_links.csv")

            log.info(f"  rds_issues:  {len(rds_issues)} rows")
            log.info(f"  rds_commits: {len(rds_commits)} rows")
            log.info(f"  rds_links:   {len(rds_links)} rows")

            # Keep Full Diff + Gemma_Summary only in unique-commit lookup (not in joined df)
            # This avoids duplicating huge text across 177K+ rows
            _fulldiff_lookup = dict(zip(rds_commits["Commit ID"],
                                        rds_commits["Full Diff"].fillna("")))
            if "Gemma_Summary" in rds_commits.columns:
                _gemma_lookup = dict(zip(rds_commits["Commit ID"],
                                         rds_commits["Gemma_Summary"].fillna("")))
            else:
                log.info("  ⚠ No Gemma_Summary column — Gemma features will be empty")
                _gemma_lookup = {cid: "" for cid in rds_commits["Commit ID"]}

            # Join with lightweight commit columns (no Full Diff / Gemma_Summary)
            commit_light_cols = ["Commit ID", "Commit Date", "Message",
                                 "Diff Summary", "File Changes"]
            df = rds_links.merge(rds_issues, on="Issue ID", how="left") \
                          .merge(rds_commits[commit_light_cols], on="Commit ID", how="left")

            # Free the full commits table — lookups have the big text
            del rds_commits, rds_issues, rds_links

            # Add Gemma_Summary as a stub so downstream code can reference the column
            # (text builders will use the lookup dict instead)
            df["Gemma_Summary"] = ""
            df["Full Diff"] = ""

            log.info(f"  Joined RDS (lightweight): {len(df)} rows, "
                     f"{df['Issue ID'].nunique()} issues, "
                     f"{df['Commit ID'].nunique()} commits, "
                     f"mem={df.memory_usage(deep=True).sum()/1e9:.2f} GB")
            n_pos = (df["Output"]==1).sum(); n_neg = (df["Output"]==0).sum()
            log.info(f"  pos={n_pos}, neg={n_neg}, ratio=1:{n_neg//max(n_pos,1)}")
        else:
            _fulldiff_lookup = None
            _gemma_lookup = None

            df_raw = pd.read_csv(ENRICHED_CSV)
            if "Gemma_Summary" not in df_raw.columns:
                raise ValueError("Gemma_Summary column not found!")
            log.info(f"Gemma_Summary: {df_raw['Gemma_Summary'].notna().sum()}/{len(df_raw)} non-null")

            for c in ["Issue Date","Commit Date"]:
                if c in df_raw.columns:
                    df_raw[c] = pd.to_datetime(df_raw[c], errors="coerce")

            if REUSE_RDS_CSV and os.path.exists(RDS_ENRICHED_CSV):
                log.info(f"Reusing enriched RDS: {RDS_ENRICHED_CSV}")
                df = pd.read_csv(RDS_ENRICHED_CSV)
                if "Gemma_Summary" not in df.columns:
                    raise ValueError("Gemma_Summary not in RDS CSV!")
            elif TRUE_LINKS_ONLY:
                df = build_rds_dataset(df_raw)
                df.to_csv(RDS_ENRICHED_CSV, index=False)
            else:
                df = df_raw

        df["Gemma_Summary"] = df["Gemma_Summary"].fillna("")
        for c in ["Issue Date","Commit Date"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")

    # ──────────────────────────────────────────────────────────
    #  Precompute metadata (once for all folds)
    # ──────────────────────────────────────────────────────────
    with stage(f"[{ds_name}] Precompute metadata"):
        meta = precompute_metadata(df)

    # ──────────────────────────────────────────────────────────
    #  Create 5-fold splits
    # ──────────────────────────────────────────────────────────
    with stage(f"[{ds_name}] Create 5-fold stratified splits"):
        folds = make_stratified_kfold_splits(df, n_folds=N_FOLDS, seed=RANDOM_SEED)

        # Save fold assignments
        fold_info = {}
        for fi in range(N_FOLDS):
            fold_info[f"fold{fi+1}"] = {
                "test_ids": [int(x) for x in folds[fi]],
                "n_test": len(folds[fi]),
            }
        with open(OUT_ROOT / "fold_assignments.json", "w") as f:
            json.dump(fold_info, f, indent=2)

    # ──────────────────────────────────────────────────────────
    #  Run each fold
    # ──────────────────────────────────────────────────────────
    all_results = []

    for fi in range(N_FOLDS):
        fold_t0 = time.perf_counter()

        test_ids  = set(folds[fi])
        train_ids = set()
        for fj in range(N_FOLDS):
            if fj != fi:
                train_ids.update(folds[fj])

        fold_out = OUT_ROOT / ds_name / f"fold{fi+1}"
        fold_out.mkdir(parents=True, exist_ok=True)

        # Save split for this fold
        with open(fold_out / "split.json", "w") as f:
            json.dump({
                "fold": fi+1,
                "n_train": len(train_ids),
                "n_test": len(test_ids),
                "train_ids": sorted(int(x) for x in train_ids),
                "test_ids": sorted(int(x) for x in test_ids),
            }, f, indent=2)

        print(f"\n{'='*70}")
        print(f"  FOLD {fi+1}/{N_FOLDS} — Train: {len(train_ids)} issues, Test: {len(test_ids)} issues")
        print(f"{'='*70}\n")

        fold_result = run_one_fold(fi, ds_name, df.copy(), train_ids, test_ids,
                                    fold_out, meta, sbert_model)
        all_results.append(fold_result)

        fold_dt = time.perf_counter() - fold_t0
        log.info(f"Fold {fi+1} completed in {fold_dt:.1f}s ({fold_dt/60:.1f} min)")

    # ──────────────────────────────────────────────────────────
    #  AGGREGATE RESULTS
    # ──────────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUT_ROOT / "all_folds_results.csv", index=False)

    print(f"\n{'='*70}")
    print(f"  5-FOLD CROSS-VALIDATION RESULTS — LinkRank v6")
    print(f"{'='*70}")
    print(f"\n  Per-Fold Results:")
    print(f"  {'Fold':>4} | {'Train':>5} | {'Test':>4} | {'KnownK-F1':>10} | {'ABSmm-F1':>10} | {'REL-F1':>10} | {'MRR':>6} | {'NDCG@K':>7} | {'P@K':>5}")
    print(f"  {'-'*4}-+-{'-'*5}-+-{'-'*4}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*6}-+-{'-'*7}-+-{'-'*5}")

    for _, row in results_df.iterrows():
        print(f"  {int(row['fold']):>4} | {int(row['n_train']):>5} | {int(row['n_test']):>4} | "
              f"{row['KnownK_F1']:>10.2f} | {row['ABSmm_F1']:>10.2f} | {row['REL_F1']:>10.2f} | "
              f"{row['MRR']:>6.2f} | {row['NDCG_at_K']:>7.2f} | {row['P_at_K']:>5.2f}")

    # Mean ± Std
    metrics = {
        "Known-K": ("KnownK_P", "KnownK_R", "KnownK_F1"),
        "ABS-mm":  ("ABSmm_P",  "ABSmm_R",  "ABSmm_F1"),
        "REL":     ("REL_P",    "REL_R",    "REL_F1"),
    }

    print(f"\n  {'='*60}")
    print(f"  AGGREGATED (Mean ± Std across {N_FOLDS} folds):")
    print(f"  {'='*60}")
    print(f"  {'Setting':<12} | {'Precision':>16} | {'Recall':>16} | {'F1':>16}")
    print(f"  {'-'*12}-+-{'-'*16}-+-{'-'*16}-+-{'-'*16}")

    agg_rows = []
    for setting, (pcol, rcol, fcol) in metrics.items():
        p_vals = results_df[pcol].values
        r_vals = results_df[rcol].values
        f_vals = results_df[fcol].values
        print(f"  {setting:<12} | {p_vals.mean():>6.2f} ± {p_vals.std():>5.2f} | "
              f"{r_vals.mean():>6.2f} ± {r_vals.std():>5.2f} | "
              f"{f_vals.mean():>6.2f} ± {f_vals.std():>5.2f}")
        agg_rows.append({
            "Setting": setting,
            "P_mean": round(p_vals.mean(), 2), "P_std": round(p_vals.std(), 2),
            "R_mean": round(r_vals.mean(), 2), "R_std": round(r_vals.std(), 2),
            "F1_mean": round(f_vals.mean(), 2), "F1_std": round(f_vals.std(), 2),
        })

    # Ranking metrics
    print(f"\n  {'Metric':<12} | {'Mean ± Std':>16}")
    print(f"  {'-'*12}-+-{'-'*16}")
    for col, label in [("MRR", "MRR"), ("NDCG_at_K", "NDCG@K"), ("P_at_K", "P@K")]:
        vals = results_df[col].values
        print(f"  {label:<12} | {vals.mean():>6.2f} ± {vals.std():>5.2f}")
        agg_rows.append({
            "Setting": label,
            "P_mean": round(vals.mean(), 2), "P_std": round(vals.std(), 2),
            "R_mean": 0, "R_std": 0, "F1_mean": 0, "F1_std": 0,
        })

    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(OUT_ROOT / "aggregated_results.csv", index=False)

    total_dt = time.perf_counter() - total_t0
    print(f"\n  Total time: {total_dt:.0f}s ({total_dt/60:.1f} min)")
    print(f"  Results saved to: {OUT_ROOT}")
    log.info(f"Total 5-fold CV time: {total_dt:.1f}s ({total_dt/60:.1f} min)")
