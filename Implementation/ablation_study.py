"""
LinkRank Ablation Study — Feature Group Combinations
=====================================================
Tests which feature groups contribute to performance.

Combinations tested:
  1. SBERT only                    (6 features)
  2. TF-IDF/SVD only              (5 features)
  3. BM25 only                    (5 features)
  4. Metadata only                (6 features)
  5. SBERT + Metadata             (12 features)
  6. SBERT + TF-IDF/SVD           (11 features)
  7. SBERT + BM25                 (11 features)
  8. SBERT + TF-IDF/SVD + Meta    (17 features)
  9. SBERT + BM25 + Meta          (17 features)
  10. TF-IDF/SVD + BM25 + Meta    (16 features)
  11. SBERT + TF-IDF/SVD + BM25   (16 features)
  12. ALL (v6 full)                (22 features)

All use same split, same Optuna tuning, same iterative refinement.
"""

import os, re, time, logging, math, json
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

# ──────────────────────────────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
# NOTE: ablation inputs are the legacy PyTorch flat CSVs (see paper Sec. 5, RQ3)
ENRICHED_CSV = str(REPO_ROOT / "Dataset/pytorch_enriched.csv")
RDS_ENRICHED_CSV = str(REPO_ROOT / "Dataset/pytorch_rds_enriched.csv")
SPLIT_JSON = REPO_ROOT / "results/stratified_split.json"

OUT_ROOT = REPO_ROOT / "results/ablation_study"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
SBERT_MODEL_NAME = "all-mpnet-base-v2"
SBERT_MAX_CHARS  = 2048
TFIDF_MIN_DF = 2; TFIDF_MAX_DF = 0.95; TFIDF_NGRAMS = (1, 2)
SVD_DIM_FULL = 128; SVD_DIM_CODE = 80; SVD_DIM_NL = 80; SVD_DIM_TITLE = 64; SVD_DIM_GEMMA = 100
TIME_TAU_DAYS = 7
USE_ITERATION_KNOWNK = True; USE_ITERATION_NOK_REL = True
ALPHA = 0.7; BETA = 0.3
OPTUNA_N_TRIALS = 50; OPTUNA_TIMEOUT = 300
TRAIN_FRACTION = 0.80
RDS_WINDOW_DAYS = 365

rng = np.random.RandomState(RANDOM_SEED)

# ──────────────────────────────────────────────────────────────────────
#  Feature group definitions
# ──────────────────────────────────────────────────────────────────────
SBERT_FEATURES = [
    "sbert_full", "sbert_nl", "sbert_title",
    "sbert_gemma_full", "sbert_gemma_nl", "sbert_gemma_title",
]
SVD_FEATURES = ["svd_full", "svd_code", "svd_nl", "svd_title", "svd_gemma"]
BM25_FEATURES = ["bm25_full", "bm25_code", "bm25_nl", "bm25_title", "bm25_gemma"]
META_FEATURES = ["feat_time", "feat_time_dir", "feat_file_bn_jaccard",
                 "feat_diff_size", "feat_nfiles", "feat_desc_len"]

ALL_22 = SVD_FEATURES + SBERT_FEATURES + BM25_FEATURES + META_FEATURES

ABLATION_CONFIGS = [
    ("01_SBERT_only",           SBERT_FEATURES),
    ("02_SVD_only",             SVD_FEATURES),
    ("03_BM25_only",            BM25_FEATURES),
    ("04_Meta_only",            META_FEATURES),
    ("05_SBERT+Meta",           SBERT_FEATURES + META_FEATURES),
    ("06_SBERT+SVD",            SBERT_FEATURES + SVD_FEATURES),
    ("07_SBERT+BM25",           SBERT_FEATURES + BM25_FEATURES),
    ("08_SBERT+SVD+Meta",       SBERT_FEATURES + SVD_FEATURES + META_FEATURES),
    ("09_SBERT+BM25+Meta",      SBERT_FEATURES + BM25_FEATURES + META_FEATURES),
    ("10_SVD+BM25+Meta",        SVD_FEATURES + BM25_FEATURES + META_FEATURES),
    ("11_SBERT+SVD+BM25",       SBERT_FEATURES + SVD_FEATURES + BM25_FEATURES),
    ("12_ALL_v6",               ALL_22),
]

# ──────────────────────────────────────────────────────────────────────
#  Logging
# ──────────────────────────────────────────────────────────────────────
log_path = OUT_ROOT / "ablation.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()],
)
log = logging.getLogger("ablation")

from contextlib import contextmanager
@contextmanager
def stage(name):
    log.info(f"▸ {name} ...")
    t0 = time.time()
    yield
    log.info(f"  ✓ {name} ({time.time()-t0:.1f}s, mem={mem_gb():.2f} GB)")

# ──────────────────────────────────────────────────────────────────────
#  Helpers (same as v6)
# ──────────────────────────────────────────────────────────────────────
def tokenize(text):
    return re.findall(r'\w+', str(text).lower())

def cos_sim_np(a, b):
    d = float(np.dot(a, b))
    n = float(np.linalg.norm(a) * np.linalg.norm(b))
    return d / n if n > 1e-12 else 0.0

def time_prox(d_issue, d_commit, tau=7):
    try:
        dt = abs((pd.Timestamp(d_issue) - pd.Timestamp(d_commit)).total_seconds()) / 86400
        return math.exp(-dt / tau)
    except Exception:
        return 0.0

def time_direction(d_issue, d_commit):
    try:
        dt = (pd.Timestamp(d_commit) - pd.Timestamp(d_issue)).total_seconds() / 86400
        return 1.0 / (1.0 + math.exp(-dt))
    except Exception:
        return 0.5

def jaccard(a, b):
    if not a and not b: return 0.0
    inter = len(a & b); union = len(a | b)
    return inter / union if union else 0.0

FILE_RE = re.compile(r'[\w\-./\\]+\.\w{1,6}')
def extract_file_paths(text):
    return set(m.lower() for m in FILE_RE.findall(str(text)))
def extract_file_basenames(text):
    return set(p.rsplit('/',1)[-1].rsplit('\\',1)[-1] for p in extract_file_paths(text))

DIFF_RE = re.compile(r'(\d+)\s+insertion|(\d+)\s+deletion|(\d+)\s+file')
def parse_diff_summary(text):
    if not text or pd.isna(text): return (0,0,0)
    a, d, f = 0, 0, 0
    for m in DIFF_RE.finditer(str(text)):
        if m.group(1): a += int(m.group(1))
        if m.group(2): d += int(m.group(2))
        if m.group(3): f += int(m.group(3))
    return (a, d, f)

def trunc(text, max_chars=2048):
    t = str(text) if text and not pd.isna(text) else ""
    return t[:max_chars]

def load_split_from_json(path):
    with open(path) as f:
        d = json.load(f)
    train = set(d["train_ids"]); test = set(d["test_ids"])
    info = {int(k): v for k, v in d.get("per_k", {}).items()}
    return train, test, info

def macro_percent(df):
    p = round(100*df["Precision"].mean(), 2) if len(df) else 0.0
    r = round(100*df["Recall"].mean(), 2) if len(df) else 0.0
    f = round(100*df["F1"].mean(), 2) if len(df) else 0.0
    return p, r, f

# ──────────────────────────────────────────────────────────────────────
#  Text builders (same as v6)
# ──────────────────────────────────────────────────────────────────────
def _agg_text(df, id_col, cols_dict, order):
    agg = {}
    for name, (col, method) in cols_dict.items():
        if col in df.columns:
            if method == "first":
                agg[name] = (col, "first")
            else:
                agg[name] = (col, lambda x: " ".join(x.dropna().astype(str)))
    if not agg:
        uniq = df[[id_col]].drop_duplicates()
        uniq["text"] = ""
        return uniq
    grp = df.groupby(id_col).agg(**agg).reset_index()
    parts = [grp[n].fillna("").astype(str) for n in order if n in grp.columns]
    grp["text"] = pd.concat(parts, axis=1).apply(lambda row: " ".join(row), axis=1)
    return grp[[id_col, "text"]]

def issue_text_full(d):
    return _agg_text(d, "Issue ID",
        {"title": ("Title","first"), "desc": ("Description","first"),
         "labels": ("Labels","first"), "comm": ("Comments","first")},
        ["title","desc","labels","comm"])
def issue_text_nl(d):
    return _agg_text(d, "Issue ID",
        {"title": ("Title","first"), "desc": ("Description","first"),
         "comm": ("Comments","first")}, ["title","desc","comm"])
def issue_text_title(d):
    return _agg_text(d, "Issue ID", {"title": ("Title","first")}, ["title"])
def issue_text_code(d):
    return _agg_text(d, "Issue ID",
        {"desc": ("Description","first"), "comm": ("Comments","first")}, ["desc","comm"])
def commit_text_full(d):
    return _agg_text(d, "Commit ID",
        {"msg": ("Message","first"), "dif": ("Diff Summary","first"),
         "files": ("File Changes","first"), "full": ("Full Diff","first")},
        ["msg","dif","files","full"])
def commit_text_code(d):
    return _agg_text(d, "Commit ID",
        {"dif": ("Diff Summary","first"), "files": ("File Changes","first"),
         "full": ("Full Diff","first")}, ["dif","files","full"])
def commit_text_msg(d):
    return _agg_text(d, "Commit ID", {"msg": ("Message","first")}, ["msg"])
def commit_text_gemma(d):
    return _agg_text(d, "Commit ID", {"gemma": ("Gemma_Summary","first")}, ["gemma"])

# ──────────────────────────────────────────────────────────────────────
#  BM25 Index
# ──────────────────────────────────────────────────────────────────────
class BM25Index:
    def __init__(self, commit_ids, commit_texts):
        self.commit_ids = list(commit_ids)
        self.cid_to_idx = {cid: i for i, cid in enumerate(self.commit_ids)}
        tokenized = [tokenize(t) for t in commit_texts]
        self.bm25 = BM25Okapi(tokenized)
    def score_batch(self, issue_text):
        query_tokens = tokenize(issue_text)
        if not query_tokens:
            return {cid: 0.0 for cid in self.commit_ids}
        scores = self.bm25.get_scores(query_tokens)
        return {cid: float(scores[i]) for i, cid in enumerate(self.commit_ids)}

# ──────────────────────────────────────────────────────────────────────
#  Precompute metadata
# ──────────────────────────────────────────────────────────────────────
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
#  COMPUTE ALL 22 FEATURES ONCE, THEN SLICE FOR EACH ABLATION
# ══════════════════════════════════════════════════════════════════════
def compute_all_features(df, train_ids, test_ids, meta, sbert_model):
    """Compute all 22 features for train and test, return DataFrames."""
    for c in ["Issue Date","Commit Date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    train_df = df[df["Issue ID"].isin(train_ids)].copy()
    test_df  = df[df["Issue ID"].isin(test_ids)].copy()
    log.info(f"Train issues={len(train_ids)}, rows={len(train_df):,}")
    log.info(f"Test  issues={len(test_ids)}, rows={len(test_df):,}")

    # ── SBERT embeddings ──
    def sbert_encode_unique(text_df, id_col, label, max_chars=SBERT_MAX_CHARS):
        with stage(f"SBERT encode [{label}] ({len(text_df)} texts)"):
            ids = text_df[id_col].tolist()
            texts = [trunc(t, max_chars) for t in text_df["text"].tolist()]
            embs = sbert_model.encode(texts, batch_size=64, show_progress_bar=True,
                                       normalize_embeddings=True)
            return {iid: embs[i] for i, iid in enumerate(ids)}

    all_issue_full  = issue_text_full(df)
    all_issue_nl    = issue_text_nl(df)
    all_issue_title = issue_text_title(df)
    all_commit_full = commit_text_full(df)
    all_commit_msg  = commit_text_msg(df)
    all_commit_gemma = commit_text_gemma(df)

    sbert_issue_full   = sbert_encode_unique(all_issue_full, "Issue ID", "issue_full")
    sbert_issue_nl     = sbert_encode_unique(all_issue_nl, "Issue ID", "issue_nl")
    sbert_issue_title  = sbert_encode_unique(all_issue_title, "Issue ID", "issue_title")
    sbert_commit_full  = sbert_encode_unique(all_commit_full, "Commit ID", "commit_full")
    sbert_commit_msg   = sbert_encode_unique(all_commit_msg, "Commit ID", "commit_msg")
    sbert_commit_gemma = sbert_encode_unique(all_commit_gemma, "Commit ID", "commit_gemma")

    SBERT_DIM = sbert_model.get_sentence_embedding_dimension()

    # ── BM25 indices ──
    with stage("Build BM25 indices"):
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
                                    all_commit_gemma["text"].fillna("").tolist())

        bm25_cache = {}
        all_issues = df["Issue ID"].unique()
        issue_full_map = dict(zip(all_issue_full["Issue ID"], all_issue_full["text"].fillna("")))
        issue_nl_map = dict(zip(all_issue_nl["Issue ID"], all_issue_nl["text"].fillna("")))
        issue_title_map = dict(zip(all_issue_title["Issue ID"], all_issue_title["text"].fillna("")))
        i_code_df = issue_text_code(df)
        issue_code_map = dict(zip(i_code_df["Issue ID"], i_code_df["text"].fillna("")))

        for iid in tqdm(all_issues, desc="BM25 scores"):
            bm25_cache[iid] = {
                "full": bm25_full_idx.score_batch(issue_full_map.get(iid, "")),
                "code": bm25_code_idx.score_batch(issue_code_map.get(iid, "")),
                "nl": bm25_nl_idx.score_batch(issue_nl_map.get(iid, "")),
                "title": bm25_title_idx.score_batch(issue_title_map.get(iid, "")),
                "gemma": bm25_gemma_idx.score_batch(issue_nl_map.get(iid, "")),
            }

    # ── TF-IDF/SVD ──
    def fit_tfidf_svd(issue_texts, commit_texts, svd_dim, label):
        with stage(f"TF-IDF+SVD [{label}] (dim={svd_dim})"):
            tfidf = TfidfVectorizer(min_df=TFIDF_MIN_DF, max_df=TFIDF_MAX_DF,
                                    ngram_range=TFIDF_NGRAMS, sublinear_tf=True)
            combined = pd.concat([issue_texts["text"], commit_texts["text"]], axis=0).fillna("")
            X = tfidf.fit_transform(combined)
            svd = SkTruncatedSVD(n_components=min(svd_dim, X.shape[1]-1), random_state=RANDOM_SEED)
            Xr = svd.fit_transform(X)
            Ei = Xr[:len(issue_texts)]; Ec = Xr[len(issue_texts):]
        return tfidf, svd, Ei, Ec

    i_full_tr = issue_text_full(train_df); c_full_tr = commit_text_full(train_df)
    i_code_tr = issue_text_code(train_df); c_code_tr = commit_text_code(train_df)
    i_nl_tr   = issue_text_nl(train_df);   c_nl_tr   = commit_text_msg(train_df)
    i_ttl_tr  = issue_text_title(train_df); c_msg_tr  = commit_text_msg(train_df)
    i_nl_gemma_tr = issue_text_nl(train_df); c_gemma_tr = commit_text_gemma(train_df)

    tf_f, sv_f, Ei_f_tr, Ec_f_tr = fit_tfidf_svd(i_full_tr, c_full_tr, SVD_DIM_FULL, "full")
    tf_c, sv_c, Ei_c_tr, Ec_c_tr = fit_tfidf_svd(i_code_tr, c_code_tr, SVD_DIM_CODE, "code")
    tf_n, sv_n, Ei_n_tr, Ec_n_tr = fit_tfidf_svd(i_nl_tr, c_nl_tr, SVD_DIM_NL, "nl")
    tf_t, sv_t, Ei_t_tr, Ec_t_tr = fit_tfidf_svd(i_ttl_tr, c_msg_tr, SVD_DIM_TITLE, "title")
    tf_g, sv_g, Ei_g_tr, Ec_g_tr = fit_tfidf_svd(i_nl_gemma_tr, c_gemma_tr, SVD_DIM_GEMMA, "gemma")

    with stage("Transform test SVD"):
        i_full_te = issue_text_full(test_df); c_full_te = commit_text_full(test_df)
        Ei_f_te = sv_f.transform(tf_f.transform(i_full_te["text"].fillna("")))
        Ec_f_te = sv_f.transform(tf_f.transform(c_full_te["text"].fillna("")))
        i_code_te = issue_text_code(test_df); c_code_te = commit_text_code(test_df)
        Ei_c_te = sv_c.transform(tf_c.transform(i_code_te["text"].fillna("")))
        Ec_c_te = sv_c.transform(tf_c.transform(c_code_te["text"].fillna("")))
        i_nl_te = issue_text_nl(test_df); c_nl_te = commit_text_msg(test_df)
        Ei_n_te = sv_n.transform(tf_n.transform(i_nl_te["text"].fillna("")))
        Ec_n_te = sv_n.transform(tf_n.transform(c_nl_te["text"].fillna("")))
        i_ttl_te = issue_text_title(test_df); c_msg_te = commit_text_msg(test_df)
        Ei_t_te = sv_t.transform(tf_t.transform(i_ttl_te["text"].fillna("")))
        Ec_t_te = sv_t.transform(tf_t.transform(c_msg_te["text"].fillna("")))
        i_nl_gemma_te = issue_text_nl(test_df); c_gemma_te = commit_text_gemma(test_df)
        Ei_g_te = sv_g.transform(tf_g.transform(i_nl_gemma_te["text"].fillna("")))
        Ec_g_te = sv_g.transform(tf_g.transform(c_gemma_te["text"].fillna("")))

    def build_idx(idf, cdf):
        return ({iid: i for i, iid in enumerate(idf["Issue ID"].tolist())},
                {cid: i for i, cid in enumerate(cdf["Commit ID"].tolist())})

    idx_f_tr = build_idx(i_full_tr, c_full_tr); idx_c_tr = build_idx(i_code_tr, c_code_tr)
    idx_n_tr = build_idx(i_nl_tr, c_nl_tr); idx_t_tr = build_idx(i_ttl_tr, c_msg_tr)
    idx_g_tr = build_idx(i_nl_gemma_tr, c_gemma_tr)
    idx_f_te = build_idx(i_full_te, c_full_te); idx_c_te = build_idx(i_code_te, c_code_te)
    idx_n_te = build_idx(i_nl_te, c_nl_te); idx_t_te = build_idx(i_ttl_te, c_msg_te)
    idx_g_te = build_idx(i_nl_gemma_te, c_gemma_te)

    # ── Build all 22 features ──
    def _svd_sim(iid, cid, idx_pair, Ei, Ec):
        i_map, c_map = idx_pair
        if iid in i_map and cid in c_map:
            return cos_sim_np(Ei[i_map[iid]], Ec[c_map[cid]])
        return 0.0

    def _sbert_sim(iid, cid, issue_dict, commit_dict):
        ei = issue_dict.get(iid); ec = commit_dict.get(cid)
        if ei is not None and ec is not None:
            return cos_sim_np(ei, ec)
        return 0.0

    def _bm25_score(iid, cid, channel):
        cache = bm25_cache.get(iid)
        if cache is None: return 0.0
        ch = cache.get(channel)
        if ch is None: return 0.0
        return ch.get(cid, 0.0)

    def build_features(d, split="train"):
        rows = []
        if split == "train":
            idxs = [(idx_f_tr, Ei_f_tr, Ec_f_tr), (idx_c_tr, Ei_c_tr, Ec_c_tr),
                    (idx_n_tr, Ei_n_tr, Ec_n_tr), (idx_t_tr, Ei_t_tr, Ec_t_tr),
                    (idx_g_tr, Ei_g_tr, Ec_g_tr)]
        else:
            idxs = [(idx_f_te, Ei_f_te, Ec_f_te), (idx_c_te, Ei_c_te, Ec_c_te),
                    (idx_n_te, Ei_n_te, Ec_n_te), (idx_t_te, Ei_t_te, Ec_t_te),
                    (idx_g_te, Ei_g_te, Ec_g_te)]

        it = tqdm(d.iterrows(), total=len(d), desc=f"features[{split}]")
        for _, row in it:
            iid, cid = row["Issue ID"], row["Commit ID"]
            # SVD (5)
            s_svd = [_svd_sim(iid, cid, idx, Ei, Ec) for idx, Ei, Ec in idxs]
            # SBERT (6)
            s_sbert = [
                _sbert_sim(iid, cid, sbert_issue_full, sbert_commit_full),
                _sbert_sim(iid, cid, sbert_issue_nl, sbert_commit_msg),
                _sbert_sim(iid, cid, sbert_issue_title, sbert_commit_msg),
                _sbert_sim(iid, cid, sbert_issue_full, sbert_commit_gemma),
                _sbert_sim(iid, cid, sbert_issue_nl, sbert_commit_gemma),
                _sbert_sim(iid, cid, sbert_issue_title, sbert_commit_gemma),
            ]
            # BM25 (5)
            s_bm25 = [_bm25_score(iid, cid, ch) for ch in ["full","code","nl","title","gemma"]]
            # Metadata (6)
            f_time = time_prox(row.get("Issue Date"), row.get("Commit Date"), TIME_TAU_DAYS)
            f_time_dir = time_direction(row.get("Issue Date"), row.get("Commit Date"))
            i_bn = meta["issue_basenames"].get(iid, set())
            c_bn = meta["commit_basenames"].get(cid, set())
            f_fbj = jaccard(i_bn, c_bn)
            added, deleted, nfiles = meta["commit_diff_stats"].get(cid, (0,0,0))
            f_ds = math.log1p(added + deleted)
            f_nf = math.log1p(nfiles)
            f_dl = math.log1p(meta["issue_desc_len"].get(iid, 0))
            s_meta = [f_time, f_time_dir, f_fbj, f_ds, f_nf, f_dl]

            rows.append([iid, cid] + s_svd + s_sbert + s_bm25 + s_meta + [row["Output"]])

        return pd.DataFrame(rows, columns=["Issue ID","Commit ID"] + ALL_22 + ["Output"])

    with stage("Build features: TRAIN"):
        train_feat = build_features(train_df, split="train")
    with stage("Build features: TEST"):
        test_feat = build_features(test_df, split="test")

    return train_feat, test_feat, sbert_issue_full, sbert_commit_full, sbert_commit_gemma, SBERT_DIM


# ══════════════════════════════════════════════════════════════════════
#  RUN ONE ABLATION CONFIG
# ══════════════════════════════════════════════════════════════════════
def run_ablation(config_name, features, train_feat, test_feat,
                 sbert_issue_full, sbert_commit_full, sbert_commit_gemma, SBERT_DIM):
    """Train LambdaMART on a subset of features, evaluate Known-K / ABS-mm / REL."""
    out_dir = OUT_ROOT / config_name
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"\n{'='*60}")
    log.info(f"  ABLATION: {config_name} ({len(features)} features)")
    log.info(f"  Features: {features}")
    log.info(f"{'='*60}")

    # Prepare rank data
    def prep_rank(df_feat):
        df_s = df_feat.sort_values("Issue ID").reset_index(drop=True)
        X = df_s[features].values
        y = df_s["Output"].astype(int).values
        groups = df_s.groupby("Issue ID").size().tolist()
        return df_s, X, y, groups

    uids = train_feat["Issue ID"].drop_duplicates().tolist()
    rng_local = np.random.RandomState(RANDOM_SEED)
    uids_shuffled = list(uids)
    rng_local.shuffle(uids_shuffled)
    n_dev = max(1, int(0.20 * len(uids_shuffled)))
    dev_ids = set(uids_shuffled[:n_dev])

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

    # Optuna tuning
    with stage(f"[{config_name}] Optuna tuning"):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        dtrain_opt = lgb.Dataset(Xtr, label=ytr, group=gtr, feature_name=features)
        dvalid_opt = lgb.Dataset(Xdv, label=ydv, group=gdv, reference=dtrain_opt, feature_name=features)

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
        log.info(f"Best NDCG@1={study.best_value:.4f}")

    # Train final model
    with stage(f"[{config_name}] Train LambdaMART"):
        dtrain = lgb.Dataset(Xtr, label=ytr, group=gtr, feature_name=features)
        dvalid = lgb.Dataset(Xdv, label=ydv, group=gdv, reference=dtrain, feature_name=features)
        params = dict(
            objective="lambdarank", metric=["ndcg"], ndcg_eval_at=[1,3,5],
            feature_pre_filter=False, bagging_freq=1,
            verbose=-1, num_threads=os.cpu_count() or 1, seed=RANDOM_SEED,
            **best_params
        )
        if TORCH_HAS_CUDA:
            params.update(device_type="gpu", gpu_platform_id=0, gpu_device_id=0)
        model = lgb.train(
            params, dtrain, valid_sets=[dtrain, dvalid], valid_names=["train","valid"],
            num_boost_round=3000,
            callbacks=[lgb.log_evaluation(0),
                       lgb.early_stopping(stopping_rounds=200, verbose=False)]
        )
        log.info(f"Best iteration: {model.best_iteration}")

        imp = model.feature_importance(importance_type='gain')
        feat_imp = sorted(zip(features, imp), key=lambda x: -x[1])
        log.info("Feature importance:")
        for fn, iv in feat_imp:
            log.info(f"  {fn}: {iv:.2f}")

    # Scoring
    def score_pool(pool_df):
        if len(pool_df) == 0:
            return pool_df.assign(score=[])
        X = pool_df[features].values
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

    # Iterative refinement (only if SBERT features present)
    has_sbert = any(f.startswith("sbert") for f in features)
    issue_vec_state = {}

    def init_issue_state(iid):
        v = sbert_issue_full.get(iid)
        if v is not None and has_sbert:
            issue_vec_state[iid] = v.copy() / (np.linalg.norm(v) + 1e-12)
        else:
            issue_vec_state[iid] = np.zeros(SBERT_DIM)

    def update_issue_state(iid, cid):
        if not has_sbert: return
        v_i = issue_vec_state[iid]
        v_c = sbert_commit_gemma.get(cid)
        if v_c is None: v_c = sbert_commit_full.get(cid)
        if v_c is not None:
            v_new = ALPHA * v_i + BETA * v_c
            issue_vec_state[iid] = v_new / (np.linalg.norm(v_new) + 1e-12)

    def refresh_features_for_issue_pool(iid, pool_df):
        if len(pool_df) == 0 or not has_sbert: return pool_df
        v_i = issue_vec_state[iid]
        new_sbert = []; new_sbert_gemma = []
        for _, r in pool_df.iterrows():
            cid = r["Commit ID"]
            vc = sbert_commit_full.get(cid)
            new_sbert.append(cos_sim_np(v_i, vc) if vc is not None else 0.0)
            vc_g = sbert_commit_gemma.get(cid)
            new_sbert_gemma.append(cos_sim_np(v_i, vc_g) if vc_g is not None else 0.0)
        result = pool_df.copy()
        if "sbert_full" in features:
            result["sbert_full"] = new_sbert
        if "sbert_gemma_full" in features:
            result["sbert_gemma_full"] = new_sbert_gemma
        return result

    # ── Known-K ──
    test_rows_by_issue = {iid: tst_s[tst_s["Issue ID"]==iid].reset_index(drop=True)
                          for iid in tst_s["Issue ID"].unique()}
    true_by_issue = tst_s[tst_s["Output"]==1].groupby("Issue ID")["Commit ID"].apply(set).to_dict()
    issues_test = sorted(true_by_issue.keys())

    rowsK = []
    for iid in issues_test:
        true_set = true_by_issue[iid]; K = len(true_set)
        pool = test_rows_by_issue[iid].copy()
        init_issue_state(iid)
        picks = []
        for _ in range(K):
            ranked = score_pool(pool)
            if len(ranked) == 0: break
            cid = ranked.iloc[0]["Commit ID"]
            picks.append(cid)
            if USE_ITERATION_KNOWNK and has_sbert:
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
                          Precision=p, Recall=r, F1=f1))
    res_knownK = pd.DataFrame(rowsK)
    res_knownK.to_csv(out_dir / "results_KnownK.csv", index=False)

    # ── No-K ──
    def mfi(pred_set, true_set):
        inter = len(pred_set & true_set)
        p = inter/max(len(pred_set),1); r = inter/max(len(true_set),1)
        f1 = (2*inter)/max(len(pred_set)+len(true_set),1)
        return dict(Precision=p, Recall=r, F1=f1)

    dev_rows_by_issue = {iid: dev_feat[dev_feat["Issue ID"]==iid].reset_index(drop=True)
                         for iid in dev_feat["Issue ID"].unique()}
    true_dev = dev_feat[dev_feat["Output"]==1].groupby("Issue ID")["Commit ID"].apply(set).to_dict()
    issues_dev = sorted(true_dev.keys())

    tst_rbi = {iid: tst_s[tst_s["Issue ID"]==iid].reset_index(drop=True)
               for iid in tst_s["Issue ID"].unique()}
    true_tst = tst_s[tst_s["Output"]==1].groupby("Issue ID")["Commit ID"].apply(set).to_dict()
    issues_tst = sorted(true_tst.keys())

    ranked_dev = {iid: score_pool(d) for iid,d in dev_rows_by_issue.items()}
    ranked_tst = {iid: score_pool(d) for iid,d in tst_rbi.items()}

    # ABS-mm
    taus = [x/100 for x in range(10,96,2)]
    best_abs = (-1, None)
    for t in taus:
        rows = [mfi(set(ranked_dev[iid][ranked_dev[iid]["score_mm"]>=t]["Commit ID"]),
                    true_dev[iid]) for iid in issues_dev]
        sc = pd.DataFrame(rows)["F1"].mean()
        if sc > best_abs[0]: best_abs = (sc, t)

    # REL
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

    # Test ABS-mm
    rows_abs = []
    for iid in issues_tst:
        rd = ranked_tst[iid]
        pred = set(rd[rd["score_mm"]>=best_abs[1]]["Commit ID"])
        rows_abs.append(dict(Issue=iid, **mfi(pred, true_tst[iid])))
    res_absmm = pd.DataFrame(rows_abs)
    res_absmm.to_csv(out_dir / "results_ABSmm.csv", index=False)

    # Test REL iterative
    rows_rel = []
    for iid in issues_tst:
        init_issue_state(iid)
        pool = ranked_tst[iid].copy()
        if len(pool)==0:
            rows_rel.append(dict(Issue=iid, **mfi(set(), true_tst[iid]))); continue
        best0 = float(pool["score"].iloc[0])
        accepted = []
        while len(pool):
            top = pool.iloc[0]
            if float(top["score"]) >= best_rel[1] * best0:
                cid = top["Commit ID"]; accepted.append(cid)
                if USE_ITERATION_NOK_REL and has_sbert:
                    update_issue_state(iid, cid)
                    pool = pool[pool["Commit ID"]!=cid].reset_index(drop=True)
                    if len(pool):
                        pool = refresh_features_for_issue_pool(iid, pool)
                        pool = score_pool(pool)
                else:
                    pool = pool[pool["Commit ID"]!=cid].reset_index(drop=True)
            else:
                break
        rows_rel.append(dict(Issue=iid, **mfi(set(accepted), true_tst[iid])))
    res_rel = pd.DataFrame(rows_rel)
    res_rel.to_csv(out_dir / "results_REL.csv", index=False)

    # Summary
    pK, rK, fK = macro_percent(res_knownK)
    pA, rA, fA = macro_percent(res_absmm)
    pR, rR, fR = macro_percent(res_rel)

    result = {
        "config": config_name, "n_features": len(features),
        "KnownK_P": pK, "KnownK_R": rK, "KnownK_F1": fK,
        "ABSmm_P": pA, "ABSmm_R": rA, "ABSmm_F1": fA,
        "REL_P": pR, "REL_R": rR, "REL_F1": fR,
    }
    log.info(f"  Known-K: P={pK}, R={rK}, F1={fK}")
    log.info(f"  ABS-mm:  P={pA}, R={rA}, F1={fA}")
    log.info(f"  REL:     P={pR}, R={rR}, F1={fR}")

    return result


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    with stage("Load SBERT model"):
        sbert_model = SentenceTransformer(SBERT_MODEL_NAME,
                                          device="cuda" if TORCH_HAS_CUDA else "cpu")

    with stage("Load data"):
        df_raw = pd.read_csv(ENRICHED_CSV)
        for c in ["Issue Date","Commit Date"]:
            if c in df_raw.columns:
                df_raw[c] = pd.to_datetime(df_raw[c], errors="coerce")

        if os.path.exists(RDS_ENRICHED_CSV):
            log.info(f"Reusing enriched RDS: {RDS_ENRICHED_CSV}")
            df = pd.read_csv(RDS_ENRICHED_CSV)
        else:
            raise FileNotFoundError(f"RDS CSV not found: {RDS_ENRICHED_CSV}")

        df["Gemma_Summary"] = df["Gemma_Summary"].fillna("")
        for c in ["Issue Date","Commit Date"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")

    with stage("Load split"):
        train_ids, test_ids, split_info = load_split_from_json(SPLIT_JSON)
        log.info(f"Split: train={len(train_ids)}, test={len(test_ids)}")

    with stage("Precompute metadata"):
        meta = precompute_metadata(df)

    with stage("Compute ALL 22 features (once)"):
        train_feat, test_feat, sbert_issue_full, sbert_commit_full, sbert_commit_gemma, SBERT_DIM = \
            compute_all_features(df, train_ids, test_ids, meta, sbert_model)

    log.info(f"\nTrain features shape: {train_feat.shape}")
    log.info(f"Test features shape:  {test_feat.shape}")

    # ── Run all ablation configs ──
    all_results = []
    for config_name, features in ABLATION_CONFIGS:
        t0 = time.time()
        result = run_ablation(config_name, features, train_feat, test_feat,
                              sbert_issue_full, sbert_commit_full, sbert_commit_gemma, SBERT_DIM)
        result["time_sec"] = round(time.time() - t0, 1)
        all_results.append(result)

    # ── Final summary table ──
    summary = pd.DataFrame(all_results)
    summary.to_csv(OUT_ROOT / "ablation_summary.csv", index=False)

    print(f"\n{'='*90}")
    print(f"  ABLATION STUDY — COMPLETE RESULTS")
    print(f"{'='*90}")
    print(f"\n{'Config':<30} {'#F':>3} | {'KnownK':>8} {'ABS-mm':>8} {'REL':>8} | {'Time':>6}")
    print(f"{'-'*30}-{'-'*3}-+-{'-'*8}-{'-'*8}-{'-'*8}-+-{'-'*6}")
    for r in all_results:
        print(f"{r['config']:<30} {r['n_features']:>3} | "
              f"{r['KnownK_F1']:>8.2f} {r['ABSmm_F1']:>8.2f} {r['REL_F1']:>8.2f} | "
              f"{r['time_sec']:>5.0f}s")

    # Best config
    best = max(all_results, key=lambda x: x["KnownK_F1"])
    print(f"\n  🏆 Best Known-K F1: {best['config']} = {best['KnownK_F1']:.2f}")
    best_abs = max(all_results, key=lambda x: x["ABSmm_F1"])
    print(f"  🏆 Best ABS-mm F1:  {best_abs['config']} = {best_abs['ABSmm_F1']:.2f}")
    best_rel = max(all_results, key=lambda x: x["REL_F1"])
    print(f"  🏆 Best REL F1:     {best_rel['config']} = {best_rel['REL_F1']:.2f}")

    print(f"\nResults saved to: {OUT_ROOT}")
    print(f"Summary CSV:      {OUT_ROOT / 'ablation_summary.csv'}")
