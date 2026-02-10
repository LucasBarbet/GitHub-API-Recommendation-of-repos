#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outputs:
- Saves a .pkl per model type in --out_dir
- Prints Eval and Test metrics (Precision@K, Recall@K, HitRate@K, MAP@K, NDCG@K)

python train_github_repo_reco_models.py \
--data_dir /info/raid-etu/m2/s2101052/data100repos \
--out_dir ./trained_models \
--top_k 10 \
--min_repo_freq 2 \
--n_eval_candidates 2000 \
--seed 42
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import pickle
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# sklearn models
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# Optional torch for BPR
try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None


# -----------------------------
# Metrics
# -----------------------------
def precision_at_k(pred: Sequence[str], truth: Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    pred_k = list(pred)[:k]
    return len(set(pred_k) & truth) / float(k)

def recall_at_k(pred: Sequence[str], truth: Set[str], k: int) -> float:
    if not truth:
        return 0.0
    pred_k = list(pred)[:k]
    return len(set(pred_k) & truth) / float(len(truth))

def hit_rate_at_k(pred: Sequence[str], truth: Set[str], k: int) -> float:
    pred_k = list(pred)[:k]
    return 1.0 if len(set(pred_k) & truth) > 0 else 0.0

def average_precision_at_k(pred: Sequence[str], truth: Set[str], k: int) -> float:
    if not truth or k <= 0:
        return 0.0
    pred_k = list(pred)[:k]
    hit_count = 0
    sum_prec = 0.0
    for i, item in enumerate(pred_k, start=1):
        if item in truth:
            hit_count += 1
            sum_prec += hit_count / float(i)
    denom = min(len(truth), k)
    return sum_prec / float(denom) if denom > 0 else 0.0

def ndcg_at_k(pred: Sequence[str], truth: Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    pred_k = list(pred)[:k]
    dcg = 0.0
    for i, item in enumerate(pred_k, start=1):
        if item in truth:
            dcg += 1.0 / math.log2(i + 1)

    ideal_hits = min(len(truth), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0

def aggregate_rank_metrics(per_user_preds: Dict[str, List[str]],
                           truth_by_user: Dict[str, Set[str]],
                           k: int) -> Dict[str, float]:
    metrics = {
        "precision@k": [],
        "recall@k": [],
        "hit_rate@k": [],
        "map@k": [],
        "ndcg@k": [],
    }
    users = [u for u in truth_by_user.keys() if len(truth_by_user[u]) > 0 and u in per_user_preds]
    for u in users:
        pred = per_user_preds[u]
        truth = truth_by_user[u]
        metrics["precision@k"].append(precision_at_k(pred, truth, k))
        metrics["recall@k"].append(recall_at_k(pred, truth, k))
        metrics["hit_rate@k"].append(hit_rate_at_k(pred, truth, k))
        metrics["map@k"].append(average_precision_at_k(pred, truth, k))
        metrics["ndcg@k"].append(ndcg_at_k(pred, truth, k))

    return {
        **{m: float(np.mean(v)) if len(v) else 0.0 for m, v in metrics.items()},
        "k": int(k),
        "n_users_eval": int(len(truth_by_user)),
        "n_users_scored": int(len(users)),
        "avg_truth_per_user": float(np.mean([len(truth_by_user[u]) for u in users])) if users else 0.0,
    }


# -----------------------------
# Data loading & splitting
# -----------------------------
def load_txt_dataset(data_dir: str) -> pd.DataFrame:
    """
    Reads files like:
      login : [owner1/repoA, owner2/repoB, ...]
    Returns df columns: user, repo, rating(=1)
    """
    paths = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    if not paths:
        raise FileNotFoundError(f"No .txt files found in {data_dir}")

    interactions: List[Tuple[str, str, int]] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                login, rest = line.split(" : [", 1)
                repos_str = rest.rstrip("]\n")
                repos = [r.strip() for r in repos_str.split(",") if r.strip()]
                # optional de-dup per user line
                for repo in dict.fromkeys(repos).keys():
                    interactions.append((login, repo, 1))

    df = pd.DataFrame(interactions, columns=["user", "repo", "rating"])
    return df


def filter_by_repo_freq(df: pd.DataFrame, min_repo_freq: int) -> pd.DataFrame:
    """
    Keep repos that appear at least min_repo_freq times (across all users) to ensure shared signal.
    """
    if min_repo_freq <= 1:
        return df
    vc = df["repo"].value_counts()
    keep = set(vc[vc >= min_repo_freq].index)
    out = df[df["repo"].isin(keep)].copy()
    return out


def per_user_split(df: pd.DataFrame,
                   train_ratio: float = 0.8,
                   eval_ratio: float = 0.1,
                   seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    train_rows, eval_rows, test_rows = [], [], []

    for u, g in df.groupby("user"):
        repos = g["repo"].values
        if len(repos) < 3:
            for r in repos:
                train_rows.append((u, r, 1))
            continue

        repos = repos.copy()
        rng.shuffle(repos)
        n = len(repos)
        n_train = int(train_ratio * n)
        n_eval = int((train_ratio + eval_ratio) * n)

        train_repos = repos[:n_train]
        eval_repos = repos[n_train:n_eval]
        test_repos = repos[n_eval:]

        train_rows += [(u, r, 1) for r in train_repos]
        eval_rows += [(u, r, 1) for r in eval_repos]
        test_rows += [(u, r, 1) for r in test_repos]

    train_df = pd.DataFrame(train_rows, columns=["user", "repo", "rating"])
    eval_df = pd.DataFrame(eval_rows, columns=["user", "repo", "rating"])
    test_df = pd.DataFrame(test_rows, columns=["user", "repo", "rating"])
    return train_df, eval_df, test_df


def make_warm_splits(train_df: pd.DataFrame,
                     eval_df: pd.DataFrame,
                     test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep only (user,repo) pairs in eval/test where:
    - user exists in train
    - repo exists in train
    """
    train_users = set(train_df["user"].unique())
    train_repos = set(train_df["repo"].unique())
    eval_warm = eval_df[eval_df["user"].isin(train_users) & eval_df["repo"].isin(train_repos)].copy()
    test_warm = test_df[test_df["user"].isin(train_users) & test_df["repo"].isin(train_repos)].copy()
    return eval_warm, test_warm


# -----------------------------
# Candidate sampling evaluation
# -----------------------------
def build_train_seen(train_df: pd.DataFrame) -> Dict[str, Set[str]]:
    return train_df.groupby("user")["repo"].apply(lambda s: set(s.values)).to_dict()

def build_truth_by_user(eval_df_warm: pd.DataFrame) -> Dict[str, Set[str]]:
    if len(eval_df_warm) == 0:
        return {}
    return eval_df_warm.groupby("user")["repo"].apply(lambda s: set(s.values)).to_dict()

def sample_candidates_for_user(
    user: str,
    truth: Set[str],
    seen: Set[str],
    all_train_repos: Sequence[str],
    n_candidates: int,
    rng: np.random.RandomState,
) -> List[str]:
    """
    Candidate pool = truth items + sampled negatives from TRAIN_REPOS \ (seen ∪ truth)
    Ensures all truth repos are actually rankable, avoiding trivial 0.0 metrics.
    """
    # Always include truth
    cand = set(truth)
    # negatives pool
    exclude = set(seen) | set(truth)
    pool = [r for r in all_train_repos if r not in exclude]
    if n_candidates <= 0:
        # Rank against all train repos (can be huge)
        return list(cand) + pool
    need = max(0, n_candidates - len(cand))
    if need > 0 and len(pool) > 0:
        # sample without replacement
        if need >= len(pool):
            sampled = pool
        else:
            idx = rng.choice(len(pool), size=need, replace=False)
            sampled = [pool[i] for i in idx]
        cand.update(sampled)
    return list(cand)


# -----------------------------
# Common model interface
# -----------------------------
class BaseRecommender:
    name: str = "base"

    def fit(self, train_df: pd.DataFrame) -> "BaseRecommender":
        raise NotImplementedError

    def score_candidates(self, user: str, candidates: Sequence[str], seen: Optional[Set[str]] = None) -> np.ndarray:
        """Return scores aligned with `candidates` (higher = better)."""
        raise NotImplementedError

    def recommend(self,
                  user: str,
                  k: int = 10,
                  candidates: Optional[Sequence[str]] = None,
                  seen: Optional[Set[str]] = None) -> List[str]:
        if candidates is None:
            raise ValueError("recommend() requires candidates in this script (evaluation uses sampled candidates).")
        scores = self.score_candidates(user, candidates, seen=seen)
        if len(scores) == 0:
            return []
        k = min(k, len(candidates))
        # partial top-k
        top_idx = np.argpartition(-scores, kth=k-1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [candidates[i] for i in top_idx]


# -----------------------------
# Model 1: Popularity baseline
# -----------------------------
@dataclass
class PopularityRecommender(BaseRecommender):
    name: str = "popularity"
    repo_score: Dict[str, float] = None

    def fit(self, train_df: pd.DataFrame) -> "PopularityRecommender":
        vc = train_df["repo"].value_counts()
        self.repo_score = vc.to_dict()
        return self

    def score_candidates(self, user: str, candidates: Sequence[str], seen: Optional[Set[str]] = None) -> np.ndarray:
        # Score = frequency in train; unseen gets 0
        return np.array([float(self.repo_score.get(r, 0.0)) for r in candidates], dtype=np.float32)


# -----------------------------
# Model 2: TF-IDF content-based
# -----------------------------
def _repo_tokenizer(s: str) -> List[str]:
    # split on non-alphanumerics + also split owner/repo
    parts = re.split(r"[^a-zA-Z0-9]+", s.lower())
    return [p for p in parts if p]

@dataclass
class TfidfContentRecommender(BaseRecommender):
    name: str = "tfidf_content"
    max_features: int = 200_000
    ngram_max: int = 2

    vectorizer: Optional[TfidfVectorizer] = None
    repo_list: Optional[List[str]] = None
    repo2idx: Optional[Dict[str, int]] = None
    repo_tfidf = None  # sparse (n_items x n_features)
    user_train_items: Optional[Dict[str, List[str]]] = None

    def fit(self, train_df: pd.DataFrame) -> "TfidfContentRecommender":
        self.repo_list = sorted(train_df["repo"].unique().tolist())
        self.repo2idx = {r: i for i, r in enumerate(self.repo_list)}

        self.user_train_items = train_df.groupby("user")["repo"].apply(list).to_dict()

        self.vectorizer = TfidfVectorizer(
            tokenizer=_repo_tokenizer,
            lowercase=True,
            max_features=self.max_features,
            ngram_range=(1, self.ngram_max),
            min_df=1,
            norm="l2",
        )
        self.repo_tfidf = self.vectorizer.fit_transform(self.repo_list)
        return self

    def _user_profile_vector(self, user: str):
        items = self.user_train_items.get(user, [])
        idx = [self.repo2idx[r] for r in items if r in self.repo2idx]
        if not idx:
            return None
        # mean of sparse rows
        v = self.repo_tfidf[idx].mean(axis=0)
        return v  # (1 x n_features) matrix-like

    def score_candidates(self, user: str, candidates: Sequence[str], seen: Optional[Set[str]] = None) -> np.ndarray:
        u_vec = self._user_profile_vector(user)
        if u_vec is None:
            return np.zeros(len(candidates), dtype=np.float32)

        cand_idx = [self.repo2idx.get(r, -1) for r in candidates]
        # unknown candidates => score 0
        scores = np.zeros(len(candidates), dtype=np.float32)
        valid_pos = [i for i, idx in enumerate(cand_idx) if idx >= 0]
        if not valid_pos:
            return scores
        valid_idx = [cand_idx[i] for i in valid_pos]
        # cosine similarity because TF-IDF is L2-normalized
        sims = (self.repo_tfidf[valid_idx] @ u_vec.T).A.ravel().astype(np.float32)
        for out_i, s in zip(valid_pos, sims):
            scores[out_i] = s
        return scores


# -----------------------------
# Model 3: TruncatedSVD MF
# -----------------------------
@dataclass
class TruncatedSVDRecommender(BaseRecommender):
    name: str = "svd_mf"
    n_components: int = 64
    random_state: int = 42

    user2idx: Optional[Dict[str, int]] = None
    idx2user: Optional[List[str]] = None
    repo2idx: Optional[Dict[str, int]] = None
    idx2repo: Optional[List[str]] = None

    user_factors: Optional[np.ndarray] = None  # (n_users x n_components)
    item_factors: Optional[np.ndarray] = None  # (n_items x n_components)

    def fit(self, train_df: pd.DataFrame) -> "TruncatedSVDRecommender":
        self.idx2user = sorted(train_df["user"].unique().tolist())
        self.idx2repo = sorted(train_df["repo"].unique().tolist())
        self.user2idx = {u: i for i, u in enumerate(self.idx2user)}
        self.repo2idx = {r: i for i, r in enumerate(self.idx2repo)}

        u_idx = train_df["user"].map(self.user2idx).values
        r_idx = train_df["repo"].map(self.repo2idx).values
        data = np.ones(len(train_df), dtype=np.float32)
        X = csr_matrix((data, (u_idx, r_idx)), shape=(len(self.idx2user), len(self.idx2repo)))

        svd = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        self.user_factors = svd.fit_transform(X).astype(np.float32)
        self.item_factors = svd.components_.T.astype(np.float32)
        return self

    def score_candidates(self, user: str, candidates: Sequence[str], seen: Optional[Set[str]] = None) -> np.ndarray:
        if user not in self.user2idx:
            return np.zeros(len(candidates), dtype=np.float32)
        u = self.user2idx[user]
        uvec = self.user_factors[u]  # (d,)
        scores = np.zeros(len(candidates), dtype=np.float32)

        cand_idx = [self.repo2idx.get(r, -1) for r in candidates]
        valid_pos = [i for i, idx in enumerate(cand_idx) if idx >= 0]
        if not valid_pos:
            return scores
        valid_idx = [cand_idx[i] for i in valid_pos]
        V = self.item_factors[valid_idx]  # (m,d)
        sims = V @ uvec  # (m,)
        for out_i, s in zip(valid_pos, sims):
            scores[out_i] = float(s)
        return scores


# -----------------------------
# Model 4: BPR-MF (PyTorch)
# -----------------------------
if torch is not None:
    class _BPRMF(nn.Module):
        def __init__(self, n_users: int, n_items: int, n_factors: int = 64):
            super().__init__()
            self.user_emb = nn.Embedding(n_users, n_factors)
            self.item_emb = nn.Embedding(n_items, n_factors)
            self.item_bias = nn.Embedding(n_items, 1)

            nn.init.normal_(self.user_emb.weight, std=0.01)
            nn.init.normal_(self.item_emb.weight, std=0.01)
            nn.init.zeros_(self.item_bias.weight)

        def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
            u = self.user_emb(users)
            v = self.item_emb(items)
            b = self.item_bias(items).squeeze(-1)
            return (u * v).sum(dim=1) + b


@dataclass
class BPRMFRecommender(BaseRecommender):
    name: str = "bpr_mf"
    n_factors: int = 64
    epochs: int = 20
    batch_size: int = 4096
    lr: float = 1e-2
    weight_decay: float = 1e-6
    seed: int = 42
    device: str = "cpu"

    user2idx: Optional[Dict[str, int]] = None
    repo2idx: Optional[Dict[str, int]] = None
    idx2repo: Optional[List[str]] = None
    user_pos: Optional[Dict[int, Set[int]]] = None

    # stored parameters for pickling
    state_dict: Optional[Dict] = None

    def fit(self, train_df: pd.DataFrame) -> "BPRMFRecommender":
        if torch is None:
            raise RuntimeError("PyTorch not available. Install torch to use BPRMFRecommender.")

        rng = np.random.RandomState(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        all_users = sorted(train_df["user"].unique().tolist())
        all_repos = sorted(train_df["repo"].unique().tolist())
        self.user2idx = {u: i for i, u in enumerate(all_users)}
        self.repo2idx = {r: i for i, r in enumerate(all_repos)}
        self.idx2repo = all_repos

        n_users = len(all_users)
        n_items = len(all_repos)

        # positives
        user_pos: Dict[int, Set[int]] = {}
        for u, r in train_df[["user", "repo"]].itertuples(index=False):
            ui = self.user2idx[u]
            ri = self.repo2idx[r]
            user_pos.setdefault(ui, set()).add(ri)
        self.user_pos = user_pos
        train_users_idx = np.array(list(user_pos.keys()), dtype=np.int64)

        model = _BPRMF(n_users=n_users, n_items=n_items, n_factors=self.n_factors).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        user_pos_lists = {u: np.fromiter(items, dtype=np.int64) for u, items in user_pos.items()}

        for epoch in range(1, self.epochs + 1):
            model.train()
            # heuristic number of steps
            n_steps = max(1, (len(train_users_idx) * 5) // self.batch_size)
            losses = []

            for _ in range(n_steps):
                users = rng.choice(train_users_idx, size=self.batch_size, replace=True)

                pos = np.empty(self.batch_size, dtype=np.int64)
                neg = np.empty(self.batch_size, dtype=np.int64)

                for i, u in enumerate(users):
                    pos_items = user_pos_lists[u]
                    pos[i] = pos_items[rng.randint(len(pos_items))]

                    j = rng.randint(n_items)
                    while j in user_pos[u]:
                        j = rng.randint(n_items)
                    neg[i] = j

                users_t = torch.tensor(users, dtype=torch.long, device=self.device)
                pos_t = torch.tensor(pos, dtype=torch.long, device=self.device)
                neg_t = torch.tensor(neg, dtype=torch.long, device=self.device)

                s_pos = model.score(users_t, pos_t)
                s_neg = model.score(users_t, neg_t)

                loss = -torch.log(torch.sigmoid(s_pos - s_neg) + 1e-8).mean()

                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.detach().cpu()))

            print(f"[BPR] epoch {epoch:02d}/{self.epochs} | bpr_loss={np.mean(losses):.4f}")

        # store for pickling
        self.state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        return self

    def _build_model(self) -> "_BPRMF":
        # rebuild a torch module from state_dict when scoring
        if torch is None:
            raise RuntimeError("PyTorch not available.")
        n_users = len(self.user2idx)
        n_items = len(self.repo2idx)
        model = _BPRMF(n_users=n_users, n_items=n_items, n_factors=self.n_factors)
        model.load_state_dict(self.state_dict)
        model.to(self.device)
        model.eval()
        return model

    def score_candidates(self, user: str, candidates: Sequence[str], seen: Optional[Set[str]] = None) -> np.ndarray:
        if user not in self.user2idx:
            return np.zeros(len(candidates), dtype=np.float32)

        cand_idx = [self.repo2idx.get(r, -1) for r in candidates]
        scores = np.zeros(len(candidates), dtype=np.float32)

        valid_pos = [i for i, idx in enumerate(cand_idx) if idx >= 0]
        if not valid_pos:
            return scores

        model = self._build_model()
        u_idx = self.user2idx[user]
        items = torch.tensor([cand_idx[i] for i in valid_pos], dtype=torch.long, device=self.device)
        users = torch.full((len(valid_pos),), u_idx, dtype=torch.long, device=self.device)
        with torch.no_grad():
            s = model.score(users, items).detach().cpu().numpy().astype(np.float32)
        for out_i, sc in zip(valid_pos, s):
            scores[out_i] = sc
        return scores


# -----------------------------
# Training + evaluation driver
# -----------------------------
def evaluate_model(
    model: BaseRecommender,
    train_df: pd.DataFrame,
    eval_df_warm: pd.DataFrame,
    top_k: int,
    n_candidates: int,
    seed: int,
) -> Dict[str, float]:
    train_seen = build_train_seen(train_df)
    truth_by_user = build_truth_by_user(eval_df_warm)
    all_train_repos = train_df["repo"].unique().tolist()
    rng = np.random.RandomState(seed)

    per_user_preds: Dict[str, List[str]] = {}
    for user, truth in truth_by_user.items():
        seen = train_seen.get(user, set())
        candidates = sample_candidates_for_user(
            user=user,
            truth=truth,
            seen=seen,
            all_train_repos=all_train_repos,
            n_candidates=n_candidates,
            rng=rng,
        )
        # rank on sampled candidates
        pred = model.recommend(user=user, k=top_k, candidates=candidates, seen=seen)
        per_user_preds[user] = pred

    return aggregate_rank_metrics(per_user_preds, truth_by_user, k=top_k)


def save_pkl(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Directory containing *.txt user -> [repos] files")
    ap.add_argument("--out_dir", type=str, required=True, help="Where to save .pkl models")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top_k", type=int, default=10)

    ap.add_argument("--min_repo_freq", type=int, default=2,
                    help="Filter repos that appear fewer than this count in the FULL dataset (helps overlap). Use 1 to disable.")

    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--eval_ratio", type=float, default=0.1)

    ap.add_argument("--n_eval_candidates", type=int, default=2000,
                    help="Candidate set size for evaluation (truth + sampled negatives). Use 0 to rank over all train repos.")

    # TF-IDF
    ap.add_argument("--tfidf_max_features", type=int, default=200000)

    # TruncatedSVD
    ap.add_argument("--svd_components", type=int, default=64)

    # BPR
    ap.add_argument("--bpr_factors", type=int, default=64)
    ap.add_argument("--bpr_epochs", type=int, default=20)
    ap.add_argument("--bpr_batch_size", type=int, default=4096)
    ap.add_argument("--bpr_lr", type=float, default=1e-2)
    ap.add_argument("--bpr_weight_decay", type=float, default=1e-6)
    ap.add_argument("--bpr_device", type=str, default="cuda" if (torch is not None and torch.cuda.is_available()) else "cpu")

    args = ap.parse_args()

    print("Loading dataset...")
    df = load_txt_dataset(args.data_dir)
    print(f"Raw: n_interactions={len(df):,} | n_users={df['user'].nunique():,} | n_repos={df['repo'].nunique():,}")

    print(f"Filtering repos with freq < {args.min_repo_freq} (global)...")
    df = filter_by_repo_freq(df, args.min_repo_freq)
    print(f"After repo-freq filter: n_interactions={len(df):,} | n_users={df['user'].nunique():,} | n_repos={df['repo'].nunique():,}")

    if df["user"].nunique() == 0 or df["repo"].nunique() == 0:
        raise RuntimeError("Dataset empty after filtering. Try --min_repo_freq 1 or a smaller threshold.")

    print("Splitting per-user (train/eval/test)...")
    train_df, eval_df, test_df = per_user_split(df, train_ratio=args.train_ratio, eval_ratio=args.eval_ratio, seed=args.seed)
    eval_warm, test_warm = make_warm_splits(train_df, eval_df, test_df)

    print(f"train={len(train_df):,} eval={len(eval_df):,} test={len(test_df):,}")
    print(f"eval_warm={len(eval_warm):,} test_warm={len(test_warm):,}")
    print(f"train users={train_df['user'].nunique():,} train repos={train_df['repo'].nunique():,}")

    if len(eval_warm) == 0:
        print("\nWARNING: eval_warm is empty => no warm items in eval.\n"
              "This usually means your repos are mostly unique per user (cold items).\n"
              "Try one or more of:\n"
              "- decrease --eval_ratio (hold out fewer interactions)\n"
              "- increase overlap by using --min_repo_freq >= 2 (already)\n"
              "- change your dataset or add repo features (descriptions) for content-based methods.\n")

    results = {}

    # --- Train 1: Popularity
    print("\nTraining Popularity...")
    pop = PopularityRecommender().fit(train_df)
    pop_eval = evaluate_model(pop, train_df, eval_warm, args.top_k, args.n_eval_candidates, args.seed)
    print("[Popularity] eval:", pop_eval)
    save_pkl(pop, os.path.join(args.out_dir, "popularity.pkl"))
    results["popularity_eval"] = pop_eval

    # --- Train 2: TF-IDF content
    print("\nTraining TF-IDF Content...")
    tfidf = TfidfContentRecommender(max_features=args.tfidf_max_features).fit(train_df)
    tfidf_eval = evaluate_model(tfidf, train_df, eval_warm, args.top_k, args.n_eval_candidates, args.seed)
    print("[TFIDF] eval:", tfidf_eval)
    save_pkl(tfidf, os.path.join(args.out_dir, "tfidf_content.pkl"))
    results["tfidf_eval"] = tfidf_eval

    # --- Train 3: TruncatedSVD MF
    print("\nTraining TruncatedSVD MF...")
    svd = TruncatedSVDRecommender(n_components=args.svd_components, random_state=args.seed).fit(train_df)
    svd_eval = evaluate_model(svd, train_df, eval_warm, args.top_k, args.n_eval_candidates, args.seed)
    print("[SVD-MF] eval:", svd_eval)
    save_pkl(svd, os.path.join(args.out_dir, "svd_mf.pkl"))
    results["svd_eval"] = svd_eval

    # --- Train 4: BPR-MF (optional)
    if torch is not None:
        print("\nTraining BPR-MF (PyTorch)...")
        bpr = BPRMFRecommender(
            n_factors=args.bpr_factors,
            epochs=args.bpr_epochs,
            batch_size=args.bpr_batch_size,
            lr=args.bpr_lr,
            weight_decay=args.bpr_weight_decay,
            seed=args.seed,
            device=args.bpr_device,
        ).fit(train_df)
        bpr_eval = evaluate_model(bpr, train_df, eval_warm, args.top_k, args.n_eval_candidates, args.seed)
        print("[BPR] eval:", bpr_eval)
        save_pkl(bpr, os.path.join(args.out_dir, "bpr_mf.pkl"))
        results["bpr_eval"] = bpr_eval
    else:
        print("\nPyTorch not available => skipping BPR-MF.")

    # --- Test evaluation (optional)
    print("\nEvaluating on TEST (warm)...")
    test_truth = build_truth_by_user(test_warm)

    def eval_on_test(m: BaseRecommender, name: str):
        m_test = evaluate_model(m, train_df, test_warm, args.top_k, args.n_eval_candidates, args.seed + 1)
        print(f"[{name}] test:", m_test)
        results[f"{name}_test"] = m_test

    eval_on_test(pop, "Popularity")
    eval_on_test(tfidf, "TFIDF")
    eval_on_test(svd, "SVD-MF")
    if torch is not None:
        eval_on_test(bpr, "BPR")

    # Save summary
    save_pkl(results, os.path.join(args.out_dir, "metrics_summary.pkl"))
    print("\nSaved models + metrics to:", args.out_dir)
    print("Summary keys:", list(results.keys()))

    # Example how to load in your app:
    print("\nExample usage in app:")
    print(f"  from train_github_repo_reco_models import load_pkl")
    print(f"  model = load_pkl('{os.path.join(args.out_dir, 'svd_mf.pkl')}')")
    print(f"  # then score on a candidate pool, e.g. top popular repos")
    print(f"  # preds = model.recommend(user='some_login', k={args.top_k}, candidates=candidates, seen=seen_set)")

if __name__ == "__main__":
    main()
