from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set
import numpy as np
import pandas as pd
from tqdm import tqdm
from .config import settings

TRAIN_REPOS: Set[str] = set()
TRAIN_USERS: Set[str] = set()


def init_train_universe(train_df: pd.DataFrame) -> None:
    global TRAIN_REPOS, TRAIN_USERS
    TRAIN_REPOS = set(train_df["repo"].unique())
    TRAIN_USERS = set(train_df["user"].unique())


def build_data_lue_train(train_df: pd.DataFrame) -> Dict[str, List[str]]:
    return train_df.groupby("user")["repo"].apply(list).to_dict()


def recommend_repos(model, user: str, data_lue_train: Dict[str, List[str]], top_k: int = 10):
    seen = set(data_lue_train.get(user, []))
    candidates = list(TRAIN_REPOS - seen)
    if not candidates:
        return []

    scored = [(r, model.predict(user, r).est) for r in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def build_index_maps(df: pd.DataFrame):
    repos2index = {repo: idx for idx, repo in enumerate(df["repo"].unique())}
    users2index = {user: idx for idx, user in enumerate(df["user"].unique())}
    return users2index, repos2index


def build_eval_matrix(eval_df: pd.DataFrame, users2index: Dict[str, int], repos2index: Dict[str, int]):
    mat_eval = np.zeros([len(users2index), len(repos2index)])
    for _, row in eval_df.iterrows():
        u = row["user"]
        r = row["repo"]
        if u in users2index and r in repos2index:
            mat_eval[users2index[u], repos2index[r]] = 1
    return mat_eval


def build_pred_matrix(model, eval_df: pd.DataFrame, data_lue_train: Dict[str, List[str]], users2index, repos2index, top_k: int):
    mat_pred = np.zeros([len(users2index), len(repos2index)], dtype=np.int8)
    recos_by_user: Dict[str, List[str]] = {}

    for user in tqdm(eval_df["user"].unique()):
        u_idx = users2index.get(user)
        if u_idx is None:
            continue

        repos_recom = recommend_repos(model, user, data_lue_train, top_k=top_k)
        recos_by_user[user] = [r for r, _ in repos_recom]

        r_idx = [repos2index[r] for r, _ in repos_recom if r in repos2index]
        if r_idx:
            mat_pred[u_idx, r_idx] = 1

    return mat_pred, recos_by_user


def precision_at_k(pred: List[str], truth: Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    pred_k = pred[:k]
    return len(set(pred_k) & truth) / k


def recall_at_k(pred: List[str], truth: Set[str], k: int) -> float:
    if not truth:
        return 0.0
    pred_k = pred[:k]
    return len(set(pred_k) & truth) / len(truth)


def hit_rate_at_k(pred: List[str], truth: Set[str], k: int) -> float:
    pred_k = pred[:k]
    return 1.0 if len(set(pred_k) & truth) > 0 else 0.0


def average_precision_at_k(pred: List[str], truth: Set[str], k: int) -> float:
    pred_k = pred[:k]
    if not truth:
        return 0.0

    hits = 0
    s = 0.0
    for i, p in enumerate(pred_k, start=1):
        if p in truth:
            hits += 1
            s += hits / i
    return s / min(len(truth), k) if hits > 0 else 0.0


def ndcg_at_k(pred: List[str], truth: Set[str], k: int) -> float:
    pred_k = pred[:k]
    dcg = 0.0
    for i, p in enumerate(pred_k, start=1):
        if p in truth:
            dcg += 1.0 / np.log2(i + 1)

    # IDCG
    ideal_hits = min(len(truth), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    return (dcg / idcg) if idcg > 0 else 0.0


def evaluate_warm_users(
    eval_df: pd.DataFrame,
    recos_by_user: Dict[str, List[str]],
    k: int | None = None,
):
    K = settings.reco.top_k if k is None else k

    truth_by_user: Dict[str, Set[str]] = (
        eval_df.groupby("user")["repo"].apply(lambda s: set(s.tolist())).to_dict()
    )
    warm_users = list(truth_by_user.keys())

    metrics = defaultdict(list)
    for user in warm_users:
        truth = truth_by_user[user]
        pred = recos_by_user.get(user, [])
        metrics["precision@k"].append(precision_at_k(pred, truth, K))
        metrics["recall@k"].append(recall_at_k(pred, truth, K))
        metrics["hit_rate@k"].append(hit_rate_at_k(pred, truth, K))
        metrics["map@k"].append(average_precision_at_k(pred, truth, K))
        metrics["ndcg@k"].append(ndcg_at_k(pred, truth, K))

    results_warm = {m: float(np.mean(v)) if len(v) else 0.0 for m, v in metrics.items()}
    results_warm["n_users_eval"] = len(truth_by_user)
    results_warm["n_users_warm_eval"] = len(warm_users)
    return results_warm


class RecoEvaluator:
    def __init__(self, model, train_df: pd.DataFrame, top_k: int | None = None):
        self.model = model
        self.top_k = settings.reco.top_k if top_k is None else top_k
        init_train_universe(train_df)
        self.data_lue_train = build_data_lue_train(train_df)

    def recommend(self, user: str):
        return recommend_repos(self.model, user, self.data_lue_train, top_k=self.top_k)

    def predict_matrix(self, df: pd.DataFrame):
        users2index, repos2index = build_index_maps(df)
        mat_eval = build_eval_matrix(df, users2index, repos2index)
        mat_pred, recos_by_user = build_pred_matrix(
            self.model, df, self.data_lue_train, users2index, repos2index, top_k=self.top_k
        )
        return mat_eval, mat_pred, recos_by_user, users2index, repos2index

    def evaluate(self, eval_df: pd.DataFrame):
        _, recos_by_user = build_pred_matrix(
            self.model,
            eval_df,
            self.data_lue_train,
            users2index={u: i for i, u in enumerate(eval_df["user"].unique())},
            repos2index={r: i for i, r in enumerate(eval_df["repo"].unique())},
            top_k=self.top_k,
        )
        return evaluate_warm_users(eval_df, recos_by_user, k=self.top_k)
