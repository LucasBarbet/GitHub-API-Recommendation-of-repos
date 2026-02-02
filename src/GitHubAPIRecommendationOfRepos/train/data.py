from __future__ import annotations

import glob
import os
from typing import Dict, List, Tuple
import pandas as pd
from surprise import Dataset, Reader
from .config import settings, rng


def load_txt_dataset(
    data_dir: str | None = None,
    file_glob: str | None = None,
) -> Tuple[Dict[str, List[str]], List[str]]:
    data_dir = data_dir or settings.data.data_dir
    file_glob = file_glob or settings.data.file_glob

    paths = glob.glob(os.path.join(data_dir, file_glob))
    paths = sorted(paths)

    data_lue: Dict[str, List[str]] = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            tmp = 0
            for ligne in f:
                login, reste = ligne.strip().split(" : [", 1)
                repos_str = reste.rstrip("]\n")
                repos = [r.strip() for r in repos_str.split(",")] if repos_str else []
                data_lue[login] = repos
                tmp += 1
            print(path, ":", tmp, "reel :", len(data_lue.keys()))

    return data_lue, paths


def build_interactions(data_lue: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    interactions: List[Tuple[str, str]] = []
    for user, repos in data_lue.items():
        for repo in repos:
            interactions.append((user, repo))
    return interactions


def build_df(interactions: List[Tuple[str, str]]) -> pd.DataFrame:
    data = []
    for user, repo in interactions:
        data.append([user, repo, 1])
    return pd.DataFrame(data, columns=["user", "repo", "rating"])


def random_split_df(df: pd.DataFrame, random_state: int | None = None):
    random_state = settings.data.random_state if random_state is None else random_state

    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    n_total = len(df_shuffled)
    train_end = int(settings.split.train_ratio * n_total)
    eval_end = int((settings.split.train_ratio + settings.split.eval_ratio) * n_total)

    train_df = df_shuffled.iloc[:train_end].copy()
    eval_df = df_shuffled.iloc[train_end:eval_end].copy()
    test_df = df_shuffled.iloc[eval_end:].copy()

    return train_df, eval_df, test_df


def per_user_split(df: pd.DataFrame, train_ratio: float = 0.8, eval_ratio: float = 0.1):
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

        if n_train < 1:
            n_train = 1
        if n_eval <= n_train:
            n_eval = min(n_train + 1, n)

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


def build_surprise_objects(train_df: pd.DataFrame, eval_df: pd.DataFrame, test_df: pd.DataFrame):
    reader = Reader(rating_scale=(0, 1))

    train_data = Dataset.load_from_df(train_df[["user", "repo", "rating"]], reader)
    eval_data = Dataset.load_from_df(eval_df[["user", "repo", "rating"]], reader)
    test_data = Dataset.load_from_df(test_df[["user", "repo", "rating"]], reader)

    trainset = train_data.build_full_trainset()
    evalset = eval_data.build_full_trainset().build_testset()
    testset = test_data.build_full_trainset().build_testset()

    return trainset, evalset, testset
