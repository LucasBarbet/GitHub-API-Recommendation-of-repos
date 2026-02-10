import os
import json
import math
import glob
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import mlflow
from mlflow import MlflowClient

from surprise import Dataset as SurpriseDataset
from surprise import Reader, SVD

# Configuration MLflow (stockage local)
MLFLOW_DIR = Path("./mlruns_demo_svd")
MLFLOW_DIR.mkdir(exist_ok=True)

mlflow.set_tracking_uri(f"file://{MLFLOW_DIR.absolute()}")
print(f"Tracking URI: {mlflow.get_tracking_uri()}")
print(f"MLflow configuré dans: {MLFLOW_DIR.absolute()}")

# Chemin vers le dossier contenant les fichiers .txt
# Format attendu par fichier :
#   login : [repoA, repoB, repoC]
#   autre_login : [repoX, repoY]
data_dir = "/info/raid-etu/m2/s2101052/data100repos/"

paths = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
print("Nb fichiers trouvés:", len(paths))
assert len(paths) > 0, "Aucun .txt trouvé — vérifie data_dir"

# Lecture des fichiers en dict {user: [repos]}
data_lue = {}
for path in paths:
    with open(path, "r", encoding="utf-8") as f:
        for ligne in f:
            login, reste = ligne.strip().split(" : [", 1)
            repos_str = reste.rstrip("]\n")
            repos = [r.strip() for r in repos_str.split(",")] if repos_str else []
            data_lue[login] = repos

print("Nb users:", len(data_lue))
print("Exemple user:", next(iter(data_lue.keys())))
print("Nb repos pour cet user:", len(next(iter(data_lue.values()))))

# Construction du DataFrame (interactions binaires)
rows = []
for user, repos in data_lue.items():
    for repo in repos:
        rows.append((user, repo, 1))

df = pd.DataFrame(rows, columns=["user", "repo", "rating"])
print("Nb interactions:", len(df))
print("Nb users uniques:", df["user"].nunique())
print("Nb repos uniques:", df["repo"].nunique())
print(df.head())

rng = np.random.RandomState(42)

def per_user_split(df, train_ratio=0.8, eval_ratio=0.1):
    """Split par utilisateur (comme dans ton notebook SVD)."""
    train_rows, eval_rows, test_rows = [], [], []
    for u, g in df.groupby("user"):
        repos = g["repo"].values
        if len(repos) < 3:
            # trop peu => tout en train
            for r in repos:
                train_rows.append((u, r, 1))
            continue

        repos = repos.copy()
        rng.shuffle(repos)

        n = len(repos)
        n_train = int(train_ratio * n)
        n_eval  = int((train_ratio + eval_ratio) * n)

        for r in repos[:n_train]:
            train_rows.append((u, r, 1))
        for r in repos[n_train:n_eval]:
            eval_rows.append((u, r, 1))
        for r in repos[n_eval:]:
            test_rows.append((u, r, 1))

    train_df = pd.DataFrame(train_rows, columns=["user", "repo", "rating"])
    eval_df  = pd.DataFrame(eval_rows,  columns=["user", "repo", "rating"])
    test_df  = pd.DataFrame(test_rows,  columns=["user", "repo", "rating"])
    return train_df, eval_df, test_df

train_df, eval_df, test_df = per_user_split(df, 0.8, 0.1)
print("Train:", train_df.shape, "Eval:", eval_df.shape, "Test:", test_df.shape)

TRAIN_REPOS = set(train_df["repo"].unique())

data_lue_train = train_df.groupby("user")["repo"].apply(list).to_dict()

def recommend_repos(model, user, data_lue_train, top_k=10):
    """Recommande top_k repos parmi ceux vus en train (sauf déjà vus par user)."""
    seen = set(data_lue_train.get(user, []))
    candidates = list(TRAIN_REPOS - seen)
    if not candidates:
        return []

    scored = [(r, model.predict(user, r).est) for r in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [r for r, _ in scored[:top_k]]

def precision_at_k(pred, truth, k):
    if k <= 0:
        return 0.0
    pred_k = pred[:k]
    return len(set(pred_k) & truth) / k

def recall_at_k(pred, truth, k):
    if not truth:
        return 0.0
    pred_k = pred[:k]
    return len(set(pred_k) & truth) / len(truth)

def hit_rate_at_k(pred, truth, k):
    pred_k = pred[:k]
    return 1.0 if len(set(pred_k) & truth) > 0 else 0.0

def average_precision_at_k(pred, truth, k):
    if not truth:
        return 0.0
    pred_k = pred[:k]
    hit_count = 0
    sum_prec = 0.0
    for i, item in enumerate(pred_k, start=1):
        if item in truth:
            hit_count += 1
            sum_prec += hit_count / i
    denom = min(len(truth), k)
    return sum_prec / denom if denom > 0 else 0.0

def ndcg_at_k(pred, truth, k):
    if k <= 0:
        return 0.0
    pred_k = pred[:k]
    dcg = 0.0
    for i, item in enumerate(pred_k, start=1):
        if item in truth:
            dcg += 1.0 / math.log2(i + 1)

    ideal_hits = min(len(truth), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_ranking(model, eval_df, data_lue_train, k=5):
    """Retourne métriques moyennes sur les users présents dans eval_df."""
    truth_by_user = eval_df.groupby("user")["repo"].apply(set).to_dict()
    users = list(truth_by_user.keys())

    precs, recs, hits, maps, ndcgs = [], [], [], [], []
    for u in tqdm(users, desc=f"Eval@{k}"):
        pred = recommend_repos(model, u, data_lue_train, top_k=k)
        truth = truth_by_user[u]

        precs.append(precision_at_k(pred, truth, k))
        recs.append(recall_at_k(pred, truth, k))
        hits.append(hit_rate_at_k(pred, truth, k))
        maps.append(average_precision_at_k(pred, truth, k))
        ndcgs.append(ndcg_at_k(pred, truth, k))

    return {
        "k": k,
        "n_users_eval": len(users),
        "precision_at_k": float(np.mean(precs)) if precs else 0.0,
        "recall_at_k": float(np.mean(recs)) if recs else 0.0,
        "hit_rate_at_k": float(np.mean(hits)) if hits else 0.0,
        "map_at_k": float(np.mean(maps)) if maps else 0.0,
        "ndcg_at_k": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }

mlflow.set_experiment("demo_svd_recommender")

BASE_PARAMS = {
    "n_factors": 50,
    "n_epochs": 20,
    "lr_all": 0.005,
    "reg_all": 0.02,
    "random_state": 42,
}
TOP_K = 5

reader = Reader(rating_scale=(0, 1))
train_data = SurpriseDataset.load_from_df(train_df[["user", "repo", "rating"]], reader)
trainset = train_data.build_full_trainset()

with mlflow.start_run(run_name="baseline_svd") as run:
    mlflow.set_tags({
        "model_type": "surprise_svd",
        "dataset": "user_repo_interactions",
        "split": "per_user_split",
    })
    mlflow.log_params({**BASE_PARAMS, "top_k": TOP_K})

    model = SVD(**BASE_PARAMS)
    model.fit(trainset)

    # Eval sur eval_df (ranking)
    metrics = evaluate_ranking(model, eval_df, data_lue_train, k=TOP_K)
    for m, v in metrics.items():
        if m != "k":
            mlflow.log_metric(m, v)

    # Sauvegarde modèle + artefacts utiles
    Path("artifacts").mkdir(exist_ok=True)
    model_path = Path("artifacts") / "svd_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    mlflow.log_artifact(str(model_path), artifact_path="model_artifacts")

print(metrics)

param_grid = {
    "n_factors": [20, 50, 100],
    "n_epochs": [10, 20],
    "lr_all": [0.002, 0.005],
    "reg_all": [0.02, 0.05],
}

search_results = []

with mlflow.start_run(run_name="hyperparameter_search") as parent_run:
    mlflow.set_tag("type", "grid_search")
    mlflow.log_param("top_k", TOP_K)

    best_score = -1.0
    best_params = None

    for n_factors in param_grid["n_factors"]:
        for n_epochs in param_grid["n_epochs"]:
            for lr_all in param_grid["lr_all"]:
                for reg_all in param_grid["reg_all"]:
                    params = {
                        "n_factors": n_factors,
                        "n_epochs": n_epochs,
                        "lr_all": lr_all,
                        "reg_all": reg_all,
                        "random_state": 42,
                    }

                    with mlflow.start_run(run_name=f"svd_f{n_factors}_e{n_epochs}", nested=True):
                        mlflow.log_params(params)

                        model = SVD(**params)
                        model.fit(trainset)

                        metrics = evaluate_ranking(model, eval_df, data_lue_train, k=TOP_K)
                        for m, v in metrics.items():
                            if m != "k":
                                mlflow.log_metric(m, v)

                        score = metrics["ndcg_at_k"]
                        search_results.append({**params, **metrics})

                        if score > best_score:
                            best_score = score
                            best_params = params

    mlflow.log_metric("best_ndcg_at_k", best_score)
    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

print("Best score", best_score)
print("Best param", best_params)

results_df = pd.DataFrame(search_results)

# Exemple de visu: NDCG@K en fonction de (n_factors, reg_all) pour n_epochs=20, lr_all=0.005 si dispo
subset = results_df[(results_df["n_epochs"] == 20) & (results_df["lr_all"] == 0.005)]
if len(subset) == 0:
    subset = results_df.copy()

pivot = subset.pivot_table(
    index="reg_all",
    columns="n_factors",
    values="ndcg_at_k",
    aggfunc="mean"
).sort_index()

plt.figure(figsize=(10, 5))
plt.imshow(pivot.values, aspect="auto")
plt.xticks(range(len(pivot.columns)), pivot.columns)
plt.yticks(range(len(pivot.index)), pivot.index)
plt.xlabel("n_factors")
plt.ylabel("reg_all")
plt.title("Grid Search — NDCG@K (moyenne)")
plt.colorbar(label="ndcg_at_k")
plt.tight_layout()
plt.show()

results_df.sort_values("ndcg_at_k", ascending=False).head(10)

client = MlflowClient()
experiment = client.get_experiment_by_name("demo_svd_recommender")
print(f"Experiment ID: {experiment.experiment_id}")
print(f"Artifact Location: {experiment.artifact_location}")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.ndcg_at_k > 0",
    order_by=["metrics.ndcg_at_k DESC"],
    max_results=10
)

cols = ["run_id", "status", "metrics.ndcg_at_k", "metrics.map_at_k", "params.n_factors", "params.n_epochs", "params.lr_all", "params.reg_all"]
runs_df = pd.DataFrame([{
    "run_id": r.info.run_id,
    "status": r.info.status,
    "ndcg_at_k": r.data.metrics.get("ndcg_at_k"),
    "map_at_k": r.data.metrics.get("map_at_k"),
    "n_factors": r.data.params.get("n_factors"),
    "n_epochs": r.data.params.get("n_epochs"),
    "lr_all": r.data.params.get("lr_all"),
    "reg_all": r.data.params.get("reg_all"),
} for r in runs])

print(runs_df.head())

# Re-entraîner avec best_params puis évaluer sur test_df
best_params = best_params or BASE_PARAMS

train_full_df = pd.concat([train_df, eval_df], ignore_index=True)
data_lue_train_full = train_full_df.groupby("user")["repo"].apply(list).to_dict()
TRAIN_REPOS = set(train_full_df["repo"].unique())

train_full_data = SurpriseDataset.load_from_df(train_full_df[["user","repo","rating"]], reader)
train_full_set = train_full_data.build_full_trainset()

with mlflow.start_run(run_name="best_on_test") as run:
    mlflow.log_params({**best_params, "top_k": TOP_K, "train": "train+eval"})
    model = SVD(**best_params)
    model.fit(train_full_set)

    test_metrics = evaluate_ranking(model, test_df, data_lue_train_full, k=TOP_K)
    for m, v in test_metrics.items():
        if m != "k":
            mlflow.log_metric("test_" + m, v)

print("Test metrics", test_metrics)

print("Dans un terminal, depuis le dossier du notebook :")
print(f"mlflow ui --backend-store-uri file://{MLFLOW_DIR.absolute()} --port 5000")