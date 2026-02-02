from __future__ import annotations

from .config import settings
from .data import (
    load_txt_dataset,
    build_interactions,
    build_df,
    random_split_df,
    per_user_split,
    build_surprise_objects,
)
from .model import create_model, fit_model


def train_svd_from_txt(
    data_dir: str | None = None,
    split_mode: str = "random",
):
    data_lue, paths = load_txt_dataset(data_dir=data_dir)

    interactions = build_interactions(data_lue)
    df = build_df(interactions)

    if split_mode == "per_user":
        train_df, eval_df, test_df = per_user_split(
            df,
            train_ratio=settings.split.train_ratio,
            eval_ratio=settings.split.eval_ratio,
        )
    else:
        train_df, eval_df, test_df = random_split_df(df, random_state=settings.data.random_state)

    trainset, evalset, testset = build_surprise_objects(train_df, eval_df, test_df)

    model = create_model()
    fit_model(model, trainset)

    return model, (train_df, eval_df, test_df), (trainset, evalset, testset), data_lue
