from __future__ import annotations

from surprise import SVD
from .config import settings
import pickle


def create_model() -> SVD:
    return SVD(
        n_factors=settings.model.n_factors,
        n_epochs=settings.model.n_epochs,
        lr_all=settings.model.lr_all,
        reg_all=settings.model.reg_all,
    )


def fit_model(model: SVD, trainset):
    return model.fit(trainset)

def save_model(model: SVD, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_model(path: str) -> SVD:
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model