from __future__ import annotations

from surprise import SVD
from .config import settings


def create_model() -> SVD:
    return SVD(
        n_factors=settings.model.n_factors,
        n_epochs=settings.model.n_epochs,
        lr_all=settings.model.lr_all,
        reg_all=settings.model.reg_all,
    )


def fit_model(model: SVD, trainset):
    return model.fit(trainset)
