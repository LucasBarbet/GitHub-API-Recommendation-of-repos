from __future__ import annotations

from pydantic import BaseModel
import yaml
import numpy as np


class DataConfig(BaseModel):
    """
    Configuration des données.

    Note: dans le notebook, le dataset est un dossier contenant des fichiers .txt
    de la forme: "<login> : [repo1, repo2, ...]".
    """
    data_dir: str = "/info/raid-etu/m2/s2101052/data100repos/"
    file_glob: str = "*.txt"
    random_state: int = 42


class SplitConfig(BaseModel):
    """
    Split de df en train/eval/test.
    - random_split: split global (shuffle puis 80/10/10)
    - per_user_split: split par user (ex: 80/10/10 pour chaque user)
    """
    train_ratio: float = 0.8
    eval_ratio: float = 0.1


class ModelParams(BaseModel):
    # Paramètres Surprise.SVD (mêmes valeurs que le notebook)
    n_factors: int = 50
    n_epochs: int = 50
    lr_all: float = 0.005
    reg_all: float = 0.02


class RecoConfig(BaseModel):
    top_k: int = 5


class Settings(BaseModel):
    data: DataConfig = DataConfig()
    split: SplitConfig = SplitConfig()
    model: ModelParams = ModelParams()
    reco: RecoConfig = RecoConfig()

    @classmethod
    def from_yaml(cls, path: str) -> "Settings":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)


settings = Settings()

rng = np.random.RandomState(settings.data.random_state)