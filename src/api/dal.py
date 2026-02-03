import os
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from src.exceptions import ModelNotLoadedError
from src.train.config import get_settings
from src.train.data import DatasetColumn


class SVDClassifierService:
    def __init__(self) -> None:
        self.model: Any = None
        self.model_uri: str | None = None
        self.model_version: str | None = None
        self.run_id: str | None = None
        self.artifact_uri: str | None = None
        self.registered_at: str | None = None

    def load_model(self, model_uri: str) -> None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or os.getenv("CML_MLFLOW_TRACKING_URI") or None

        if tracking_uri is None:
            settings = get_settings()
            tracking_uri = settings.mlflow.tracking_uri

        mlruns_path = Path(tracking_uri.replace("file://", "")) if tracking_uri else Path("mlruns")

        pkl_files = sorted(
            mlruns_path.glob("**/python_model.pkl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not pkl_files:
            raise FileNotFoundError(f"No model pickle found in {mlruns_path}")

        model_path = pkl_files[0]
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        self.model_uri = str(model_path)
        print(f"[+] Model loaded from: {model_path}")

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, message: str) -> dict:
        if not self.is_loaded:
            raise ModelNotLoadedError

        df = pd.DataFrame({DatasetColumn.MESSAGE: [message]})
        result = self.model.predict(df)

        prediction_label = int(result["predictions"][0])
        spam_prob = float(result["probabilities"][0][1])

        return {
            "prediction": "spam" if prediction_label == 1 else "ham",
            "probability": spam_prob,
            "is_spam": prediction_label == 1,
        }