from datetime import UTC
from datetime import datetime
from enum import StrEnum
from typing import Annotated
from typing import List

from pydantic import BaseModel
from pydantic import Field

class PredictInput(BaseModel):
    user: Annotated[
        str,
        Field(
            min_length=1,
            max_length=100,
            description="Nom de l'utilisateur"
        )
    ]
    k: Annotated[
        int,
        Field(
            ge=1,
            le=20,
            default=5,
            description="Nombre de recommandations"
        )
    ]

class Recommendation(BaseModel):
    title: str
    confidence: Annotated[float, Field(ge=0, le=1)]

class PredictOutput(BaseModel):
    user: str
    probability: Annotated[
        float,
        Field(ge=0, le=1, description="Probability")
    ]
    recommendations: List[Recommendation]

class ModelInfoOutput(BaseModel):
    model_name: str
    model_version: str | None = None
    run_id: str | None = None
    mlflow_ui_url: str
    artifact_uri: str | None = None
    registered_at: int | None = None




