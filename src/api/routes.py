from datetime import UTC
from datetime import datetime
from typing import Annotated
from typing import cast

from fastapi import APIRouter
from fastapi import Depends
from fastapi import Request
from fastapi import status

from src.api.dal import SVDClassifierService
from src.api.models import PredictInput
from src.api.models import PredictOutput
from src.api.models import Recommendation
from src.api.models import ModelInfoOutput

def get_classifier_service(request: Request) -> SVDClassifierService:
    if not hasattr(request.app.state, "classifier_service"):
        raise ModelNotLoadedError("Classifier service not initialized")
    return cast(SVDClassifierService, request.app.state.classifier_service)

ClassifierServiceDep = Annotated[SVDClassifierService, Depends(get_classifier_service)]

router = APIRouter(prefix="/api", tags=["predictions"])

@router.post("/predict", response_model=PredictOutput, status_code=status.HTTP_200_OK)
async def predict(
    classifier: ClassifierServiceDep,
    request: PredictInput,
) -> PredictOutput:
    result = classifier.predict(request.user,request.k)
    recommendations = [
        Recommendation(
            title=r["title"],
            confidence=r["confidence"],
        )
        for r in result["recommendations"]
    ]

    return PredictOutput(
        user=request.user,
        probability=result["probability"],
        recommendations=recommendations,
    )

@router.get("/model/info", response_model=ModelInfoOutput, status_code=status.HTTP_200_OK)
async def model_info(classifier: ClassifierServiceDep) -> ModelInfoOutput:
    return ModelInfoOutput(
        model_name="SVD",
        model_version=classifier.model_version,
        run_id=classifier.run_id,
        mlflow_ui_url=f"{MLFLOW_UI_BASE}/#/runs/{classifier.run_id}" if classifier.run_id else MLFLOW_UI_BASE,
        artifact_uri=classifier.artifact_uri,
        registered_at=classifier.registered_at,
    )