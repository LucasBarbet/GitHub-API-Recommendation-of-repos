from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from mlflow.exceptions import MlflowException

from src.api.config import API_TITLE
from src.api.config import API_VERSION
from src.api.dal import SpamClassifierService
from src.api.exception_handlers import course_mlops_exception_handler
from src.api.exception_handlers import general_exception_handler
from src.api.routes import router
from src.exceptions import CourseMLOpsError
from src.exceptions import ModelMetadataFetchError


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[Any, None]:
    classifier = SpamClassifierService()
    try:
        classifier.load_model("models:/spam-classifier/latest")
    except (ModelMetadataFetchError, MlflowException, OSError) as e:
        print(f"[-] Model loading failed: {e}")
        print("[!] Starting without model. Train the model first with 'inv train'")

    app.state.classifier_service = classifier
    yield


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="Peepo",
    lifespan=lifespan,
)

app.add_exception_handler(CourseMLOpsError, course_mlops_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

app.include_router(router)


@app.get("/", tags=["info"])
async def root() -> dict:
    return {
        "service": API_TITLE,
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/api/v1/health",
    }