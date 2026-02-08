from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from src.api.config import API_TITLE
from src.api.config import API_VERSION
from src.api.services import UserService, RecommendationService
from src.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[Any, None]:
    # Initialize services
    print("[+] Initializing Services...")
    user_service = UserService()
    recommendation_service = RecommendationService()
    
    # Attach to app state
    app.state.user_service = user_service
    app.state.recommendation_service = recommendation_service
    
    print("[+] Services Initialized (MongoDB & Recommendation Pipeline)")
    yield
    print("[-] Shutting down services...")


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="GitHub Repository Recommendation API",
    lifespan=lifespan,
)

# app.add_exception_handler(Exception, general_exception_handler) # Keep generic handler if needed

app.include_router(router)


@app.get("/", tags=["info"])
async def root() -> dict:
    return {
        "service": API_TITLE,
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/api/health",
    }