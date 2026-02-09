from typing import Annotated, cast
from fastapi import APIRouter, Depends, Query, HTTPException, status, Request

from src.api.services import UserService, RecommendationService
from src.api.models import UserInput, UserOutput, PredictInput, PredictOutput, RepoInput
import os
from src.GitHubAPIRecommendationOfRepos.components.models import ModelLoader

# Dependency Injection for Services
def get_user_service(request: Request) -> UserService:
    return request.app.state.user_service

def get_recommendation_service(request: Request) -> RecommendationService:
    return request.app.state.recommendation_service

UserServiceDep = Annotated[UserService, Depends(get_user_service)]
RecommendationServiceDep = Annotated[RecommendationService, Depends(get_recommendation_service)]

router = APIRouter(prefix="/api", tags=["github-recommender"])

@router.get("/users/{username}", response_model=UserOutput)
async def get_user(username: str, service: UserServiceDep):
    repos = service.get_user_repos(username)
    if repos is None:
        raise HTTPException(status_code=404, detail="User not found")
    return UserOutput(username=username, repos=repos)

@router.post("/users", status_code=status.HTTP_201_CREATED)
async def add_user(user: UserInput, service: UserServiceDep):
    success = service.add_user(user.username)
    if not success:
        raise HTTPException(status_code=409, detail="User already exists")
    return {"message": f"User {user.username} created successfully"}

@router.post("/users/{username}/repos")
async def add_repo(username: str, input_data: RepoInput, service: UserServiceDep):
    import logging
    logger = logging.getLogger("uvicorn")
    logger.info(f"DEBUG Route: Received request to add {input_data.repo_name} for {username}")
    
    success = service.add_repo_to_user(username, input_data.repo_name)
    if not success:
        logger.error(f"DEBUG Route: Failed to add repo for {username}")
        raise HTTPException(status_code=404, detail="User not found (Update failed)")
    
    logger.info(f"DEBUG Route: Successfully added repo for {username}")
    return {"message": "Repository added successfully"}

@router.post("/predict", response_model=PredictOutput)
async def predict(input_data: PredictInput, 
                  user_service: UserServiceDep, 
                  rec_service: RecommendationServiceDep):
    
    # 1. Fetch user repos (backend verification)
    repos = user_service.get_user_repos(input_data.user)
    if repos is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    # 2. Fetch all repos (candidates)
    # In a real heavy production system, we wouldn't fetch ALL repos every time. 
    # We might use a pre-computed list or a vector search.
    # For this SVD implementation without an item map file, we need the candidates from DB.
    all_repos = user_service.get_all_repos()

    # 3. Run prediction with selected model
    recommendations = rec_service.predict(
        input_data.user, 
        repos, 
        all_repos=all_repos, 
        top_k=input_data.k,
        model_name=input_data.model_name
    )
    
    return PredictOutput(user=input_data.user, recommendations=recommendations)

@router.get("/models")
async def list_models():
    """
    List available recommendation models (only those that load successfully).
    """
    # Scan via ModelLoader to get only valid models
    model_dir = os.path.join("src", "GitHubAPIRecommendationOfRepos", "model")
    models = ModelLoader.get_available_models(model_dir)
    return {"models": models}

@router.get("/health")
async def health():
    return {"status": "ok"}