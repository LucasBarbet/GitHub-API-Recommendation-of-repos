from typing import Annotated
from typing import List

from pydantic import BaseModel
from pydantic import Field

class UserInput(BaseModel):
    username: Annotated[str, Field(min_length=1, description="GitHub Username")]

class UserOutput(BaseModel):
    username: str
    repos: List[str]

class PredictInput(BaseModel):
    user: Annotated[str, Field(min_length=1, description="Nom de l'utilisateur")]
    k: Annotated[int, Field(ge=1, le=20, default=5, description="Nombre de recommandations")]
    # Optional: pass repos directly if we want to avoid DB lookup in predict service, 
    # but for now we follow the plan where service does lookup.

class Recommendation(BaseModel):
    title: str
    confidence: float = 0.0 # Pipeline might not return confidence yet, optional

class PredictOutput(BaseModel):
    user: str
    recommendations: List[str] # Simple list of strings for now as per current pipeline output

class RepoInput(BaseModel):
    repo_name: str
