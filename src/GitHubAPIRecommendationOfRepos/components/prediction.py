import os
from pathlib import Path
from ..utils.common import load_bin
from ..entity.config_entity import ModelTrainerConfig, DataTransformationConfig
import sys
from typing import List, Optional
from src.GitHubAPIRecommendationOfRepos.components.models import ModelLoader, BaseRecommender

class PredictionPipeline:
    def __init__(self):
        # Base directory for models
        self.model_dir = os.path.join("src", "GitHubAPIRecommendationOfRepos", "model")
        
    def predict(self, username: str, user_repos: List[str], all_repos: List[str] = None, top_k: int = 5, model_name: str = "svd_model") -> List[str]:
        """
        Generic predict method that delegates to the specific model implementation.
        """
        try:
            # Construct model path based on name
            # Assuming model names map to filenames like 'svd_model' -> 'svd_model.pkl'
            filename = f"{model_name}.pkl"
            model_path = os.path.join(self.model_dir, filename)

            if not os.path.exists(model_path):
                 raise FileNotFoundError(f"Model file not found at {model_path}")

            # Get or Load the model via Factory
            recommender = ModelLoader.get_model(model_name, model_path)
            
            # Execute prediction
            return recommender.predict(username, user_repos, all_repos, top_k)

        except Exception as e:
            print(f"Error in prediction pipeline: {e}")
            # print stack trace for debugging
            import traceback
            traceback.print_exc()
            raise e