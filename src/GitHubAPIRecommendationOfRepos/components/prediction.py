import os
from pathlib import Path
from ..utils.common import load_bin
from ..entity.config_entity import ModelTrainerConfig, DataTransformationConfig
from src.GitHubAPIRecommendationOfRepos.train.model import load_model

class PredictionPipeline:
    def __init__(self):
        # Path to the SVD model manually placed
        # Adjusting path relative to project root or using absolute path strategy if needed
        # Here we assume the workspace structure is preserved
        self.model_path = os.path.join("src", "GitHubAPIRecommendationOfRepos", "model", "svd_model.pkl")
        
    def predict(self, username, user_repos, all_repos=None, top_k=5):
        """
        Reçoit un utilisateur, ses repos, et une liste de tous les repos candidats.
        Renvoie une recommandation basée sur le modèle SVD.
        """
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at {self.model_path}")

            # 1. Load the SVD model
            model = load_model(self.model_path)

            if not all_repos:
                # Fallback if no candidates provided
                return []

            # 2. Filter out repos the user already has
            user_repos_set = set(user_repos)
            candidates = [repo for repo in all_repos if repo not in user_repos_set]

            # 3. Predict score for each candidate
            predictions = []
            for repo in candidates:
                # model.predict returns a Prediction object (uid, iid, r_ui, est, details)
                pred = model.predict(username, repo)
                predictions.append((repo, pred.est))

            # 4. Sort by estimated score in descending order
            predictions.sort(key=lambda x: x[1], reverse=True)

            # 5. Return top k repo names
            recommendations = [repo for repo, score in predictions[:top_k]]
            
            return recommendations

        except Exception as e:
            # Log the error potentially
            print(f"Error in prediction: {e}")
            raise e