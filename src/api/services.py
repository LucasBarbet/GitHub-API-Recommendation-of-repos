import os
import sys
from typing import List, Optional
from pymongo import MongoClient
import pandas as pd

from src.GitHubAPIRecommendationOfRepos.constants import MONGO_DATABASE_NAME, MONGO_COLLECTION_NAME
from src.GitHubAPIRecommendationOfRepos.components.prediction import PredictionPipeline

class UserService:
    def __init__(self):
        # Initialize DB connection using the same logic as db_connector
        # But here we encapsulate it in the service
        mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
        self.client = MongoClient(mongo_uri)
        self.db = self.client[MONGO_DATABASE_NAME]
        self.collection = self.db[MONGO_COLLECTION_NAME]

    def get_user_repos(self, username: str) -> Optional[List[str]]:
        """
        Fetch user repositories from MongoDB using _id.
        Returns None if user not found.
        """
        user_data = self.collection.find_one({"_id": username})
        if user_data:
            return user_data.get('repos', [])
        return None

    def add_user(self, username: str) -> bool:
        """
        Add a new user to MongoDB if they don't exist.
        Returns True if added, False if already exists.
        """
        if self.collection.find_one({"_id": username}):
            return False
        
        self.collection.insert_one({"_id": username, "repos": []})
        return True

class RecommendationService:
    def __init__(self):
        self.pipeline = PredictionPipeline()

    def predict(self, username: str, user_repos: List[str], top_k: int = 5) -> List[str]:
        """
        Run the recommendation pipeline.
        """
        try:
            # The pipeline currently returns a list of strings
            recommendations = self.pipeline.predict(username, user_repos, top_k=top_k)
            return recommendations
        except TypeError:
            # Fallback for older pipeline signature
            recommendations = self.pipeline.predict(username, user_repos)
            return recommendations[:top_k]
