import os
import sys
from typing import List, Optional
from pymongo import MongoClient

from src.GitHubAPIRecommendationOfRepos.constants import MONGO_DATABASE_NAME, MONGO_COLLECTION_NAME
from src.GitHubAPIRecommendationOfRepos.components.prediction import PredictionPipeline

class UserService:
    def __init__(self):
        # Initialize DB connection using the same logic as db_connector
        # But here we encapsulate it in the service
        mongo_uri = os.environ.get("MONGO_URI") or "mongodb://localhost:27017/"
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

    def get_all_repos(self) -> List[str]:
        """
        Fetch all unique repositories from MongoDB.
        """
        # distinct("repos") will return a list of all items found in the 'repos' array across all documents
        return self.collection.distinct("repos")

    def add_repo_to_user(self, username: str, repo_name: str) -> bool:
        """
        Add a repository to the user's list if it doesn't already exist.
        Returns True if successful (or if already exists), False if user not found.
        """
        import logging
        logger = logging.getLogger("uvicorn")
        logger.info(f"DEBUG: Attempting to add repo {repo_name} to user {username}")
        # Use upsert=True to create the user if they don't exist (self-healing)
        result = self.collection.update_one(
            {"_id": username},
            {"$addToSet": {"repos": repo_name}},
            upsert=True
        )
        logger.info(f"DEBUG: Update result matched_count: {result.matched_count}, upserted_id: {result.upserted_id}, raw_result: {result.raw_result}")
        # Success if we matched (updated) or upserted (created)
        return result.matched_count > 0 or result.upserted_id is not None


class RecommendationService:
    def __init__(self):
        self.pipeline = PredictionPipeline()

    def predict(self, username: str, user_repos: List[str], all_repos: List[str], top_k: int = 5, model_name: str = "svd_model") -> List[str]:
        try:
            return self.pipeline.predict(username, user_repos, all_repos, top_k, model_name=model_name)
        except Exception as e:
            # Fallback for older pipeline signature or if new one fails oddly
            print(f"Warning: Pipeline predict error: {e}")
            # If standard prediction fails, try a fallback? Or just return empty.
            # For now, let's re-raise or return empty safe list
            print(f"Error during prediction: {e}")
            return []
