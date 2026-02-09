
import os
import pickle
import sys
from abc import ABC, abstractmethod
from typing import List, Any
import pandas as pd
from contextlib import contextmanager

# =============================================================================
# 1. Define Dummy/Rescued Classes
# These must match the structure expected by the pickle files.
# Based on the error "Can't get attribute 'Popularity'...", 
# we'll define them and map them during unpickling.
# =============================================================================

class Popularity:
    """
    Rescued class for Popularity model.
    Assumed to hold a dictionary or dataframe of popular repos.
    """
    def __init__(self):
        self.popular_repos = [] # Placeholder

    def recommend(self, user_id, n=5):
        # We will override this logic in the wrapper, 
        # but we need the data from the pickled object.
        pass

class TfidfContent:
    """
    Rescued class for TF-IDF/Content-Based model.
    Assumed to hold TF-IDF matrices and repository lists.
    """
    def __init__(self):
        pass

    def recommend(self, user_id, n=5):
        pass

# =============================================================================
# 2. Custom Unpickler
# =============================================================================

class DummyImpl:
    """Fallback for any missing class."""
    def __init__(self, *args, **kwargs):
        pass

# =============================================================================
# 2. Custom Unpickler
# =============================================================================

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Debugging
        print(f"Finding class: module={module}, name={name}", flush=True) 
        
        # Map missing classes to our local definitions
        if name == 'Popularity':
            return Popularity
        if name == 'TfidfContent':
            return TfidfContent
        
        if 'Popularity' in name:
            return Popularity
        if 'TfidfContent' in name:
            return TfidfContent
        if 'BPR' in name or 'MF' in name: # Catch/Rescue generic BPR/MF classes
             return DummyImpl
            
        try:
            return super().find_class(module, name)
        except (AttributeError, ImportError) as e:
            print(f"WARNING: Handling missing class {module}.{name} with DummyImpl. Error: {e}", flush=True)
            return DummyImpl

def load_rescued_model(path: str) -> Any:
    """Loads a pickle file using the CustomUnpickler."""
    with open(path, 'rb') as f:
        return CustomUnpickler(f).load()

# =============================================================================
# 3. Standard Recommender Interface
# =============================================================================

class BaseRecommender(ABC):
    @abstractmethod
    def load(self, path: str):
        pass

    @abstractmethod
    def predict(self, username: str, user_repos: List[str], all_repos: List[str], top_k: int) -> List[str]:
        pass

# =============================================================================
# 4. Concrete Implementations
# =============================================================================

class SurpriseRecommender(BaseRecommender):
    def __init__(self):
        self.model = None

    def load(self, path: str):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, username: str, user_repos: List[str], all_repos: List[str], top_k: int) -> List[str]:
        if not self.model:
            return []
        
        user_repos_set = set(user_repos)
        candidates = [repo for repo in all_repos if repo not in user_repos_set]
        
        predictions = []
        for repo in candidates:
            try:
                # Surprise prediction object: (uid, iid, r_ui, est, details)
                pred = self.model.predict(username, repo)
                predictions.append((repo, pred.est))
            except:
                pass
            
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [repo for repo, score in predictions[:top_k]]


class PopularityRecommenderWrapper(BaseRecommender):
    def __init__(self):
        self.internal_model = None

    def load(self, path: str):
        self.internal_model = load_rescued_model(path)

    def predict(self, username: str, user_repos: List[str], all_repos: List[str], top_k: int) -> List[str]:
        # Logic based on verification: internal_model has 'repo_score' which is a DataFrame-like structure
        # with repos and scores.
        
        candidates = []
        if not self.internal_model:
            return []

        if hasattr(self.internal_model, 'repo_score'):
            rs = self.internal_model.repo_score
            try:
                # Based on verification, rs is a DataFrame with an Index of repo names
                if hasattr(rs, 'index') and hasattr(rs, 'head'): 
                    # It's a DataFrame or Series. The logs showed it's likely sorted descending.
                    # We accept the order as is.
                    candidates = rs.index.tolist()
                elif isinstance(rs, dict):
                     # If dict {repo: score}
                    sorted_repos = sorted(rs.items(), key=lambda x: x[1], reverse=True)
                    candidates = [r[0] for r in sorted_repos]
                
            except Exception as e:
                print(f"Error processing repo_score: {e}")
                return []
        
        # Filter out user repos
        user_repos_set = set(user_repos)
        final_recs = [repo for repo in candidates if repo not in user_repos_set]
        
        return final_recs[:top_k]

class TfidfRecommenderWrapper(BaseRecommender):
    def __init__(self):
        self.internal_model = None

    def load(self, path: str):
        self.internal_model = load_rescued_model(path)

    def predict(self, username: str, user_repos: List[str], all_repos: List[str], top_k: int) -> List[str]:
        # Safe fallback for TF-IDF since vectorizer component is missing
        if self.internal_model and hasattr(self.internal_model, 'repo_tfidf'):
            # If we had the vectorizer, we would transform user_repos and compute dot product.
            # Without it, we can't map user repos to the TF-IDF space.
            print("Warning: TfidfContent model loaded but vectorizer is missing. Returning empty.")
            return []
        
        return []

# =============================================================================
# 5. Model Loader / Factory
# =============================================================================

class ModelLoader:
    _models = {}
    _available_models_cache = None

    @classmethod
    def get_model(cls, model_name: str, model_path: str) -> BaseRecommender:
        if model_name in cls._models:
            return cls._models[model_name]

        print(f"Loading model: {model_name} from {model_path}")
        
        recommender = None
        if "svd" in model_name.lower():
            recommender = SurpriseRecommender()
        elif "popularity" in model_name.lower():
            recommender = PopularityRecommenderWrapper()
        elif "tfidf" in model_name.lower() or "content" in model_name.lower():
            recommender = TfidfRecommenderWrapper()
        elif "bpr" in model_name.lower() or "mf" in model_name.lower():
            # Try Surprise first, then generic fallback (Tfidf/Dummy wrapper style)
            # If it's a Surprise algorithm (e.g. SVD, NMF), SurpriseRecommender works.
            # If it works like Popularity (custom class), we need a wrapper.
            # We'll try SurpriseRecommender first as best guess for 'mf' (Matrix Factorization).
            # If load fails with pickle error, we might need a custom one.
            # But here we let it try.
            recommender = SurpriseRecommender()
        else:
            # Default to Surprise for unknown pkls as they are likely from the same library
            recommender = SurpriseRecommender()

        try:
            recommender.load(model_path)
            cls._models[model_name] = recommender
            return recommender
        except Exception as e:
            # If standard pickle load failed, try rescuing with CustomUnpickler for BPR/MF
            if "bpr" in model_name.lower():
                print(f"Standard load failed for {model_name}, trying rescue... Error: {e}")
                try:
                    recommender = TfidfRecommenderWrapper() # Re-use generic wrapper (just loads into internal_model)
                    recommender.load(model_path) # Uses load_rescued_model
                    cls._models[model_name] = recommender
                    return recommender
                except Exception as e2:
                    print(f"Rescue failed for {model_name}: {e2}")
                    raise e2

            print(f"Failed to load {model_name}: {e}")
            raise e

    @classmethod
    def get_available_models(cls, model_dir: str) -> List[str]:
        """
        Scans the directory for .pkl files. 
        Does NOT load the models to ensure fast response (Lazy Loading).
        Models will be loaded on-demand during prediction.
        """
        # If we have a cache of names, return it. 
        # Note: If files are added at runtime, we might need to invalidate this.
        if cls._available_models_cache is not None:
             return cls._available_models_cache

        valid_models = []
        if not os.path.exists(model_dir):
            return []

        for f in os.listdir(model_dir):
            if f.endswith(".pkl"):
                model_name = f.replace(".pkl", "")
                valid_models.append(model_name)
        
        cls._available_models_cache = valid_models
        return valid_models

