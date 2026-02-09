import json
import pickle
import pandas as pd
import mlflow.pyfunc


class RecommenderPyFuncModel(mlflow.pyfunc.PythonModel):
    """Minimal wrapper around a pickled recommender.

    Input DataFrame columns:
      - user (str) required
      - k (float/int) optional (defaults to metadata default_k)
      - seen_repos (json list or python list) optional
    Output:
      - user
      - recommendations (list[str])
    """

    def load_context(self, context):
        with open(context.artifacts["recommender_pkl"], "rb") as f:
            self.recommender = pickle.load(f)
        with open(context.artifacts["metadata_json"], "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.all_train_repos = self.metadata["all_train_repos"]
        self.default_k = int(self.metadata.get("default_k", 10))

    def predict(self, context, model_input):
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)
        if "user" not in model_input.columns:
            raise ValueError("Input DataFrame must contain a 'user' column.")

        users = model_input["user"].astype(str).tolist()
        if "k" in model_input.columns:
            ks = model_input["k"].fillna(self.default_k).astype(float).tolist()
        else:
            ks = [float(self.default_k)] * len(users)

        seen_col = model_input["seen_repos"] if "seen_repos" in model_input.columns else None
        recos = []
        for i, user in enumerate(users):
            k = int(ks[i])
            seen = set()
            if seen_col is not None:
                v = seen_col.iloc[i]
                if isinstance(v, str):
                    try:
                        v = json.loads(v)
                    except Exception:
                        v = []
                if isinstance(v, (list, tuple, set)):
                    seen = set(map(str, v))

            candidates = [r for r in self.all_train_repos if r not in seen]
            recos.append(self.recommender.recommend(user=user, k=k, candidates=candidates))

        return pd.DataFrame({"user": users, "recommendations": recos})
