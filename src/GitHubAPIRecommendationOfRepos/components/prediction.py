import os
import pandas as pd
from pathlib import Path
from ..utils.common import load_bin
from ..entity.config_entity import ModelTrainerConfig, DataTransformationConfig

class PredictionPipeline:
    def __init__(self):
        # Chemins vers les fichiers générés par l'entraînement
        self.model_path = ModelTrainerConfig().trained_model_file_path
        self.preprocessor_path = DataTransformationConfig().preprocessor_obj_file_path

    def predict(self, username, user_repos, top_k=5):
        """
        Reçoit un utilisateur et ses repos, et renvoie une recommandation.
        """
        try:
            # 1. Charger le modèle et le préprocesseur via load_bin
            # Note : load_bin attend un objet Path, donc on convertit le chemin string
            model = load_bin(Path(self.model_path))
            preprocessor = load_bin(Path(self.preprocessor_path))

            # 2. Préparer les données (DataFrame)
            # Attention : Le nom de la colonne doit correspondre à celui utilisé lors du fit (voir DataTransformation)
            input_data = pd.DataFrame({"repos": [user_repos]}) 
            
            # 3. Transformer les données
            data_scaled = preprocessor.transform(input_data)

            # 4. Prédire 
            # (Ceci retourne un résultat brut, ex: un chiffre ou une classe)
            prediction = model.predict(data_scaled)

            # --- LOGIQUE DE RECOMMANDATION (À ADAPTER) ---
            # Pour l'instant, le modèle Random Forest Regressor/Classifier sort un chiffre.
            # Ici, nous allons simuler un retour de dépôts basés sur cette prédiction.
            
            # Exemple : Si c'est une classification de cluster, on pourrait renvoyer les tops repos du cluster.
            # Pour cet exemple MLOps, on renvoie une liste statique ou basée sur la logique métier.
            
            recommendations = ["tensorflow/tensorflow", "keras-team/keras", "pytorch/pytorch", "huggingface/transformers", "scikit-learn/scikit-learn", "pandas-dev/pandas", "numpy/numpy"]
            return recommendations[:top_k]

        except Exception as e:
            raise e