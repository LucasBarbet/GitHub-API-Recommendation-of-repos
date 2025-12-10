import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from entity.config_entity import ModelTrainerConfig
from utils.common import save_object

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def initiate_model_trainer(self, train_path, test_path):
        try:
            # Chargement
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Séparation X, y (On suppose que la cible est la dernière colonne)
            X_train = train_df.iloc[:, :-1]
            y_train = train_df.iloc[:, -1]
            X_test = test_df.iloc[:, :-1]
            y_test = test_df.iloc[:, -1]

            # Entraînement
            model = RandomForestRegressor()
            model.fit(X_train, y_train)

            # Sauvegarde du modèle (Artifact)
            save_object(
                file_path=self.config.trained_model_file_path,
                obj=model
            )
            
            return self.config.trained_model_file_path

        except Exception as e:
            raise e