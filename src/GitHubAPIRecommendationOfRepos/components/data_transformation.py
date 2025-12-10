import sys
import os
import pandas as pd
import ast
from sklearn.base import BaseEstimator, TransformerMixin
from entity.config_entity import DataTransformationConfig
from utils.common import save_object, create_directories

# Cette classe convertit la colonne 'repos' (liste) en 'nb_repos' (int)
class FeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        # Si 'repos' est une chaine "['a', 'b']", on la remet en liste Python
        if 'repos' in df.columns:
            if isinstance(df['repos'].iloc[0], str):
                # ast.literal_eval est plus sûr que eval()
                df['repos'] = df['repos'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            
            # Création de la feature
            df['nb_repos'] = df['repos'].apply(len)
            
            # On ne garde que la colonne numérique pour le modèle simple
            # (Vous pourrez ajouter d'autres features ici plus tard)
            return df[['nb_repos']]
        return df

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.feature_generator = FeatureGenerator()

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # 1. Chargement des données brutes
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            print("Transformation des données en cours...")

            # 2. Transformation
            # On fit sur le train (si besoin de calculer des moyennes, etc.)
            self.feature_generator.fit(train_df)
            
            input_feature_train_df = self.feature_generator.transform(train_df)
            input_feature_test_df = self.feature_generator.transform(test_df)

            # Si on avait une colonne cible (target), on la rajouterait ici
            # Pour l'instant, supposons que 'nb_repos' est notre feature X
            # et imaginons une fausse target 'score' si elle n'existe pas, 
            # ou on utilise nb_repos comme input pour prédire autre chose.
            # *Note : Pour cet exemple, je sauvegarde juste les features transformées.*

            # 3. Sauvegarde des CSV propres
            create_directories([self.config.dir_path])
            
            input_feature_train_df.to_csv(self.config.transformed_train_file_path, index=False, header=True)
            input_feature_test_df.to_csv(self.config.transformed_test_file_path, index=False, header=True)

            # 4. Sauvegarde de l'objet "processeur" (le FeatureGenerator)
            # C'est CRUCIAL en MLOps : on veut appliquer exactement la même transformation
            # aux nouvelles données qui arriveront plus tard.
            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=self.feature_generator
            )

            print(f"Transformation terminée. Fichiers sauvegardés dans {self.config.dir_path}")

            return (
                self.config.transformed_train_file_path,
                self.config.transformed_test_file_path,
                self.config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise e