# --- Fichier : components/data_ingestion.py ---
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from entity.config_entity import DataIngestionConfig
from utils.common import create_directories
from utils.db_connector import get_database 
from constants import MONGO_COLLECTION_NAME

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):
        try:
            db = get_database()
            collection = db[MONGO_COLLECTION_NAME]
            
            # Récupération des données
            # On ne prend que les champs utiles (pas besoin de la date_ajout pour le modèle par exemple)
            cursor = collection.find({}, {"_id": 1, "repos": 1})
            df = pd.DataFrame(list(cursor))
            
            # MongoDB renvoie '_id', on le renomme en 'username' pour la clarté
            if "_id" in df.columns:
                df.rename(columns={"_id": "username"}, inplace=True)

            # Note : La colonne 'repos' contient des listes []. 
            # Pandas gère ça, mais attention au format CSV (ça deviendra des chaînes de caractères "['a', 'b']").
            
            create_directories([self.config.dir_path])
            
            # Sauvegarde
            df.to_csv(self.config.feature_store_file_path, index=False, header=True)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.config.train_file_path, index=False, header=True)
            test_set.to_csv(self.config.test_file_path, index=False, header=True)

            return (
                self.config.train_file_path,
                self.config.test_file_path
            )

        except Exception as e:
            raise e