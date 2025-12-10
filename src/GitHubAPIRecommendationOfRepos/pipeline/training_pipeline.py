from entity.config_entity import DataIngestionConfig, DataTransformationConfig, ModelTrainerConfig
from components.data_ingestion import DataIngestion
from components.data_transformation import DataTransformation
from components.model_trainer import ModelTrainer

class TrainingPipeline:
    def start_training(self):
        try:
            # --- ÉTAPE 1 : Ingestion (MongoDB -> CSV brut) ---
            print(">>> [1/3] Démarrage de l'Ingestion des données <<<")
            ingestion_config = DataIngestionConfig()
            data_ingestion = DataIngestion(config=ingestion_config)
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            print(f"Ingestion terminée. Train: {train_data_path}, Test: {test_data_path}")

            # --- ÉTAPE 2 : Transformation (CSV brut -> CSV propre avec features) ---
            print("\n>>> [2/3] Démarrage de la Transformation des données <<<")
            transformation_config = DataTransformationConfig()
            data_transformation = DataTransformation(config=transformation_config)
            train_arr_path, test_arr_path, _ = data_transformation.initiate_data_transformation(
                train_path=train_data_path, 
                test_path=test_data_path
            )
            print(f"Transformation terminée. Données prêtes dans : {train_arr_path}")

            # --- ÉTAPE 3 : Entraînement (CSV propre -> Modèle .pkl) ---
            print("\n>>> [3/3] Démarrage de l'Entraînement du modèle <<<")
            trainer_config = ModelTrainerConfig()
            model_trainer = ModelTrainer(config=trainer_config)
            
            # Note : Assurez-vous que ModelTrainer lit bien le CSV transformé
            model_path = model_trainer.initiate_model_trainer(train_arr_path, test_arr_path)
            
            print(f"\n>>> SUCCÈS TOTAL ! Modèle final disponible : {model_path} <<<")

        except Exception as e:
            print(f"\n!!! ERREUR CRITIQUE DANS LE PIPELINE !!!\n{e}")

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.start_training()