from dataclasses import dataclass
import os
from constants import ARTIFACTS_DIR

@dataclass
class DataIngestionConfig:
    dir_path: str = os.path.join(ARTIFACTS_DIR, "data_ingestion")
    feature_store_file_path: str = os.path.join(dir_path, "feature_store.csv")
    train_file_path: str = os.path.join(dir_path, "train.csv")
    test_file_path: str = os.path.join(dir_path, "test.csv")

@dataclass
class DataTransformationConfig:
    dir_path: str = os.path.join(ARTIFACTS_DIR, "data_transformation")
    preprocessor_obj_file_path: str = os.path.join(dir_path, "preprocessor.pkl")
    transformed_train_file_path: str = os.path.join(dir_path, "train_transformed.csv")
    transformed_test_file_path: str = os.path.join(dir_path, "test_transformed.csv")

@dataclass
class ModelTrainerConfig:
    dir_path: str = os.path.join(ARTIFACTS_DIR, "model_trainer")
    trained_model_file_path: str = os.path.join(dir_path, "model.pkl")