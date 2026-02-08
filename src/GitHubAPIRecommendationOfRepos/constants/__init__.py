# --- Fichier : constants/__init__.py ---
import os
from datetime import date

ARTIFACTS_DIR = os.path.join(os.getcwd(), "artifacts", f"{date.today()}")

MONGO_DATABASE_NAME = "idx_database"
MONGO_COLLECTION_NAME = "users"

TEST_SIZE = 0.2
RANDOM_STATE = 42