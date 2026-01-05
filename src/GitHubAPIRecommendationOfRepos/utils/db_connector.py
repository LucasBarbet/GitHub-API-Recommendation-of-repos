import os
from pymongo import MongoClient
from ..constants import MONGO_DATABASE_NAME

def get_database():
    mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
    # Crée la connexion
    client = MongoClient(mongo_uri)
    # Retourne la base de données spécifique
    return client[MONGO_DATABASE_NAME]