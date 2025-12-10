from pymongo import MongoClient
import constants as const

def get_database():
    # Crée la connexion
    client = MongoClient(const.MONGO_URI)
    # Retourne la base de données spécifique
    return client[const.DB_NAME]