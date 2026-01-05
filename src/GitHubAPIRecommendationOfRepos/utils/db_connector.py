from pymongo import MongoClient
import os
from ..constants import MONGO_DATABASE_NAME


def get_database():
    mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
    client = MongoClient(mongo_uri)
    return client[MONGO_DATABASE_NAME]