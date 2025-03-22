import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve MongoDB Atlas URI and Database name from environment variables
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "my_database")  # Default DB name if not provided

def get_db_client():
    """
    Create and return a MongoClient instance connected to MongoDB Atlas.
    """
    if not MONGO_URI:
        raise Exception("MONGO_URI is not set in your environment variables.")
    client = MongoClient(MONGO_URI)
    return client

def get_database():
    """
    Retrieve the database instance.
    """
    client = get_db_client()
    db = client[DB_NAME]
    return db

if __name__ == "__main__":
    # Test connection by retrieving the database name
    db = get_database()
    print(f"Connected to database: {db.name}")