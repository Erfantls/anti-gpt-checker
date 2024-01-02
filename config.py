import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_PORT = int(os.getenv("MONGODB_PORT"))
MONGODB_DB_NAME= os.getenv("MONGODB_DB_NAME")
MONGODB_EMAILS_SRC_COLLECTIONS = {
    "email_spam_data": os.getenv("MONGODB_COLLECTION_EMAIL_SPAM_DATASET"),
    "email_spam_assassin": os.getenv("MONGODB_COLLECTION_EMAIL_SPAM_ASSASSIN_DATASET"),
    "email_class_git": os.getenv("MONGODB_COLLECTION_EMAIL_CLASSIFICATION_GITHUB"),
}

MONGO_CLIENT = MongoClient(MONGODB_URI, MONGODB_PORT)
