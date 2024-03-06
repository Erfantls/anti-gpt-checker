import os

import spacy
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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_EMAIL_GENERATION_PROMPT_NAME = os.getenv("DEFAULT_EMAIL_GENERATION_PROMPT_NAME")
OPENAI_GPT3_5_MODEL_NAME = os.getenv("OPENAI_GPT3_5_MODEL_NAME")
OPENAI_GPT4_MODEL_NAME = os.getenv("OPENAI_GPT4_MODEL_NAME")
MAX_API_RETRIES = int(os.getenv("MAX_API_RETRIES"))

SPACY_POLISH_NLP_MODEL = None
SPACY_ENGLISH_NLP_MODEL = None

SPACY_POLISH_NLP_MODEL_SMALL = None
SPACY_ENGLISH_NLP_MODEL_SMALL = None


def init_spacy_polish_nlp_model() -> None:
    global SPACY_POLISH_NLP_MODEL
    SPACY_POLISH_NLP_MODEL = spacy.load("pl_core_news_lg")

def init_spacy_english_nlp_model() -> None:
    global SPACY_ENGLISH_NLP_MODEL
    SPACY_ENGLISH_NLP_MODEL = spacy.load("en_core_web_trf")


def init_spacy_polish_nlp_model_small() -> None:
    global SPACY_POLISH_NLP_MODEL_SMALL
    SPACY_POLISH_NLP_MODEL_SMALL = spacy.load("pl_core_news_sm")

def init_spacy_english_nlp_model_small() -> None:
    global SPACY_ENGLISH_NLP_MODEL_SMALL
    SPACY_ENGLISH_NLP_MODEL_SMALL = spacy.load("en_core_web_sm")
