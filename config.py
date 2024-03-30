import os

import nltk
import spacy
import torch
from dotenv import load_dotenv
from pymongo import MongoClient
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AutoTokenizer, AutoModelForCausalLM

load_dotenv()
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('pl196x')
nltk.download('wordnet')

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

def init_spacy_polish_nlp_model() -> None:
    global SPACY_POLISH_NLP_MODEL
    SPACY_POLISH_NLP_MODEL = spacy.load("pl_core_news_lg")

def init_spacy_english_nlp_model() -> None:
    global SPACY_ENGLISH_NLP_MODEL
    SPACY_ENGLISH_NLP_MODEL = spacy.load("en_core_web_trf")

SPACY_POLISH_NLP_MODEL_SMALL = None
SPACY_ENGLISH_NLP_MODEL_SMALL = None


def init_spacy_polish_nlp_model_small() -> None:
    global SPACY_POLISH_NLP_MODEL_SMALL
    SPACY_POLISH_NLP_MODEL_SMALL = spacy.load("pl_core_news_sm")

def init_spacy_english_nlp_model_small() -> None:
    global SPACY_ENGLISH_NLP_MODEL_SMALL
    SPACY_ENGLISH_NLP_MODEL_SMALL = spacy.load("en_core_web_sm")


PERPLEXITY_POLISH_MODEL = None
PERPLEXITY_ENGLISH_MODEL = None
PERPLEXITY_POLISH_GPT2_MODEL_ID = os.getenv("PERPLEXITY_POLISH_GPT2_MODEL_ID")
PERPLEXITY_POLISH_QRA_MODEL_ID = os.getenv("PERPLEXITY_POLISH_QRA_MODEL_ID")
PERPLEXITY_ENGLISH_GPT2_MODEL_ID = os.getenv("PERPLEXITY_ENGLISH_GPT2_MODEL_ID")
PERPLEXITY_MODEL_ID = os.getenv("PERPLEXITY_MODEL_ID")
PERPLEXITY_POLISH_TOKENIZER = None
PERPLEXITY_ENGLISH_TOKENIZER = None

def init_polish_perplexity_model(model_name: str = PERPLEXITY_POLISH_GPT2_MODEL_ID) -> None:
    global PERPLEXITY_POLISH_MODEL
    PERPLEXITY_POLISH_MODEL = AutoModelForCausalLM.from_pretrained(model_name)
    global PERPLEXITY_POLISH_TOKENIZER
    PERPLEXITY_POLISH_TOKENIZER = AutoTokenizer.from_pretrained(model_name)

def init_english_perplexity_model(model_name: str = PERPLEXITY_ENGLISH_GPT2_MODEL_ID) -> None:
    global PERPLEXITY_ENGLISH_MODEL
    PERPLEXITY_ENGLISH_MODEL = GPT2LMHeadModel.from_pretrained(model_name)
    global PERPLEXITY_ENGLISH_TOKENIZER
    PERPLEXITY_ENGLISH_TOKENIZER = GPT2TokenizerFast.from_pretrained(model_name)




