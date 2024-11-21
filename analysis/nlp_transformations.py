import string
from typing import Tuple, List

import re

import nltk
from nltk.corpus import stopwords

from config import RELATIVE_PATH_TO_PROJECT


def lemmatize_text(text: str, lang_code: str) -> Tuple[str, List[str]]:
    """
    Lemmatize the text using the appropriate Spacy NLP model
    :param text: text to be lemmatized
    :param lang_code: language of the text
    :return: str and list of lemmatized words
    """
    from config import SPACY_POLISH_NLP_MODEL, SPACY_ENGLISH_NLP_MODEL
    if lang_code == "pl" and SPACY_POLISH_NLP_MODEL is None:
        raise ValueError("Polish NLP model is not initialized")
    elif lang_code == "en" and SPACY_ENGLISH_NLP_MODEL is None:
        raise ValueError("English NLP model is not initialized")

    if lang_code == "pl":
        nlp = SPACY_POLISH_NLP_MODEL
    elif lang_code == "en":
        nlp = SPACY_ENGLISH_NLP_MODEL
    else:
        raise ValueError(f"Language {lang_code} is not supported")

    doc = nlp(text)
    lemma_list = [token.lemma_ for token in doc]
    lemma_text = " ".join(lemma_list)
    return lemma_text, lemma_list


def remove_stopwords_punctuation_emojis_and_splittings(lemmatize_text: str, lang_code: str) -> List[str]:
    """
    Remove stopwords and punctuation from the text
    :param lemmatize_text: lemmatize_text to be cleaned
    :param lang_code: language of the text
    :return: list of cleaned words
    """
    lemmatize_text = deemojify(lemmatize_text)
    lemmatize_text = remove_footers(lemmatize_text)
    lemmatize_text = re.sub(r'[=]+', ' ', lemmatize_text)
    lemmatize_text = re.sub(r'[-]+', ' ', lemmatize_text)
    lemmatize_text = re.sub(r'\d+', ' ', lemmatize_text)
    lemmatize_text = re.sub(r'[^\w\s]', '', lemmatize_text)
    tokens = lemmatize_text.split()
    if lang_code == "pl":
        # read polish stopwords from file
        with open(f"{RELATIVE_PATH_TO_PROJECT}static/polish.stopwords.txt", "r") as file:
            stop_words = file.read().splitlines()
    elif lang_code == "en":
        stop_words = set(stopwords.words('english'))
    else:
        raise ValueError(f"Language {lang_code} is not supported")
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
    return tokens


def clean_text(input_text: str) -> str:
    # Replace lines with only a dot or spaces with a blank line
    cleaned_text = re.sub(r'^[\s.]*$', '\n', input_text, flags=re.MULTILINE)
    # Replace multiple blank lines with a single blank line
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
    # Remove leading and trailing blank lines
    cleaned_text = cleaned_text.strip('\n')
    return cleaned_text


def deemojify(text: str) -> str:
    # Unicode ranges for emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251" 
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r'', text)


def separate_previous_conversation(text: str) -> Tuple[str, str]:
    lines = text.split('\n')
    previous_messages = []
    response = []

    for line in lines:
        # Check if the line is a previous message or part of the response
        if line.startswith('>'):
            previous_messages.append(line)
        else:
            response.append(line)

    # Join the lists back into strings
    previous_text = '\n'.join(previous_messages)
    response_text = '\n'.join(response)

    return previous_text, response_text

def remove_footers(text: str) -> str:
    # Remove the footer
    return re.sub(r"\w{2}\., \d{1,2} \w{3} \d{4} o \d{2}:\d{2} .+ <.+@.+> napisał(a|\(a\)|):", "", text)


def replace_links_with_text(text: str, replacement: str="link") -> str:
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+|\bwww\.\S+\.\S+'
    replaced_text = re.sub(url_pattern, replacement, text)
    return replaced_text


def split_into_sentences(text: str, lang_code: str) -> List[str]:
    if lang_code == "pl":
        sentence_tokenizer = nltk.data.load('tokenizers/punkt/polish.pickle')
    elif lang_code == "en":
        sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    else:
        raise ValueError(f"Language {lang_code} is not supported")

    sentences = sentence_tokenizer.tokenize(text)
    return sentences

def replace_meaningful_report_tags(text: str) -> str:
    # Replace tags with placeholders
    text = text.replace("<<Imię>>", "IMIĘ").replace("<<Nazwisko>>", "NAZWISKO").replace("<<adres e-mail>>", "ADRES E-MAIL")
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    return text
def remove_report_tags(text: str) -> str:
    # Remove the tags from the report
    tags = ["<<Imię>>", "<<Nazwisko>>", "<<nr albumu>>", "<<adres e-mail>>", "<<tabela>>", "<<obrazek>>", "<<obcy język>>"]
    for tag in tags:
        text = text.replace(tag, "")
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    return text

def replace_whitespaces(text: str) -> str:
    # Replace multiple whitespaces with a single whitespace
    return text.replace('\u200B', " ")
