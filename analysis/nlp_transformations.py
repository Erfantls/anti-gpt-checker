from typing import Tuple, List

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
