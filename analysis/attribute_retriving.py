import quopri
import re
import math
from typing import Dict, List, Union, Optional, Tuple

import nltk
import numpy as np
from bs4 import BeautifulSoup
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import stylo_metrix as sm
from numpy import std, var, mean
from pandas import DataFrame
from collections import Counter
from langdetect import detect as langdetect_detect, DetectorFactory, LangDetectException
import langid

import pycld2 as cld2
import cld3
from textblob import TextBlob

import torch

from analysis.nlp_transformations import replace_links_with_text, remove_stopwords_and_punctuation

from models.stylometrix_metrics import StyloMetrixMetrics
import html2text

html2text_handler = html2text.HTML2Text()




def average_word_length(text: str) -> float:
    """
    Calculate the average length of words in the given text.

    Parameters:
    - text (str): Input text.

    Returns:
    - float: Average word length.
    """
    words = word_tokenize(text)
    word_lengths = [len(word) for word in words]
    return sum(word_lengths) / len(word_lengths) if len(word_lengths) > 0 else 0


def average_sentence_length(text: str) -> float:
    """
    Calculate the average length of sentences in the given text.

    Parameters:
    - text (str): Input text.

    Returns:
    - float: Average sentence length.
    """
    sentences = re.split(r'[.!?]', text)
    sentence_lengths = [len(sentence.split()) for sentence in sentences]
    return sum(sentence_lengths) / len(sentence_lengths) if len(sentence_lengths) > 0 else 0


def count_pos_tags_eng(text: str, pos_tags=None) -> Dict[str, int]:
    """
    Count the occurrences of specified part-of-speech tags in the given English text.

    Parameters:
    - text (str): Input text.
    - pos_tags (List[str]): List of part-of-speech tags to count.

    Returns:
    - dict: Dictionary containing counts for each specified part-of-speech tag.
    """
    if pos_tags is None:
        pos_tags = ['NN', 'VB', 'JJ']

    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    pos_counts = {tag: 0 for tag in pos_tags}

    for word, tag in tagged_words:
        if tag in pos_counts:
            pos_counts[tag] += 1

    return pos_counts


def sentiment_score_eng(text: str) -> float:
    """
    Calculate the sentiment polarity score of the given English text.

    Parameters:
    - text (str): Input text.

    Returns:
    - float: Sentiment polarity score.
    """
    blob = TextBlob(text)
    return blob.sentiment.polarity


def count_punctuation(text: str) -> float:
    """
    Count the occurrences of punctuation marks in the given text.

    Parameters:
    - text (str): Input text.

    Returns:
    - float: Count of punctuation marks per length of text
    """
    punctuation_count = sum([1 for char in text if char in '.,;:!?'])
    if len(text) == 0:
        return 0
    return punctuation_count/len(text)


def stylo_metrix_analysis(texts: List[str]) -> List[StyloMetrixMetrics]:
    stylo = sm.StyloMetrix('pl')
    metrics = stylo.transform(texts)
    converted_metrics = stylo_metrix_output_to_model(metrics)
    return converted_metrics

def stylo_metrix_output_to_model(metrics_df: DataFrame) -> List[StyloMetrixMetrics]:
    model_instances = []
    for index, row in metrics_df.iterrows():
        model_instance = StyloMetrixMetrics(**row.to_dict())
        model_instances.append(model_instance)

    return model_instances

def calculate_perplexity_old(text: str, language_word_probabilities: Dict[str, float]) -> float:
    """
    Calculate perplexity for a given text based on a language model.

    Parameters:
    - text (str): The input text.
    - language_model (Dict[str, float]): A language model that provides probabilities for each word.

    Returns:
    - float: Perplexity value.
    """
    words = text.split()
    N = len(words)
    log_prob_sum = 0.0

    for word in words:
        # Assuming language_model is a dictionary with word probabilities
        # If you have a language model from a library, adjust this part accordingly
        word_prob = language_word_probabilities.get(word, 1e-10)  # Use a small default probability for unseen words
        log_prob_sum += math.log2(word_prob)

    average_log_prob = log_prob_sum / N
    perplexity = 2 ** (-average_log_prob)

    return perplexity


def calculate_perplexity(text: str, language_code: str, per_token: Optional[str] = "word", return_base_ppl: bool = False) -> Optional[float]:
    text = replace_links_with_text(text)
    if per_token not in ["word", "char"]:
        raise ValueError("per_token must be either 'word' or 'char'")

    match per_token:
        case "word":
            denominator = len(text.split())
        case "char":
            denominator = len(text)
        case _:
            raise ValueError("per_token must be either 'word' or 'char'")

    if language_code == "pl":
        from config import PERPLEXITY_POLISH_TOKENIZER, PERPLEXITY_POLISH_MODEL
        tokenizer = PERPLEXITY_POLISH_TOKENIZER
        model = PERPLEXITY_POLISH_MODEL
    elif language_code == "en":
        from config import PERPLEXITY_ENGLISH_TOKENIZER, PERPLEXITY_ENGLISH_MODEL
        tokenizer = PERPLEXITY_ENGLISH_TOKENIZER
        model = PERPLEXITY_ENGLISH_MODEL
    else:
        raise ValueError("Language code must be either 'pl' or 'en'")

    encodings = tokenizer(text, return_tensors="pt")

    max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to('cuda' if torch.cuda.is_available() else 'cpu')
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    if len(nlls) == 0:
        return None

    if return_base_ppl:
        ppl = float((torch.stack(nlls).sum()).float())
        return ppl

    ppl = float(torch.exp((torch.stack(nlls).sum())/denominator).float())
    return ppl



#https://github.com/AIAnytime/GPT-Shield-AI-Plagiarism-Detector

def calculate_burstiness(lemmatize_text: str, language_code: str) -> float:
    tokens = remove_stopwords_and_punctuation(lemmatize_text, language_code)

    word_freq = nltk.FreqDist(tokens)
    avg_freq = float(sum(word_freq.values()) / len(word_freq))
    variance = float(sum((freq - avg_freq) ** 2 for freq in word_freq.values()) / len(word_freq))

    burstiness_score = variance / (avg_freq ** 2)
    return burstiness_score

#https://github.com/thinkst/zippy/blob/main/burstiness.py
def calc_distribution_sentence_length(sentences : List[str]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    '''
    Given a list of sentences returns the standard deviation, variance and average of sentence length in terms of both chars and words
    '''
    lens = []
    for sentence in sentences:
        chars = len(sentence)
        if chars < 1:
            continue
        words = len(sentence.split(' '))
        lens.append((chars, words))
    char_data = [x[0] for x in lens]
    word_data = [x[1] for x in lens]
    char_length_distribution: Tuple[float, float, float] = (std(char_data), var(char_data), mean(char_data))
    word_length_distribution: Tuple[float, float, float] = (std(word_data), var(word_data), mean(word_data))
    return char_length_distribution, word_length_distribution


def calculate_burstiness_old(text: str, window_size: int, language_word_probabilities: Dict[str, float]) -> float:
    """
    Calculate burstiness of the text based on the standard deviation of perplexity over windows.

    Parameters:
    - text (str): Input text.
    - window_size (int): Size of the window for calculating perplexity.
    - language_model (LanguageModel): Language model object.

    Returns:
    - float: Burstiness value.
    """
    perplexities = []

    # Split the text into windows
    windows = [text[i:i+window_size] for i in range(0, len(text), window_size)]

    # Calculate perplexity for each window
    for window in windows:
        perplexity = calculate_perplexity(window, language_word_probabilities)
        perplexities.append(perplexity)

    # Calculate standard deviation of perplexity
    burstiness = np.std(perplexities)

    return burstiness


def extract_strings_from_html(html_text: str, ignore_links: bool = True) -> str:
    """
    Extract text strings from HTML.

    Parameters:
    - html_text (str): HTML text.

    Returns:
    - str with concatenated text strings.
    """
    html2text_handler.ignore_links = ignore_links
    return html2text_handler.handle(html_text)


def detect_language(text: str) -> str:
    """
    Detect the language of the given text.

    Parameters:
    - text (str): Input text.

    Returns:
    - str: Detected language or 'unknown'.
    """
    # lang, _ = langid.classify(text.lower())
    DetectorFactory.seed = 0
    try:
        lang = langdetect_detect(text)
    except LangDetectException:
        lang = 'unknown'
    return lang


def detect_language_by_voting(text):
    # Ensure consistent results with langdetect
    DetectorFactory.seed = 0

    # Initialize a list to store detected languages
    detected_languages = []

    # langdetect
    try:
        detected_languages.append(langdetect_detect(text))
    except LangDetectException:
        pass

    # langid
    try:
        langid_result = langid.classify(text)
        detected_languages.append(langid_result[0])
    except Exception:
        pass

    # cld2
    try:
        isReliable, _, details = cld2.detect(text)
        if isReliable:
            detected_languages.append(details[0][1])
    except Exception:
        pass

    # cld3
    try:
        cld3_result = cld3.get_language(text)
        if cld3_result is not None:
            detected_languages.append(cld3_result.language)
    except Exception:
        pass

    if len(detected_languages) == 0:
        return 'unknown'

    # Perform voting
    language_counts = Counter(detected_languages)
    most_common_language, _ = language_counts.most_common(1)[0]

    return most_common_language
