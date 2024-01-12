import quopri
import re
import math
from typing import Dict, List, Union

from bs4 import BeautifulSoup
from textblob import TextBlob
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import stylo_metrix as sm
from pandas import DataFrame
import langid

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


def count_punctuation(text: str) -> int:
    """
    Count the occurrences of punctuation marks in the given text.

    Parameters:
    - text (str): Input text.

    Returns:
    - int: Count of punctuation marks.
    """
    punctuation_count = sum([1 for char in text if char in '.,;:!?'])
    return punctuation_count


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

def calculate_perplexity(text: str, language_word_probabilities: Dict[str, float]) -> float:
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

def calculate_burstiness(text: str, overall_counts: Dict[str, int]) -> float:
    """
    Calculate burstiness for each word based on the chi-squared statistic.

    Parameters:
    - text (str): Text to analyze.
    - overall_counts (Dict[str, int]): Overall word frequencies.

    Returns:
    - float: Burstiness value.
    """
    window_counts = {}  # Initialize window counts

    words = text.split()
    for word in words:
        window_counts[word] = window_counts.get(word, 0) + 1

    burstiness = 0.0

    for word, overall_freq in overall_counts.items():
        window_freq = window_counts.get(word, 0)

        # Calculate expected frequency (assuming uniform distribution)
        expected_freq = overall_freq / len(overall_counts)

        # Calculate chi-squared statistic
        chi_squared = (window_freq - expected_freq) ** 2 / expected_freq

        # Update burstiness
        burstiness += chi_squared

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
    lang, _ = langid.classify(text.lower())
    return lang
