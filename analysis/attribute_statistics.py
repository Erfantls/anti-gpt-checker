from typing import List, Tuple, Dict


def compose_word_probabilities(texts: List[str]) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    Build overall word counts and language model dictionaries from a list of texts.

    Parameters:
    - texts (List[str]): List of texts.

    Returns:
    - Tuple[Dict[str, int], Dict[str, float]]: Tuple containing overall word counts and language model.
    """
    overall_counts = {}
    total_words = 0

    # Build overall word counts
    for text in texts:
        words = text.split()
        total_words += len(words)

        for word in words:
            overall_counts[word] = overall_counts.get(word, 0) + 1

    # Build language model
    language_word_probabilities = {word: count / total_words for word, count in overall_counts.items()}

    return overall_counts, language_word_probabilities
