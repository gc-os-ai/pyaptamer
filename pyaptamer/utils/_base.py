__author__ = ["nennomp"]
__all__ = [
    "filter_words",
]

import numpy as np


def filter_words(words: dict[str, float]) -> dict[str, int]:
    """Retain words with strictly above-average frequency and map to indices.

    Parameters
    ----------
    words : dict[str, float]
        A dictionary containing words and their absolute or relative frequencies.

    Returns
    -------
    dict[str, int]
        A dictionary mapping the retained words to unique integer indices
        (1-indexed).

    Examples
    --------
    >>> from pyaptamer.utils import filter_words
    >>> freqs = {"A": 10.0, "C": 2.0, "G": 8.0, "T": 1.0}
    >>> print(filter_words(freqs))
    {'A': 1, 'G': 2}
    """
    if not words:
        return {}

    # Calculate the threshold across all provided words
    mean_freq = np.mean(list(words.values()))

    # Keep only words that appear more often than the average
    filtered_list = [seq for seq, freq in words.items() if freq > mean_freq]

    # Map the remaining words to a 1-based index
    word_to_idx = {word: i + 1 for i, word in enumerate(filtered_list)}

    return word_to_idx