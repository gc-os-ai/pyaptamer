__author__ = ["nennomp"]
__all__ = [
    "filter_words",
]

import numpy as np


def filter_words(words: dict[str, float]) -> dict[str, int]:
    """Filter words with below average frequency.

    Parameters
    ----------
    words : dict[str, float]
        A dictionary containing words and their frequencies.

    Returns
    -------
    dict[str, int]
        A dictionary mapping filtered words to unique integer indices.
    """
    mean_freq = np.mean(list(words.values()))
    words = [seq for seq, freq in words.items() if freq > mean_freq]
    words = {word: i + 1 for i, word in enumerate(words)}

    return words
