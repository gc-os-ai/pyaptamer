__author__ = ["nennomp"]
__all__ = [
    "filter_words",
    "compute_protein_word_frequencies",
]

from collections import Counter

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


def compute_protein_word_frequencies(
    sequences: list[str], n: int = 3
) -> dict[str, float]:
    """Compute protein n-gram frequencies from a collection of protein sequences.

    This function is used to generate the protein word frequency dictionary
    required by AptaTransPipeline for a specific dataset.

    Parameters
    ----------
    sequences : list of str
        Protein sequences (strings of amino acid single-letter codes).
    n : int, optional, default=3
        The length of n-grams (word size). Typically 3 for AptaTrans.

    Returns
    -------
    dict[str, float]
        A dictionary mapping each n-gram to its raw count (frequency) in the dataset.

    Examples
    --------
    >>> sequences = ["MKTVR", "MKTVE"]
    >>> freq = compute_protein_word_frequencies(sequences, n=3)
    >>> sorted(freq.items())[:3]
    [('KTV', 2), ('MKT', 2), ('TVR', 1)]
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if not sequences:
        return {}

    counts: Counter[str] = Counter()
    for seq in sequences:
        seq = seq.upper()
        # Skip sequences shorter than n
        if len(seq) < n:
            continue
        for i in range(len(seq) - n + 1):
            word = seq[i : i + n]
            counts[word] += 1

    return dict(counts)
