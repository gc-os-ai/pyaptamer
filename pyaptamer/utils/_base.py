__author__ = ["nennomp"]
__all__ = [
    "dna2rna",
    "filter_words",
    "generate_triplets",
]

from itertools import product

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


def generate_triplets(letters: list[str]) -> dict[str, int]:
    """Generate a dictionary of all possible triplets combinations from given letters.

    Parameters
    ----------
    letters : list[str]
        List of characters to form triplets from.

    Returns
    -------
    dict[str, int]
        A dictionary mapping each triplet to a unique integer ID.
    """
    triplets = {}
    for idx, triplet in enumerate(product(letters, repeat=3)):
        triplets["".join(triplet)] = idx + 1

    return triplets


def dna2rna(sequence: str) -> str:
    """
    Convert a DNA sequence to an RNA sequence.

    Nucleotides 'T' in the DNA sequence are replaced with 'U' in the RNA sequence.
    Unknown nucleotides are replaced with 'N'. Other nucleotides ('A', 'C', 'G') remain
    unchanged.

    Parameters
    ----------
    seq : str
        The DNA sequence to be converted.

    Returns
    -------
    str
        The converted RNA sequence.
    """
    # replace nucleotides 'T' with 'U'
    result = sequence.translate(str.maketrans("T", "U"))
    for char in result:
        if char not in "ACGU":
            result = result.replace(char, "N")  # replace unknown nucleotides with 'N'
    return result
