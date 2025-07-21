__author__ = ["nennomp"]
__all__ = ["dna2rna", "rna2vec", "word2idx",]

from itertools import product

import numpy as np


def word2idx(word: str, words: dict[str, int]) -> int:
    """Convert a word to its index representation based on a given dictionary.

    Parameters
    ----------
    word : str
        The word to be converted.
    words : dict[str, int]
        A dictionary mapping words to their indices.

    Returns
    -------
    int
        The index representing the word in the dictionary, or 0 if the word is not 
        found.
    """
    return words.get(word, 0)

def dna2rna(sequence: str) -> str:
    """
    Convert a DNA sequence to an RNA sequence.

    Nucleotides 'T' in the DNA sequence are replaced with 'U' in the RNA sequence. 
    Unknown nucleotides are replaced with 'N'.

    Parameters
    ----------
    seq : str
        The DNA sequence to be converted.

    Returns
    -------
    str
        The converted RNA sequence.
    """
    # mapping DNA nucleotides to RNA nucleotides
    map = {'A': 'A', 'C': 'C', 'G': 'G', 'U': 'U', 'T': 'U'}
    result = ''
    for char in sequence:
        if char in map.keys():
            result += map[char]
        else:
            result += 'N' # placeholder for unknown nucleotides
    return result

def rna2vec(sequence_list: list[str], max_sequence_length: int = 275) -> np.ndarray:
    """Convert a list of RNA sequences into a numerical representation.

    Parameters
    ----------
    sequence_list : list[str]
        A list of RNA sequences to be converted.
    
    Returns
    -------
    np.ndarray
        A numpy array containing the numerical representation of the RNA sequences.
    """
    nucleotides = ['A', 'C', 'G', 'U', 'N'] # 'N' marks unknown nucleotides

    # create a dictionary mapping every possible 3-nucleotide combination (triplet) to 
    # a unique index, Should be 5^3 = 125 possible triplets (AAA, AAC, AAG, ..., NNN).
    words = {
        ''.join(triplet): i + 1 
        for i, triplet in enumerate(product(nucleotides, repeat=3))
    }

    result = []
    for sequence in sequence_list:
        sequence = dna2rna(sequence)

        # extract all overlapping triplets from the sequence
        # e.g., 'ACGUA' -> ['ACG', 'CGU', 'GUA']
        converted = [
            word2idx(sequence[i:i + 3], words)
            for i in range(len(sequence) - 2)
        ]

        # skip sequences that convert to an empy list
        if any(converted):
            padded_sequence = np.pad(
                array=converted,
                pad_width=(0, max_sequence_length - len(converted)), 
                constant_values=0,
            )
            result.append(padded_sequence)

    return np.array(result, dtype=np.int32)