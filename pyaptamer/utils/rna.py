__author__ = ["nennomp"]
__all__ = [
    "dna2rna",
    "generate_all_aptamer_triplets",
    "rna2vec",
]

from itertools import product

import numpy as np


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


def generate_all_aptamer_triplets() -> dict[str, int]:
    """
    Generate a dictionary mapping all possible 3-mer RNA subsequences (triplets) to
    unique indices.

    Returns
    -------
    dict[str, int]
        A dictionary where keys are 3-mer RNA subsequences and values are unique
        indices.
    """
    nucleotides = ["A", "C", "G", "U", "N"]  # 'N' marks unknown nucleotides
    # create a dictionary mapping every possible 3-nucleotide combination (triplet) to
    # a unique index, Should be 5^3 = 125 possible triplets (AAA, AAC, AAG, ..., NNN).
    words = {
        "".join(triplet): i + 1
        for i, triplet in enumerate(product(nucleotides, repeat=3))
    }
    return words


def rna2vec(sequence_list: list[str], max_sequence_length: int = 275) -> np.ndarray:
    """Convert a list of RNA sequences into a numerical representation.

    First, if not already in RNA format, the sequences are converted from DNA to RNA.
    Then, all overlapping triplets (3-nucleotide combinations) are extracted from each
    RNA sequence and mapped to unique indices. Finally, the sequences are zero padded
    to length `max_sequence_length`. The result is a numpy array where each row
    corresponds to a sequence, and each column corresponds to an integer representing
    the triplet's index in dictionary `words`.

    If the number of extracted triplets is grerater than `max_sequence_length`, the
    sequence is truncated to fit.

    Parameters
    ----------
    sequence_list : list[str]
        A list containing RNA sequences as strings.

    Returns
    -------
    np.ndarray
        A numpy array containing the numerical representation of the RNA sequences.

    Raises
    ------
    ValueError
        If `max_sequence_length` is less than or equal to 0.

    Examples
    --------
    >>> from pyaptamer.utils.rna import rna2vec
    >>> # two triplets: 'AAAC' -> ['AAA', 'AAC']
    >>> rna = rna2vec(["AAAC"], max_sequence_length=4)
    >>> print(rna)
    [[1 2 0 0]]
    """
    if max_sequence_length <= 0:
        raise ValueError("`max_sequence_length` must be greater than 0.")

    words = generate_all_aptamer_triplets()

    result = []
    for sequence in sequence_list:
        sequence = dna2rna(sequence)

        # extract all overlapping triplets from the sequence
        # e.g., 'ACGUA' -> ['ACG', 'CGU', 'GUA']
        converted = [
            words.get(sequence[i : i + 3], 0) for i in range(len(sequence) - 2)
        ]

        # skip sequences that convert to an empty list
        if any(converted):
            padded_sequence = np.pad(
                array=converted,
                pad_width=(0, max_sequence_length - len(converted)),
                constant_values=0,
            )
            result.append(padded_sequence)

    return np.array(result)
