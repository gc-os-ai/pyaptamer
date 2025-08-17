__author__ = ["nennomp"]
__all__ = [
    "dna2rna",
    "rna2vec",
]

import numpy as np

from pyaptamer.utils._base import generate_triplets


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


def rna2vec(
    sequence_list: list[str], sequence_type: str = "rna", max_sequence_length: int = 275
) -> np.ndarray:
    """
    Convert a list of RNA sequence or RNA secondary structures into a numerical
    representation.

    For RNA sequences, if not already in RNA format, the sequences are converted from
    DNA to RNA. For both RNA and secondary structure sequences, all overlapping
    triplets (3-nucleotide/character combinations) are extracted from each sequence and
    mapped to unique indices. Finally, the sequences are zero padded to length
    `max_sequence_length`. The result is a numpy array where each row corresponds to a
    sequence, and each column corresponds to an integer representing the triplet's
    index in the dictionary.

    If the number of extracted triplets is greater than `max_sequence_length`, the
    sequence is truncated to fit.

    Parameters
    ----------
    sequence_list : list[str]
        A list containing sequences as strings (RNA sequences or secondary structure
        sequences).
    sequence_type : str, optional, default="rna"
        The type of sequence to process. Either "rna" for RNA sequences or "ss" for
        secondary structure sequences.
    max_sequence_length : int, optional, default=275
        The maximum length of the output sequences.

    Returns
    -------
    np.ndarray
        A numpy array containing the numerical representation of the sequences, of
        shape (len(sequence_list), `max_sequence_length`).

    Raises
    ------
    ValueError
        If `max_sequence_length` is less than or equal to 0, or if `sequence_type`
        is not "rna" or "ss".

    Examples
    --------
    >>> from pyaptamer.utils import rna2vec
    >>> rna = rna2vec(["AAAC"], sequence_type="rna", max_sequence_length=4)
    >>> print(rna)
    [[1 2 0 0]]
    >>> # Secondary structure sequences
    >>> ss = rna2vec(["SSHH"], sequence_type="ss", max_sequence_length=4)
    >>> print(ss)
    [[2 9 0 0]]
    """
    if max_sequence_length <= 0:
        raise ValueError("`max_sequence_length` must be greater than 0.")

    if sequence_type not in ["rna", "ss"]:
        raise ValueError("`sequence_type` must be either 'rna' or 'ss'.")

    if sequence_type == "rna":
        # generate all rna triplets, 'N' marks unknown nucleotides
        letters = ["A", "C", "G", "U", "N"]
    else:  # sequence_type == "ss"
        # generate all ss triplets
        letters = ["S", "H", "M", "I", "B", "X", "E"]

    triplets = generate_triplets(letters=letters)

    result = []
    for sequence in sequence_list:
        # convert DNA to RNA only for RNA sequences
        if sequence_type == "rna":
            sequence = dna2rna(sequence)

        # extract all overlapping triplets from the sequence
        # e.g., 'ACGUA' -> ['ACG', 'CGU', 'GUA']
        converted = [
            triplets.get(sequence[i : i + 3], 0) for i in range(len(sequence) - 2)
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
