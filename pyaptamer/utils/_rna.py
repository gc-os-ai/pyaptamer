__author__ = ["nennomp"]
__all__ = [
    "dna2rna",
    "encode_rna",
    "generate_all_aptamer_triplets",
    "rna2vec",
]

from itertools import product

import numpy as np
import torch
from torch import Tensor


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


def generate_all_aptamer_triplets(letters: list[str]) -> dict[str, int]:
    """
    Generate a dictionary mapping all possible 3-mer RNA subsequences (triplets) to
    unique indices.

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

    triplets = generate_all_aptamer_triplets(letters=letters)

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
            # truncate if too long
            if max_sequence_length is not None and len(converted) > max_sequence_length:
                converted = converted[:max_sequence_length]

            # pad if too short
            if max_sequence_length is not None:
                pad_length = max_sequence_length - len(converted)
                padded_sequence = np.pad(
                    array=converted,
                    pad_width=(0, pad_length),
                    constant_values=0,
                )
            else:
                padded_sequence = np.array(converted)

            result.append(padded_sequence)

    return np.array(result)


def encode_rna(
    sequences: list[str],
    words: dict[str, int],
    max_len: int,
    word_max_len: int = 3,
) -> Tensor:
    """Encode RNA sequences into their numerical representations.

    This function tokenizes protein sequences using a greedy longest-match approach,
    where longer amino acid patterns are preferred over shorter ones. Sequences are
    either trunacted or zero-padded to `max_len` tokens.

    Parameters
    ----------
    sequences : list[str]
        List of RNA sequences to be encoded.
    words : dict[str, int]
        A dictionary mappings RNA 3-mers to unique indices.
    max_len : int
        Maximum length of each encoded sequence. Sequences will be truncated
        or padded to this length.
    word_max_len : int, optional, default=3
        Maximum length of amino acid patterns to consider during tokenization.

    Returns
    -------
    Tensor
        Encoded sequences with shape (n_sequences, `max_len`).

    Examples
    --------
    >>> from pyaptamer.utils import encode_rna
    >>> words = {"A": 1, "C": 2, "D": 3, "AC": 4}
    >>> print(encode_rna("ACD", words, max_len=5))
    tensor([[4, 3, 0, 0, 0]])
    """
    # handle single protein input
    if isinstance(sequences, str):
        sequences = [sequences]

    encoded_sequences = []
    for seq in sequences:
        tokens = []
        i = 0

        while i < len(seq):
            # try to match the longest possible pattern first
            matched = False
            for pattern_len in range(min(word_max_len, len(seq) - i), 0, -1):
                pattern = seq[i : i + pattern_len]
                if pattern in words:
                    tokens.append(words[pattern])
                    i += pattern_len
                    matched = True
                    break

            # if no pattern matched, use unknown token (0) and advance by 1
            if not matched:
                tokens.append(0)
                i += 1

            # stop if we've reached max_len tokens
            if len(tokens) >= max_len:
                tokens = tokens[:max_len]
                break

        # pad sequence to max_len
        padded_tokens = tokens + [0] * (max_len - len(tokens))
        encoded_sequences.append(padded_tokens)

    # convert to numpy array first
    result = np.array(encoded_sequences, dtype=np.int64)

    return torch.tensor(result, dtype=torch.int64)
