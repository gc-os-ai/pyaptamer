__author__ = ["nennomp"]
__all__ = [
    "dna2rna",
    "encode_rna",
    "generate_nplets",
    "rna2vec",
]

from collections.abc import Iterable
from itertools import product

import numpy as np

_VALID_NUCLEOTIDES = frozenset("ACGU")


def dna2rna(sequence: str) -> str:
    """
    Convert a DNA sequence to an RNA sequence.

    Nucleotides 'T' in the DNA sequence are replaced with 'U' in the RNA sequence.
    Unknown nucleotides are replaced with 'N'. Other nucleotides ('A', 'C', 'G') remain
    unchanged.

    Parameters
    ----------
    sequence : str
        The DNA sequence to be converted.

    Returns
    -------
    str
        The converted RNA sequence.
    """
    # replace nucleotides 'T' with 'U'
    result = sequence.translate(str.maketrans("T", "U"))
    result = "".join(char if char in _VALID_NUCLEOTIDES else "N" for char in result)
    return result


def generate_nplets(letters: list[str], repeat: int | Iterable[int]) -> dict[str, int]:
    """
    Generate a dictionary containing all possible n-plets of given characters.

    This method generates all possible n-plets, specified by the `repeat` parameter, of
    characters contained in `letters`. Each n-plet is mapped to a unique integer ID.

    Parameters
    ----------
    letters : list[str]
        List of characters to form n-plets from.
    repeat : int or Iterable[int]
        The length(s) of the sequences to generate. If an int is given, only that length
        is generated (e.g., triplets). If a list or range is given, all lengths are
        generated.

    Returns
    -------
    dict[str, int]
        A dictionary mapping each n-plet to a unique integer ID (1-indexed).
    """
    if isinstance(repeat, int):
        repeat = [repeat]

    nplets = {}
    idx = 1
    for r in repeat:
        for combo in product(letters, repeat=r):
            nplets["".join(combo)] = idx
            idx += 1

    return nplets


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

    triplets = generate_nplets(letters=letters, repeat=3)

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
    sequences: list[str] | str,
    words: dict[str, int],
    max_len: int,
    word_max_len: int = 3,
    return_type: str = "tensor",
) -> "np.ndarray | torch.Tensor":
    """Encode RNA sequences into their numerical representations.

    This function tokenizes RNA sequences using a greedy longest-match approach,
    where longer patterns are preferred over shorter ones. Sequences are
    either truncated or zero-padded to `max_len` tokens.

    Parameters
    ----------
    sequences : list[str] or str
        List of RNA sequences to be encoded. A single string is also accepted
        and treated as a list of one sequence.
    words : dict[str, int]
        A dictionary mapping RNA sub-sequences (e.g. 3-mers) to unique integer
        indices.
    max_len : int
        Maximum number of tokens per encoded sequence. Sequences will be
        truncated or zero-padded to this length.
    word_max_len : int, optional, default=3
        Maximum length of patterns to consider during greedy tokenization.
    return_type : str, optional, default="tensor"
        Format of the returned array:

        * ``"tensor"``: returns a PyTorch ``Tensor`` of dtype ``torch.int64``
        * ``"numpy"``: returns a NumPy ``ndarray`` of dtype ``np.int64``

    Returns
    -------
    torch.Tensor or np.ndarray
        Encoded sequences of shape ``(n_sequences, max_len)``.
        Returns a PyTorch ``Tensor`` when ``return_type="tensor"`` (default),
        or a NumPy ``ndarray`` when ``return_type="numpy"``.

    Raises
    ------
    ValueError
        If ``return_type`` is not ``"tensor"`` or ``"numpy"``.

    Examples
    --------
    >>> from pyaptamer.utils import encode_rna
    >>> words = {"A": 1, "C": 2, "D": 3, "AC": 4}
    >>> print(encode_rna("ACD", words, max_len=5))
    tensor([[4, 3, 0, 0, 0]])
    """
    _VALID_RETURN_TYPES = ("tensor", "numpy")
    if return_type not in _VALID_RETURN_TYPES:
        raise ValueError(
            f"return_type must be one of {_VALID_RETURN_TYPES}, got {return_type!r}."
        )

    # handle single sequence input
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

    if return_type == "numpy":
        return result
    else:  # return_type == "tensor"
        import torch

        return torch.tensor(result, dtype=torch.int64)
