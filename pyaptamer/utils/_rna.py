__author__ = ["nennomp"]
__all__ = [
    "dna2rna",
    "encode_rna",
    "generate_nplets",
    "rna2vec",
]

import re
from collections.abc import Iterable
from itertools import product
from typing import Literal

import numpy as np


def _build_greedy_pattern(
    words: dict[str, int], word_max_len: int
) -> re.Pattern[str] | None:
    """Build longest-first regex pattern for greedy tokenization."""
    vocab = sorted(
        [
            word
            for word, word_idx in words.items()
            if word and 0 < len(word) <= word_max_len and word_idx != 0
        ],
        key=lambda word: (len(word), word),
        reverse=True,
    )

    if not vocab:
        return None

    return re.compile("|".join(re.escape(word) for word in vocab))


def _pad_token_chunks(outputs: list[np.ndarray], seq_max_len: int) -> np.ndarray:
    """Pad variable-length token chunks to fixed length arrays."""
    if not outputs:
        return np.zeros((0, seq_max_len))

    padded_outputs = np.zeros((len(outputs), seq_max_len))
    for idx, seq_array in enumerate(outputs):
        seq_len = len(seq_array)
        padded_outputs[idx, :seq_len] = seq_array

    return padded_outputs


def _greedy_tokenize(
    sequence: str,
    words: dict[str, int],
    word_max_len: int,
    *,
    unknown_policy: Literal["append_zero", "skip"] = "append_zero",
    return_spans: bool = False,
    pattern: re.Pattern[str] | None = None,
    max_tokens: int | None = None,
) -> tuple[list[int], list[tuple[int, int]] | None]:
    """Tokenize a sequence using a configurable greedy longest-first strategy."""
    if unknown_policy not in {"append_zero", "skip"}:
        raise ValueError("`unknown_policy` must be either 'append_zero' or 'skip'.")

    if max_tokens is not None and max_tokens <= 0:
        return [], [] if return_spans else None

    tokens: list[int] = []
    spans: list[tuple[int, int]] | None = [] if return_spans else None

    if unknown_policy == "skip":
        compiled_pattern = pattern or _build_greedy_pattern(words, word_max_len)
        if compiled_pattern is None:
            return tokens, spans

        for match in compiled_pattern.finditer(sequence):
            tokens.append(words[match.group()])
            if spans is not None:
                spans.append(match.span())

            if max_tokens is not None and len(tokens) >= max_tokens:
                break

        return tokens, spans

    index = 0
    while index < len(sequence):
        matched = False
        for pattern_len in range(min(word_max_len, len(sequence) - index), 0, -1):
            candidate = sequence[index : index + pattern_len]
            if candidate in words:
                tokens.append(words[candidate])
                if spans is not None:
                    spans.append((index, index + pattern_len))
                index += pattern_len
                matched = True
                break

        if not matched:
            tokens.append(0)
            if spans is not None:
                spans.append((index, index + 1))
            index += 1

        if max_tokens is not None and len(tokens) >= max_tokens:
            break

    return tokens, spans


def _chunk_tokens(
    tokens: list[int],
    chunk_size: int | None,
    spans: list[tuple[int, int]] | None = None,
) -> tuple[list[list[int]], list[list[tuple[int, int]]] | None]:
    """Split token (and optional span) streams into fixed-size chunks."""
    if chunk_size is None:
        token_chunks = [tokens] if tokens else []
        if spans is None:
            return token_chunks, None
        span_chunks = [spans] if spans else []
        return token_chunks, span_chunks

    if chunk_size <= 0:
        raise ValueError("`chunk_size` must be greater than 0 when provided.")

    token_chunks = [
        tokens[idx : idx + chunk_size] for idx in range(0, len(tokens), chunk_size)
    ]
    if spans is None:
        return token_chunks, None

    span_chunks = [
        spans[idx : idx + chunk_size] for idx in range(0, len(spans), chunk_size)
    ]
    return token_chunks, span_chunks


def _greedy_tokenize_to_chunks(
    sequence: str,
    words: dict[str, int],
    word_max_len: int,
    *,
    chunk_size: int | None,
    unknown_policy: Literal["append_zero", "skip"] = "append_zero",
    return_spans: bool = False,
    pattern: re.Pattern[str] | None = None,
    max_tokens: int | None = None,
) -> tuple[list[list[int]], list[list[tuple[int, int]]] | None]:
    """Tokenize greedily and optionally chunk the output for downstream encoders."""
    tokens, spans = _greedy_tokenize(
        sequence=sequence,
        words=words,
        word_max_len=word_max_len,
        unknown_policy=unknown_policy,
        return_spans=return_spans,
        pattern=pattern,
        max_tokens=max_tokens,
    )
    return _chunk_tokens(tokens=tokens, chunk_size=chunk_size, spans=spans)


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
    sequences: list[str],
    words: dict[str, int],
    max_len: int,
    word_max_len: int = 3,
    return_type: str = "tensor",
):
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
    return_type : str, optional, default="tensor"
        The type of the returned encoded sequences.

        * "tensor": returns a PyTorch Tensor
        * "numpy": returns a NumPy ndarray

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
        tokens, _ = _greedy_tokenize(
            sequence=seq,
            words=words,
            word_max_len=word_max_len,
            unknown_policy="append_zero",
            return_spans=False,
            max_tokens=max_len,
        )
        if max_len > 0:
            tokens = tokens[:max_len]
        else:
            tokens = []

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
