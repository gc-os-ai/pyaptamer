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
    """Build a longest-first regular expression for greedy tokenization.

    Only non-empty words with positive ids and length up to ``word_max_len`` are
    included. Tokens are ordered from longest to shortest so regex matching preserves
    greedy preference.
    """
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
    """Pad variable-length token chunks to a fixed 2D array.

    Parameters
    ----------
    outputs : list[np.ndarray]
        List of one-dimensional token-id arrays.
    seq_max_len : int
        Target width for each output row.

    Returns
    -------
    np.ndarray
        Padded array with shape ``(len(outputs), seq_max_len)``. If ``outputs`` is
        empty, an empty array with shape ``(0, seq_max_len)`` is returned.
    """
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
    """Tokenize a sequence with a greedy longest-first strategy.

    Parameters
    ----------
    sequence : str
        Input sequence to tokenize.
    words : dict[str, int]
        Token vocabulary mapping token strings to ids.
    word_max_len : int
        Maximum candidate token length checked at each position.
    unknown_policy : {"append_zero", "skip"}, optional, default="append_zero"
        Policy for unknown regions. ``append_zero`` emits a zero token and advances by
        one character. ``skip`` skips unknown regions and returns only matched tokens.
    return_spans : bool, optional, default=False
        Whether to also return character spans for each produced token.
    pattern : re.Pattern[str] or None, optional, default=None
        Precompiled regex pattern used when ``unknown_policy`` is ``skip``.
    max_tokens : int or None, optional, default=None
        Optional cap on number of emitted tokens.

    Returns
    -------
    tuple[list[int], list[tuple[int, int]] or None]
        Token ids and optional token spans.
    """
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
    """Split token streams and optional span streams into fixed-size chunks.

    Parameters
    ----------
    tokens : list[int]
        Token-id stream.
    chunk_size : int or None
        Chunk size. If ``None``, the full stream is returned as a single chunk.
    spans : list[tuple[int, int]] or None, optional, default=None
        Optional token spans aligned with ``tokens``.

    Returns
    -------
    tuple[list[list[int]], list[list[tuple[int, int]]] or None]
        Token chunks and optional aligned span chunks.

    Raises
    ------
    ValueError
        If ``chunk_size`` is provided and is not positive.
    """
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
    """Tokenize with greedy matching and chunk results for downstream encoders.

    This helper combines :func:`_greedy_tokenize` and :func:`_chunk_tokens` in one
    call.
    """
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

    Nucleotides ``T`` are replaced by ``U``. Any character outside ``ACGU`` is
    normalized to ``N``.

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
    for char in result:
        if char not in "ACGU":
            result = result.replace(char, "N")  # replace unknown nucleotides with 'N'
    return result


def generate_nplets(letters: list[str], repeat: int | Iterable[int]) -> dict[str, int]:
    """
    Generate a dictionary containing all possible n-plets of given characters.

    Each generated n-plet is assigned a unique 1-indexed integer id in iteration order.

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
        Mapping from each generated n-plet to its integer id.
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
    Convert RNA sequences or RNA secondary-structure strings to fixed-length vectors.

    RNA inputs are normalized with :func:`dna2rna` first. Then overlapping triplets
    are mapped to integer ids and padded or truncated to ``max_sequence_length``.

    Parameters
    ----------
    sequence_list : list[str]
        Input sequences as strings.
    sequence_type : str, optional, default="rna"
        Sequence domain. Use ``"rna"`` for nucleotide sequences and ``"ss"`` for
        secondary-structure strings.
    max_sequence_length : int, optional, default=275
        Maximum output length per encoded sequence.

    Returns
    -------
    np.ndarray
        Encoded array of shape ``(n_sequences, max_sequence_length)``.

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
    """Encode RNA sequences into integer token ids.

    The function performs greedy longest-match tokenization against ``words``, then
    truncates or zero-pads each sequence to ``max_len``.

    Parameters
    ----------
    sequences : list[str]
        RNA sequences to encode.
    words : dict[str, int]
        Mapping from RNA tokens to integer ids.
    max_len : int
        Maximum length of each encoded sequence.
    word_max_len : int, optional, default=3
        Maximum token length considered during greedy matching.
    return_type : str, optional, default="tensor"
        Output container type.

        * "tensor": returns a PyTorch Tensor
        * "numpy": returns a NumPy ndarray

    Returns
    -------
    torch.Tensor or np.ndarray
        Encoded sequences with shape ``(n_sequences, max_len)``.

    Examples
    --------
    >>> from pyaptamer.utils import encode_rna
    >>> words = {"A": 1, "C": 2, "D": 3, "AC": 4}
    >>> print(encode_rna("ACD", words, max_len=5))
    tensor([[4, 3, 0, 0, 0]])
    """
    # handle single sequence input
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
