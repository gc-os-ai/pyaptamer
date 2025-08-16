__author__ = ["nennomp"]
__all__ = ["encode_protein"]

import numpy as np
import torch
from torch import Tensor


def encode_protein(
    sequences: list[str],
    words: dict[str, int],
    max_len: int,
    word_max_len: int = 3,
) -> Tensor:
    """Encode protein sequences into their numerical representations.

    This function tokenizes protein sequences using a greedy longest-match approach,
    where longer amino acid patterns are preferred over shorter ones. Sequences are
    either trunacted or zero-padded to `max_len` tokens.

    Parameters
    ----------
    sequences : list[str]
        List of protein sequences to be encoded.
    words : dict[str, int]
        A dictionary mappings protein 3-mers to unique indices.
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
    >>> from pyaptamer.utils import encode_protein
    >>> words = {"A": 1, "C": 2, "D": 3, "AC": 4}
    >>> print(encode_protein("ACD", words, max_len=5))
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