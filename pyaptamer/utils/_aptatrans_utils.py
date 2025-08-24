"Generic utilities."

__author__ = ["nennomp"]
__all__ = ["encode_protein", "filter_words", "generate_triplets", "seq2vec"]

from itertools import product

import numpy as np
import torch
from torch import Tensor


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


def seq2vec(
    sequence_list: tuple[list[str], list[str]],
    words: dict[str, int],
    seq_max_len: int,
    word_max_len: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert sequences to vector representations using word dictionaries.

    TODO: see if this can be merged in a more generic version of `_rna.rna2vec(...)`.
    TODO: look into ways to speed it up.

    Tokenizes input sequences by matching substrings of varying lengths against
    provided vocabularies, then converts matches to indices and pads to uniform length.
    The tokenization process uses a greedy approach, attempting to match the longest
    possible substring first (from word_max_len down to 1). Unknown words are mapped
    to index 0. Sequences longer than seq_max_len are split into multiple sequences.

    Parameters
    ----------
    sequence_list : tuple[list[str], list[str]]
        Tuple of lists containing paired sequences (primary sequence, secondary
        structure).
    words : dict[str, int]
        Dictionary mapping primary sequence words to indices.
    seq_max_len : int
        Maximum sequence length for padding.
    word_max_len : int, optional, default=3
        Maximum word length to consider during tokenization.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple of numpy arrays, containing the padded primary sequence indices array
        and the padded secondary structure indices array, respectively. Both have shape
        (n_sequences, seq_max_len).

    Examples
    --------
    >>> from pyaptamer.utils._aptatrans_utils import seq2vec
    >>> words = {"AA": 1, "AC": 2, "A": 3}
    >>> sequences = (["AAAC"], ["HHHC"])
    >>> seq2vec(sequences, words, seq_max_len=4)
    (array([[1., 2., 0., 0.]]), array([[91.,  0.,  0.,  0.]]))
    """
    words_ss = generate_triplets(letters=["", "H", "B", "E", "G", "I", "T", "S", "-"])

    outputs = []
    outputs_ss = []

    for seq, ss in zip(*sequence_list, strict=False):
        output = []
        output_ss = []
        i = 0

        while i < len(seq):
            matched = False

            # try to match longest possible substring first
            for j in range(word_max_len, 0, -1):
                if i + j <= len(seq):
                    substring = seq[i : i + j]
                    substring_ss = ss[i : i + j]

                    # check if substring exists in vocabulary (0 is unknown token)
                    word_idx = words.get(substring, 0)
                    if word_idx != 0:
                        matched = True
                        output.append(word_idx)
                        # 0 marks unknown secondary structure tokens
                        output_ss.append(words_ss.get(substring_ss, 0))

                        # if at `seq_max_len`, store and reset
                        if len(output) == seq_max_len:
                            outputs.append(np.array(output))
                            outputs_ss.append(np.array(output_ss))
                            output = []
                            output_ss = []

                        i += j
                        break

            # skip character if no match found
            if not matched:
                i += 1

        # add remaining output if not empty
        if len(output) > 0:
            outputs.append(np.array(output))
            outputs_ss.append(np.array(output_ss))

    # pad all sequences to `seq_max_len`
    if outputs:
        padded_outputs = np.zeros((len(outputs), seq_max_len))
        padded_outputs_ss = np.zeros((len(outputs_ss), seq_max_len))

        for idx, (seq_array, ss_array) in enumerate(
            zip(outputs, outputs_ss, strict=False)
        ):
            seq_len = len(seq_array)
            padded_outputs[idx, :seq_len] = seq_array
            padded_outputs_ss[idx, :seq_len] = ss_array

        return padded_outputs, padded_outputs_ss
    else:
        return np.zeros((0, seq_max_len)), np.zeros((0, seq_max_len))


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
    >>> from pyaptamer.utils._aptatrans_utils import encode_protein
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
