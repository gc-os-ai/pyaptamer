"Generic utilities."

__author__ = ["nennomp"]
__all__ = ["seq2vec"]

import numpy as np

from pyaptamer.utils._rna import (
    _build_greedy_pattern,
    _greedy_tokenize_to_chunks,
    _pad_token_chunks,
    generate_nplets,
)

WORDS_SS = generate_nplets(
    letters=["H", "B", "E", "G", "I", "T", "S", "-"], repeat=range(1, 4)
)


def seq2vec(
    sequence_list: tuple[list[str], list[str]],
    words: dict[str, int],
    seq_max_len: int,
    word_max_len: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert sequences to vector representations using word dictionaries.

    TODO: look into further ways to speed this up.

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
        and the padded secondary structure indices array, respectively. The former has
        sequence length (n_sequences, seq_max_len), the latter (n_sequences, len
        (`words_ss`) = 584).

    Examples
    --------
    >>> from pyaptamer.utils._aptatrans_utils import seq2vec
    >>> words = {"AA": 1, "AC": 2, "A": 3}
    >>> sequences = (["AAAC"], ["HHHC"])
    >>> seq2vec(sequences, words, seq_max_len=4)
    (array([[1., 2., 0., 0.]]), array([[9., 0., 0., 0.]]))
    """
    pattern = _build_greedy_pattern(words=words, word_max_len=word_max_len)
    if pattern is None:
        return np.zeros((0, seq_max_len)), np.zeros((0, seq_max_len))

    outputs = []
    outputs_ss = []
    for seq, ss in zip(*sequence_list, strict=False):
        token_chunks, span_chunks = _greedy_tokenize_to_chunks(
            sequence=seq,
            words=words,
            word_max_len=word_max_len,
            chunk_size=seq_max_len,
            unknown_policy="skip",
            return_spans=True,
            pattern=pattern,
        )

        if not token_chunks or span_chunks is None:
            continue

        for token_chunk, span_chunk in zip(token_chunks, span_chunks, strict=False):
            outputs.append(np.array(token_chunk))
            outputs_ss.append(
                np.array([WORDS_SS.get(ss[start:end], 0) for start, end in span_chunk])
            )

    return _pad_token_chunks(outputs, seq_max_len), _pad_token_chunks(
        outputs_ss, seq_max_len
    )
