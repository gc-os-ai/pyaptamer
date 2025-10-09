"Generic utilities."

__author__ = ["nennomp"]
__all__ = ["seq2vec"]

import numpy as np

from pyaptamer.utils import generate_all_aptamer_triplets


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
    words_ss = generate_all_aptamer_triplets(
        letters=["", "H", "B", "E", "G", "I", "T", "S", "-"]
    )

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
