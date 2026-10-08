__author__ = "satvshr"
__all__ = ["preprocess_seq_ohe", "preprocess_seq_shape", "preprocess_y"]

import numpy as np

from pyaptamer.utils._deepaptamer_utils import (
    ohe,
    pad_sequence,
    remove_na,
    run_deepdna_prediction,
)


def preprocess_seq_ohe(seq, seq_len=35):
    """
    Preprocesses a single DNA sequence for DeepAptamer.

    The function pads the sequence to length `seq_len` using 'N' and one-hot encodes
    it. The resulting array has shape (`seq_len`, 4), where each base is encoded as:
    - A → [1, 0, 0, 0]
    - T → [0, 1, 0, 0]
    - C → [0, 0, 1, 0]
    - G → [0, 0, 0, 1]
    - N or unknown → [0, 0, 0, 0]

    Parameters
    ----------
    seq : str
        A DNA sequence of length ≤ `seq_len`.
    seq_len : int, optional, default=35
        The length to which the sequence will be padded or truncated.

    Returns
    -------
    np.ndarray
        A NumPy array of shape (`seq_len`, 4) representing the one-hot
        encoded sequence.
    """
    seq_pad = pad_sequence(seq, seq_len)  # pads to `seq_len`
    seq_ohe = ohe(seq_pad)  # one-hot encode (shape `seq_len` × 4)
    return seq_ohe


def preprocess_seq_shape(seqs, full_dna_shape=True):
    """
    Preprocesses a batch of DNA sequences into normalized shape vectors.

    The function runs DeepDNA prediction on all input sequences at once,
    normalizes each sequence's features, flattens them into a single row
    vector per sequence, and removes any "NA" values.

    Parameters
    ----------
    seqs : str or list of str
        DNA sequence(s) to be processed. All sequences must have the same length.
    full_dna_shape : bool, optional, default=True
        If True, uses the 138-length long `DeepDNAShape` vector.
        If False, uses the 126-length long `DNAshapeR` like vector.

    Returns
    -------
    np.ndarray
        A 3D NumPy array of shape (n_seqs, 1, new_length), where `new_length`
        depends on the DeepDNA prediction output after flattening and
        removing "NA" values.
    """
    seq_shapes = run_deepdna_prediction(seqs)
    return np.stack(
        [_normalize_shape(seq_shape, full_dna_shape) for seq_shape in seq_shapes]
    )


def _normalize_shape(seq_shape, full_dna_shape=True):
    """
    Normalize and flatten the shape predictions of a single sequence.

    Parameters
    ----------
    seq_shape : list of list of float
        A list of 4 lists in order [MGW, HelT, ProT, Roll].
    full_dna_shape : bool, optional, default=True
        Passed on from `preprocess_seq_shape`.

    Returns
    -------
    np.ndarray
        A 2D NumPy array of shape (1, new_length).
    """
    if full_dna_shape:
        seq_shape = remove_na(seq_shape)

    norm_features = []
    for feat in seq_shape:  # each feat is a list of floats
        arr = np.array(feat, dtype=np.float32)

        # Normalize per feature
        mean = arr.mean()
        std = arr.std() if arr.std() > 0 else 1.0
        arr_norm = (arr - mean) / std

        norm_features.append(arr_norm)

    # Concatenate all features into one flat vector
    seq_flat = np.concatenate(norm_features).reshape(1, -1)

    return seq_flat


def preprocess_y(y):
    """
    Preprocess labels into one-hot vectors.

    Converts:
        1 -> [1, 0]  (binder)
        0 -> [0, 1]  (non-binder)

    Parameters
    ----------
    y : np.ndarray
        A 1D NumPy array of binary labels (0 or 1).

    Returns
    -------
    np.ndarray
        A 2D NumPy array of shape (len(y), 2) with one-hot encoded labels.
    """
    one_hot = np.zeros((len(y), 2), dtype=int)
    one_hot[y == 1] = [1, 0]
    one_hot[y == 0] = [0, 1]
    return one_hot
