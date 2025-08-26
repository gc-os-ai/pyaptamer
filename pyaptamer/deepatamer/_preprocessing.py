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

    Returns
    -------
    np.ndarray
        A NumPy array of shape (`seq_len`, 4) representing the one-hot
        encoded sequence.
    """
    seq_pad = pad_sequence(seq, seq_len)  # pads to `seq_len`
    seq_ohe = ohe(seq_pad)  # one-hot encode (shape `seq_len` × 4)
    return seq_ohe


# input will be of size (n_shapes(4), shape_vector_size)
def preprocess_seq_shape(seq, full_dna_shape=True):
    """
    Preprocesses a single DNA sequence into a normalized shape vector.

    The function runs DeepDNA prediction on the input sequence, normalizes
    the resulting feature matrix column-wise, flattens it into a single row
    vector, and removes any "NA" values.

    Parameters
    ----------
    seq : str
        A DNA sequence to be processed.

    Returns
    -------
    np.ndarray
        A 2D NumPy array of shape (1, new_length), where `new_length`
        depends on the DeepDNA prediction output after flattening and
        removing "NA" values.
    """

    # Step 1: Get raw predictions
    seq_shape = run_deepdna_prediction(seq)
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

    # Step 2: Concatenate all features into one flat vector
    seq_flat = np.concatenate(norm_features).reshape(1, -1)

    return seq_flat


def preprocess_y(y):
    """
    Preprocess labels into one-hot vectors.

    Converts:
        1 -> [1, 0]  (binder)
        0 -> [0, 1]  (non-binder)

    Args:
        y (array-like): list or numpy array of binary labels (0 or 1)

    Returns:
        np.ndarray: one-hot encoded labels with shape (n_samples, 2)
    """
    one_hot = np.zeros((len(y), 2), dtype=int)
    one_hot[y == 1] = [1, 0]
    one_hot[y == 0] = [0, 1]
    return one_hot
