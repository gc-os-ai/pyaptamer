import numpy as np

from pyaptamer.utils._deepaptamer_utils import (
    ohe,
    pad_sequence,
    remove_na,
    run_deepdna_prediction,
)


def preprocess_seq_ohe(seq):
    """
    Preprocesses a single DNA sequence for DeepAptamer.

    The function pads the sequence to length 35 using 'N' and one-hot encodes
    it. The resulting array has shape (35, 4), where each base is encoded as:
    - A → [1, 0, 0, 0]
    - T → [0, 1, 0, 0]
    - C → [0, 0, 1, 0]
    - G → [0, 0, 0, 1]
    - N or unknown → [0, 0, 0, 0]

    Parameters
    ----------
    seq : str
        A DNA sequence of length ≤ 35.

    Returns
    -------
    np.ndarray
        A NumPy array of shape (35, 4) representing the one-hot
        encoded sequence.
    """
    seq_pad = pad_sequence(seq)  # pads to 35
    seq_ohe = ohe(seq_pad)  # one-hot encode (shape 35 × 4)
    return seq_ohe


# input will be of size (n_shapes(4), shape_vector_size)
def preprocess_seq_shape(seq, use_126_shape=True):
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

    # Step 1: Pad each row with two zeros on each side (?)
    # seq_padded = np.pad(
    #     seq, pad_width=((0, 0), (2, 2)), mode="constant", constant_values=0
    # )

    # Step 2: Get shapes
    seq_shape = run_deepdna_prediction(seq)
    if use_126_shape:
        seq_shape = remove_na(seq_shape)

    # Step 3: Normalize each feature (column-wise)
    mean = seq_shape.mean(axis=1, keepdims=True)  # shape (n_shapes, 1)
    std = seq_shape.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    seq_norm = (seq_shape - mean) / std

    # Step 4: Flatten to shape (1, total_length) and remove 'NA' values
    seq_flat = np.array(seq_norm.flatten().reshape(1, -1))

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
