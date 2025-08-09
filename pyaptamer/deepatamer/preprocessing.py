import numpy as np

from pyaptamer.utils._deepaptamer_utils import ohe, pad_sequences


def preprocess_x_ohe(X):
    """
    Preprocesses input DNA sequences for DeepAptamer.

    This function pads each input sequence to length 35 using 'N' and one-hot encodes
    them. The resulting array has shape (n_sequences, 35, 4), where each base is
    encoded as:
    - A → [1, 0, 0, 0]
    - T → [0, 1, 0, 0]
    - C → [0, 0, 1, 0]
    - G → [0, 0, 0, 1]
    - N or unknown → [0, 0, 0, 0]

    Parameters
    ----------
    X : list of str or np.ndarray
        A list or array of DNA sequences (strings), each of length ≤ 35.

    Returns
    -------
    np.ndarray
        A NumPy array of shape (n_sequences, 35, 4) representing one-hot
        encoded sequences.
    """
    X_pad = pad_sequences(X)
    X_ohe = ohe(X_pad)
    return X_ohe


def preprocess_x_pad(X):
    # pad x with 2 zeros on both sides
    # X_pad = 2 * ["0"] + X + 2 * ["0"]
    return X


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
