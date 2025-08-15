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


# input will be of size (n_shapes(4), shape_vector_size)
def preprocess_x_shape(X):
    """
    X: numpy array of shape (n_shapes, shape_vector_size)
    Returns: flattened array of shape (1, new_length)
    """

    # Step 1: Pad each row with two zeros on each side
    X_padded = np.pad(X, pad_width=((0, 0), (2, 2)), mode="constant", constant_values=0)

    # TODO: Step 2: Get shapes using padded array

    # Step 3: Normalize each feature (column-wise)
    mean = X_padded.mean(axis=1, keepdims=True)  # shape (4, 1)
    std = X_padded.std(axis=1, keepdims=True)
    # avoid division by zero
    std[std == 0] = 1.0
    X_norm = (X_padded - mean) / std

    # Step 4: Flatten to shape (1, total_length)
    X_flat = X_norm.flatten().reshape(1, -1)

    return X_flat


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
