import numpy as np


def preprocess_y(y):
    """
    Preprocess labels into one-hot vectors.

    Converts:
        1 -> [1, 0]  (binder)
        0 -> [0, 1]  (non-binder)

    Args:
        y (array-like): numpy array of binary labels (0 or 1)

    Returns:
        np.ndarray: one-hot encoded labels with shape (n_samples, 2)
    """
    one_hot = np.zeros((len(y), 2), dtype=int)
    one_hot[y == 1] = [1, 0]
    one_hot[y == 0] = [0, 1]
    return one_hot


print(preprocess_y(np.array([1, 0, 1, 0])))  # Example usage
