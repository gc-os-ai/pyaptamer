__author__ = ["nennomp"]
__all__ = ["augment_reverse"]

import numpy as np


def augment_reverse(*sequence_arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Augment arrays of sequences by adding their reverse complement.

    Parameters
    ----------
    *sequence_arrays : np.ndarray
        Variable number of numpy arrays of sequences (containing strings).

    Returns
    -------
    tuple[np.ndarray, ...]
        A tuple of arrays, each containing sequences with their reverse complements
        added.
    """
    results = []
    for sequences in sequence_arrays:
        # create array of reversed sequences
        reversed_sequences = np.array([seq[::-1] for seq in sequences])
        # concatenate original and reversed sequences
        result = np.concatenate([sequences, reversed_sequences])
        results.append(result)

    return tuple(results)
