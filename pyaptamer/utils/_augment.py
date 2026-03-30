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

    Raises
    ------
    TypeError
        If any argument is not a numpy ndarray.
    ValueError
        If any array is empty.
    """
    if len(sequence_arrays) == 0:
        raise ValueError("At least one sequence array must be provided")

    results = []
    for idx, sequences in enumerate(sequence_arrays):
        # Validate input type
        if not isinstance(sequences, np.ndarray):
            raise TypeError(
                f"All arguments must be numpy arrays, got {type(sequences).__name__} "
                f"for argument {idx + 1}"
            )

        # Validate non-empty array
        if sequences.size == 0:
            raise ValueError(
                f"Sequence array {idx + 1} is empty, at least one sequence required"
            )

        # create array of reversed sequences
        reversed_sequences = np.array([seq[::-1] for seq in sequences])
        # concatenate original and reversed sequences
        result = np.concatenate([sequences, reversed_sequences])
        results.append(result)

    return tuple(results)
