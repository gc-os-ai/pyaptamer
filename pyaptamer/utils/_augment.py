__author__ = ["nennomp"]
__all__ = ["augment_reverse"]

import numpy as np

_RNA_COMPLEMENT = str.maketrans("ACGUacgu", "UGCAugca")


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
        rc_sequences = np.array(
            [seq[::-1].translate(_RNA_COMPLEMENT) for seq in sequences]
        )
        result = np.concatenate([sequences, rc_sequences])
        results.append(result)

    return tuple(results)
