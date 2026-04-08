__author__ = ["nennomp"]
__all__ = ["augment_reverse"]

import numpy as np


def augment_reverse(*sequence_arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Augment arrays of sequences by adding their reversed versions.

    Parameters
    ----------
    *sequence_arrays : np.ndarray
        Variable number of numpy arrays of sequences (containing strings).

    Returns
    -------
    tuple[np.ndarray, ...]
        A tuple of arrays, each containing the original sequences concatenated
        with their reversed versions.

    Examples
    --------
    >>> import numpy as np
    >>> from pyaptamer.utils import augment_reverse
    >>> seqs = np.array(["ACGT", "AACC"])
    >>> res = augment_reverse(seqs)
    >>> print(res[0])
    ['ACGT' 'AACC' 'TGCA' 'CCAA']
    """
    results = []
    for sequences in sequence_arrays:
        # Create array of reversed sequences
        reversed_sequences = np.array([seq[::-1] for seq in sequences])

        # Concatenate original and reversed sequences
        result = np.concatenate([sequences, reversed_sequences])
        results.append(result)

    return tuple(results)