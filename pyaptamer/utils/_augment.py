__author__ = ["nennomp"]
__all__ = ["augment_reverse"]

import numpy as np

# Watson-Crick complement table covering both DNA (T) and RNA (U) nucleotides.
# Characters not in the table (e.g. 'N', non-biological characters) are left unchanged.
_COMPLEMENT = str.maketrans("ACGTUacgtu", "TGCAAtgcaa")


def _reverse_complement(seq: str) -> str:
    """Return the reverse complement of a nucleotide sequence."""
    return seq.translate(_COMPLEMENT)[::-1]


def augment_reverse(*sequence_arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Augment arrays of sequences by adding their reverse complement.

    Computes the Watson-Crick reverse complement of each sequence using
    standard base-pairing rules (A↔T/U, C↔G) and appends the results
    to the original arrays.

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
        # compute the reverse complement of each sequence
        rc_sequences = np.array([_reverse_complement(seq) for seq in sequences])
        # concatenate original and reverse-complemented sequences
        result = np.concatenate([sequences, rc_sequences])
        results.append(result)

    return tuple(results)
