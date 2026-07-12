__author__ = ["satvshr", "siddharth7113"]
__all__ = ["generate_kmer_vecs"]

from itertools import product

import numpy as np


def generate_kmer_vecs(aptamer_sequence, k=4):
    """
    Generate normalized k-mer frequency vectors for the aptamer sequence.

    For all possible k-mers from length 1 to k, count their occurrences in the sequence
    and normalize to form a frequency vector.

    Parameters
    ----------
    aptamer_sequence : str
        The DNA sequence of the aptamer.
    k : int, optional
        Maximum k-mer length (default is 4).

    Returns
    -------
    np.ndarray
        1D numpy array of normalized frequency vector for all possible k-mers from
        length 1 to k.

    Raises
    ------
    ValueError
        If the sequence contains non-DNA characters (e.g., 'U' from RNA).
    """
    DNA_BASES = list("ACGT")

    invalid_chars = sorted(set(aptamer_sequence) - set(DNA_BASES))
    if invalid_chars:
        raise ValueError(
            f"Sequence contains non-DNA characters: {', '.join(invalid_chars)}. "
            "This function only supports DNA sequences (A, C, G, T)."
        )

    # Generate all possible k-mers from 1 to k
    all_kmers = []
    for i in range(1, k + 1):
        all_kmers.extend(["".join(p) for p in product(DNA_BASES, repeat=i)])

    # Count occurrences of each k-mer in the aptamer_sequence
    kmer_counts = dict.fromkeys(all_kmers, 0)
    for i in range(len(aptamer_sequence)):
        for j in range(1, k + 1):
            if i + j <= len(aptamer_sequence):
                kmer = aptamer_sequence[i : i + j]
                if kmer in kmer_counts:
                    kmer_counts[kmer] += 1

    # Normalize counts to frequencies
    total_kmers = sum(kmer_counts.values())
    kmer_freq = np.array(
        [
            kmer_counts[kmer] / total_kmers if total_kmers > 0 else 0
            for kmer in all_kmers
        ]
    )

    return kmer_freq
