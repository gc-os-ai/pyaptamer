__author__ = "satvshr"
__all__ = ["generate_kmer_vecs", "pairs_to_features"]

from itertools import product

import numpy as np
import pandas as pd

from pyaptamer.pseaac import AptaNetPSeAAC


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
    """
    DNA_BASES = list("ACGT")

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


def pairs_to_features(X, k=4):
    """
    Convert aptamer-protein pairs into concatenated numeric feature vectors.

    Accepts either an iterable of ``(aptamer_sequence, protein_sequence)`` pairs
    or a pandas DataFrame with ``aptamer`` and ``protein`` columns.

    This function generates feature vectors for each (aptamer, protein) pair using:

    - k-mer representation of the aptamer sequence
    - Pseudo amino acid composition (PSeAAC) representation of the protein sequence

    Parameters
    ----------
    X : iterable of tuple[str, str] or pandas.DataFrame
        Sequence pairs as ``(aptamer, protein)`` tuples, or a DataFrame containing
        ``aptamer`` and ``protein`` columns.

    k : int, optional
        The k-mer size used to generate the k-mer vector from the aptamer sequence.
        Default is 4.

    Returns
    -------
    np.ndarray
        A 2D NumPy array where each row corresponds to the concatenated feature vector
        for a given (aptamer, protein) pair.

    Raises
    ------
    ValueError
        If input is empty, DataFrame columns are missing, pair structure is invalid,
        sequence entries are not strings, or sequences are empty.

    Notes
    -----
    Aptamer and protein sequences are normalized to uppercase before feature
    extraction.
    """
    pseaac = AptaNetPSeAAC()
    feats = []

    if isinstance(X, pd.DataFrame):
        if "aptamer" not in X.columns or "protein" not in X.columns:
            raise ValueError("DataFrame must contain 'aptamer' and 'protein'")
        pairs = list(zip(X["aptamer"], X["protein"], strict=False))
    else:
        pairs = list(X)

    if not pairs:
        raise ValueError("Empty input")

    for pair in pairs:
        if not isinstance(pair, (list | tuple)) or len(pair) != 2:
            raise ValueError("Each element must be (aptamer, protein)")

        aptamer_seq, protein_seq = pair

        if not isinstance(aptamer_seq, str) or not isinstance(protein_seq, str):
            raise ValueError("Sequences must be strings")

        if not aptamer_seq or not protein_seq:
            raise ValueError("Sequences cannot be empty")

        aptamer_seq = aptamer_seq.upper()
        protein_seq = protein_seq.upper()

        kmer = generate_kmer_vecs(aptamer_seq, k=k)
        pseaac_vec = np.asarray(pseaac.transform(protein_seq))

        feats.append(np.concatenate([kmer, pseaac_vec]))

    # Ensure float32 for PyTorch compatibility
    return np.vstack(feats).astype(np.float32)
