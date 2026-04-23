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
    Convert a list of (aptamer_sequence, protein_sequence) pairs into feature vectors.
    Also supports a pandas DataFrame with 'aptamer' and 'protein' columns.

    This function generates feature vectors for each (aptamer, protein) pair using:

    - k-mer representation of the aptamer sequence
    - Pseudo amino acid composition (PSeAAC) representation of the protein sequence

    Parameters
    ----------
    X : iterable of tuple[str, str] or pandas.DataFrame
        An iterable where each element is a tuple
        `(aptamer_sequence, protein_sequence)`, or a DataFrame containing
        'aptamer' and 'protein' columns.

    k : int, optional
        The k-mer size used to generate the k-mer vector from the aptamer sequence.
        Default is 4.

    Returns
    -------
    np.ndarray
    A 2D NumPy array where each row corresponds to the concatenated feature vector
    for a given (aptamer, protein) pair.
    """
    pseaac = AptaNetPSeAAC()
    feats = []

    if X is None:
        raise ValueError("pairs_to_features() requires at least one pair.")

    if isinstance(X, pd.DataFrame):
        required_columns = {"aptamer", "protein"}
        missing_columns = required_columns.difference(X.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(
                "DataFrame input to pairs_to_features() is missing required "
                f"column(s): {missing}."
            )
        if X.empty:
            raise ValueError("pairs_to_features() requires at least one pair.")

        pairs = zip(X["aptamer"], X["protein"], strict=False)
    else:
        try:
            pairs = list(X)
        except TypeError as exc:
            raise ValueError(
                "pairs_to_features() expects an iterable of pairs or a DataFrame."
            ) from exc

        if len(pairs) == 0:
            raise ValueError("pairs_to_features() requires at least one pair.")

    for pair in pairs:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(
                "Each input pair must contain exactly two values: "
                "(aptamer_sequence, protein_sequence)."
            )

        aptamer_seq, protein_seq = pair
        if not isinstance(aptamer_seq, str) or not isinstance(protein_seq, str):
            raise ValueError(
                "Each input pair must contain two strings: "
                "(aptamer_sequence, protein_sequence)."
            )

        kmer = generate_kmer_vecs(aptamer_seq, k=k)
        pseaac_vec = np.asarray(pseaac.transform(protein_seq))
        feats.append(np.concatenate([kmer, pseaac_vec]))

    # Ensure float32 for PyTorch compatibility
    return np.vstack(feats).astype(np.float32)
