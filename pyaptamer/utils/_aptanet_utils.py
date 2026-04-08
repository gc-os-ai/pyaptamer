__author__ = "satvshr"
__all__ = ["generate_kmer_vecs", "pairs_to_features"]

from itertools import product
from typing import Union

import numpy as np
import pandas as pd

from pyaptamer.pseaac import AptaNetPSeAAC


def generate_kmer_vecs(aptamer_sequence: str, k: int = 4) -> np.ndarray:
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

    Examples
    --------
    >>> from pyaptamer.utils import generate_kmer_vecs
    >>> vec = generate_kmer_vecs("ACGT", k=2)
    >>> print(vec.shape)
    (20,)
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


def pairs_to_features(X: Union[list[tuple[str, str]], pd.DataFrame], k: int = 4) -> np.ndarray:
    """
    Convert a list of (aptamer_sequence, protein_sequence) pairs into feature vectors.
    Also supports a pandas DataFrame with 'aptamer' and 'protein' columns.

    This function generates feature vectors for each (aptamer, protein) pair using:

    - k-mer representation of the aptamer sequence
    - Pseudo amino acid composition (PSeAAC) representation of the protein sequence

    Parameters
    ----------
    X : list[tuple[str, str]] or pandas.DataFrame
        A list where each element is a tuple `(aptamer_sequence, protein_sequence)`,
        or a DataFrame containing 'aptamer' and 'protein' columns.
    k : int, optional
        The k-mer size used to generate the k-mer vector from the aptamer sequence.
        Default is 4.

    Returns
    -------
    np.ndarray
        A 2D NumPy array where each row corresponds to the concatenated feature vector
        for a given (aptamer, protein) pair.

    Examples
    --------
    >>> import pandas as pd
    >>> from pyaptamer.utils import pairs_to_features
    >>> data = [("ACGT", "ACDEFGH")]
    >>> feats = pairs_to_features(data, k=2)
    >>> print(feats.shape)
    (1, 40)
    """
    pseaac = AptaNetPSeAAC()
    feats = []

    if isinstance(X, pd.DataFrame):
        pairs = zip(X["aptamer"], X["protein"], strict=False)
    else:
        pairs = X

    for aptamer_seq, protein_seq in pairs:
        kmer = generate_kmer_vecs(aptamer_seq, k=k)
        pseaac_vec = np.asarray(pseaac.transform(protein_seq))
        feats.append(np.concatenate([kmer, pseaac_vec]))

    # Ensure float32 for PyTorch compatibility
    return np.vstack(feats).astype(np.float32)