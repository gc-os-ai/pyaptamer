__author__ = "satvshr"
__all__ = ["generate_kmer_vecs", "pairs_to_features"]

from collections import Counter
from itertools import product

import numpy as np
import pandas as pd

from pyaptamer.pseaac import AptaNetPSeAAC

# Cache k-mer vocabularies by k; rebuilt per k only once
_KMER_VOCAB_CACHE = {}


def _get_kmer_vocab(k):
    """Return ordered list of all k-mers (length 1..k) over ACGT."""
    if k not in _KMER_VOCAB_CACHE:
        _KMER_VOCAB_CACHE[k] = [
            "".join(p)
            for i in range(1, k + 1)
            for p in product("ACGT", repeat=i)
        ]
    return _KMER_VOCAB_CACHE[k]


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
    all_kmers = _get_kmer_vocab(k)
    counts = Counter(
        aptamer_sequence[i : i + j]
        for j in range(1, k + 1)
        for i in range(len(aptamer_sequence) - j + 1)
    )
    total = sum(counts.get(m, 0) for m in all_kmers)
    if total == 0:
        return np.zeros(len(all_kmers), dtype=np.float64)
    return np.array([counts.get(m, 0) / total for m in all_kmers], dtype=np.float64)


def pairs_to_features(X, k=4):
    """
    Convert a list of (aptamer_sequence, protein_sequence) pairs into feature vectors.
    Also supports a pandas DataFrame with 'aptamer' and 'protein' columns.

    This function generates feature vectors for each (aptamer, protein) pair using:

    - k-mer representation of the aptamer sequence
    - Pseudo amino acid composition (PSeAAC) representation of the protein sequence

    Parameters
    ----------
    X : list of tuple of str or pandas.DataFrame
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
