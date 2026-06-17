__author__ = "satvshr"
__all__ = ["generate_kmer_vecs", "pairs_to_features"]

import numpy as np
import pandas as pd

from pyaptamer.trafos.feature import AptaNetKmerTransformer, AptaNetPairTransformer


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
    X = pd.DataFrame({"aptamer": [aptamer_sequence]})
    return AptaNetKmerTransformer(k=k).fit_transform(X).to_numpy()[0]


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
    if isinstance(X, pd.DataFrame):
        X_inner = X
    else:
        X_inner = pd.DataFrame(X, columns=["aptamer", "protein"])

    # Ensure float32 for PyTorch compatibility
    return AptaNetPairTransformer(k=k).fit_transform(X_inner).to_numpy().astype(
        np.float32
    )
