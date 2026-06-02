__author__ = "satvshr"
__all__ = ["generate_kmer_vecs", "pairs_to_features"]

from itertools import product

import numpy as np
import pandas as pd

from pyaptamer.pseaac import AptaNetPSeAAC


import warnings

import numpy as np
import pandas as pd


def generate_kmer_vecs(aptamer_sequence, k=4):
    """
    Generate normalized k-mer frequency vectors for the aptamer sequence.

    .. deprecated:: 0.1.0
        `generate_kmer_vecs` will be removed in a future version.
        Use `pyaptamer.trafos.encode.KMerEncoder` instead.

    Parameters
    ----------
    aptamer_sequence : str
        The DNA sequence of the aptamer.
    k : int, optional
        Maximum k-mer length (default is 4).

    Returns
    -------
    np.ndarray
        1D numpy array of normalized frequency vector.
    """
    warnings.warn(
        "`generate_kmer_vecs` is deprecated and will be removed in a future version. "
        "Use `pyaptamer.trafos.encode.KMerEncoder` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from pyaptamer.trafos.encode import KMerEncoder

    encoder = KMerEncoder(k=k)
    # KMerEncoder expects a DataFrame
    X = pd.DataFrame([aptamer_sequence])
    return encoder.transform(X).values[0]


def pairs_to_features(X, k=4):
    """
    Convert a list of (aptamer_sequence, protein_sequence) pairs into feature vectors.

    .. deprecated:: 0.1.0
        `pairs_to_features` will be removed in a future version.
        Use `pyaptamer.trafos.encode.AptaNetFeatureExtractor` instead.

    Parameters
    ----------
    X : list of tuple of str or pandas.DataFrame
        A list where each element is a tuple `(aptamer_sequence, protein_sequence)`,
        or a DataFrame containing 'aptamer' and 'protein' columns.

    k : int, optional
        The k-mer size. Default is 4.

    Returns
    -------
    np.ndarray
        A 2D NumPy array of feature vectors.
    """
    warnings.warn(
        "`pairs_to_features` is deprecated and will be removed in a future version. "
        "Use `pyaptamer.trafos.encode.AptaNetFeatureExtractor` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from pyaptamer.trafos.encode import AptaNetFeatureExtractor

    extractor = AptaNetFeatureExtractor(k=k)

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=["aptamer", "protein"])

    return extractor.transform(X).values.astype(np.float32)
