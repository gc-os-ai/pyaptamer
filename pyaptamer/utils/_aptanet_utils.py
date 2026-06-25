__author__ = "satvshr"
__all__ = ["generate_kmer_vecs", "pairs_to_features"]

import numpy as np
import pandas as pd

from pyaptamer.pseaac import AptaNetPSeAAC


def generate_kmer_vecs(aptamer_sequence, k=4, alphabet=None):
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
    alphabet : list[str] or str or None, optional
        Alphabet to use for encoding. If None, it is inferred from sequence.

    Returns
    -------
    np.ndarray
        1D numpy array of normalized frequency vector for all possible k-mers from
        length 1 to k.
    """
    import warnings

    warnings.warn(
        "`generate_kmer_vecs` is deprecated and will be removed in a future version. "
        "Use `pyaptamer.trafos.encode.KMerEncoder` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from pyaptamer.trafos.encode import KMerEncoder

    encoder = KMerEncoder(k=k, alphabet=alphabet)
    # KMerEncoder expects a DataFrame
    X = pd.DataFrame([aptamer_sequence])
    return encoder.fit_transform(X).values[0]


def pairs_to_features(X, k=4, alphabet=None):
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

    alphabet : list[str] or str or None, optional
        Alphabet to use for encoding. If None, it is inferred across the sequences in the batch.

    Returns
    -------
    np.ndarray
        A 2D NumPy array where each row corresponds to the concatenated feature vector
        for a given (aptamer, protein) pair.
    """
    pseaac = AptaNetPSeAAC()
    feats = []

    if isinstance(X, pd.DataFrame):
        pairs = list(zip(X["aptamer"], X["protein"], strict=False))
    else:
        pairs = list(X)

    if alphabet is None:
        # Determine unique alphabet across all sequences in the batch to keep
        # shapes consistent
        unique_chars = set()
        for aptamer_seq, _ in pairs:
            unique_chars.update(aptamer_seq)
        alphabet = sorted(unique_chars)
    else:
        alphabet = list(alphabet)

    for aptamer_seq, protein_seq in pairs:
        kmer = generate_kmer_vecs(aptamer_seq, k=k, alphabet=alphabet)
        pseaac_vec = np.asarray(pseaac.transform(protein_seq))
        feats.append(np.concatenate([kmer, pseaac_vec]))

    # Ensure float32 for PyTorch compatibility
    return np.vstack(feats).astype(np.float32)
