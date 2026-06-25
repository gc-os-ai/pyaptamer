__author__ = ["satvshr", "siddharth7113"]
__all__ = ["generate_kmer_vecs"]


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

    import pandas as pd

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
