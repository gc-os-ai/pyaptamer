import numpy as np


def ohe(X):
    """
    One-hot encodes a batch of DNA sequences.

    Each sequence is converted into a matrix of one-hot vectors.
    Unknown characters are encoded as [0, 0, 0, 0].

    Parameters
    ----------
    X : np.ndarray or str
        - If str: a single DNA sequence.
        - If np.ndarray: a 1D array of DNA sequences (strings), shape (n_sequences,).

    Returns
    -------
    np.ndarray
        A 3D NumPy array of shape (n_sequences, seq_len, 4),
        where each sequence is one-hot encoded.
        Column order is [A, T, C, G].
    """
    alphabet = "ATCG"
    mapping = {base: i for i, base in enumerate(alphabet)}

    if isinstance(X, str):
        X = np.array([X])

    n_seqs = len(X)
    seq_len = len(X[0])

    ohe_batch = np.zeros((n_seqs, seq_len, 4), dtype=int)

    for i, seq in enumerate(X):
        for j, base in enumerate(seq):
            idx = mapping.get(base)
            if idx is not None:
                ohe_batch[i, j, idx] = 1

    return ohe_batch


def pad_sequences(X):
    """
    Pads DNA sequences to length 35 using 'N'. Raises an error if any sequence is longer
    than 35.

    Parameters
    ----------
    X : list of str or np.ndarray
        List or array of DNA sequences (strings) of length ≤ 35.

    Returns
    -------
    np.ndarray
        A NumPy array of shape (n_sequences,) with sequences padded to exactly
        35 characters.
    """
    X = np.asarray(X, dtype=str)
    for seq in X:
        if len(seq) > 35:
            raise ValueError(f"Sequence length {len(seq)} exceeds 35: '{seq}'")

    padded = np.array([seq.ljust(35, "N") for seq in X])
    return padded
