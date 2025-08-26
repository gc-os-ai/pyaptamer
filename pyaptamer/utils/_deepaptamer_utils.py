__author__ = "satvshr"
__all__ = ["ohe", "pad_sequence", "run_deepdna_prediction", "remove_na"]


import numpy as np
from deepDNAshape import predictor


def ohe(seq):
    """
    One-hot encodes a single DNA sequence.

    Each character is converted into a one-hot vector. Unknown characters are encoded
    as [0, 0, 0, 0].

    Parameters
    ----------
    seq : str
        A DNA sequence.

    Returns
    -------
    np.ndarray
        A 2D NumPy array of shape (seq_len, 4), where the sequence is one-hot encoded.
        Column order is [A, T, C, G].
    """
    alphabet = "ATCG"
    mapping = {base: i for i, base in enumerate(alphabet)}

    seq_len = len(seq)
    ohe_matrix = np.zeros((seq_len, 4), dtype=int)

    for j, base in enumerate(seq):
        idx = mapping.get(base)
        if idx is not None:
            ohe_matrix[j, idx] = 1

    return ohe_matrix


def pad_sequence(seq, seq_len=35):
    """
    Pads a single DNA sequence to length `seq_len` using 'N'. Raises an error if the
    sequence is longer than `seq_len`.

    Parameters
    ----------
    seq : str
        DNA sequence of length ≤ `seq_len`.

    Returns
    -------
    str
        The padded sequence of exactly `seq_len` characters.
    """
    if len(seq) > seq_len:
        raise ValueError(f"Sequence length {len(seq)} exceeds {seq_len}: '{seq}'")

    return seq.ljust(seq_len, "N")


def run_deepdna_prediction(seq, mode="cpu"):
    """
    Run deepDNAshape prediction for all DNA structural features (MGW, HelT, ProT, Roll)
    on a single DNA sequence.

    Parameters
    ----------
    seq : str
        DNA sequence (e.g., "AAGGTTCC") to predict structural features for.
    mode : {"cpu", "gpu"}, optional
        Compute mode for the predictor. Default is "cpu".

    Returns
    -------
    list of list of float
        A list of length 4, where each element is a list of floats containing
        predictions for one structural feature. The order is [MGW, HelT, ProT, Roll].
        Lengths differ depending on the feature.
    """
    # Always use layer 2 (sliding window of 5)
    layer = 2

    model = predictor.predictor(mode=mode)
    features = ["MGW", "HelT", "ProT", "Roll"]

    results = [model.predict(feat, seq, layer).tolist() for feat in features]
    return results


def remove_na(shape_vectors):
    """
    Trim deepDNAShape predictions to match DeepAptamer's convention
    (remove edge positions that correspond to NA in original DNAshape).

    Parameters
    ----------
    shape_vectors : list of list of float
        A list of 4 lists in order [MGW, HelT, ProT, Roll],

    Returns
    -------
    list of lists of float
        A list of 4 lists after trimming:
        - MGW (drop first 2 and last 2 → len=31)
        - HelT (drop first and last → len=32)
        - ProT (drop first 2 and last 2 → len=31)
        - Roll (drop first and last → len=32)
    """
    mgw, helt, prot, roll = shape_vectors

    mgw = mgw[2:-2]
    prot = prot[2:-2]
    helt = helt[1:-1]
    roll = roll[1:-1]

    return [mgw, helt, prot, roll]
