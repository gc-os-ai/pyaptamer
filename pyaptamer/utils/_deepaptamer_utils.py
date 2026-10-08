__author__ = "satvshr"
__all__ = ["ohe", "pad_sequence", "run_deepdna_prediction", "remove_na"]


import numpy as np
import pandas as pd

from pyaptamer.deepdnashape import DeepDNAShape


def ohe(seq):
    """
    One-hot encodes a single DNA sequence.

    Each character is converted into a one-hot vector. Unknown characters are encoded
    as [0, 0, 0, 0]. Column order is [A, T, C, G].

    Parameters
    ----------
    seq : str
        A DNA sequence.

    Returns
    -------
    np.ndarray
        A 2D NumPy array of shape (seq_len, 4), where the sequence is one-hot encoded.
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
    seq_len : int, optional, default=35
        The length to which the sequence will be padded or truncated.

    Returns
    -------
    str
        The padded sequence of exactly `seq_len` characters.

    Raises
    -------
    ValueError
        If the input sequence length exceeds `seq_len`.

    """
    if len(seq) > seq_len:
        raise ValueError(f"Sequence length {len(seq)} exceeds {seq_len}: '{seq}'")

    return seq.ljust(seq_len, "N")


def run_deepdna_prediction(seqs, layer=2):
    """
    Run `DeepDNAShape` prediction for all DNA structural features (MGW, HelT, ProT,
    Roll) on a batch of DNA sequences.

    Each feature model is loaded once and applied to the whole batch.

    The four DNA shape features are:
        - MGW: Minor Groove Width
        - HelT: Helical Twist
        - ProT: Propeller Twist
        - Roll: Roll angle

    Parameters
    ----------
    seqs : str or list of str
        DNA sequence(s) (e.g., "AAGGTTCC") to predict structural features for.
    layer : int, optional, default=2
        Message-passing layer of `DeepDNAShape` to read. Layer 2 corresponds to a
        sliding window of 5 bases.

    Returns
    -------
    list of list of list of float
        One entry per sequence. Each entry is a list of length 4, where each element
        is a list of floats containing predictions for one structural feature. The
        order is [MGW, HelT, ProT, Roll]. MGW and ProT have `len(seq)` values, HelT
        and Roll have `len(seq) - 1`.
    """
    if isinstance(seqs, str):
        seqs = [seqs]

    X = pd.DataFrame({"seq": seqs})
    features = ["MGW", "HelT", "ProT", "Roll"]

    # (n_features, n_seqs, max_len), NaN padded for shorter sequences
    preds = [
        DeepDNAShape(feature=feat, layer=layer).fit_transform(X).to_numpy()
        for feat in features
    ]

    results = [
        [feat_preds[i][~np.isnan(feat_preds[i])].tolist() for feat_preds in preds]
        for i in range(len(seqs))
    ]
    return results


def remove_na(shape_vectors):
    """
    Trim deepDNAShape predictions to match DeepAptamer's convention
    (remove edge positions that correspond to NA in original DNAshape).

    The four DNA shape features used are:
        - MGW: Minor Groove Width
        - HelT: Helical Twist
        - ProT: Propeller Twist
        - Roll: Roll angle

    Parameters
    ----------
    shape_vectors : list of list of float
        A list of 4 lists in order [MGW, HelT, ProT, Roll],

    Returns
    -------
    list of lists of float
        A list of 4 lists after trimming:
        - MGW (drop first 2 and last 2 -> len=31)
        - HelT (drop first and last -> len=32)
        - ProT (drop first 2 and last 2 -> len=31)
        - Roll (drop first and last -> len=32)
    """
    mgw, helt, prot, roll = shape_vectors

    mgw = mgw[2:-2]
    prot = prot[2:-2]
    helt = helt[1:-1]
    roll = roll[1:-1]

    return [mgw, helt, prot, roll]
