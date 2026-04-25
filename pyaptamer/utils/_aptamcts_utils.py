__author__ = "aditi-dsi"
__all__ = [
    "get_reduced_protein_letter_dict",
    "rna_to_ictf",
    "protein_to_ictf",
    "pairs_to_features",
]

from itertools import product

import numpy as np
import pandas as pd


def get_reduced_protein_letter_dict():
    """
    Generate a mapping dictionary for a reduced amino acid alphabet.

    This groups standard amino acids into 7 distinct classes (labeled 'A' through 'G')
    based on their physicochemical properties, which is required for calculating
    the Improved Conjoint Triad Feature (iCTF).

    Returns
    -------
    dict
        A dictionary mapping standard amino acid string characters to their
        corresponding reduced class characters.
    """

    rpdict = {}

    reduced_letters = [
        ["A", "G", "V"],
        ["I", "L", "F", "P"],
        ["Y", "M", "T", "S"],
        ["H", "N", "Q", "W"],
        ["R", "K"],
        ["D", "E"],
        ["C"],
    ]

    changed_letter = ["A", "B", "C", "D", "E", "F", "G"]

    for class_letters, target_letter in zip(
        reduced_letters, changed_letter, strict=False
    ):
        for letter in class_letters:
            rpdict[letter] = target_letter

    return rpdict


def rna_to_ictf(aptamer_sequence, k=4):
    """
    Generate the Improved Conjoint Triad Feature (iCTF) representation
    for an RNA sequence.

    This function normalizes the input sequence (converting DNA to RNA by replacing
    'T' with 'U') and calculates the occurrence frequencies of all possible k-mers
    from length 1 up to `k` using the alphabet ["A", "C", "G", "U"]. The counts are
    then normalized by the total length of the sequence.

    Parameters
    ----------
    aptamer_sequence : str
        The RNA or DNA sequence of the aptamer.
    k : int, optional
        Maximum k-mer length to consider. Default is 4.

    Returns
    -------
    np.ndarray
        1D numpy array of the normalized iCTF frequency vector for all possible
        k-mers from length 1 to k.
    """

    RNA_BASES = list("ACGU")

    r_mers = []
    for i in range(1, k + 1):
        r_mers.extend(["".join(p) for p in product(RNA_BASES, repeat=i)])

    rmer_counts = dict.fromkeys(r_mers, 0)

    aptamer_sequence = aptamer_sequence.upper()
    aptamer_sequence = aptamer_sequence.replace("T", "U")

    for mer in range(1, k + 1):
        for i in range(len(aptamer_sequence)):
            if i + mer <= len(aptamer_sequence):
                pattern = aptamer_sequence[i : i + mer]
                if pattern in rmer_counts:
                    rmer_counts[pattern] += 1

    r_feature = np.array([rmer_counts[rmer] / len(aptamer_sequence) for rmer in r_mers])

    return r_feature


def protein_to_ictf(protein_sequence, k=3):
    """
    Generate the Improved Conjoint Triad Feature (iCTF) representation
    for a protein sequence.

    This function first translates the input amino acid sequence into a reduced 7-letter
    alphabet (["A", "B", "C", "D", "E", "F", "G"]) based on physicochemical properties.
    It then calculates the occurrence frequencies of all possible k-mers from length 1
    up to `k` using this reduced alphabet. The counts are normalized by the total length
    of the original sequence.

    Parameters
    ----------
    protein_sequence : str
        The amino acid sequence of the protein.
    k : int, optional
        Maximum k-mer length to consider. Default is 3.

    Returns
    -------
    np.ndarray
        1D numpy array of the normalized iCTF frequency vector for all possible
        k-mers from length 1 to k.
    """

    PROT_BASES = list("ABCDEFG")
    rpdict = get_reduced_protein_letter_dict()

    p_mers = []
    for i in range(1, k + 1):
        p_mers.extend(["".join(p) for p in product(PROT_BASES, repeat=i)])

    pmer_counts = dict.fromkeys(p_mers, 0)

    protein_sequence = protein_sequence.upper()
    rpseq = []
    for p in protein_sequence:
        rpseq.append(rpdict.get(p, "X"))

    pseq = "".join(rpseq)

    for mer in range(1, k + 1):
        for i in range(len(pseq)):
            if i + mer <= len(pseq):
                pattern = pseq[i : i + mer]
                if pattern in pmer_counts:
                    pmer_counts[pattern] += 1

    p_feature = np.array([pmer_counts[pmer] / len(protein_sequence) for pmer in p_mers])

    return p_feature


def pairs_to_features(X, rna_k=4, prot_k=3):
    """
    Convert a list of (aptamer, protein) sequence pairs into iCTF feature vectors.
    Also supports a pandas DataFrame with 'aptamer' and 'protein' columns.

    This function generates feature vectors for each (aptamer, protein) pair by
    concatenating:
    - The Improved Conjoint Triad Feature (iCTF) representation of the aptamer.
    - The Improved Conjoint Triad Feature (iCTF) representation of the protein.

    Parameters
    ----------
    X : list of tuple of str or pandas.DataFrame
        A list where each element is a tuple `(aptamer_sequence, protein_sequence)`,
        or a DataFrame containing 'aptamer' and 'protein' columns.
    rna_k : int, optional
        The k-mer size used to generate the iCTF vector for the aptamer sequence.
        Default is 4.
    prot_k : int, optional
        The k-mer size used to generate the iCTF vector for the protein sequence.
        Default is 3.

    Returns
    -------
    np.ndarray
        A 2D NumPy array where each row corresponds to the concatenated iCTF feature
        vector for a given (aptamer, protein) pair.
    """

    feats = []

    if isinstance(X, pd.DataFrame):
        pairs = zip(X["aptamer"], X["protein"], strict=False)
    else:
        pairs = X

    for aptamer_seq, protein_seq in pairs:
        rx = rna_to_ictf(aptamer_seq, k=rna_k)
        px = protein_to_ictf(protein_seq, k=prot_k)
        feats.append(np.concatenate([rx, px]))

    # Ensure float32 for PyTorch compatibility
    return np.vstack(feats).astype(np.float32)
