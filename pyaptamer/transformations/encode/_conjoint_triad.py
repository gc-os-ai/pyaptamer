"""Conjoint triad features of protein sequences."""

__author__ = ["aditi-dsi", "siddharth7113"]
__all__ = ["ConjointTriad"]

import string

from pyaptamer.transformations.base import BaseTransform
from pyaptamer.transformations.encode._kmer import KMerFrequencies

GROUPS = {
    "A": "AGV",
    "B": "ILFP",
    "C": "YMTS",
    "D": "HNQW",
    "E": "RK",
    "F": "DE",
    "G": "C",
}


def _group_table():
    """Translation table from amino acid letters to group letters, X otherwise."""
    table = dict.fromkeys(string.ascii_uppercase, "X")
    for group, residues in GROUPS.items():
        for residue in residues:
            table[residue] = group
    return str.maketrans(table)


class ConjointTriad(BaseTransform):
    """Conjoint triad k-mer frequencies of protein sequences.

    The 20 amino acids are put into 7 groups by the dipole moment and the
    volume of their side chains, after Shen et al. [1]_. Each sequence is
    rewritten in the 7-letter group alphabet. Then every k-mer of length 1 to
    ``k`` over that alphabet is counted and divided by the sequence length.
    This is the improved conjoint triad feature, iCTF, of Lee and Han [2]_.
    The classic conjoint triad of Shen et al. is the 343 trimer frequencies
    alone.

    ===== ==============
    Group Amino acids
    ===== ==============
    A     A, G, V
    B     I, L, F, P
    C     Y, M, T, S
    D     H, N, Q, W
    E     R, K
    F     D, E
    G     C
    ===== ==============

    Features are ordered by k-mer length, then alphabetically within each
    length: ``A, B, ..., G, AA, AB, ..., GG, AAA, ...``. With ``k=3`` the
    width is 7 + 49 + 343 = 399.

    Lower-case letters are upper-cased. A non amino acid letter, such as
    ``X`` or the ambiguity code ``B``, belongs to no group. K-mers that
    contain it are not counted, but it still counts toward the sequence
    length.

    Input is a one-column DataFrame or ``MoleculeLoader`` of strings.

    Parameters
    ----------
    k : int, default=3
        Longest k-mer length.

    References
    ----------
    .. [1] Shen, J. et al. Predicting protein-protein interactions based only
       on sequences information. *PNAS* 104, 4337-4341 (2007).
       https://doi.org/10.1073/pnas.0607879104
    .. [2] Lee, W. and Han, K. Constructive prediction of potential RNA
       aptamers for a protein target. *IEEE/ACM Transactions on Computational
       Biology and Bioinformatics* 17, 1476-1482 (2020).
       https://doi.org/10.1109/TCBB.2019.2951114

    Examples
    --------
    >>> import pandas as pd
    >>> from pyaptamer.transformations.encode import ConjointTriad
    >>> X = pd.DataFrame({"protein": ["ACDEFGHIKLMNPQRSTVWY"]})
    >>> ConjointTriad().fit_transform(X).shape
    (1, 399)
    >>> ConjointTriad(k=1).fit_transform(X).iloc[0].round(2).tolist()
    [0.15, 0.2, 0.2, 0.2, 0.1, 0.1, 0.05]
    """

    _tags = {
        "authors": ["aditi-dsi", "siddharth7113"],
        "maintainers": ["siddharth7113"],
        "output_type": "numeric",
        "property:fit_is_empty": True,
        "capability:multivariate": False,
    }

    def __init__(self, k=3):
        self.k = k
        super().__init__()

    def _transform(self, X):
        """Encode every sequence in the single column of X.

        Parameters
        ----------
        X : pd.DataFrame
            One column of strings.

        Returns
        -------
        pd.DataFrame
            Shape ``(len(X), sum(7 ** i for i in 1..k))``, indexed like X.
        """
        groups = X.iloc[:, 0].str.upper().str.translate(_group_table())
        encoder = KMerFrequencies(k=self.k, alphabet="ABCDEFG", normalize="length")
        return encoder.fit_transform(groups.to_frame())

    @classmethod
    def get_test_params(cls):
        """Return parameter sets for the shared transformer tests."""
        return [{}, {"k": 1}]
