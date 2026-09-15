"""Normalized k-mer frequencies of sequences."""

__author__ = ["satvshr"]
__all__ = ["KMerFrequencies"]

from itertools import product

import numpy as np
import pandas as pd

from pyaptamer.trafos.base import BaseTransform


class KMerFrequencies(BaseTransform):
    """Normalized frequencies of all k-mers of length 1 to ``k``.

    For every k-mer length from 1 to ``k``, every string over ``alphabet`` of
    that length is a feature. Each sequence is encoded as the count of every
    such k-mer divided by the total count, so the vector sums to one. Any
    substring containing a letter outside ``alphabet`` is not counted.

    The features are ordered by length, then in ``itertools.product`` order
    over ``alphabet``. With the default alphabet and ``k=4`` the width is
    ``4 + 16 + 64 + 256 = 340``.

    Input is a one-column DataFrame or a ``MoleculeLoader`` of strings.

    Parameters
    ----------
    k : int, default=4
        Longest k-mer length.
    alphabet : str, default="ACGT"
        Letters the k-mers are built from. Use ``"ACGU"`` for RNA.

    Examples
    --------
    >>> import pandas as pd
    >>> from pyaptamer.trafos.encode import KMerFrequencies
    >>> X = pd.DataFrame({"seq": ["ACGTACGT", "AAAA"]})
    >>> KMerFrequencies(k=2).fit_transform(X).shape
    (2, 20)
    >>> KMerFrequencies(k=1, alphabet="ACGU").fit_transform(X).iloc[1].tolist()
    [1.0, 0.0, 0.0, 0.0]
    """

    _tags = {
        "authors": ["satvshr"],
        "maintainers": ["siddharth7113"],
        "output_type": "numeric",
        "property:fit_is_empty": True,
        "capability:multivariate": False,
    }

    def __init__(self, k=4, alphabet="ACGT"):
        self.k = k
        self.alphabet = alphabet
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
            Shape ``(len(X), sum(len(alphabet) ** i for i in 1..k))``,
            indexed like X.
        """
        kmers = [
            "".join(p)
            for i in range(1, self.k + 1)
            for p in product(self.alphabet, repeat=i)
        ]
        rows = [self._encode(seq, kmers) for seq in X.iloc[:, 0]]
        return pd.DataFrame(np.vstack(rows), index=X.index)

    def _encode(self, sequence, kmers):
        """Return the normalized k-mer frequency vector of one sequence."""
        counts = dict.fromkeys(kmers, 0)
        for i in range(len(sequence)):
            for j in range(1, self.k + 1):
                kmer = sequence[i : i + j]
                if kmer in counts:
                    counts[kmer] += 1

        total = sum(counts.values())
        if total == 0:
            return np.zeros(len(kmers))
        return np.array([counts[kmer] / total for kmer in kmers])

    @classmethod
    def get_test_params(cls):
        """Return parameter sets for the shared transformer tests."""
        return [{}, {"k": 2, "alphabet": "ACGU"}]
