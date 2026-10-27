"""Integer index encoding of the overlapping k-mers of sequences."""

__author__ = ["nennomp", "siddharth7113"]
__all__ = ["KMerIndexEncoder"]

from itertools import product

import numpy as np
import pandas as pd

from pyaptamer.transformations.base import BaseTransform


class KMerIndexEncoder(BaseTransform):
    """Encode sequences as the ids of their overlapping k-mers.

    A sequence of length ``L`` has ``L - k + 1`` overlapping k-mers. With
    ``k=3``, ``ACGUA`` has ``ACG``, ``CGU`` and ``GUA``. Each k-mer made of
    letters from ``alphabet`` has an id from 1 to ``len(alphabet) ** k``,
    in the order ``AAA, AAC, AAG, ...`` for ``alphabet="ACGT"``. Id 0 is
    padding, and also marks a k-mer with a letter outside ``alphabet``.

    Sequences are upper-cased before encoding. Each row is truncated or
    zero-padded to ``max_len`` ids. A sequence shorter than ``k`` has no
    k-mer and encodes to a row of zeros. Every input row gives one output
    row.

    Input is a one-column DataFrame or a ``MoleculeLoader`` of strings.

    Parameters
    ----------
    k : int, default=3
        Length of the k-mers.
    alphabet : str, default="ACGT"
        Upper-case letters the k-mers are built from. Use ``"ACGU"`` for RNA.
    max_len : int or None, default=None
        Number of ids per row. If None, the number of k-mers of the longest
        sequence.

    Examples
    --------
    >>> import pandas as pd
    >>> from pyaptamer.transformations.encode import KMerIndexEncoder
    >>> X = pd.DataFrame({"seq": ["AAAC", "AC"]})
    >>> KMerIndexEncoder(k=3, alphabet="ACGT", max_len=4).fit_transform(X)
       0  1  2  3
    0  1  2  0  0
    1  0  0  0  0
    """

    _tags = {
        "authors": ["nennomp", "siddharth7113"],
        "maintainers": ["siddharth7113"],
        "output_type": "numeric",
        "property:fit_is_empty": True,
        "capability:multivariate": False,
    }

    def __init__(self, k=3, alphabet="ACGT", max_len=None):
        self.k = k
        self.alphabet = alphabet
        self.max_len = max_len
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
            Shape ``(len(X), max_len)``, int64, indexed like X.
        """
        if self.k < 1:
            raise ValueError(f"k must be at least 1, got {self.k}.")
        if self.max_len is not None and self.max_len < 1:
            raise ValueError(f"max_len must be at least 1 or None, got {self.max_len}.")

        ids = {
            "".join(kmer): i
            for i, kmer in enumerate(product(self.alphabet, repeat=self.k), start=1)
        }
        rows = []
        for sequence in X.iloc[:, 0]:
            sequence = sequence.upper()
            n_kmers = len(sequence) - self.k + 1
            rows.append([ids.get(sequence[i : i + self.k], 0) for i in range(n_kmers)])

        width = self.max_len
        if width is None:
            width = max((len(row) for row in rows), default=0)

        encoded = np.zeros((len(rows), width), dtype=np.int64)
        for i, row in enumerate(rows):
            row = row[:width]
            encoded[i, : len(row)] = row
        return pd.DataFrame(encoded, index=X.index)

    @classmethod
    def get_test_params(cls):
        """Return parameter sets for the shared transformer tests."""
        return [{}, {"k": 2, "alphabet": "ACGU", "max_len": 10}]
