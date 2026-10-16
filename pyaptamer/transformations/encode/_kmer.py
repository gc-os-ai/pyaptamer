"""Normalized k-mer frequencies of sequences."""

__author__ = ["satvshr", "siddharth7113"]
__all__ = ["KMerFrequencies"]

from itertools import product

import numpy as np
import pandas as pd

from pyaptamer.transformations.base import BaseTransform


class KMerFrequencies(BaseTransform):
    """Normalized frequencies of all k-mers of length 1 to ``k``.

    A k-mer is a substring of length 1 to ``k`` made of letters from
    ``alphabet``. There is one feature per possible k-mer. With
    ``alphabet="ACGT"`` and ``k=2`` these are the 4 single letters and the
    16 pairs, 20 features in all. Each feature is how often that k-mer
    occurs in the sequence, divided by a normalizer.

    With ``normalize="total"`` the normalizer is the total number of counted
    k-mers, so each row sums to one and the values are comparable between
    sequences of different length. With ``normalize="length"`` it is the
    length of the sequence, so the features of each k-mer length keep their
    own scale: the single letters sum to one, the pairs to a little less,
    and so on. For ``k=1`` both give the same values. A substring with a
    letter outside ``alphabet`` is not counted.

    The features are ordered by length, then alphabetically within each
    length: ``A, C, G, T, AA, AC, AG, AT, CA, ...`` for the default alphabet.
    With ``k=4`` the width is ``4 + 16 + 64 + 256 = 340``.

    Input is a one-column DataFrame or a ``MoleculeLoader`` of strings.

    Parameters
    ----------
    k : int, default=4
        Longest k-mer length.
    alphabet : str, default="ACGT"
        Letters the k-mers are built from. Use ``"ACGU"`` for RNA.
    normalize : {"total", "length"}, default="total"
        What each k-mer count is divided by. ``"total"`` is the number of
        k-mers counted in the sequence, over all lengths. ``"length"`` is
        the number of letters in the sequence.

    Examples
    --------
    >>> import pandas as pd
    >>> from pyaptamer.transformations.encode import KMerFrequencies
    >>> X = pd.DataFrame({"seq": ["ACGTACGT", "AAAA"]})
    >>> KMerFrequencies(k=2).fit_transform(X).shape
    (2, 20)
    >>> KMerFrequencies(k=1, alphabet="ACGU").fit_transform(X).iloc[1].tolist()
    [1.0, 0.0, 0.0, 0.0]
    >>> Xt = KMerFrequencies(k=2, normalize="length").fit_transform(X)
    >>> Xt.iloc[1, :5].tolist()
    [1.0, 0.0, 0.0, 0.0, 0.75]
    """

    _tags = {
        "authors": ["satvshr", "siddharth7113"],
        "maintainers": ["siddharth7113"],
        "output_type": "numeric",
        "property:fit_is_empty": True,
        "capability:multivariate": False,
    }

    def __init__(self, k=4, alphabet="ACGT", normalize="total"):
        self.k = k
        self.alphabet = alphabet
        self.normalize = normalize
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
        if self.normalize not in ("total", "length"):
            raise ValueError(
                f'normalize must be "total" or "length", got {self.normalize!r}.'
            )
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
            for j in range(1, min(self.k, len(sequence) - i) + 1):
                kmer = sequence[i : i + j]
                if kmer in counts:
                    counts[kmer] += 1

        if self.normalize == "length":
            denominator = len(sequence)
        else:
            denominator = sum(counts.values())
        if denominator == 0:
            return np.zeros(len(kmers))
        return np.array([counts[kmer] / denominator for kmer in kmers])

    @classmethod
    def get_test_params(cls):
        """Return parameter sets for the shared transformer tests."""
        return [{}, {"k": 2, "alphabet": "ACGU"}, {"k": 2, "normalize": "length"}]
