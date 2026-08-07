"""K-mer frequency encoder transformer."""

__author__ = ["Jayant-kernel"]
__all__ = ["KMerEncoder"]

from itertools import product

import numpy as np
import pandas as pd

from pyaptamer.trafos.base import BaseTransform


class KMerEncoder(BaseTransform):
    """K-mer frequency encoder for DNA/RNA aptamer sequences.

    Encodes each sequence as a normalized k-mer frequency vector by counting
    occurrences of all possible k-mers of length 1 to ``k`` over the DNA
    alphabet ``{A, C, G, T}`` and normalizing the counts to frequencies.

    Parameters
    ----------
    k : int, optional, default=4
        Maximum k-mer length. All k-mers of length 1 to ``k`` are included,
        yielding a feature vector of length ``sum(4**i for i in range(1, k+1))``.

    Examples
    --------
    >>> import pandas as pd
    >>> from pyaptamer.trafos.encode import KMerEncoder
    >>> enc = KMerEncoder(k=2)
    >>> X = pd.DataFrame({"sequence": ["ACGT", "AACC"]})
    >>> enc.fit_transform(X).shape
    (2, 20)
    """

    _tags = {
        "authors": ["Jayant-kernel"],
        "maintainers": ["Jayant-kernel"],
        "output_type": "numeric",
        "property:fit_is_empty": True,
        "capability:multivariate": False,
    }

    def __init__(self, k=4):
        self.k = k
        super().__init__()

    def _transform(self, X):
        """Transform aptamer sequences to k-mer frequency vectors.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, 1)
            DataFrame whose first column contains aptamer sequence strings.

        Returns
        -------
        Xt : pd.DataFrame, shape (n_samples, n_features)
            Numeric DataFrame where each row is the k-mer frequency vector
            for the corresponding input sequence.
            ``n_features = sum(4**i for i in range(1, k+1))``.
        """
        k = self.k
        dna_bases = list("ACGT")

        all_kmers = []
        for i in range(1, k + 1):
            all_kmers.extend(["".join(p) for p in product(dna_bases, repeat=i)])

        sequences = X.iloc[:, 0].tolist()

        rows = []
        for seq in sequences:
            kmer_counts = dict.fromkeys(all_kmers, 0)
            for i in range(len(seq)):
                for j in range(1, k + 1):
                    if i + j <= len(seq):
                        kmer = seq[i : i + j]
                        if kmer in kmer_counts:
                            kmer_counts[kmer] += 1

            total = sum(kmer_counts.values())
            freq = np.array(
                [kmer_counts[km] / total if total > 0 else 0.0 for km in all_kmers],
                dtype=np.float32,
            )
            rows.append(freq)

        result = np.vstack(rows)
        return pd.DataFrame(result, index=X.index)

    def get_test_params(self):
        """Return test parameter sets for KMerEncoder.

        Returns
        -------
        params : list of dict
            Two parameter sets: one with k=1 and one with k=2.
        """
        return [{"k": 1}, {"k": 2}]
