"""K-mer frequency encoder."""

from itertools import product

import numpy as np
import pandas as pd

from pyaptamer.trafos.base import BaseTransform


class KMerEncoder(BaseTransform):
    """K-mer frequency encoder.

    For all possible k-mers from length 1 up to k, count their occurrences in
    each sequence and normalize to form a frequency vector.

    Parameters
    ----------
    k : int, optional, default=4
        Maximum k-mer length.
    """

    _tags = {
        "authors": ["satvshr"],
        "maintainers": ["satvshr"],
        "output_type": "numeric",
        "property:fit_is_empty": True,
        "capability:multivariate": False,
    }

    def __init__(self, k: int = 4):
        self.k = k
        super().__init__()

    def _transform(self, X):
        """Transform the data.

        Parameters
        ----------
        X : pd.DataFrame
            Input data to transform.

        Returns
        -------
        X : pd.DataFrame, shape (n_samples, n_features_transformed)
            Transformed data.
        """
        k = self.k
        DNA_BASES = list("ACGT")

        # Generate all possible k-mers from 1 to k
        all_kmers = []
        for i in range(1, k + 1):
            all_kmers.extend(["".join(p) for p in product(DNA_BASES, repeat=i)])

        raw_sequences = X.values[:, 0].tolist()
        sequences = ["".join(seq) for seq in raw_sequences]

        feats = []
        for seq in sequences:
            # Count occurrences of each k-mer in the sequence
            kmer_counts = dict.fromkeys(all_kmers, 0)
            for i in range(len(seq)):
                for j in range(1, k + 1):
                    if i + j <= len(seq):
                        kmer = seq[i : i + j]
                        if kmer in kmer_counts:
                            kmer_counts[kmer] += 1

            # Normalize counts to frequencies
            total_kmers = sum(kmer_counts.values())
            kmer_freq = [
                kmer_counts[kmer] / total_kmers if total_kmers > 0 else 0
                for kmer in all_kmers
            ]
            feats.append(kmer_freq)

        result_np = np.array(feats, dtype=np.float32)
        result_df = pd.DataFrame(result_np, index=X.index)

        return result_df

    def get_test_params(self):
        """Get test parameters for KMerEncoder.

        Returns
        -------
        params : dict
            Test parameters for KMerEncoder.
        """
        return [{"k": 1}, {"k": 2}]
