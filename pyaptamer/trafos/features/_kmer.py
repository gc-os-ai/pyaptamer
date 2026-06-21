"""K-mer feature generation."""

from itertools import product

import numpy as np
import pandas as pd

from pyaptamer.trafos.base import BaseTransform


class KMerFeatures(BaseTransform):
    """
    Generate normalized k-mer frequency vectors for aptamer sequences.

    For all possible k-mers from length 1 to k, count their occurrences in the sequence
    and normalize to form a frequency vector.

    Parameters
    ----------
    k : int, optional
        Maximum k-mer length (default is 4).
    """

    _tags = {
        "authors": ["satvshr"],
        "maintainers": ["satvshr"],
        "output_type": "numeric",
        "property:fit_is_empty": True,
        "capability:multivariate": False,
    }

    def __init__(self, k=4):
        self.k = k
        super().__init__()

    def _transform(self, X):
        """Transform the data.

        Parameters
        ----------
        X : pd.DataFrame
            Input data to transform. Expected to have a single column containing strings.

        Returns
        -------
        X : pd.DataFrame, shape (n_samples, n_features_transformed)
            Transformed data.
        """
        k = self.k
        DNA_BASES = list("ACGT")

        all_kmers = []
        for i in range(1, k + 1):
            all_kmers.extend(["".join(p) for p in product(DNA_BASES, repeat=i)])

        raw_sequences = X.values[:, 0].tolist()

        result_np = []
        for aptamer_sequence in raw_sequences:
            if not isinstance(aptamer_sequence, str):
                aptamer_sequence = ""
            kmer_counts = dict.fromkeys(all_kmers, 0)
            seq_len = len(aptamer_sequence)
            for i in range(seq_len):
                for j in range(1, k + 1):
                    if i + j <= seq_len:
                        kmer = aptamer_sequence[i : i + j]
                        if kmer in kmer_counts:
                            kmer_counts[kmer] += 1

            total_kmers = sum(kmer_counts.values())
            kmer_freq = np.array(
                [
                    kmer_counts[kmer] / total_kmers if total_kmers > 0 else 0
                    for kmer in all_kmers
                ]
            )
            result_np.append(kmer_freq)

        result_df = pd.DataFrame(np.vstack(result_np), index=X.index)
        return result_df

    def get_test_params(self):
        """Get test parameters for KMerFeatures."""
        param0 = {"k": 4}
        param1 = {"k": 3}
        return [param0, param1]
