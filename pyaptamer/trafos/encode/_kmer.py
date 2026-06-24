"""K-mer frequency encoder."""

from itertools import product

import numpy as np
import pandas as pd

from pyaptamer.trafos.base import BaseTransform


class KMerEncoder(BaseTransform):
    """K-mer frequency encoder.

    For all possible k-mers from length 1 up to k, count their occurrences in
    each sequence and normalize to form a frequency vector.

    The alphabet used to generate the k-mer vocabulary is either inferred
    automatically from the sequences seen during ``fit`` or supplied
    explicitly via the ``alphabet`` parameter.

    Parameters
    ----------
    k : int, optional, default=4
        Maximum k-mer length.
    alphabet : list[str] or str or None, optional, default=None
        Characters used to build the k-mer vocabulary.

        * ``None`` (default) – infer the alphabet from the unique characters
          found in the input sequences during ``fit``.
        * ``str`` or ``list[str]`` – use the provided characters, e.g.
          ``"ACGU"`` or ``["A", "C", "G", "U"]``.
    """

    _tags = {
        "authors": ["satvshr"],
        "maintainers": ["satvshr"],
        "output_type": "numeric",
        "property:fit_is_empty": False,
        "capability:multivariate": False,
    }

    def __init__(self, k: int = 4, alphabet=None):
        self.k = k
        self.alphabet = alphabet
        super().__init__()

    def _fit(self, X, y=None):
        """Fit the encoder by determining the alphabet.

        Parameters
        ----------
        X : pd.DataFrame
            Input data whose first column contains sequence strings.
        y : ignored

        Returns
        -------
        self
        """
        if self.alphabet is not None:
            self.alphabet_ = list(self.alphabet)
        else:
            # Auto-infer: extract all unique characters from input sequences
            raw_sequences = X.values[:, 0].tolist()
            sequences = ["".join(seq) for seq in raw_sequences]
            unique_chars = set()
            for seq in sequences:
                unique_chars.update(seq)
            self.alphabet_ = sorted(list(unique_chars))
        return self

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
        bases = self.alphabet_

        # Generate all possible k-mers from 1 to k
        all_kmers = []
        for i in range(1, k + 1):
            all_kmers.extend(["".join(p) for p in product(bases, repeat=i)])

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
        return [{"k": 1}, {"k": 2}, {"k": 1, "alphabet": "ACGU"}]
