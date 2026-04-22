"""K-mer encoding transformer for aptamer sequences."""

__author__ = ["satvshr"]
__all__ = ["KMerEncoder"]

from itertools import product

import numpy as np
import pandas as pd

from pyaptamer.trafos.base import BaseTransform


class KMerEncoder(BaseTransform):
    """
    Encode DNA sequences as normalized k-mer frequency vectors.

    This transformer converts DNA sequences into numeric feature vectors by counting
    the frequency of all possible k-mers (subsequences of length 1 to k) and
    normalizing the counts.

    Parameters
    ----------
    k : int, optional, default=4
        Maximum k-mer length. The encoder will generate features for all k-mers
        from length 1 to k. For k=4, this produces 4+16+64+256 = 340 features.

    Attributes
    ----------
    all_kmers_ : list of str
        List of all possible k-mers from length 1 to k, generated after initialization.

    Examples
    --------
    >>> from pyaptamer.trafos.encode import KMerEncoder
    >>> import pandas as pd
    >>> encoder = KMerEncoder(k=2)
    >>> X = pd.DataFrame({"seq": ["ACGT", "GGAA"]})
    >>> X_transformed = encoder.fit_transform(X)
    >>> X_transformed.shape
    (2, 20)

    Notes
    -----
    The k-mer encoding approach is used in the AptaNet algorithm [1]_ for
    representing aptamer sequences in machine learning models.

    References
    ----------
    .. [1] Emami, N., Ferdousi, R. AptaNet as a deep learning approach for
        aptamer–protein interaction prediction. *Scientific Reports*, 11, 6074 (2021).
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
        self._generate_kmers()

    def _generate_kmers(self):
        """Generate all possible k-mers from length 1 to k."""
        DNA_BASES = list("ACGT")
        self.all_kmers_ = []
        for i in range(1, self.k + 1):
            self.all_kmers_.extend(["".join(p) for p in product(DNA_BASES, repeat=i)])

    def _generate_kmer_vec(self, sequence: str) -> np.ndarray:
        """
        Generate normalized k-mer frequency vector for a single sequence.

        Parameters
        ----------
        sequence : str
            DNA sequence.

        Returns
        -------
        np.ndarray
            1D array of normalized k-mer frequencies.
        """
        # Count occurrences of each k-mer
        kmer_counts = dict.fromkeys(self.all_kmers_, 0)
        for i in range(len(sequence)):
            for j in range(1, self.k + 1):
                if i + j <= len(sequence):
                    kmer = sequence[i : i + j]
                    if kmer in kmer_counts:
                        kmer_counts[kmer] += 1

        # Normalize counts to frequencies
        total_kmers = sum(kmer_counts.values())
        kmer_freq = np.array(
            [
                kmer_counts[kmer] / total_kmers if total_kmers > 0 else 0
                for kmer in self.all_kmers_
            ]
        )

        return kmer_freq

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform DNA sequences to k-mer frequency vectors.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame with DNA sequences in the first column.

        Returns
        -------
        pd.DataFrame
            DataFrame with k-mer frequency features. Each row corresponds to
            an input sequence, and columns represent k-mer frequencies.
        """
        # Extract sequences from the first column
        sequences = X.iloc[:, 0].tolist()

        # Generate k-mer vectors for each sequence
        feature_vectors = [self._generate_kmer_vec(seq) for seq in sequences]

        # Stack into a 2D array
        feature_array = np.vstack(feature_vectors)

        # Create DataFrame with feature columns
        feature_columns = [f"kmer_{kmer}" for kmer in self.all_kmers_]
        result_df = pd.DataFrame(feature_array, index=X.index, columns=feature_columns)

        return result_df

    @classmethod
    def get_test_params(cls):
        """
        Get test parameters for KMerEncoder.

        Returns
        -------
        list of dict
            List of parameter dictionaries for testing.
        """
        return [{"k": 2}, {"k": 3}, {"k": 4}]
