"""Frequent Consecutive Subsequence (FCS) Word Transformer."""

__author__ = ["nennomp", "fkiraly"]
__all__ = ["FCSWordTransformer"]

from collections import Counter

import numpy as np
import pandas as pd

from pyaptamer.trafos.base import BaseTransform
from pyaptamer.utils._base import filter_words


class FCSWordTransformer(BaseTransform):
    """Frequent Consecutive Subsequence (FCS) Word Transformer.

    Identifies and extracts frequent protein subsequences (words) from a dataset
    based on their occurrence frequency. This methodology follows the FCS mining
    approach used in AptaTrans [1]_.

    The transformer fits a vocabulary by counting k-mers (default up to length 3)
    across all sequences in the input data. It then filters out subsequences with
    below-average frequency to form a final set of "frequent" words, which are
    mapped to unique integer indices.

    During transformation, it performs a greedy longest-match tokenization of
    sequences based on the fitted vocabulary.

    Parameters
    ----------
    k_max : int, optional, default=3
        Maximum length of subsequences (k-mers) to consider during fitting.
    max_len : int, optional, default=None
        Maximum length of each encoded sequence. Sequences will be truncated
        or padded to this length. If None, padded to the length of the longest
        sequence in the input data.

    Attributes
    ----------
    words_ : dict[str, int]
        Mapping from frequent protein words to unique integer indices (starting from 1).
        Index 0 is reserved for unknown tokens.
    counts_ : dict[str, int]
        Raw occurrence counts of all k-mers found during fitting.

    References
    ----------
    .. [1] Shin, Incheol, et al. "AptaTrans: a deep neural network for predicting
    aptamer-protein interaction using pretrained encoders." BMC bioinformatics 24.1
    (2023): 447.

    Examples
    --------
    >>> from pyaptamer.trafos.encode import FCSWordTransformer
    >>> import pandas as pd
    >>>
    >>> X = pd.DataFrame({"seq": ["DHRNENIAIQ", "DHRNEN"]})
    >>> transformer = FCSWordTransformer(k_max=2)
    >>> transformer.fit(X)
    >>> encoded_X = transformer.transform(X)
    """

    _tags = {
        "authors": ["nennomp", "fkiraly"],
        "maintainers": ["nennomp", "fkiraly"],
        "output_type": "numeric",
        "property:fit_is_empty": False,
        "capability:multivariate": False,
    }

    def __init__(
        self,
        k_max: int = 3,
        max_len: int = None,
    ):
        self.k_max = k_max
        self.max_len = max_len

        super().__init__()

    def _fit(self, X, y=None):
        """Fit the transformer by counting k-mers and filtering frequencies.

        Parameters
        ----------
        X : pd.DataFrame
            Input data containing protein sequences in the first column.
        y : None
            Ignored.

        Returns
        -------
        self : object
            Returns self.
        """
        raw_sequences = X.values[:, 0].tolist()
        sequences = ["".join(seq) for seq in raw_sequences]

        counts = Counter()
        for seq in sequences:
            for k in range(1, self.k_max + 1):
                # optimized counting using zip/slicing
                for i in range(len(seq) - k + 1):
                    counts[seq[i : i + k]] += 1

        self.counts_ = dict(counts)
        self.words_ = filter_words(self.counts_)

        return self

    def _transform(self, X):
        """Transform sequences into indices using the fitted vocabulary.

        Parameters
        ----------
        X : pd.DataFrame
            Input data to transform.

        Returns
        -------
        result_df : pd.DataFrame
            Encoded sequences padded with zeros.
        """
        if not hasattr(self, "words_"):
            raise ValueError("Transformer must be fitted before calling transform.")

        words = self.words_
        max_len = self.max_len
        word_max_len = self.k_max

        raw_sequences = X.values[:, 0].tolist()
        sequences = ["".join(seq) for seq in raw_sequences]

        encoded_seqs = []
        for seq in sequences:
            tokens = []
            i = 0

            while i < len(seq):
                matched = False
                # greedy longest-match
                for pattern_len in range(min(word_max_len, len(seq) - i), 0, -1):
                    pattern = seq[i : i + pattern_len]
                    if pattern in words:
                        tokens.append(words[pattern])
                        i += pattern_len
                        matched = True
                        break

                if not matched:
                    tokens.append(0)
                    i += 1

                if max_len is not None and len(tokens) >= max_len:
                    tokens = tokens[:max_len]
                    break

            encoded_seqs.append(tokens)

        if max_len is None:
            max_len = max(len(t) for t in encoded_seqs) if encoded_seqs else 0

        # padding
        encoded_seqs = [seq + [0] * (max_len - len(seq)) for seq in encoded_seqs]

        result_np = np.array(encoded_seqs, dtype=np.int64)
        result_df = pd.DataFrame(result_np, index=X.index)

        return result_df
