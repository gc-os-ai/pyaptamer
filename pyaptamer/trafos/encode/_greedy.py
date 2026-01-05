"""Base transformation class."""

import numpy as np
import pandas as pd

from pyaptamer.trafos.base import BaseTransform


class GreedyEncoder(BaseTransform):
    """Greedy string encoder.

    Carries out greedy longest-match tokenization of sequences
    based on a provided dictionary of words (k-mers) to indices.

    The encoder scans each sequence from left to right,
    at each position attempting to match the longest possible word
    from the dictionary. If a match is found, the corresponding index
    is added to the encoded sequence, and the scan advances by the length
    of the matched word. If no match is found, a special unknown token (index 0)
    is added, and the scan advances by one character.

    Encoded sequences are padded with zeros to ensure uniform length,
    and truncated if they exceed the specified maximum length, if
    ``max_len`` is provided.

    Parameters
    ----------
    words : dict[str, int]
        A dictionary mapping k-mers to unique indices.
        Keys should normally not contain the special unknown token (0).
    max_len : int, optional, default=None
        Maximum length of each encoded sequence. Sequences will be truncated
        or padded to this length.
        If None, sequences will not be truncated, and padded to the length of the
        longest sequence in the input data.
    word_max_len : int, optional, default=None
        Maximum length of words to consider during tokenization.
        If None, defaults to the length of the longest word in ``words``.

    Examples
    --------
    >>> from pyaptamer.trafos.encode import GreedyEncoder
    >>> from pyaptamer.datasets import load_1gnh
    >>>
    >>> data = load_1gnh()
    >>> words = {"QT": 1, "QTA": 2, "S": 3, "G": 4}
    >>>
    >>> encoder = GreedyEncoder(words=words, max_len=5, word_max_len=2)
    >>> encoded_data = encoder.fit_transform(data.to_df_seq())
    """

    _tags = {
        "authors": ["nennomp", "fkiraly"],
        "maintainers": ["nennomp", "fkiraly"],
        "output_type": "numeric",
        "property:fit_is_empty": True,
        "capability:multivariate": False,
    }

    def __init__(
        self,
        words: dict[str, int],
        max_len: int,
        word_max_len: int = None,
    ):
        self.words = words
        self.max_len = max_len
        self.word_max_len = word_max_len

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
        words = self.words
        max_len = self.max_len
        word_max_len = self.word_max_len

        if word_max_len is None:
            word_max_len = max(len(word) for word in words.keys())

        raw_sequences = X.values[:, 0].tolist()
        sequences = ["".join(seq) for seq in raw_sequences]

        encoded_seqs = []
        for seq in sequences:
            tokens = []
            i = 0

            while i < len(seq):
                # try to match the longest possible pattern first
                matched = False
                for pattern_len in range(min(word_max_len, len(seq) - i), 0, -1):
                    pattern = seq[i : i + pattern_len]
                    if pattern in words:
                        tokens.append(words[pattern])
                        i += pattern_len
                        matched = True
                        break

                # if no pattern matched, use unknown token (0) and advance by 1
                if not matched:
                    tokens.append(0)
                    i += 1

                # stop if we've reached max_len tokens
                if max_len is not None and len(tokens) >= max_len:
                    tokens = tokens[:max_len]
                    break

            encoded_seqs.append(tokens)

        if max_len is None:
            max_len = max(len(t) for t in encoded_seqs)

        encoded_seqs = [seq + [0] * (max_len - len(seq)) for seq in encoded_seqs]

        # convert to numpy array
        result_np = np.array(encoded_seqs, dtype=np.int64)

        result_df = pd.DataFrame(result_np, index=X.index)

        return result_df

    def get_test_params(self):
        """Get test parameters for GreedyEncoder.

        Returns
        -------
        params : dict
            Test parameters for GreedyEncoder.
        """
        param0 = {
            "words": {"A": 1, "C": 2, "G": 3, "U": 4, "AC": 5, "GU": 6},
        }
        param1 = {
            "words": {"A": 1, "C": 2, "G": 3, "U": 4},
            "max_len": 10,
            "word_max_len": 2,
        }
        return [param0, param1]
