"""RaptGen end-to-end generation pipeline"""

__author__ = ["NoorMajdoub"]
__all__ = ["RaptgenGenerator"]

import torch
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from pyaptamer.raptgen._estimator import RaptGenModel

# NOTE: `one_hot_index` is being renamed in #741
from pyaptamer.raptgen.layers._utils import one_hot_index


class RaptgenGenerator(BaseEstimator):
    """
    RaptGen algorithm for unsupervised aptamer sequence generation.

    Parameters
    ----------
    motif_len : int, optional, default=10
        Length of the motif to be learned by the model.
    """

    def __init__(self, motif_len=10):
        self.motif_len = motif_len

    def _encode_sequences(self, X):
        """Helper function to map from seqeunce to integer indices."""
        lengths = {len(seq) for seq in X}
        if len(lengths) > 1:
            raise ValueError(
                "All sequences passed in a single call must have the same "
                f"length for batching, got lengths {sorted(lengths)}."
            )
        indices = [one_hot_index(seq) for seq in X]
        return torch.tensor(indices, dtype=torch.long)

    def _to_frame(self, X):
        return self._encode_sequences(X)

    def fit(self, X, y=None):
        self.model_ = RaptGenModel(motif_len=self.motif_len)
        X_idx = self._to_frame(X)
        self.model_.fit(X_idx, y)
        return self

    def transform(self, X):
        X_idx = self._to_frame(X)
        return self.model_.transform(X_idx)

    def inverse_transform(self, Z, most_likely=True):
        check_is_fitted(self, "model_")
        return self.model_.inverse_transform(Z, most_likely=most_likely)
