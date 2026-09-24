"""RaptGen end-to-end generation pipeline"""

__author__ = ["NoorMajdoub"]
__all__ = ["RaptgenGenerator"]

import torch
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from pyaptamer.raptgen._estimator import RaptGenModel

from pyaptamer.raptgen.layers._utils import seq_to_indices, nt_index


class RaptgenGenerator(BaseEstimator):
    """
    RaptGen algorithm for unsupervised aptamer sequence generation.
    This is the end-to-end, user-facing entry point: it accepts raw
    A/T/G/C sequence strings directly, encodes them into integer
    indices, and delegates the actual training/generation work to
    `RaptGenModel`, which operates purely on numeric arrays.
    Parameters
    ----------
    motif_len : int, optional, default=10
        Length of the motif to be learned by the model.
    Attributes
    ----------
    model_ : RaptGenModel
        The underlying fitted engine, doing the actual training and
        generation. Only present after calling `fit`.
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
        indices = [seq_to_indices(seq) for seq in X]
        return torch.tensor(indices, dtype=torch.long)

    def _to_frame(self, X):
        """Convert raw sequence strings into the integer-encoded form
        RaptGenModel expects.
        """
        return self._encode_sequences(X)

    def fit(self, X, y=None):
        """Train the model on of raw aptamer
        sequences.

        Parameters
        ----------
        X : list of str
            Aptamer sequences (A/T/G/C only), all the same length.
        y : ignored
            Present for sklearn API compatibility. RaptGen is
            unsupervised and does not use labels.

        Returns
        -------
        self : RaptgenGenerator
            The fitted generator.
        """
        self.model_ = RaptGenModel(motif_len=self.motif_len)
        X_idx = self._to_frame(X)
        self.model_.fit(X_idx, y)
        return self

    def transform(self, X):
        """Encode raw sequences into latent-space points.

        Parameters
        ----------
        X : list of str
            Aptamer sequences (A/T/G/C only), all the same length.

        Returns
        -------
        ndarray of shape (n_samples, embed_size)
            Latent-space coordinates.
        """
        check_is_fitted(self, "model_")
        X_idx = self._to_frame(X)
        return self.model_.transform(X_idx)

    def inverse_transform(self, Z, most_likely=True):
        """Generate sequences from latent-space points.

        Parameters
        ----------
        Z : array of shape (n_points, embed_size)
            Latent-space points.
        most_likely : bool, optional, default=True
            To specify the sampling strategy.

        Returns
        -------
        list of str
            One generated sequence per point in `Z`.
        """
        check_is_fitted(self, "model_")
        return self.model_.inverse_transform(Z, most_likely=most_likely)
