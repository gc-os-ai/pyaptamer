"""Feature-encoding transforms for the AptaDiff algorithm."""

__author__ = ["aditi-dsi"]
__all__ = ["AptamerOneHotEncoder"]

import pandas as pd
import torch
import torch.nn.functional as F

from pyaptamer.data import MoleculeLoader
from pyaptamer.trafos.base import BaseTransform


class AptamerOneHotEncoder(BaseTransform):
    """Transform aptamer sequences into one-hot encoded PyTorch tensors.

    This transformer standardizes variable-length aptamer sequences by
    truncating long sequences and padding short sequences to a fixed length.
    It maps nucleotide characters to integers and converts them into a 3D
    one-hot tensor suitable for the AptaDiff diffusion process.

    Input must be a :class:`~pyaptamer.data.loader.MoleculeLoader`.

    Parameters
    ----------
    target_length : int, default=40
        The strictly enforced sequence length. Sequences longer than this
        will be truncated, and sequences shorter will be padded.
    pad_token : str, default='P'
        The character used to pad sequences shorter than `target_length`.
        Mapped to the highest integer index in the vocabulary.
    aptamer_col : str, default="aptamer"
        Name of the column holding aptamer sequences.

    Examples
    --------
    >>> import torch
    >>> from pyaptamer.aptadiff import AptamerOneHotEncoder
    >>> from pyaptamer.data import MoleculeLoader
    >>> X = MoleculeLoader(
    ...     data={
    ...         "aptamer": ["ATGC", "GCTATA"],
    ...     }
    ... )
    >>> enc = AptamerOneHotEncoder(target_length=5, pad_token="P")
    >>> Xt = enc.fit_transform(X)
    >>> Xt.shape
    torch.Size([2, 5, 5])
    >>> decoded = enc.inverse_transform(Xt)
    >>> decoded["aptamer"].iloc[0]
    'ATGC'
    """

    _tags = {
        "property:fit_is_empty": True,
        "output_type": "tensor",
    }

    def __init__(self, target_length=40, pad_token="P", aptamer_col="aptamer"):
        self.target_length = target_length
        self.pad_token = pad_token
        self.aptamer_col = aptamer_col

        self.vocab_ = {"A": 0, "T": 1, "G": 2, "C": 3, "U": 1, self.pad_token: 4}
        self.inverse_vocab_ = {v: k for k, v in self.vocab_.items() if k != "U"}

        super().__init__()

    def _check_X(self, X):  # noqa: N802
        """Require a MoleculeLoader, then defer to the base coercion/checks."""
        if not isinstance(X, MoleculeLoader):
            raise TypeError(
                f"{type(self).__name__} accepts only a MoleculeLoader as input, "
                f"got {type(X).__name__}."
            )
        return super()._check_X(X)

    def _transform(self, X):
        """Standardize sequence lengths and convert to one-hot tensors.

        Parameters
        ----------
        X : pandas.DataFrame
            Contains the ``aptamer_col`` column.

        Returns
        -------
        torch.Tensor
            A 3D float32 tensor of shape (n_samples, target_length, num_classes).
        """
        encoded_seqs = []
        pad_idx = self.vocab_[self.pad_token]

        for seq in X[self.aptamer_col]:
            seq = seq[: self.target_length]
            indices = [self.vocab_.get(char, pad_idx) for char in seq]

            if len(indices) < self.target_length:
                indices += [pad_idx] * (self.target_length - len(indices))

            encoded_seqs.append(indices)

        int_tensor = torch.tensor(encoded_seqs, dtype=torch.long)
        num_classes = len(set(self.inverse_vocab_))

        return F.one_hot(int_tensor, num_classes=num_classes).float()

    def inverse_transform(self, X_tensor):
        """Convert one-hot or index tensors back to aptamer sequences.

        Parameters
        ----------
        X_tensor : torch.Tensor
            A 3D one-hot tensor or 2D integer tensor.

        Returns
        -------
        pandas.DataFrame
            Contains decoded aptamer strings in the ``aptamer_col``.
        """
        if X_tensor.dim() == 3:
            indices = X_tensor.argmax(dim=-1)
        else:
            indices = X_tensor

        decoded_seqs = []
        for row in indices:
            seq = "".join([self.inverse_vocab_.get(int(idx), "X") for idx in row])
            seq = seq.replace(self.pad_token, "")
            decoded_seqs.append(seq)

        return pd.DataFrame({self.aptamer_col: decoded_seqs})
