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

    This transformer encodes fixed-length aptamer sequences. It maps nucleotide
    characters to integers and converts them into a 3D one-hot tensor suitable
    for the AptaDiff diffusion process.

    Input must be a :class:`~pyaptamer.data.loader.MoleculeLoader`.

    Parameters
    ----------
    aptamer_col : str, default="aptamer"
        Name of the column holding aptamer sequences.

    Examples
    --------
    >>> import torch
    >>> from pyaptamer.aptadiff import AptamerOneHotEncoder
    >>> from pyaptamer.data import MoleculeLoader
    >>> X = MoleculeLoader(
    ...     data={
    ...         "aptamer": ["ATGC", "GCTA"],
    ...     }
    ... )
    >>> enc = AptamerOneHotEncoder()
    >>> Xt = enc.fit_transform(X)
    >>> Xt.shape
    torch.Size([2, 4, 4])
    >>> decoded = enc.inverse_transform(Xt)
    >>> decoded["aptamer"].iloc[0]
    'ATGC'
    """

    _tags = {
        "property:fit_is_empty": True,
        "output_type": "tensor",
    }

    _VOCAB = {"A": 0, "T": 1, "G": 2, "C": 3, "U": 1}
    _INVERSE_VOCAB = {0: "A", 1: "T", 2: "G", 3: "C"}

    def __init__(self, aptamer_col="aptamer"):
        self.aptamer_col = aptamer_col
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
            A 3D float32 tensor of shape (n_samples, num_classes, sequence_length).
        """

        encoded_seqs = []

        for seq in X[self.aptamer_col]:
            indices = [self._VOCAB.get(char.upper(), 0) for char in seq]
            encoded_seqs.append(indices)

        int_tensor = torch.tensor(encoded_seqs, dtype=torch.long)
        num_classes = len(self._INVERSE_VOCAB)

        one_hot = F.one_hot(int_tensor, num_classes=num_classes).float()

        return one_hot.permute(0, 2, 1)

    def inverse_transform(self, X_tensor):
        """Convert one-hot or index tensors back to aptamer sequences.

        Parameters
        ----------
        X_tensor : torch.Tensor
            A 3D one-hot tensor of shape (batch_size, num_classes, sequence_length)
            or a 2D integer tensor of shape (batch_size, sequence_length).

        Returns
        -------
        pandas.DataFrame
            Contains decoded aptamer strings in the ``aptamer_col``.
        """

        if X_tensor.dim() == 3:
            indices = X_tensor.argmax(dim=1)
        else:
            indices = X_tensor

        decoded_seqs = []
        for row in indices:
            seq = "".join([self._INVERSE_VOCAB.get(int(idx), "X") for idx in row])
            decoded_seqs.append(seq)

        return pd.DataFrame({self.aptamer_col: decoded_seqs})
