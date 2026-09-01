"""Feature-encoding transforms for the AptaDiff algorithm."""

__author__ = ["aditi-dsi"]
__all__ = ["AptamerOneHotEncoder"]

from typing import Literal

import pandas as pd
import torch
import torch.nn.functional as F

from pyaptamer.trafos.base import BaseTransform


class AptamerOneHotEncoder(BaseTransform):
    """Transform aptamer sequences into one-hot encoded PyTorch tensors.

    This transformer encodes fixed-length aptamer sequences. It maps nucleotide
    characters to integers and converts them into a 3D one-hot tensor suitable
    for the AptaDiff diffusion process.

    Input can be a :class:`~pyaptamer.data.loader.MoleculeLoader` or a
    ``pandas.DataFrame`` containing ``aptamer_col``.

    Parameters
    ----------
    aptamer_col : str, default="aptamer"
        Name of the column holding aptamer sequences.
    handle_unknown : {"raise", "drop"}, default="raise"
        What to do when a sequence is missing (``NaN``/``None``) or contains a
        character outside ``{A, T, G, C, U}``.

        - "raise" : raise a ``ValueError`` naming the problem.
        - "drop" : skip that row in the encoded output.


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

    def __init__(
        self,
        aptamer_col: str = "aptamer",
        handle_unknown: Literal["raise", "drop"] = "raise",
    ):
        self.aptamer_col = aptamer_col
        self.handle_unknown = handle_unknown
        super().__init__()

    def _validate_params(self):
        """Check the constructor arguments, raising ValueError if not supported."""
        valid_handle_unknown = {"raise", "drop"}
        if self.handle_unknown not in valid_handle_unknown:
            raise ValueError(
                f"handle_unknown must be one of {sorted(valid_handle_unknown)}, "
                f"got {self.handle_unknown!r}."
            )

    def _check_X(self, X):  # noqa: N802
        """Coerce X to a DataFrame and require the configured column.

        Parameters
        ----------
        X : MoleculeLoader or pandas.DataFrame
            Input data to validate.

        Returns
        -------
        pandas.DataFrame
            ``X`` coerced to a DataFrame, guaranteed to contain ``aptamer_col``.

        Raises
        ------
        TypeError
            If ``X`` is not a MoleculeLoader or DataFrame.
        KeyError
            If ``aptamer_col`` is not a column of ``X``.
        """
        X = super()._check_X(X)
        if self.aptamer_col not in X.columns:
            raise KeyError(
                f"{type(self).__name__} expects a column named "
                f"{self.aptamer_col!r}, but X has columns {list(X.columns)}. "
                "Pass aptamer_col= to match the column produced by your "
                "loader or upstream transform."
            )
        return X

    def _transform(self, X):
        """Validate and convert aptamer sequences to one-hot tensors.

        Parameters
        ----------
        X : pandas.DataFrame
            Contains the ``aptamer_col`` column.

        Returns
        -------
        torch.Tensor
            A 3D float32 tensor of shape (n_samples, num_classes, sequence_length).
            ``n_samples`` is the input row count minus any rows dropped under
            ``handle_unknown="drop"``.

        Raises
        ------
        ValueError
            If sequences have differing lengths, or if
            ``handle_unknown="raise"`` and a sequence is missing, or
            contains a character outside ``{A, T, G, C, U}``.
        """
        self._validate_params()

        reads = X[self.aptamer_col]
        lengths = reads.dropna().str.len()
        if lengths.nunique() > 1:
            raise ValueError(
                f"{type(self).__name__} requires all sequences to be the "
                f"same fixed length, got lengths {sorted(lengths.unique())}. "
                "Route raw reads through PrimerTrimmer first to produce "
                "fixed-length regions."
            )

        encoded_seqs = []

        for seq in reads:
            if pd.isna(seq):
                if self.handle_unknown == "raise":
                    raise ValueError(
                        f"{type(self).__name__} found a missing value in "
                        f"{self.aptamer_col!r}. Set handle_unknown='drop' to "
                        "skip these rows instead."
                    )
                continue

            seq = seq.upper()
            unknown = set(seq) - self._VOCAB.keys()

            if unknown:
                if self.handle_unknown == "raise":
                    raise ValueError(
                        f"{type(self).__name__} found unsupported "
                        f"character(s) {sorted(unknown)} in "
                        f"{self.aptamer_col!r}; expected only "
                        f"{sorted(self._VOCAB)}. Set handle_unknown='drop' "
                        "to skip these rows instead."
                    )
                continue

            encoded_seqs.append([self._VOCAB[char] for char in seq])

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
