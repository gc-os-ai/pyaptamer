"""K-hot encoding of fixed-length biological sequences."""

__author__ = ["aditi-dsi"]
__all__ = ["SequenceKHotEncoder"]

import warnings
from typing import Literal

import pandas as pd
import torch
import torch.nn.functional as F

from pyaptamer import logger
from pyaptamer.trafos.base import BaseTransform


class SequenceKHotEncoder(BaseTransform):
    """Encode fixed-length sequences as K-hot tensors.

    Each sequence of length ``L`` becomes a ``(V, L)`` matrix with one active
    entry per position, so ``L`` in total. Hence K-hot, with ``K = L``.

    Input can be a :class:`~pyaptamer.data.loader.MoleculeLoader` or a
    ``pandas.DataFrame`` with a ``sequence_col`` column.

    Parameters
    ----------
    sequence_col : str
        Name of the column holding the sequences.
    vocab : dict[str, int], optional, default=None
        Maps characters to integer indices. Several characters can share an
        index, e.g. ``U`` and ``T``. If None, uses the nucleotide default.
    inverse_vocab : dict[int, str], optional, default=None
        Maps indices back to characters, for decoding. If None, uses the
        nucleotide default.
    handle_unknown : {"raise", "drop"}, default="raise"
        What to do when a sequence is missing (``NaN``/``None``) or contains a
        character outside ``{A, T, G, C, U}``.

        - "raise" : raise a ``ValueError`` naming the problem.
        - "drop" : skip that row in the encoded output.

    Examples
    --------
    >>> import torch
    >>> from pyaptamer.trafos.encode import SequenceKHotEncoder
    >>> from pyaptamer.data import MoleculeLoader
    >>> X = MoleculeLoader(
    ...     data={
    ...         "seq": ["ATGC", "GCTA"],
    ...     }
    ... )
    >>> enc = SequenceKHotEncoder(sequence_col="seq")
    >>> Xt = enc.fit_transform(X)
    >>> Xt.shape
    torch.Size([2, 4, 4])
    >>> decoded = enc.inverse_transform(Xt)
    >>> decoded["seq"].iloc[0]
    'ATGC'
    """

    _tags = {
        "authors": ["aditi-dsi"],
        "maintainers": ["aditi-dsi"],
        "property:fit_is_empty": True,
        "capability:multivariate": False,
        "output_type": "tensor",
    }

    _VOCAB = {"A": 0, "T": 1, "G": 2, "C": 3, "U": 1}
    _INVERSE_VOCAB = {0: "A", 1: "T", 2: "G", 3: "C"}

    def __init__(
        self,
        sequence_col: str,
        vocab: dict[str, int] | None = None,
        inverse_vocab: dict[int, str] | None = None,
        handle_unknown: Literal["raise", "drop"] = "raise",
    ):
        self.sequence_col = sequence_col
        self.vocab = vocab
        self.inverse_vocab = inverse_vocab
        self.handle_unknown = handle_unknown
        super().__init__()

    @property
    def _active_vocab(self):
        """Vocabulary in use, or the default if none was given."""
        if self.vocab is not None:
            return self.vocab
        return self._VOCAB

    @property
    def _active_inverse_vocab(self):
        """Inverse vocabulary in use, or the default if none was given."""
        if self.inverse_vocab is not None:
            return self.inverse_vocab
        return self._INVERSE_VOCAB

    def _validate_params(self):
        """Check the constructor arguments, raising ValueError if unusable."""
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
            ``X`` as a DataFrame, with the ``sequence_col`` column present.

        Raises
        ------
        TypeError
            If ``X`` is not a MoleculeLoader or DataFrame, or if
            ``sequence_col`` does not hold one str sequence per row.
        KeyError
            If ``sequence_col`` is not a column of ``X``.
        """
        X = super()._check_X(X)
        if self.sequence_col not in X.columns:
            raise KeyError(
                f"{type(self).__name__} expects a column named "
                f"{self.sequence_col!r}, but X has columns {list(X.columns)}. "
                "Pass sequence_col= to match the column produced by your "
                "loader or upstream transform."
            )

        kind = pd.api.types.infer_dtype(X[self.sequence_col], skipna=True)
        if kind not in ("string", "empty"):
            raise TypeError(
                f"{type(self).__name__} expects one str sequence per row, but "
                f"{self.sequence_col!r} is {kind!r}. If it holds several "
                "sequences per cell, the MoleculeLoader was built with the "
                'default tiling="bag" - use tiling="samples" so each sequence '
                "is one row."
            )
        return X

    def _transform(self, X):
        """Validate and convert sequences to K-hot tensors.

        Parameters
        ----------
        X : pandas.DataFrame
            Contains the ``sequence_col`` column.

        Returns
        -------
        torch.Tensor
            A 3D float32 tensor of shape (n_samples, num_classes, sequence_length).
            Rows skipped by ``handle_unknown="drop"`` are not included.

        Raises
        ------
        ValueError
            If sequences have differing lengths, or if
            ``handle_unknown="raise"`` and a sequence is missing, or
            contains a character outside ``{A, T, G, C, U}``.
        """
        self._validate_params()

        vocab = self._active_vocab

        reads = X[self.sequence_col]

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
                        f"{self.sequence_col!r}. Set handle_unknown='drop' to "
                        "skip these rows instead."
                    )
                continue

            seq = seq.upper()
            unknown = set(seq) - vocab.keys()

            if unknown:
                if self.handle_unknown == "raise":
                    raise ValueError(
                        f"{type(self).__name__} found unsupported "
                        f"character(s) {sorted(unknown)} in "
                        f"{self.sequence_col!r}; expected only "
                        f"{sorted(vocab)}. Set handle_unknown='drop' "
                        "to skip these rows instead."
                    )
                continue

            encoded_seqs.append([vocab[char] for char in seq])

        dropped = len(reads) - len(encoded_seqs)
        if dropped:
            if not encoded_seqs:
                warnings.warn(
                    f"{type(self).__name__} dropped all {len(reads)} sequences: "
                    "none could be encoded.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                logger.info(
                    f"{type(self).__name__} dropped {dropped} of {len(reads)} "
                    "sequences that could not be encoded."
                )

        num_classes = max(vocab.values()) + 1

        if not encoded_seqs:
            seq_len = int(lengths.iloc[0]) if len(lengths) else 0
            return torch.zeros((0, num_classes, seq_len), dtype=torch.float32)

        int_tensor = torch.tensor(encoded_seqs, dtype=torch.long)

        one_hot = F.one_hot(int_tensor, num_classes=num_classes).float()

        return one_hot.permute(0, 2, 1)

    def inverse_transform(self, X_tensor):
        """Convert K-hot or index tensors back to sequences.

        Parameters
        ----------
        X_tensor : torch.Tensor
            A 3D K-hot tensor of shape (batch_size, num_classes, sequence_length)
            or a 2D integer tensor of shape (batch_size, sequence_length).

        Returns
        -------
        pandas.DataFrame
            Decoded sequences in the ``sequence_col`` column, with a fresh
            index. Rows dropped during ``transform`` cannot be mapped back to
            the index of the original input.

        Notes
        -----
        Indices outside ``inverse_vocab`` decode to ``"X"``. Such sequences
        cannot be passed back through ``transform``, since ``"X"`` is not in
        the vocabulary.
        """
        inverse_vocab = self._active_inverse_vocab

        if X_tensor.dim() == 3:
            indices = X_tensor.argmax(dim=1)
        else:
            indices = X_tensor

        decoded_seqs = []
        for row in indices.tolist():
            seq = "".join([inverse_vocab.get(idx, "X") for idx in row])
            decoded_seqs.append(seq)

        return pd.DataFrame({self.sequence_col: decoded_seqs})

    @classmethod
    def get_test_params(cls):
        """Get test parameters for SequenceKHotEncoder.

        Returns
        -------
        params : list of dict
            Test parameters for SequenceKHotEncoder.
        """
        param0 = {"sequence_col": "seq"}
        param1 = {
            "sequence_col": "seq",
            "vocab": {"A": 0, "C": 1, "G": 2, "U": 3},
            "inverse_vocab": {0: "A", 1: "C", 2: "G", 3: "U"},
        }
        return [param0, param1]
