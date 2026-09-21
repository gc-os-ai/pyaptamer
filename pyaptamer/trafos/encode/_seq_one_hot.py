"""One-hot encoding of fixed-length biological sequences."""

__author__ = ["aditi-dsi"]
__all__ = ["SequenceOneHotEncoder"]

import warnings
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from pyaptamer import logger
from pyaptamer.trafos.base import BaseTransform


class SequenceOneHotEncoder(BaseTransform):
    """Encode fixed-length sequences as one-hot frames.

    Each sequence of length ``L`` over a vocab of size ``V`` becomes ``V * L``
    one-hot columns, indexed like the input. Column ``v * L + l`` holds class
    ``v`` at position ``l``. Reshape with ``.to_numpy().reshape(n, V, L)`` to
    recover the ``(V, L)`` matrix per sequence.

    Input can be a :class:`~pyaptamer.data.loader.MoleculeLoader` or a
    ``pandas.DataFrame`` with exactly one column of sequences.

    Parameters
    ----------
    vocab : dict[str, int], optional, default=None
        Maps characters to integer indices. Several characters can share an
        index, e.g. ``U`` and ``T``. If None, uses the nucleotide default.
        The vocab must cover every character in the data and anything outside it
        is handled by ``handle_unknown``.
    inverse_vocab : dict[int, str], optional, default=None
        Maps indices back to characters, for decoding. If None, uses the
        nucleotide default.
    handle_unknown : {"raise", "drop"}, default="raise"
        What to do when a sequence is ``NaN``/``None`` or contains a
        character outside the vocab.

        - "raise" : raise a ``ValueError`` naming the problem.
        - "drop" : skip that row in the encoded output.

    Notes
    -----
    The default vocab maps ``T`` and ``U`` to the same index,
    so DNA and RNA can be encoded without changing the vocab.
    Decoding that index gives ``T``, so RNA in comes back
    out as DNA.

    Examples
    --------
    >>> from pyaptamer.trafos.encode import SequenceOneHotEncoder
    >>> from pyaptamer.data import MoleculeLoader
    >>> X = MoleculeLoader(
    ...     data={
    ...         "seq": ["ATGCAT", "GCTAGC"],
    ...     }
    ... )
    >>> enc = SequenceOneHotEncoder()
    >>> Xt = enc.fit_transform(X)
    >>> Xt.shape
    (2, 24)
    >>> decoded = enc.inverse_transform(Xt)
    >>> decoded["sequence"].iloc[0]
    'ATGCAT'
    """

    _tags = {
        "authors": ["aditi-dsi"],
        "maintainers": ["aditi-dsi"],
        "output_type": "numeric",
        "property:fit_is_empty": True,
        "capability:multivariate": False,
    }

    _VOCAB = {"A": 0, "T": 1, "G": 2, "C": 3, "U": 1}
    _INVERSE_VOCAB = {0: "A", 1: "T", 2: "G", 3: "C"}

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        inverse_vocab: dict[int, str] | None = None,
        handle_unknown: Literal["raise", "drop"] = "raise",
    ):
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
        """Coerce X to a DataFrame and check its column holds sequences.

        Parameters
        ----------
        X : MoleculeLoader or pandas.DataFrame
            Input data to validate.

        Returns
        -------
        pandas.DataFrame
            ``X`` as a DataFrame.

        Raises
        ------
        TypeError
            If ``X`` is not a MoleculeLoader or DataFrame, or if its column
            does not hold one sequence per row.
        """
        X = super()._check_X(X)

        kind = pd.api.types.infer_dtype(X.iloc[:, 0], skipna=True)
        if kind not in ("string", "empty"):
            raise TypeError(
                f"{type(self).__name__} expects one str sequence per row, but "
                f"column {X.columns[0]!r} is {kind!r}. If it holds several "
                "sequences per cell, the MoleculeLoader was built with the "
                'default tiling="bag" - use tiling="samples" so each sequence '
                "is one row."
            )
        return X

    def _transform(self, X):
        """Validate and convert sequences to one-hot frames.

        Parameters
        ----------
        X : pandas.DataFrame
            One column of sequences, selected by position.

        Returns
        -------
        pandas.DataFrame
            Shape ``(n_samples, num_classes * sequence_length)``, float32,
            indexed by the input rows that were encoded. Rows skipped by
            ``handle_unknown="drop"`` are not included.

        Raises
        ------
        ValueError
            If sequences have varying lengths, or if
            ``handle_unknown="raise"`` and a sequence has None/NaN value, or
            contains a character outside the vocab.
        """
        self._validate_params()

        vocab = self._active_vocab

        reads = X.iloc[:, 0]
        lengths = reads.dropna().str.len()

        if lengths.nunique() > 1:
            raise ValueError(
                f"{type(self).__name__} requires all sequences to be the "
                f"same fixed length, got lengths {sorted(lengths.unique())}. "
                "Route raw reads through PrimerTrimmer first to produce "
                "fixed-length regions."
            )

        encoded_seqs, kept_pos = [], []
        vocab_chars = vocab.keys()

        for pos, seq in enumerate(reads):
            if pd.isna(seq):
                if self.handle_unknown == "raise":
                    raise ValueError(
                        f"{type(self).__name__} found a None/NaN value in "
                        f"{reads.name!r}. Set handle_unknown='drop' to "
                        "skip these rows instead."
                    )
                continue

            seq = seq.upper()
            unknown = set(seq) - vocab_chars

            if unknown:
                if self.handle_unknown == "raise":
                    raise ValueError(
                        f"{type(self).__name__} found unsupported "
                        f"character(s) {sorted(unknown)} in "
                        f"{reads.name!r}, expected only "
                        f"{sorted(vocab)}. Set handle_unknown='drop' "
                        "to skip these rows instead."
                    )
                continue

            encoded_seqs.append([vocab[char] for char in seq])
            kept_pos.append(pos)

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
        index = reads.index[kept_pos]

        if not encoded_seqs:
            seq_len = int(lengths.iloc[0]) if not lengths.empty else 0
            return pd.DataFrame(
                np.zeros((0, num_classes * seq_len), dtype=np.float32),
                index=index,
            )

        int_tensor = torch.tensor(encoded_seqs, dtype=torch.long)
        one_hot = F.one_hot(int_tensor, num_classes=num_classes).to(torch.float32)
        flat = one_hot.permute(0, 2, 1).reshape(len(encoded_seqs), -1).numpy()

        return pd.DataFrame(flat, index=index)

    def inverse_transform(self, X):
        """Convert encoded frames, one-hot tensors or index tensors back to sequences.

        Parameters
        ----------
        X : pandas.DataFrame or torch.Tensor
            The frame returned by ``transform``, a 3D one-hot tensor of shape
            (batch_size, num_classes, sequence_length), or a 2D integer tensor
            of shape (batch_size, sequence_length).

        Returns
        -------
        pandas.DataFrame
            Decoded sequences in a column named ``"sequence"``. A frame input
            keeps its index and a tensor input gets a fresh one.

        Raises
        ------
        ValueError
            If a frame's column count is not a multiple of the vocab size, or
            if a tensor is not 2D or 3D.

        Notes
        -----
        Indices outside ``inverse_vocab`` decode to ``"X"``. Such sequences
        cannot be passed back through ``transform``, since ``"X"`` is not in
        the vocabulary.
        """
        inverse_vocab = self._active_inverse_vocab

        index = None
        if isinstance(X, pd.DataFrame):
            num_classes = max(self._active_vocab.values()) + 1
            if X.shape[1] % num_classes:
                raise ValueError(
                    f"{type(self).__name__} expects a frame with a multiple of "
                    f"{num_classes} columns, got {X.shape[1]}."
                )
            index = X.index
            seq_len = X.shape[1] // num_classes
            X = torch.tensor(X.to_numpy()).reshape(len(X), num_classes, seq_len)

        if X.dim() not in (2, 3):
            raise ValueError(
                f"{type(self).__name__}.inverse_transform expects a 2D index "
                f"tensor or a 3D one-hot tensor, got {X.dim()}D."
            )

        if X.dim() == 3:
            indices = X.argmax(dim=1)
        else:
            indices = X

        decoded_seqs = []
        for row in indices.tolist():
            seq = "".join([inverse_vocab.get(idx, "X") for idx in row])
            decoded_seqs.append(seq)

        return pd.DataFrame({"sequence": decoded_seqs}, index=index)

    @classmethod
    def get_test_params(cls):
        """Get test parameters for SequenceOneHotEncoder.

        Returns
        -------
        params : list of dict
            Test parameters for SequenceOneHotEncoder.
        """
        param0 = {}
        param1 = {
            "vocab": {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3},
            "inverse_vocab": {0: "A", 1: "C", 2: "G", 3: "T"},
        }
        return [param0, param1]
