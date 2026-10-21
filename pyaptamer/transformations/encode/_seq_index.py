"""Integer index encoding of fixed-length biological sequences."""

__author__ = ["siddharth7113"]
__all__ = ["SequenceIndexEncoder"]

import warnings
from typing import Literal

import numpy as np
import pandas as pd

from pyaptamer import logger
from pyaptamer.transformations.base import BaseTransform


class SequenceIndexEncoder(BaseTransform):
    """Encode fixed-length sequences as integer indices, one column per position.

    Each sequence of length ``L`` becomes ``L`` integer columns. Column ``l``
    holds the vocab index of the character at position ``l``. This is the
    input format of models with an embedding layer, such as RaptGen, and the
    first step of :class:`SequenceOneHotEncoder`.

    Input can be a :class:`~pyaptamer.data.loader.MoleculeLoader` or a
    ``pandas.DataFrame`` with exactly one column of sequences.

    Parameters
    ----------
    vocab : dict[str, int], optional, default=None
        Maps characters to integer indices. Several characters can share an
        index, e.g. ``U`` and ``T``. If None, uses
        ``{"A": 0, "T": 1, "G": 2, "C": 3, "U": 1}``.
        Keys must be upper case, since sequences are upper-cased before
        lookup. Characters outside the vocab are handled by ``handle_unknown``.
    handle_unknown : {"raise", "drop"}, default="raise"
        What to do when a sequence is ``NaN``/``None`` or contains a
        character outside the vocab.

        - "raise" : raise a ``ValueError`` naming the problem.
        - "drop" : skip that row in the encoded output.

    Examples
    --------
    >>> from pyaptamer.data import MoleculeLoader
    >>> from pyaptamer.transformations.encode import SequenceIndexEncoder
    >>> X = MoleculeLoader(data={"seq": ["ATGC", "GCTA"]})
    >>> Xt = SequenceIndexEncoder().fit_transform(X)
    >>> Xt.shape
    (2, 4)
    >>> Xt.iloc[0].tolist()
    [0, 1, 2, 3]
    """

    _tags = {
        "authors": ["siddharth7113"],
        "maintainers": ["siddharth7113"],
        "property:fit_is_empty": True,
        "capability:multivariate": False,
    }

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        handle_unknown: Literal["raise", "drop"] = "raise",
    ):
        self.vocab = vocab
        self.handle_unknown = handle_unknown
        super().__init__()

    def _resolve_vocab(self):
        """Return the vocab in use."""
        if self.vocab is None:
            return {"A": 0, "T": 1, "G": 2, "C": 3, "U": 1}
        return self.vocab

    def _validate_params(self):
        """Check the constructor arguments, raising ValueError if unusable."""
        valid_handle_unknown = {"raise", "drop"}
        if self.handle_unknown not in valid_handle_unknown:
            raise ValueError(
                f"handle_unknown must be one of {sorted(valid_handle_unknown)}, "
                f"got {self.handle_unknown!r}."
            )

    def _check_sequences(self, X):
        """Check that the first column of X holds one str sequence per row.

        Parameters
        ----------
        X : pd.DataFrame
            Input data. Only the first column is read.

        Raises
        ------
        TypeError
            If the first column of X does not hold str sequences.
        """
        kind = pd.api.types.infer_dtype(X.iloc[:, 0], skipna=True)
        if kind not in ("string", "empty"):
            raise TypeError(
                f"{type(self).__name__} expects one str sequence per row, but "
                f"column {X.columns[0]!r} is {kind!r}. If it holds several "
                "sequences per cell, the MoleculeLoader was built with the "
                'default tiling="bag" - use tiling="samples" so each sequence '
                "is one row."
            )

    def _transform(self, X):
        """Validate and convert sequences to integer index frames.

        Parameters
        ----------
        X : pandas.DataFrame
            One column of sequences, selected by position.

        Returns
        -------
        pandas.DataFrame
            Shape ``(n_samples, sequence_length)``, int64, indexed by the
            input rows that were encoded. Rows skipped by
            ``handle_unknown="drop"`` are not included.

        Raises
        ------
        TypeError
            If the first column of X does not hold str sequences.
        ValueError
            If sequences have varying lengths, or if
            ``handle_unknown="raise"`` and a sequence has None/NaN value, or
            contains a character outside the vocab.
        """
        self._validate_params()
        self._check_sequences(X)

        vocab = self._resolve_vocab()
        name = type(self).__name__

        reads = X.iloc[:, 0]
        lengths = reads.dropna().str.len()

        if lengths.nunique() > 1:
            raise ValueError(
                f"{name} requires all sequences to be the same fixed length, got "
                f"lengths {sorted(lengths.unique())}. Route raw reads through "
                "PrimerTrimmer first to produce fixed-length regions."
            )

        encoded_seqs, kept_pos = [], []
        vocab_chars = vocab.keys()

        for pos, seq in enumerate(reads):
            if pd.isna(seq):
                if self.handle_unknown == "raise":
                    raise ValueError(
                        f"{name} found a None/NaN value in {reads.name!r}. Set "
                        "handle_unknown='drop' to skip these rows instead."
                    )
                continue

            seq = seq.upper()
            unknown = set(seq) - vocab_chars

            if unknown:
                if self.handle_unknown == "raise":
                    raise ValueError(
                        f"{name} found unsupported character(s) {sorted(unknown)} "
                        f"in {reads.name!r}, expected only {sorted(vocab)}. Set "
                        "handle_unknown='drop' to skip these rows instead."
                    )
                continue

            encoded_seqs.append([vocab[char] for char in seq])
            kept_pos.append(pos)

        dropped = len(reads) - len(encoded_seqs)
        if dropped:
            if not encoded_seqs:
                warnings.warn(
                    f"{name} dropped all {len(reads)} sequences: none could be "
                    "encoded.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                logger.info(
                    f"{name} dropped {dropped} of {len(reads)} sequences that "
                    "could not be encoded."
                )

        seq_len = int(lengths.iloc[0]) if not lengths.empty else 0
        encoded = np.array(encoded_seqs, dtype=np.int64).reshape(
            len(encoded_seqs), seq_len
        )
        return pd.DataFrame(encoded, index=reads.index[kept_pos])

    @classmethod
    def get_test_params(cls):
        """Get test parameters for SequenceIndexEncoder.

        Returns
        -------
        params : list of dict
            Test parameters for SequenceIndexEncoder.
        """
        return [{}, {"vocab": {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}}]
