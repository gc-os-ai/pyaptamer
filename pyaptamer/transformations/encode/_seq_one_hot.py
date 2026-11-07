"""One-hot encoding of fixed-length biological sequences."""

__author__ = ["aditi-dsi"]
__all__ = ["SequenceOneHotEncoder"]

from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from pyaptamer.transformations.base import BaseTransform
from pyaptamer.transformations.encode._seq_index import SequenceIndexEncoder


class SequenceOneHotEncoder(BaseTransform):
    """Encode fixed-length sequences as one-hot frames.

    Each sequence of length ``L`` over a vocab of size ``V`` becomes ``V * L``
    one-hot columns, indexed like the input. Column ``v * L + l`` holds class
    ``v`` at position ``l``. Reshape with ``.to_numpy().reshape(n, V, L)`` to
    recover the ``(V, L)`` matrix per sequence.

    By default all sequences must have the same length. Set ``seq_len`` to
    encode sequences of different lengths: each one is right-padded to
    ``seq_len`` positions with all-zero vectors, so ``L`` is ``seq_len``.

    Sequences are validated and mapped to indices by
    :class:`SequenceIndexEncoder`, then one-hot encoded.

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
    inverse_vocab : dict[int, str], optional, default=None
        Maps indices back to characters, for decoding. If None, each index
        decodes to the first character mapped to it in ``vocab``.
    handle_unknown : {"raise", "drop"}, default="raise"
        What to do when a sequence is ``NaN``/``None`` or contains a
        character outside the vocab.

        - "raise" : raise a ``ValueError`` naming the problem.
        - "drop" : skip that row in the encoded output.
    seq_len : int or None, default=None
        Number of positions to right-pad every sequence to. If None, all
        sequences must have the same length. If an int, the positions after
        the end of a shorter sequence are all-zero vectors, and a longer
        sequence raises a ``ValueError``.

    Notes
    -----
    The default vocab maps ``T`` and ``U`` to the same index,
    so DNA and RNA can be encoded without changing the vocab.
    Decoding that index gives ``T``, so RNA in comes back
    out as DNA.

    Examples
    --------
    >>> from pyaptamer.transformations.encode import SequenceOneHotEncoder
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

    Sequences of different lengths are padded to ``seq_len``, and decoding
    drops the padding again:

    >>> X = MoleculeLoader(data={"seq": ["ATGCAT", "GCTA"]})
    >>> enc = SequenceOneHotEncoder(seq_len=8)
    >>> Xt = enc.fit_transform(X)
    >>> Xt.shape
    (2, 32)
    >>> enc.inverse_transform(Xt)["sequence"].tolist()
    ['ATGCAT', 'GCTA']
    """

    _tags = {
        "authors": ["aditi-dsi"],
        "maintainers": ["aditi-dsi"],
        "property:fit_is_empty": True,
        "capability:multivariate": False,
    }

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        inverse_vocab: dict[int, str] | None = None,
        handle_unknown: Literal["raise", "drop"] = "raise",
        seq_len: int | None = None,
    ):
        self.vocab = vocab
        self.inverse_vocab = inverse_vocab
        self.handle_unknown = handle_unknown
        self.seq_len = seq_len
        super().__init__()

    def _resolve_vocab(self):
        """Return the vocab in use."""
        if self.vocab is None:
            return {"A": 0, "T": 1, "G": 2, "C": 3, "U": 1}
        return self.vocab

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
            indexed by the input rows that were encoded.
            ``sequence_length`` is ``seq_len`` if it is set. Rows skipped by
            ``handle_unknown="drop"`` are not included.

        Raises
        ------
        TypeError
            If the first column of X does not hold str sequences.
        ValueError
            If ``seq_len`` is None and sequences have varying lengths, if a
            sequence is longer than ``seq_len``, or if
            ``handle_unknown="raise"`` and a sequence has None/NaN value, or
            contains a character outside the vocab.
        """
        vocab = self._resolve_vocab()
        num_classes = max(vocab.values()) + 1

        indices = SequenceIndexEncoder(
            vocab=vocab, handle_unknown=self.handle_unknown, seq_len=self.seq_len
        ).fit_transform(X)

        if indices.empty:
            return pd.DataFrame(
                np.zeros((0, num_classes * indices.shape[1]), dtype=np.float32),
                index=indices.index,
            )

        int_tensor = torch.tensor(indices.to_numpy(), dtype=torch.long)
        # padding is -1: shifted by one it is class 0, which is dropped, so
        # padding positions become all-zero vectors
        one_hot = F.one_hot(int_tensor + 1, num_classes=num_classes + 1)[..., 1:]
        one_hot = one_hot.to(torch.float32)
        flat = one_hot.permute(0, 2, 1).reshape(len(indices), -1).numpy()

        return pd.DataFrame(flat, index=indices.index)

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
            If a frame's column count is not a multiple of the vocab size, if
            a tensor is not 2D or 3D, or if a 2D tensor is not of integer type.

        Notes
        -----
        Indices outside ``inverse_vocab`` decode to ``"X"``. Such sequences
        cannot be passed back through ``transform``, since ``"X"`` is not in
        the vocabulary.

        If ``seq_len`` is set, padding positions are dropped, so each sequence
        decodes to its length before padding. Padding is an all-zero vector
        in a frame or 3D tensor and ``-1`` in a 2D index tensor.
        """
        vocab = self._resolve_vocab()
        if self.inverse_vocab is not None:
            inverse_vocab = self.inverse_vocab
        else:
            inverse_vocab = {}
            for char, idx in vocab.items():
                inverse_vocab.setdefault(idx, char)

        index = None
        if isinstance(X, pd.DataFrame):
            num_classes = max(vocab.values()) + 1
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

        if X.dim() == 2 and (X.is_floating_point() or X.is_complex()):
            raise ValueError(
                f"{type(self).__name__}.inverse_transform expects a 2D tensor "
                f"of integer indices, got dtype {X.dtype}."
            )

        if X.dim() == 3:
            indices = X.argmax(dim=1)
            if self.seq_len is not None:
                indices[(X == 0).all(dim=1)] = -1
        else:
            indices = X

        decoded_seqs = []
        for row in indices.tolist():
            if self.seq_len is not None:
                row = [idx for idx in row if idx != -1]
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
        param2 = {"seq_len": 50}
        return [param0, param1, param2]
