__author__ = ["satvshr", "geetu040"]
__all__ = ["DeepAptamerFeatures"]

import numpy as np
import pandas as pd

from pyaptamer.deepdnashape import DeepDNAShape
from pyaptamer.trafos.base import BaseTransform


def _one_hot(seq, seq_len):
    """
    One-hot encode a DNA sequence into a fixed number of positions.

    Each base is encoded as:
    - A -> [1, 0, 0, 0]
    - T -> [0, 1, 0, 0]
    - C -> [0, 0, 1, 0]
    - G -> [0, 0, 0, 1]
    - any other character -> [0, 0, 0, 0]

    Positions after the end of ``seq`` are all zero.

    Parameters
    ----------
    seq : str
        A DNA sequence of length at most ``seq_len``.
    seq_len : int
        Number of positions in the output.

    Returns
    -------
    np.ndarray
        A 2D NumPy array of shape (``seq_len``, 4).

    Raises
    ------
    ValueError
        If ``seq`` is longer than ``seq_len``.
    """
    if len(seq) > seq_len:
        raise ValueError(f"Sequence length {len(seq)} exceeds {seq_len}: '{seq}'")

    encoded = np.zeros((seq_len, 4), dtype=int)
    for i, base in enumerate(seq):
        j = "ATCG".find(base)
        if j >= 0:
            encoded[i, j] = 1
    return encoded


def _shape_features(X, seq_len, full_dna_shape):
    """
    Turn a batch of DNA sequences into normalized shape vectors.

    For each shape feature (MGW, HelT, ProT, Roll), `DeepDNAShape` predicts the
    values of every sequence, read from message-passing layer 2 (a 5-base
    window). Each sequence's values are then trimmed, normalized to zero mean and
    unit variance, and right-padded with zeros to their length for a ``seq_len``
    sequence. The four features are concatenated in that order.

    Each sequence is normalized with its own mean and standard deviation. The
    paper (Equation 2) may compute them over all samples instead.

    Trimming drops ``edge`` values at each end: the positions that `DNAshapeR`
    leaves NA, as in the DeepAptamer paper. That is 2 for MGW and ProT and 1 for
    HelT and Roll.

    Parameters
    ----------
    X : pd.DataFrame
        One column of DNA sequences, each of length at most ``seq_len``.
    seq_len : int
        Length the shape features are padded to.
    full_dna_shape : bool
        If True, skip trimming (138 values for a 35-mer instead of 126).

    Returns
    -------
    np.ndarray
        A 2D NumPy array of shape (n_seqs, shape_len), where ``shape_len``
        depends on ``seq_len`` and ``full_dna_shape``.
    """
    seqs = X.iloc[:, 0]
    blocks = []
    for feature, edge in {"MGW": 2, "HelT": 1, "ProT": 2, "Roll": 1}.items():
        predictions = DeepDNAShape(feature=feature, layer=2).fit_transform(X)
        if full_dna_shape:
            edge = 0
        block = []
        for seq, values in zip(seqs, predictions.to_numpy(), strict=True):
            values = values[~np.isnan(values)]
            values = values[edge : len(values) - edge]
            values = (values - values.mean()) / (values.std() or 1.0)
            block.append(np.pad(values, (0, seq_len - len(seq))))
        blocks.append(np.stack(block))
    return np.hstack(blocks)


class DeepAptamerFeatures(BaseTransform):
    """DeepAptamer input features of aptamer sequences.

    Each sequence is encoded as its one-hot matrix of ``seq_len`` positions, with
    all-zero rows after the end of the sequence, flattened row-wise from shape
    (``seq_len``, 4). It is followed by the normalized DNA shape vector (MGW, HelT,
    ProT, Roll) from `DeepDNAShape`.
    This is the input layout expected by `DeepAptamerClassifier`.

    Shape is predicted on the unpadded sequence. Each shape feature is then
    normalized and right-padded with zeros, so padding does not change the
    values of the real positions.

    The width is ``4 * seq_len + shape_len``, where ``shape_len`` is
    ``4 * seq_len - 14`` if ``full_dna_shape`` is False and ``4 * seq_len - 2``
    otherwise. With the defaults it is ``140 + 126 = 266``.

    Input is a one-column DataFrame or a ``MoleculeLoader`` of DNA strings. Only
    DNA is supported: sequences must be uppercase and contain only A, C, G, T and
    N. Other characters, including U in RNA, raise a ValueError from
    `DeepDNAShape`.

    Parameters
    ----------
    seq_len : int, default=35
        Length the sequences are padded to. Longer sequences raise an error.
    full_dna_shape : bool, default=False
        If True, keep the full `DeepDNAShape` output. If False, drop the edge
        positions that are NA in `DNAshapeR`, as in the DeepAptamer paper.

    Examples
    --------
    >>> from pyaptamer.data import MoleculeLoader
    >>> from pyaptamer.deepaptamer import DeepAptamerFeatures
    >>> aptamers = [
    ...     "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCC",
    ...     "TGCATGCTAGCTAGCTAGCT",
    ... ]
    >>> X = MoleculeLoader(data={"aptamer": aptamers})
    >>> DeepAptamerFeatures().fit_transform(X).shape
    (2, 266)
    """

    _tags = {
        "authors": ["satvshr", "geetu040"],
        "maintainers": ["geetu040"],
        "output_type": "numeric",
        "property:fit_is_empty": True,
        "capability:multivariate": False,
    }

    def __init__(self, seq_len=35, full_dna_shape=False):
        self.seq_len = seq_len
        self.full_dna_shape = full_dna_shape
        super().__init__()

    def _transform(self, X):
        """Encode every sequence in the single column of X.

        Parameters
        ----------
        X : pd.DataFrame
            One column of DNA strings.

        Returns
        -------
        pd.DataFrame
            Shape ``(len(X), 4 * seq_len + shape_len)``, indexed like X.
        """
        seqs = X.iloc[:, 0].tolist()
        X_ohe = np.stack([_one_hot(seq, self.seq_len).ravel() for seq in seqs])
        X_shape = _shape_features(X, self.seq_len, self.full_dna_shape)
        return pd.DataFrame(np.hstack([X_ohe, X_shape]), index=X.index)

    @classmethod
    def get_test_params(cls):
        """Return parameter sets for the shared transformer tests.

        ``seq_len`` must be at least the length of the test sequences (40).
        """
        return [{"seq_len": 40}, {"seq_len": 50, "full_dna_shape": True}]
