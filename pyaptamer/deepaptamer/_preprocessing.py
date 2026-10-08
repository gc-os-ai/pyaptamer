__author__ = ["satvshr", "geetu040"]
__all__ = ["DeepAptamerFeatures", "preprocess_seq_ohe", "preprocess_seq_shape"]

import numpy as np
import pandas as pd

from pyaptamer.trafos.base import BaseTransform
from pyaptamer.utils._deepaptamer_utils import (
    ohe,
    pad_sequence,
    remove_na,
    run_deepdna_prediction,
)


def preprocess_seq_ohe(seq, seq_len=35):
    """
    Preprocesses a single DNA sequence for DeepAptamer.

    The function pads the sequence to length `seq_len` using 'N' and one-hot encodes
    it. The resulting array has shape (`seq_len`, 4), where each base is encoded as:
    - A → [1, 0, 0, 0]
    - T → [0, 1, 0, 0]
    - C → [0, 0, 1, 0]
    - G → [0, 0, 0, 1]
    - N or unknown → [0, 0, 0, 0]

    Parameters
    ----------
    seq : str
        A DNA sequence of length ≤ `seq_len`.
    seq_len : int, optional, default=35
        The length to which the sequence will be padded or truncated.

    Returns
    -------
    np.ndarray
        A NumPy array of shape (`seq_len`, 4) representing the one-hot
        encoded sequence.
    """
    seq_pad = pad_sequence(seq, seq_len)  # pads to `seq_len`
    seq_ohe = ohe(seq_pad)  # one-hot encode (shape `seq_len` × 4)
    return seq_ohe


def preprocess_seq_shape(seqs, full_dna_shape=False):
    """
    Preprocesses a batch of DNA sequences into normalized shape vectors.

    The function runs DeepDNA prediction on all input sequences at once,
    normalizes each sequence's features, flattens them into a single row
    vector per sequence, and removes any "NA" values.

    Parameters
    ----------
    seqs : str or list of str
        DNA sequence(s) to be processed. All sequences must have the same length.
    full_dna_shape : bool, optional, default=False
        If True, keep the full `DeepDNAShape` output (138 values for a 35-mer).
        If False, drop the edge positions that are NA in `DNAshapeR`, as in the
        DeepAptamer paper (126 values for a 35-mer).

    Returns
    -------
    np.ndarray
        A 3D NumPy array of shape (n_seqs, 1, new_length), where `new_length`
        depends on the DeepDNA prediction output after flattening and
        removing "NA" values.
    """
    seq_shapes = run_deepdna_prediction(seqs)
    return np.stack(
        [_normalize_shape(seq_shape, full_dna_shape) for seq_shape in seq_shapes]
    )


def _normalize_shape(seq_shape, full_dna_shape=False):
    """
    Normalize and flatten the shape predictions of a single sequence.

    Parameters
    ----------
    seq_shape : list of list of float
        A list of 4 lists in order [MGW, HelT, ProT, Roll].
    full_dna_shape : bool, optional, default=False
        Passed on from `preprocess_seq_shape`.

    Returns
    -------
    np.ndarray
        A 2D NumPy array of shape (1, new_length).
    """
    if not full_dna_shape:
        seq_shape = remove_na(seq_shape)

    norm_features = []
    for feat in seq_shape:  # each feat is a list of floats
        arr = np.array(feat, dtype=np.float32)

        # Normalize per feature
        mean = arr.mean()
        std = arr.std() if arr.std() > 0 else 1.0
        arr_norm = (arr - mean) / std

        norm_features.append(arr_norm)

    # Concatenate all features into one flat vector
    seq_flat = np.concatenate(norm_features).reshape(1, -1)

    return seq_flat


class DeepAptamerFeatures(BaseTransform):
    """DeepAptamer input features of aptamer sequences.

    Each sequence is padded with ``N`` to ``seq_len`` and encoded as its one-hot
    matrix, flattened row-wise from shape (``seq_len``, 4), followed by its
    normalized DNA shape vector (MGW, HelT, ProT, Roll) from `DeepDNAShape`.
    This is the input layout expected by `DeepAptamerClassifier`.

    The width is ``4 * seq_len + shape_len``, where ``shape_len`` is
    ``4 * seq_len - 14`` if ``full_dna_shape`` is False and ``4 * seq_len - 2``
    otherwise. With the defaults it is ``140 + 126 = 266``.

    Input is a one-column DataFrame or a ``MoleculeLoader`` of DNA strings.

    Parameters
    ----------
    seq_len : int, default=35
        Length the sequences are padded to. Longer sequences raise an error.
    full_dna_shape : bool, default=False
        If True, keep the full `DeepDNAShape` output. If False, drop the edge
        positions that are NA in `DNAshapeR`, as in the DeepAptamer paper.

    Examples
    --------
    >>> import pandas as pd
    >>> from pyaptamer.deepaptamer import DeepAptamerFeatures
    >>> X = pd.DataFrame({"aptamer": ["AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCC"]})
    >>> DeepAptamerFeatures().fit_transform(X).shape
    (1, 266)
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
        seqs = [pad_sequence(seq, self.seq_len) for seq in X.iloc[:, 0]]
        X_ohe = np.stack(
            [preprocess_seq_ohe(seq, self.seq_len).ravel() for seq in seqs]
        )
        X_shape = preprocess_seq_shape(seqs, self.full_dna_shape).reshape(len(seqs), -1)
        return pd.DataFrame(np.hstack([X_ohe, X_shape]), index=X.index)

    @classmethod
    def get_test_params(cls):
        """Return parameter sets for the shared transformer tests.

        ``seq_len`` must be at least the length of the test sequences (40).
        """
        return [{"seq_len": 40}, {"seq_len": 50, "full_dna_shape": True}]
