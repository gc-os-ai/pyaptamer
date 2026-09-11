"""Feature-extraction transforms for the AptaTrans algorithm."""

__author__ = ["siddharth7113"]
__all__ = ["PairsToTokens"]
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from pyaptamer.trafos.base import BaseTransform
from pyaptamer.utils import encode_rna, rna2vec
from pyaptamer.utils._base import filter_words


class PairsToTokens(BaseTransform):
    """Transform (aptamer, protein) sequence pairs into AptaTrans token ids.

    Each row is encoded as the concatenation of:

    - overlapping-triplet ids of the aptamer sequence, zero-padded to
      ``apta_max_len``, and
    - greedy longest-match token ids of the protein sequence, zero-padded to
      ``prot_max_len``.

    The two blocks sit side by side, so a row has ``apta_max_len + prot_max_len``
    columns and the boundary between them is at index ``apta_max_len``.

    Input must be a :class:`~pyaptamer.data.loader.MoleculeLoader`, or a
    ``pandas.DataFrame`` holding the aptamer and protein columns.

    Parameters
    ----------
    prot_words : dict[str, float]
        Protein n-mer subsequences mapped to their frequency. Filtered during
        ``fit``; see ``prot_words_``. Should come from the same dataset used to
        pretrain the protein encoder.
    apta_max_len : int, default=275
        Width of the aptamer token block. Sequences are truncated or zero-padded
        to this length.
    prot_max_len : int, default=867
        Width of the protein token block. Sequences are truncated or zero-padded
        to this length.
    aptamer_col : str, default="aptamer"
        Name of the column holding aptamer sequences.
    protein_col : str, default="protein"
        Name of the column holding protein sequences.

    Attributes
    ----------
    prot_words_ : dict[str, int]
        Protein words with above-average frequency, mapped to unique integer ids.
        Derived from ``prot_words`` during ``fit``.

    Examples
    --------
    >>> from pyaptamer.aptatrans import PairsToTokens
    >>> from pyaptamer.data import MoleculeLoader
    >>> X = MoleculeLoader(
    ...     data={"aptamer": ["AGCUUAGCGUAC"], "protein": ["ACDEFGHIKLMN"]}
    ... )
    >>> prot_words = {"ACD": 5.0, "EFG": 4.0, "HIK": 3.0, "LMN": 1.0}
    >>> Xt = PairsToTokens(
    ...     prot_words=prot_words, apta_max_len=10, prot_max_len=8
    ... ).fit_transform(X)
    >>> Xt.shape
    (1, 18)
    """

    _tags = {
        "capability:multivariate": True,
        "property:fit_is_empty": False,
        "output_type": "numeric",
    }

    def __init__(
        self,
        prot_words: dict[str, float],
        apta_max_len: int = 275,
        prot_max_len: int = 867,
        aptamer_col: str = "aptamer",
        protein_col: str = "protein",
    ):
        self.prot_words = prot_words
        self.apta_max_len = apta_max_len
        self.prot_max_len = prot_max_len
        self.aptamer_col = aptamer_col
        self.protein_col = protein_col
        super().__init__()

    def _transform(self, X):
        """Encode each (aptamer, protein) row into token ids."""
        check_is_fitted(self)
        aptamers = X[self.aptamer_col].astype(str).tolist()
        proteins = X[self.protein_col].astype(str).tolist()

        aptamer_ids = rna2vec(
            aptamers,
            max_sequence_length=self.apta_max_len,
        )
        protein_ids = encode_rna(
            proteins,
            words=self.prot_words_,
            max_len=self.prot_max_len,
            return_type="numpy",
        )

        encoded = np.hstack([aptamer_ids, protein_ids])
        return pd.DataFrame(encoded, index=X.index)

    def _fit(self, X, y=None):
        """Derive the filtered protein vocabulary. ``X`` is not used."""
        self.prot_words_ = filter_words(self.prot_words)
