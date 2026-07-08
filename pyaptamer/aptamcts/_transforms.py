"""Feature-extraction transforms for the AptaMCTS algorithm."""

__author__ = ["aditi-dsi"]
__all__ = ["PairsToFeatures"]

import numpy as np
import pandas as pd

from pyaptamer.data import MoleculeLoader
from pyaptamer.trafos.base import BaseTransform
from pyaptamer.utils._aptamcts_utils import protein_to_ictf, rna_to_ictf


class PairsToFeatures(BaseTransform):
    """Transform (aptamer, protein) sequence pairs into AptaMCTS feature vectors.

    Each row is encoded as the concatenation of:
    - The Improved Conjoint Triad Feature (iCTF) representation of the aptamer.
    - The Improved Conjoint Triad Feature (iCTF) representation of the protein.

    Input must be a :class:`~pyaptamer.data.loader.MoleculeLoader`.

    Parameters
    ----------
    rna_k : int, default=4
        The k-mer size used to generate the iCTF vector for the aptamer sequence.
    prot_k : int, default=3
        The k-mer size used to generate the iCTF vector for the protein sequence.
    aptamer_col : str, default="aptamer"
        Name of the column holding aptamer sequences.
    protein_col : str, default="protein"
        Name of the column holding protein sequences.

    Examples
    --------
    >>> from pyaptamer.aptamcts import PairsToFeatures
    >>> from pyaptamer.data import MoleculeLoader
    >>> X = MoleculeLoader(
    ...     data={
    ...         "aptamer": ["AGCUUAGCGUACAGCUUAAAAGGGUUUCCCCUGCCCGCGTAC"],
    ...         "protein": ["ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"],
    ...     }
    ... )
    >>> Xt = PairsToFeatures().fit_transform(X)
    >>> len(Xt)
    1
    """

    _tags = {
        "capability:multivariate": True,
        "property:fit_is_empty": True,
        "output_type": "numeric",
    }

    def __init__(self, rna_k=4, prot_k=3, aptamer_col="aptamer", protein_col="protein"):
        self.rna_k = rna_k
        self.prot_k = prot_k
        self.aptamer_col = aptamer_col
        self.protein_col = protein_col
        super().__init__()

    def _check_X_y(self, X, y):  # noqa: N802
        """Require a MoleculeLoader, then defer to the base coercion/checks."""
        if not isinstance(X, MoleculeLoader):
            raise TypeError(
                f"{type(self).__name__} accepts only a MoleculeLoader as input, "
                f"got {type(X).__name__}."
            )
        return super()._check_X_y(X, y)

    def _transform(self, X):
        """Encode each (aptamer, protein) row into an iCTF feature vector.

        Parameters
        ----------
        X : pandas.DataFrame
            Contains the ``aptamer_col`` and ``protein_col`` columns.

        Returns
        -------
        pandas.DataFrame
            One row per input row; columns are the concatenated iCTF features
            (``float32``), indexed like ``X``.
        """
        feats = [
            np.concatenate(
                [
                    rna_to_ictf(aptamer_seq, k=self.rna_k),
                    protein_to_ictf(protein_seq, k=self.prot_k),
                ]
            )
            for aptamer_seq, protein_seq in zip(
                X[self.aptamer_col], X[self.protein_col], strict=True
            )
        ]
        return pd.DataFrame(np.vstack(feats).astype(np.float32), index=X.index)
