"""Feature-extraction transforms for the AptaNet algorithm."""

__author__ = ["siddharth7113"]
__all__ = ["PairsToFeatures"]

import numpy as np
import pandas as pd

from pyaptamer.data import MoleculeLoader
from pyaptamer.pseaac import AptaNetPSeAAC
from pyaptamer.trafos.base import BaseTransform
from pyaptamer.utils._aptanet_utils import generate_kmer_vecs


class PairsToFeatures(BaseTransform):
    """Transform (aptamer, protein) sequence pairs into AptaNet feature vectors.

    Each row is encoded as the concatenation of:

    - normalized k-mer frequencies of the aptamer sequence, and
    - pseudo amino-acid composition (PSeAAC) of the protein sequence.

    Input must be a :class:`~pyaptamer.data.loader.MoleculeLoader`

    Parameters
    ----------
    k : int, default=4
        Maximum k-mer length used for the aptamer k-mer frequency vector.
    aptamer_col : str, default="aptamer"
        Name of the column holding aptamer sequences.
    protein_col : str, default="protein"
        Name of the column holding protein sequences.

    Examples
    --------
    >>> from pyaptamer.aptanet import PairsToFeatures
    >>> from pyaptamer.data import MoleculeLoader
    >>> X = MoleculeLoader(
    ...     data={
    ...         "aptamer": ["AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"],
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

    def __init__(self, k=4, aptamer_col="aptamer", protein_col="protein"):
        self.k = k
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
        """Encode each (aptamer, protein) row into a feature vector.

        Parameters
        ----------
        X : pandas.DataFrame
            Contains the ``aptamer_col`` and ``protein_col`` columns.

        Returns
        -------
        pandas.DataFrame
            One row per input row; columns are the concatenated k-mer + PSeAAC
            features (``float32``), indexed like ``X``.
        """
        pseaac = AptaNetPSeAAC()
        feats = [
            np.concatenate(
                [
                    generate_kmer_vecs(aptamer_seq, k=self.k),
                    np.asarray(pseaac.transform(protein_seq)),
                ]
            )
            for aptamer_seq, protein_seq in zip(
                X[self.aptamer_col], X[self.protein_col], strict=True
            )
        ]
        return pd.DataFrame(np.vstack(feats).astype(np.float32), index=X.index)
