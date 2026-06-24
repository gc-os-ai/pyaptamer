"""AptaNet feature extractor."""

import pandas as pd

from pyaptamer.trafos.base import BaseTransform
from pyaptamer.trafos.encode._kmer import KMerEncoder
from pyaptamer.trafos.encode._pseaac import PSeAACTransformer


class AptaNetFeatureExtractor(BaseTransform):
    """AptaNet feature extractor.

    Composite transformer that extracts features for aptamer-protein pairs.
    It applies KMerEncoder to the aptamer sequences and PSeAACTransformer
    to the protein sequences, then concatenates the results.

    Parameters
    ----------
    k : int, optional, default=4
        The k-mer size used to generate the k-mer vector from the aptamer sequence.
    lambda_val : int, optional, default=30
        The lambda parameter for PseAAC.
    weight : float, optional, default=0.05
        The weight factor for PseAAC.
    aptamer_col : str, optional, default="aptamer"
        The name of the column containing aptamer sequences.
    protein_col : str, optional, default="protein"
        The name of the column containing protein sequences.
    alphabet : list[str] or str or None, optional, default=None
        Characters used to build the k-mer vocabulary. Passed through to
        ``KMerEncoder``.

        * ``None`` (default) – infer the alphabet from the aptamer sequences
          seen during ``fit``.
        * ``str`` or ``list[str]`` – use the provided characters.
    """

    _tags = {
        "authors": ["satvshr"],
        "maintainers": ["satvshr"],
        "output_type": "numeric",
        "property:fit_is_empty": False,
        "capability:multivariate": True,
    }

    def __init__(
        self,
        k: int = 4,
        lambda_val: int = 30,
        weight: float = 0.05,
        aptamer_col: str = "aptamer",
        protein_col: str = "protein",
        alphabet=None,
    ):
        self.k = k
        self.lambda_val = lambda_val
        self.weight = weight
        self.aptamer_col = aptamer_col
        self.protein_col = protein_col
        self.alphabet = alphabet

        self._kmer_encoder = KMerEncoder(k=k, alphabet=alphabet)
        self._pseaac_transformer = PSeAACTransformer(
            lambda_val=lambda_val, weight=weight
        )

        super().__init__()

    def _resolve_aptamer_col(self, X):
        """Resolve the aptamer column name from the DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            Input data containing aptamer and protein columns.

        Returns
        -------
        str
            The resolved aptamer column name.
        """
        if self.aptamer_col not in X.columns:
            if X.shape[1] >= 2 and self.aptamer_col == "aptamer":
                return X.columns[0]
            else:
                raise ValueError(f"Column '{self.aptamer_col}' not found in X.")
        return self.aptamer_col

    def _resolve_protein_col(self, X):
        """Resolve the protein column name from the DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            Input data containing aptamer and protein columns.

        Returns
        -------
        str
            The resolved protein column name.
        """
        if self.protein_col not in X.columns:
            if X.shape[1] >= 2 and self.protein_col == "protein":
                return X.columns[1]
            else:
                raise ValueError(f"Column '{self.protein_col}' not found in X.")
        return self.protein_col

    def _fit(self, X, y=None):
        """Fit the feature extractor.

        Fits the internal ``KMerEncoder`` on the aptamer column so that it
        can infer the alphabet from the input sequences.

        Parameters
        ----------
        X : pd.DataFrame
            Input data containing aptamer and protein columns.
        y : ignored

        Returns
        -------
        self
        """
        aptamer_col = self._resolve_aptamer_col(X)
        self._kmer_encoder.fit(X[[aptamer_col]], y)
        return self

    def _transform(self, X):
        """Transform the data.

        Parameters
        ----------
        X : pd.DataFrame
            Input data containing aptamer and protein columns.

        Returns
        -------
        Xt : pd.DataFrame
            Concatenated feature matrix.
        """
        aptamer_col = self._resolve_aptamer_col(X)
        protein_col = self._resolve_protein_col(X)

        # Transform columns
        X_aptamer = X[[aptamer_col]]
        X_protein = X[[protein_col]]

        feat_aptamer = self._kmer_encoder.transform(X_aptamer)
        feat_protein = self._pseaac_transformer.transform(X_protein)

        # Concatenate results along columns
        # pairs_to_features returns a single numpy array, so we return a single DF.
        Xt = pd.concat([feat_aptamer, feat_protein], axis=1)

        # Ensure unique column names for the result
        Xt.columns = range(Xt.shape[1])

        return Xt

    def get_test_params(self):
        """Get test parameters for AptaNetFeatureExtractor.

        Returns
        -------
        params : dict
            Test parameters for AptaNetFeatureExtractor.
        """
        return [{"k": 1, "lambda_val": 5}]
