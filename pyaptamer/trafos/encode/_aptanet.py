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
    """

    _tags = {
        "authors": ["satvshr"],
        "maintainers": ["satvshr"],
        "output_type": "numeric",
        "property:fit_is_empty": True,
        "capability:multivariate": True,
    }

    def __init__(
        self,
        k: int = 4,
        lambda_val: int = 30,
        weight: float = 0.05,
        aptamer_col: str = "aptamer",
        protein_col: str = "protein",
    ):
        self.k = k
        self.lambda_val = lambda_val
        self.weight = weight
        self.aptamer_col = aptamer_col
        self.protein_col = protein_col

        self._kmer_encoder = KMerEncoder(k=k)
        self._pseaac_transformer = PSeAACTransformer(
            lambda_val=lambda_val, weight=weight
        )

        super().__init__()

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
        # Validate columns
        if self.aptamer_col not in X.columns:
            # Fallback to first column if names don't match and it's 2-column df
            if X.shape[1] >= 2 and self.aptamer_col == "aptamer":
                aptamer_col = X.columns[0]
            else:
                raise ValueError(f"Column '{self.aptamer_col}' not found in X.")
        else:
            aptamer_col = self.aptamer_col

        if self.protein_col not in X.columns:
            # Fallback to second column
            if X.shape[1] >= 2 and self.protein_col == "protein":
                protein_col = X.columns[1]
            else:
                raise ValueError(f"Column '{self.protein_col}' not found in X.")
        else:
            protein_col = self.protein_col

        # Transform columns
        X_aptamer = X[[aptamer_col]]
        X_protein = X[[protein_col]]

        feat_aptamer = self._kmer_encoder.transform(X_aptamer)
        feat_protein = self._pseaac_transformer.transform(X_protein)

        # Concatenate results along columns
        # Resetting columns names for feature matrix if desired, or keep as is.
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
