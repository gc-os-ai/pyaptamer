"""Composite transformer for aptamer-protein pair feature extraction."""

__author__ = ["satvshr"]
__all__ = ["PairsToFeaturesTransformer"]

import pandas as pd

from pyaptamer.trafos.base import BaseTransform
from pyaptamer.trafos.encode._kmer import KMerEncoder
from pyaptamer.trafos.encode._pseaac_trafo import PSeAACEncoder


class PairsToFeaturesTransformer(BaseTransform):
    """
    Transform aptamer-protein sequence pairs into combined feature vectors.

    This composite transformer applies k-mer encoding to aptamer sequences and
    PSeAAC encoding to protein sequences, then concatenates the results into
    a single feature vector for each pair.

    Parameters
    ----------
    k : int, optional, default=4
        K-mer size for aptamer encoding. Generates features for all k-mers
        from length 1 to k.
    aptamer_col : str, optional, default="aptamer"
        Name of the column containing aptamer sequences.
    protein_col : str, optional, default="protein"
        Name of the column containing protein sequences.
    lambda_val : int, optional, default=30
        Number of sequence-order correlation factors for PSeAAC.
    weight : float, optional, default=0.05
        Weight factor for PSeAAC sequence-order correlation features.
    prop_indices : list of int or None, optional, default=None
        Indices of physicochemical properties for PSeAAC. If None, uses defaults.
    group_props : int or None, optional, default=None
        Number of properties per group for PSeAAC. If None, uses defaults.
    custom_groups : list of list of int or None, optional, default=None
        Custom property groupings for PSeAAC. If None, uses defaults.

    Examples
    --------
    >>> from pyaptamer.trafos.encode import PairsToFeaturesTransformer
    >>> import pandas as pd
    >>> transformer = PairsToFeaturesTransformer(k=2, lambda_val=5)
    >>> X = pd.DataFrame(
    ...     {
    ...         "aptamer": ["ACGTACGT", "GGGGAAAA"],
    ...         "protein": ["ACDEFGHIKLMNPQRSTVWY" * 2, "ACDEFGHIKLMNPQRSTVWY" * 2],
    ...     }
    ... )
    >>> X_transformed = transformer.fit_transform(X)
    >>> X_transformed.shape[0]
    2

    Notes
    -----
    This transformer is designed to replace the `pairs_to_features` function
    in the AptaNet pipeline, providing a scikit-learn compatible interface
    that follows the BaseTransform pattern.

    The output feature vector consists of:
    - K-mer frequencies for the aptamer (4+16+64+256 = 340 features for k=4)
    - PSeAAC features for the protein ((20 + lambda_val) * n_groups features)

    References
    ----------
    .. [1] Emami, N., Ferdousi, R. AptaNet as a deep learning approach for
        aptamer–protein interaction prediction. *Scientific Reports*, 11, 6074 (2021).
    """

    _tags = {
        "authors": ["satvshr"],
        "maintainers": ["satvshr"],
        "output_type": "numeric",
        "property:fit_is_empty": True,
        "capability:multivariate": True,  # Requires 2 columns
    }

    def __init__(
        self,
        k: int = 4,
        aptamer_col: str = "aptamer",
        protein_col: str = "protein",
        lambda_val: int = 30,
        weight: float = 0.05,
        prop_indices: list[int] | None = None,
        group_props: int | None = None,
        custom_groups: list[list[int]] | None = None,
    ):
        self.k = k
        self.aptamer_col = aptamer_col
        self.protein_col = protein_col
        self.lambda_val = lambda_val
        self.weight = weight
        self.prop_indices = prop_indices
        self.group_props = group_props
        self.custom_groups = custom_groups
        super().__init__()

    def _check_X_y(self, X, y):  # noqa: N802
        """
        Check X and y inputs and coerce to DataFrame.

        Overrides parent method to handle list of tuples in addition to DataFrames.

        Parameters
        ----------
        X : list of tuple or pd.DataFrame
            List of (aptamer, protein) tuples or DataFrame with sequence columns.
        y : array-like or None
            Target values (unused in transform).

        Returns
        -------
        X : pd.DataFrame
            Coerced DataFrame with aptamer and protein columns.
        y : array-like or None
            Target values (unchanged).
        """
        # Handle list of tuples
        if isinstance(X, list):
            # Convert list of tuples to DataFrame
            if len(X) > 0 and isinstance(X[0], tuple):
                X = pd.DataFrame(X, columns=[self.aptamer_col, self.protein_col])
            else:
                raise TypeError(
                    "If X is a list, it must contain tuples of "
                    "(aptamer_sequence, protein_sequence)"
                )

        # Call parent _check_X_y to validate DataFrame
        return super()._check_X_y(X, y)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform aptamer-protein pairs to combined feature vectors.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame with aptamer and protein sequence columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with concatenated k-mer and PSeAAC features.

        Raises
        ------
        ValueError
            If the specified columns are not found in X.
        """
        # Validate that required columns exist
        if self.aptamer_col not in X.columns:
            raise ValueError(
                f"Column '{self.aptamer_col}' not found in input DataFrame. "
                f"Available columns: {list(X.columns)}"
            )
        if self.protein_col not in X.columns:
            raise ValueError(
                f"Column '{self.protein_col}' not found in input DataFrame. "
                f"Available columns: {list(X.columns)}"
            )

        # Create K-mer encoder for aptamers
        kmer_encoder = KMerEncoder(k=self.k)
        aptamer_df = X[[self.aptamer_col]].copy()
        aptamer_features = kmer_encoder.fit_transform(aptamer_df)

        # Create PSeAAC encoder for proteins
        pseaac_encoder = PSeAACEncoder(
            lambda_val=self.lambda_val,
            weight=self.weight,
            prop_indices=self.prop_indices,
            group_props=self.group_props,
            custom_groups=self.custom_groups,
        )
        protein_df = X[[self.protein_col]].copy()
        protein_features = pseaac_encoder.fit_transform(protein_df)

        # Concatenate features horizontally
        # Use pd.concat to preserve the index
        result_df = pd.concat([aptamer_features, protein_features], axis=1)

        # Ensure the index matches the input
        result_df.index = X.index

        return result_df

    @classmethod
    def get_test_params(cls):
        """
        Get test parameters for PairsToFeaturesTransformer.

        Returns
        -------
        list of dict
            List of parameter dictionaries for testing.
        """
        return [
            {"k": 2, "lambda_val": 5},
            {"k": 3, "lambda_val": 10, "weight": 0.1},
            {
                "k": 4,
                "aptamer_col": "seq1",
                "protein_col": "seq2",
                "lambda_val": 5,
            },
        ]
