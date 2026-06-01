"""Pseudo Amino Acid Composition (PSeAAC) transformer for protein sequences."""

__author__ = ["satvshr"]
__all__ = ["PSeAACEncoder"]

import numpy as np
import pandas as pd

from pyaptamer.pseaac import PSeAAC
from pyaptamer.trafos.base import BaseTransform


class PSeAACEncoder(BaseTransform):
    """
    Encode protein sequences using Pseudo Amino Acid Composition (PSeAAC).

    This transformer converts protein amino acid sequences into numeric feature
    vectors that capture both amino acid composition and sequence-order information
    based on physicochemical properties.

    Parameters
    ----------
    lambda_val : int, optional, default=30
        Number of sequence-order correlation factors to compute.
    weight : float, optional, default=0.05
        Weight factor for the sequence-order correlation features.
    prop_indices : list of int or None, optional, default=None
        Indices of physicochemical properties to use. If None, uses default properties.
    group_props : int or None, optional, default=None
        Number of properties per group. If None, uses default grouping.
    custom_groups : list of list of int or None, optional, default=None
        Custom property groupings. If None, uses default groups.

    Examples
    --------
    >>> from pyaptamer.trafos.encode import PSeAACEncoder
    >>> import pandas as pd
    >>> encoder = PSeAACEncoder(lambda_val=5)
    >>> X = pd.DataFrame({"seq": ["ACDEFGHIKLMNPQRSTVWY" * 2]})
    >>> X_transformed = encoder.fit_transform(X)
    >>> X_transformed.shape[0]
    1

    Notes
    -----
    PSeAAC encoding is used in the AptaNet algorithm [1]_ for representing
    protein sequences in aptamer-protein interaction prediction.

    The feature vector consists of:
    - 20 normalized amino acid composition features
    - lambda_val * n_groups sequence-order correlation features

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
        "capability:multivariate": False,
    }

    def __init__(
        self,
        lambda_val: int = 30,
        weight: float = 0.05,
        prop_indices: list[int] | None = None,
        group_props: int | None = None,
        custom_groups: list[list[int]] | None = None,
    ):
        self.lambda_val = lambda_val
        self.weight = weight
        self.prop_indices = prop_indices
        self.group_props = group_props
        self.custom_groups = custom_groups
        super().__init__()

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform protein sequences to PSeAAC feature vectors.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame with protein sequences in the first column.

        Returns
        -------
        pd.DataFrame
            DataFrame with PSeAAC features. Each row corresponds to an input
            sequence, and columns represent PSeAAC feature values.
        """
        # Instantiate PSeAAC with stored parameters
        pseaac = PSeAAC(
            lambda_val=self.lambda_val,
            weight=self.weight,
            prop_indices=self.prop_indices,
            group_props=self.group_props,
            custom_groups=self.custom_groups,
        )

        # Extract sequences from the first column
        sequences = X.iloc[:, 0].tolist()

        # Generate PSeAAC vectors for each sequence
        feature_vectors = []
        for seq in sequences:
            pseaac_vec = np.asarray(pseaac.transform(seq))
            feature_vectors.append(pseaac_vec)

        # Stack into a 2D array
        feature_array = np.vstack(feature_vectors)

        # Create DataFrame with feature columns
        n_features = feature_array.shape[1]
        feature_columns = [f"pseaac_{i}" for i in range(n_features)]
        result_df = pd.DataFrame(feature_array, index=X.index, columns=feature_columns)

        return result_df

    @classmethod
    def get_test_params(cls):
        """
        Get test parameters for PSeAACEncoder.

        Returns
        -------
        list of dict
            List of parameter dictionaries for testing.
        """
        return [
            {"lambda_val": 5},
            {"lambda_val": 10, "weight": 0.1},
            {"lambda_val": 15, "weight": 0.05},
        ]
