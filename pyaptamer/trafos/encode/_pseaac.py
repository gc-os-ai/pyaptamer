"""Pseudo Amino Acid Composition (PseAAC) transformer."""

import numpy as np
import pandas as pd

from pyaptamer.pseaac import PSeAAC
from pyaptamer.trafos.base import BaseTransform


class PSeAACTransformer(BaseTransform):
    """Pseudo Amino Acid Composition (PseAAC) transformer.

    Encodes protein sequences into numerical feature vectors that capture
    both composition and local order correlations.

    Parameters
    ----------
    lambda_val : int, optional, default=30
        The lambda parameter defining the number of sequence-order correlation factors.
    weight : float, optional, default=0.05
        The weight factor for the sequence-order correlation features.
    prop_indices : list[int] or None, optional
        Indices of properties to use (0-based).
    group_props : int or None, optional
        Group size for selected properties.
    custom_groups : list[list[int]] or None, optional
        Explicit groupings of local property indices.
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
        prop_indices=None,
        group_props=None,
        custom_groups=None,
    ):
        self.lambda_val = lambda_val
        self.weight = weight
        self.prop_indices = prop_indices
        self.group_props = group_props
        self.custom_groups = custom_groups

        self._pseaac = PSeAAC(
            lambda_val=lambda_val,
            weight=weight,
            prop_indices=prop_indices,
            group_props=group_props,
            custom_groups=custom_groups,
        )

        super().__init__()

    def _transform(self, X):
        """Transform the data.

        Parameters
        ----------
        X : pd.DataFrame
            Input data to transform.

        Returns
        -------
        X : pd.DataFrame, shape (n_samples, n_features_transformed)
            Transformed data.
        """
        raw_sequences = X.values[:, 0].tolist()
        sequences = ["".join(seq) for seq in raw_sequences]

        feats = [self._pseaac.transform(seq) for seq in sequences]

        result_np = np.array(feats, dtype=np.float32)
        result_df = pd.DataFrame(result_np, index=X.index)

        return result_df

    def get_test_params(self):
        """Get test parameters for PSeAACTransformer.

        Returns
        -------
        params : dict
            Test parameters for PSeAACTransformer.
        """
        return [
            {"lambda_val": 5},
            {"lambda_val": 3, "weight": 0.1, "prop_indices": [0, 1, 2], "group_props": 3},
        ]
