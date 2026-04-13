__author__ = ["nennomp", "satvshr", "fkiraly"]
__all__ = ["PSeAAC"]

from collections import Counter

import numpy as np
import pandas as pd

from pyaptamer.pseaac._props import aa_props
from pyaptamer.trafos.base import BaseTransform
from pyaptamer.utils._pseaac_utils import AMINO_ACIDS, clean_protein_seq


class PSeAAC(BaseTransform):
    """
    Compute Pseudo Amino Acid Composition (PseAAC) features for a protein sequence.

    This class generates a numerical feature vector that encodes both the composition
    and local order of amino acids in a protein sequence. The features are derived from
    selected physicochemical properties and sequence-order correlations as described in
    the PseAAC model by Chou.

    The PSeAAC algorithm uses normalized physicochemical (NP) properties of amino
    acids, loaded from a predefined matrix using `aa_props`. Properties can be grouped
    in one of three ways:

    - `prop_indices`: A list of property indices (0-based) to select from the 21
      available properties. If None, all 21 properties are used.
    - `group_props`: If provided as an integer, the selected properties are grouped
      into chunks of this size (e.g., `group_props=3` groups into sets of 3).
      If None, the default is groups of size 3 (7 groups for 21 properties).
    - `custom_groups`: A list of lists, where each sublist contains local column
      indices into the selected property matrix. This overrides all other grouping
      logic.

    Each feature vector consists of:

    - 20 normalized amino acid composition features (frequency of each standard
      amino acid)
    - `lambda_val` sequence-order correlation features (theta values) computed
      from the selected physicochemical property groups.

    For each property group, the above (20 + `lambda_val`) features are computed,
    resulting in a final vector of length (20 + lambda_val) * number of normalized
    physiochemical (NP) property groups of amino acids (default 7).

    Parameters
    ----------
    lambda_val : int, optional, default=30
        The lambda parameter defining the number of sequence-order correlation factors.
        This also determines the minimum length allowed for input protein sequences,
        which should be of length greater than `lambda_val`.
    weight : float, optional, default=0.05
        The weight factor for the sequence-order correlation features.
    prop_indices : list[int] or None, optional
        Indices of properties to use (0-based). If None, all 21 properties are used.
    group_props : int or None, optional
        Group size for selected properties. If None, defaults to groups of 3.
    custom_groups : list[list[int]] or None, optional
        Explicit groupings of local property indices. Overrides `group_props`.

    Attributes
    ----------
    np_matrix : np.ndarray of shape (20, n_props)
        Normalized property values for the selected amino acids and properties.
    prop_groups : list[list[int]]
        Groupings of local property indices into `np_matrix`.
    """

    _tags = {
        "authors": ["nennomp", "satvshr", "fkiraly"],
        "maintainers": ["nennomp", "satvshr", "fkiraly"],
        "output_type": "numeric",
        "property:fit_is_empty": True,
        "capability:multivariate": False,
    }

    def __init__(
        self,
        lambda_val=30,
        weight=0.05,
        prop_indices=None,
        group_props=None,
        custom_groups=None,
    ):
        self.lambda_val = lambda_val
        self.weight = weight
        self.prop_indices = prop_indices
        self.group_props = group_props
        self.custom_groups = custom_groups

        if group_props is not None and custom_groups is not None:
            raise ValueError(
                "Specify only one of `group_props` or `custom_groups`,not both."
            )

        self.np_matrix = aa_props(
            prop_indices=prop_indices, type="numpy", normalize=True
        )
        self._n_cols = self.np_matrix.shape[1]  # The number of properties selected

        if custom_groups:
            self.prop_groups = custom_groups
        elif group_props is None:
            if self._n_cols % 3 != 0:
                raise ValueError(
                    "Default grouping expects number of properties divisible by 3."
                )
            self.prop_groups = [
                list(range(i, i + 3)) for i in range(0, self._n_cols, 3)
            ]
        else:
            if self._n_cols % group_props != 0:
                raise ValueError(
                    f"Number of properties ({self._n_cols}) must be divisible by"
                    f"group_props ({group_props})."
                )
            self.prop_groups = [
                list(range(i, i + group_props))
                for i in range(0, self._n_cols, group_props)
            ]
        
        super().__init__()

    def _normalized_aa(self, seq):
        """Compute the normalized amino acid composition for a sequence."""
        counts = Counter(seq)
        total = len(seq)
        return np.array([counts.get(aa, 0) / total for aa in AMINO_ACIDS])

    def _avg_theta_val(self, seq_vec, seq_len, n, prop_group):
        """Compute the average theta value for a sequence and property group."""
        props = self.np_matrix[:, prop_group]

        ri = props[seq_vec[: seq_len - n]]
        rj = props[seq_vec[n:]]

        diffs = rj - ri
        return np.mean(diffs**2)

    def _transform_sequence(self, protein_sequence):
        """Generate the PseAAC feature vector for a single protein sequence."""
        seq = clean_protein_seq(protein_sequence)
        seq_len = len(seq)
        if seq_len <= self.lambda_val:
            raise ValueError(
                f"Protein sequence is too short, should be longer than `lambda_val`. "
                f"Sequence length: {seq_len}, `lambda_val`: {self.lambda_val}."
            )

        aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
        seq_vec = np.array([aa_to_idx[aa] for aa in seq], dtype=np.int32)

        aa_freq = self._normalized_aa(seq)
        sum_all_aa_freq = aa_freq.sum()

        all_pseaac = []
        for prop_group in self.prop_groups:
            all_theta_val = np.array(
                [
                    self._avg_theta_val(seq_vec, seq_len, n, prop_group)
                    for n in range(1, self.lambda_val + 1)
                ]
            )

            sum_all_theta_val = np.sum(all_theta_val)
            denominator_val = sum_all_aa_freq + (self.weight * sum_all_theta_val)

            # First 20 features: normalized amino acid composition
            all_pseaac.extend(np.round(aa_freq / denominator_val, 3))

            # Next `self.lambda_val` features: theta values
            all_pseaac.extend(
                np.round((self.weight * all_theta_val) / denominator_val, 3)
            )

        return np.array(all_pseaac)

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
        # X is coerced to DataFrame with single column "sequence" by BaseTransform
        sequences = X.iloc[:, 0]
        
        feature_vectors = [self._transform_sequence(seq) for seq in sequences]
        
        result_np = np.stack(feature_vectors)
        result_df = pd.DataFrame(result_np, index=X.index)
        
        return result_df

    def get_test_params(self):
        """Get test parameters for PSeAAC."""
        return [
            {"lambda_val": 5, "weight": 0.05},
            {"lambda_val": 10, "prop_indices": list(range(6)), "group_props": 3},
        ]
