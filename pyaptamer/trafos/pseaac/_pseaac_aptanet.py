__author__ = ["nennomp", "satvshr", "fkiraly"]
__all__ = ["AptaNetPSeAAC"]

from collections import Counter

import numpy as np
import pandas as pd

from pyaptamer.pseaac._props import aa_props
from pyaptamer.trafos.base import BaseTransform
from pyaptamer.utils._pseaac_utils import AMINO_ACIDS, clean_protein_seq


class AptaNetPSeAAC(BaseTransform):
    """
    Compute Pseudo Amino Acid Composition (PseAAC) features for a protein sequence.

    This class generates a numerical feature vector that encodes both the composition
    and local order of amino acids in a protein sequence. The features are derived from
    selected physicochemical properties and sequence-order correlations as described in
    the PseAAC model by Chou.

    The PSeAAC algorithm uses 21 normalized physiochemical (NP) properties of amino
    acids, which we load from a predefined matrix using `aa_props`.These 21 properties
    are grouped into 7 distinct property groups, with each group containing
    3 consecutive properties. Specifically, the groups are arranged in order as follows:
    Group 1 includes properties 1–3, Group 2 includes properties 4–6, and so on, up to
    Group 7, which includes properties 19–21.

    Each feature vector consists of:

    - 20 normalized amino acid composition features (frequency of each standard
    amino acid)
    - `self.lambda_val` sequence-order correlation features based on physicochemical
    similarity between residues.
    These (20 + `self.lambda_val`) features are computed for each of 7 predefined
    property groups, resulting in a final vector of length (20 + `self.lambda_val`) * 7.

    Parameters
    ----------
    lambda_val : int, optional, default=30
        The lambda parameter defining the number of sequence-order correlation factors.
        This also determines the minimum length allowed for input protein sequences,
        which should be of length greater than `lambda_val`.
    weight : float, optional, default=0.05
        The weight factor for the sequence-order correlation features.

    Attributes
    ----------
    np_matrix : np.ndarray
        A 20x21 matrix of normalized physicochemical properties for the 20 standard
        amino acids.
    prop_groups : list of tuple
        List of 7 tuples, each containing indices of 3 properties that form a property
        group.
    """

    _tags = {
        "authors": ["nennomp", "satvshr", "fkiraly"],
        "maintainers": ["nennomp", "satvshr", "fkiraly"],
        "output_type": "numeric",
        "property:fit_is_empty": True,
        "capability:multivariate": False,
    }

    def __init__(self, lambda_val=30, weight=0.05):
        self.lambda_val = lambda_val
        self.weight = weight

        # Load normalized property matrix (20x21, rows=AA, cols=NP1-NP21)
        self.np_matrix = aa_props(type="numpy", normalize=True)
        # Each prop_group is a tuple of 3 columns (property indices)
        self.prop_groups = [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (9, 10, 11),
            (12, 13, 14),
            (15, 16, 17),
            (18, 19, 20),
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
        """Get test parameters for AptaNetPSeAAC."""
        return [
            {"lambda_val": 5, "weight": 0.05},
            {"lambda_val": 20, "weight": 0.1},
        ]
