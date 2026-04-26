__author__ = ["nennomp", "satvshr"]
__all__ = ["PSeAAC"]

from collections import Counter

import numpy as np

from pyaptamer.pseaac._props import aa_props
from pyaptamer.utils._pseaac_utils import AMINO_ACIDS, clean_protein_seq


class PSeAAC:
    """Compute Pseudo Amino Acid Composition features for a protein sequence."""

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

        if group_props is not None and custom_groups is not None:
            raise ValueError(
                "Specify only one of `group_props` or `custom_groups`,not both."
            )

        self.np_matrix = aa_props(
            prop_indices=prop_indices, type="numpy", normalize=True
        )
        self._n_cols = self.np_matrix.shape[1]

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

    def _normalized_aa(self, seq):
        counts = Counter(seq)
        total = len(seq)
        return np.array([counts.get(aa, 0) / total for aa in AMINO_ACIDS])

    def _avg_theta_val(self, seq_vec, seq_len, n, prop_group):
        props = self.np_matrix[:, prop_group]

        ri = props[seq_vec[: seq_len - n]]
        rj = props[seq_vec[n:]]

        diffs = rj - ri
        return np.mean(diffs**2)

    def transform(self, protein_sequence):
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

            all_pseaac.extend(np.round(aa_freq / denominator_val, 3))
            all_pseaac.extend(np.round((self.weight * all_theta_val) / denominator_val, 3))

        return np.array(all_pseaac)

    def get_test_params(self):
        return [
            {"lambda_val": 5, "weight": 0.05},
            {"lambda_val": 10, "prop_indices": list(range(6)), "group_props": 3},
        ]
