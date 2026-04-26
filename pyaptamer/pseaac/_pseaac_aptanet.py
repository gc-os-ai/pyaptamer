__author__ = ["nennomp", "satvshr"]
__all__ = ["AptaNetPSeAAC"]

from collections import Counter

import numpy as np

from pyaptamer.pseaac._props import aa_props
from pyaptamer.utils._pseaac_utils import AMINO_ACIDS, clean_protein_seq


class AptaNetPSeAAC:
    """Compute Pseudo Amino Acid Composition features for a protein sequence."""

    def __init__(self, lambda_val=30, weight=0.05):
        self.lambda_val = lambda_val
        self.weight = weight

        self.np_matrix = aa_props(type="numpy", normalize=True)
        self.prop_groups = [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (9, 10, 11),
            (12, 13, 14),
            (15, 16, 17),
            (18, 19, 20),
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
            {"lambda_val": 20, "weight": 0.1},
        ]
