__author__ = ["nennomp", "satvshr"]
__all__ = ["AptaNetPSeAAC"]

# from collections import Counter

# import numpy as np

# from pyaptamer.pseaac._props import aa_props
# from pyaptamer.utils._pseaac_utils import AMINO_ACIDS, clean_protein_seq
from pyaptamer.pseaac._pseaac_general import PSeAAC

class AptaNetPSeAAC(PSeAAC):
    """
    Compute Pseudo Amino Acid Composition (PseAAC) features for a protein sequence,
    using the fixed 21-property, 7-group configuration as used in AptaNet.

    This is a thin subclass of `PSeAAC` with fixed physicochemical property
    configuration matching the original AptaNet implementation. All 21 properties
    are used, grouped into 7 groups of 3.

    For full documentation see `PSeAAC`.

    Parameters
    ----------
    lambda_val : int, optional, default=30
        The lambda parameter defining the number of sequence-order correlation factors.
        This also determines the minimum length allowed for input protein sequences,
        which should be of length greater than `lambda_val`.
    weight : float, optional, default=0.05
        The weight factor for the sequence-order correlation features.

    Example
    -------
    >>> from pyaptamer.pseaac import AptaNetPSeAAC
    >>> pseaac = AptaNetPSeAAC()
    >>> features = pseaac.transform("ACDEFGHIKLMNPQRHIKLMNPQRSTVWHIKLMNPQRSTVWY")
    >>> print(features[:10])
    [0.006 0.006 0.006 0.006 0.006 0.006 0.018 0.018 0.018 0.018]
    """

    def __init__(self, lambda_val=30, weight=0.05):
        super().__init__(
            lambda_val = lambda_val,
            weight = weight,
            prop_indices=None,
            group_props=3,
            custom_groups=None,
        )

        # Load normalized property matrix (20x21, rows=AA, cols=NP1-NP21)
        # self.np_matrix = aa_props(type="numpy", normalize=True)
        # Each prop_group is a tuple of 3 columns (property indices)
        # self.prop_groups = [
        #     (0, 1, 2),
        #     (3, 4, 5),
        #     (6, 7, 8),
        #     (9, 10, 11),
        #     (12, 13, 14),
        #     (15, 16, 17),
        #     (18, 19, 20),
        # ]

    # def _normalized_aa(self, seq):
    #     """
    #     Compute the normalized amino acid composition for a sequence.

    #     Parameters
    #     ----------
    #     seq : str
    #         Protein sequence.

    #     Returns
    #     -------
    #     np.ndarray
    #         A 1D NumPy array of length 20, where each entry corresponds to the frequency
    #         of a standard amino acid in the input sequence. The order of amino acids is:
    #         ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
    #         'S', 'T', 'V', 'W', 'Y']
    #     """
    #     counts = Counter(seq)
    #     total = len(seq)
    #     return np.array([counts.get(aa, 0) / total for aa in AMINO_ACIDS])

    # def _avg_theta_val(self, seq_vec, seq_len, n, prop_group):
    #     """
    #     Compute the average theta value for a sequence and property group.

    #     Parameters
    #     ----------
    #     seq_vec : np.ndarray
    #         Sequence converted to integer indices (shape: [seq_len]).
    #     seq_len : int
    #         Length of the sequence.
    #     n : int
    #         Offset for theta calculation.
    #     prop_group : tuple of int
    #         Tuple of property indices.

    #     Returns
    #     -------
    #     float
    #         Average theta value.
    #     """
    #     props = self.np_matrix[:, prop_group]

    #     ri = props[seq_vec[: seq_len - n]]
    #     rj = props[seq_vec[n:]]

    #     diffs = rj - ri
    #     return np.mean(diffs**2)

    # def transform(self, protein_sequence):
    #     """
    #     Generate the PseAAC feature vector for the given protein sequence.

    #     This method computes a set of features based on amino acid composition
    #     and sequence-order correlations using physicochemical properties, as
    #     described in the Pseudo Amino Acid Composition (PseAAC) model. The protein
    #     sequence should be of length greater than `self.lambda_val`.

    #     Parameters
    #     ----------
    #     protein_sequence : str
    #         The input protein sequence consisting of valid amino acid characters
    #         (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y).

    #     Returns
    #     -------
    #     np.ndarray
    #         A 1D NumPy array of length (20 + `self.lambda_val) * number of normalized
    #         physiochemical (NP) property groups of amino acids (7).
    #         Each element consists of:
    #         - 20 normalized amino acid composition features
    #         - `self.lambda_val` normalized sequence-order correlation factors (theta
    #         values)

    #     Raises
    #     ------
    #     ValueError
    #         If the input sequence contains invalid amino acids or is shorter than
    #         `self.lambda_val`.
    #     """
    #     seq = clean_protein_seq(protein_sequence)
    #     seq_len = len(seq)
    #     if seq_len <= self.lambda_val:
    #         raise ValueError(
    #             f"Protein sequence is too short, should be longer than `lambda_val`. "
    #             f"Sequence length: {seq_len}, `lambda_val`: {self.lambda_val}."
    #         )

    #     aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    #     seq_vec = np.array([aa_to_idx[aa] for aa in seq], dtype=np.int32)

    #     aa_freq = self._normalized_aa(seq)
    #     sum_all_aa_freq = aa_freq.sum()

    #     all_pseaac = []
    #     for prop_group in self.prop_groups:
    #         all_theta_val = np.array(
    #             [
    #                 self._avg_theta_val(seq_vec, seq_len, n, prop_group)
    #                 for n in range(1, self.lambda_val + 1)
    #             ]
    #         )

    #         sum_all_theta_val = np.sum(all_theta_val)
    #         denominator_val = sum_all_aa_freq + (self.weight * sum_all_theta_val)

    #         # First 20 features: normalized amino acid composition
    #         all_pseaac.extend(np.round(aa_freq / denominator_val, 3))

    #         # Next `self.lambda_val` features: theta values
    #         all_pseaac.extend(
    #             np.round((self.weight * all_theta_val) / denominator_val, 3)
    #         )

    #     return np.array(all_pseaac)
