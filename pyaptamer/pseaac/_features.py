import numpy as np

from pyaptamer.pseaac._props import aa_props
from pyaptamer.utils._pseaac_utils import AMINO_ACIDS, is_valid_aa


class PSeAAC:
    """
    Compute Pseudo Amino Acid Composition (PseAAC) features for a protein sequence.

    This class generates a numerical feature vector that encodes both the composition
    and local order of amino acids in a protein sequence. The features are derived from
    selected physicochemical properties and sequence-order correlations as described in
    the PseAAC model by Chou.

    The PSeAAC algorithm uses normalized physicochemical (NP) properties of amino
    acids, which we load from a predefined matrix using `aa_props`. These properties
    can be grouped using one of the following options:

    - If `custom_groups` is provided, it overrides all other logic. It must be a list of
    lists, where each sublist contains indices (0-based) of properties to group
    together.
    - If `group_props` is an integer, the selected properties are grouped into chunks of
    that size (e.g., `group_props=3` groups into sets of 3).
    - If `group_props` is None, then each property is treated as its own group.


    The properties in order are:


    0. Hydrophobicity
    1. Hydrophilicity
    2. Side-chain Mass
    3. Polarity
    4. Molecular Weight
    5. Melting Point
    6. Transfer Free Energy
    7. Buriability
    8. Bulkiness
    9. Solvation Free Energy
    10. Relative Mutability
    11. Residue Volume
    12. Volume
    13. Amino Acid Distribution
    14. Hydration Number
    15. Isoelectric Point
    16. Compressibility
    17. Chromatographic Index
    18. Unfolding Entropy Change
    19. Unfolding Enthalpy Change
    20. Unfolding Gibbs Free Energy Change


    Each feature vector consists of:

    - 20 normalized amino acid composition features (frequency of each standard
    amino acid)
    - 30 sequence-order correlation features based on physicochemical similarity between
    residues.
    These 50 features are computed for each of the defined property groups,
    resulting in a final vector of length 50 × number of property groups.
    For example, if there are 7 groups, the output length is 350 ((20 + 30) * 7 = 350).

    References
    ----------
    Shen HB, Chou KC. PseAAC: a flexible web server for generating various kinds of
    protein pseudo amino acid composition. Anal Biochem. 2008 Feb 15;373(2):386-8.
    doi: 10.1016/j.ab.2007.10.012. Epub 2007 Oct 13. PMID: 17976365.

    Parameters
    ----------
    prop_indices : list of int, optional
        List of property indices to use (0-based). If None, all 21 are used.
    group_props : int or None, optional
        Group properties into chunks of this size. If None, no grouping
        (each property is its own group).
    custom_groups : list of list of int, optional
        Explicit custom groupings of property indices. Overrides group_props logic.

    Example
    -------
    >>> pse = PSeAAC()
    >>> features = pse.transform("ACDEFGHIKLMNPQRSTVWY")
    >>> print(features[:10])
    np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])

    Methods
    -------
    transform(protein_sequence)
        Generate the PseAAC feature vector for the given protein sequence.
    """

    def __init__(self, prop_indices=None, group_props=None, custom_groups=None):
        indices = list(range(21)) if prop_indices is None else prop_indices

        if custom_groups:
            self.prop_groups = custom_groups
        elif group_props is None:
            # No grouping; each property becomes its own group
            self.prop_groups = [[i] for i in indices]
        else:
            if len(indices) % group_props != 0:
                raise ValueError(
                    f"Number of properties ({len(indices)}) must be divisible by "
                    f"group_props ({group_props})."
                )
            self.prop_groups = [
                indices[i : i + group_props]
                for i in range(0, len(indices), group_props)
            ]

        # Load only needed properties from aa_props
        self.np_matrix = aa_props(list_props=indices, type="numpy", normalize=True)
        self.index_map = {idx: i for i, idx in enumerate(indices)}
        # Remap property groups to local indices
        self.prop_groups = [
            [self.index_map[i] for i in group] for group in self.prop_groups
        ]

    # Function to average the amino acid composition
    def _average_aa(self, seq):
        """
        Compute the average amino acid composition for a sequence.

        Parameters
        ----------
        seq : str
            Protein sequence.

        Returns
        -------
        dict
            Dictionary mapping amino acid to its average frequency.
        """
        from collections import Counter

        counts = Counter(seq)
        total = len(AMINO_ACIDS)
        return {aa: counts.get(aa, 0) / total if total > 0 else 0 for aa in AMINO_ACIDS}

    def _theta_rirj(self, ri, rj, prop_group):
        """
        Compute the theta value between two amino acids for a group of properties.

        Parameters
        ----------
        ri : str
            First amino acid.
        rj : str
            Second amino acid.
        prop_group : tuple of int
            Tuple of property indices.

        Returns
        -------
        float
            Theta value.
        """
        idx_ri = AMINO_ACIDS.index(ri)
        idx_rj = AMINO_ACIDS.index(rj)
        diffs = (
            self.np_matrix[idx_rj, list(prop_group)]
            - self.np_matrix[idx_ri, list(prop_group)]
        )
        return np.mean(diffs**2)

    def _sum_theta_val(self, seq, seq_len, lambda_val, n, prop_group):
        """
        Compute the average theta value for a sequence and property group.

        Parameters
        ----------
        seq : str
            Protein sequence.
        seq_len : int
            Length of the sequence.
        lambda_val : int
            Lambda parameter.
        n : int
            Offset for theta calculation.
        prop_group : tuple of int
            Tuple of property indices.

        Returns
        -------
        float
            Average theta value.
        """
        return sum(
            self._theta_rirj(seq[i], seq[i + n], prop_group)
            for i in range(seq_len - lambda_val)
        ) / (seq_len - n)

    def transform(self, protein_sequence, lambda_val=30, weight=0.15):
        """
        Generate the PseAAC feature vector for the given protein sequence.

        This method computes a set of features based on amino acid composition
        and sequence-order correlations using physicochemical properties, as
        described in the Pseudo Amino Acid Composition (PseAAC) model.

        Parameters
        ----------
        protein_sequence : str
            The input protein sequence consisting of valid amino acid characters
            (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y).
        lambda_val : int, default=30
            The maximum distance between residues considered in the sequence-order
            correlation (θ) calculations.
        weight : float, default=0.15
            The weight factor that balances the contribution of sequence-order
            correlation features relative to amino acid composition features.

        Returns
        -------
        np.ndarray
            A 1D NumPy array of length 50 × number of normalized physicochemical
            (NP) property groups of amino acids. Each 50-element block consists of:
            - 20 normalized amino acid composition features
            - 30 normalized sequence-order correlation factors (theta values)
            For example, if there are 7 property groups, the output length is 350.

        Raises
        ------
        ValueError
            If the sequence contains invalid amino acids or is shorter than
            the required lambda value.
        """
        if not is_valid_aa(protein_sequence):
            raise ValueError(
                "Invalid amino acid found in protein_sequence. Only "
                f"{''.join(AMINO_ACIDS)} are allowed."
            )

        self.lambda_val = lambda_val
        self.weight = weight
        all_pseaac = []

        seq_len = len(protein_sequence)
        if seq_len <= self.lambda_val:
            raise ValueError(
                f"Protein sequence too short for {self.lambda_val}."
                f"Must be > {self.lambda_val}."
            )

        for prop_group in self.prop_groups:
            aa_freq = self._average_aa(protein_sequence)
            sum_all_aa_freq = sum(aa_freq.values())

            all_theta_val = np.array(
                [
                    self._sum_theta_val(
                        protein_sequence, seq_len, self.lambda_val, n, prop_group
                    )
                    for n in range(1, self.lambda_val + 1)
                ]
            )
            sum_all_theta_val = np.sum(all_theta_val)

            denominator_val = sum_all_aa_freq + (self.weight * sum_all_theta_val)

            # First 20 features: normalized amino acid composition
            aa_composition = np.array([aa_freq[aa] for aa in AMINO_ACIDS])
            all_pseaac.extend(np.round(aa_composition / denominator_val, 3))

            # Next 30 features: theta values
            all_pseaac.extend(
                np.round((self.weight * all_theta_val) / denominator_val, 3)
            )

        return np.array(all_pseaac)
