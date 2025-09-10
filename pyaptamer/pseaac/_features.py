__author__ = ["nennomp", "satvshr"]
__all__ = ["PSeAAC"]

from collections import Counter

import numpy as np

from pyaptamer.pseaac._props import aa_props
from pyaptamer.utils._pseaac_utils import AMINO_ACIDS, clean_protein_seq


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


    - `prop_indices`: A list of property indices (0-based) to use (e.g.,
    `prop_indices=[3, 4, 8]` uses property groups [Polarity, Molecular Weight,
    Bulkiness] (refer table below)).
    - If `group_props` is an integer, the selected properties are grouped into chunks of
    that size (e.g., `group_props=3` groups into sets of 3). If `group_props` is None,
    then each property is treated as its own group.
    - If `custom_groups` is provided, it overrides all other logic. It must be a list of
    lists, where each sublist contains indices (0-based) of properties to group
    together. (e.g., `prop_indices=[[3, 4], [8]]` groups Polarity and Molecular Weight
    together, and Bulkiness is its own group).


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
    - `self.lambda_val` sequence-order correlation features based on physicochemical
    similarity between residues.
    These (20 + `self.lambda_val`) features are computed for each of 7 predefined
    property groups, resulting in a final vector of length (20 + `self.lambda_val`) * 7.

    See `transform` method for usage.

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

    Methods
    -------
    transform(protein_sequence)
        Generate the PseAAC feature vector for the given protein sequence.

    References
    ----------
    Shen HB, Chou KC. PseAAC: a flexible web server for generating various kinds of
    protein pseudo amino acid composition. Anal Biochem. 2008 Feb 15;373(2):386-8.
    doi: 10.1016/j.ab.2007.10.012. Epub 2007 Oct 13. PMID: 17976365.

    Parameters
    ----------
    prop_indices : list of int, optional
        List of property indices to use (0-based). If None, all properties are used.
    group_props : int or None, optional
        Group properties into chunks of this size. If None, no grouping
        (each property is its own group).
    custom_groups : list of list of int, optional
        Explicit custom groupings of property indices. Overrides group_props logic.

    Example
    -------
    >>> from pyaptamer.pseaac import PSeAAC
    >>> pseaac = PSeAAC()
    >>> features = pseaac.transform("ACDEFGHIKLMNPQRHIKLMNPQRSTVWHIKLMNPQRSTVWY")
    >>> print(features[:10])
    [0.006 0.006 0.006 0.006 0.006 0.006 0.018 0.018 0.018 0.018]
    """

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

        self._n_props = aa_props(type="numpy", normalize=True).shape[1]
        self._indices = (
            list(range(self._n_props)) if prop_indices is None else prop_indices
        )

        if custom_groups:
            self.prop_groups = custom_groups
        elif group_props is None:
            # No grouping; each property becomes its own group
            self.prop_groups = [[i] for i in self._indices]
        else:
            if len(self._indices) % group_props != 0:
                raise ValueError(
                    f"Number of properties ({len(self._indices)}) must be divisible by "
                    f"group_props ({group_props})."
                )
            self.prop_groups = [
                self._indices[i : i + group_props]
                for i in range(0, len(self._indices), group_props)
            ]

        # Load only needed properties from aa_props
        self.np_matrix = aa_props(
            list_props=self._indices, type="numpy", normalize=True
        )
        self.index_map = {idx: i for i, idx in enumerate(self._indices)}
        # Remap property groups to local indices
        self.prop_groups = [
            [self.index_map[i] for i in group] for group in self.prop_groups
        ]

    def _normalized_aa(self, seq):
        """
        Compute the normalized amino acid composition for a sequence.

        Parameters
        ----------
        seq : str
            Protein sequence.

        Returns
        -------
        np.ndarray
            A 1D NumPy array of length 20, where each entry corresponds to the frequency
            of a standard amino acid in the input sequence. The order of amino acids is:
            ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
            'S', 'T', 'V', 'W', 'Y']
        """
        counts = Counter(seq)
        total = len(seq)
        return np.array([counts.get(aa, 0) / total for aa in AMINO_ACIDS])

    def _avg_theta_val(self, seq_vec, seq_len, n, prop_group):
        """
        Compute the average theta value for a sequence and property group.

        Parameters
        ----------
        seq_vec : np.ndarray
            Sequence converted to integer indices (shape: [seq_len]).
        seq_len : int
            Length of the sequence.
        n : int
            Offset for theta calculation.
        prop_group : tuple of int
            Tuple of property indices.

        Returns
        -------
        float
            Average theta value.
        """
        props = self.np_matrix[:, prop_group]

        ri = props[seq_vec[: seq_len - n]]
        rj = props[seq_vec[n:]]

        diffs = rj - ri
        return np.mean(diffs**2)

    def transform(self, protein_sequence):
        """
        Generate the PseAAC feature vector for the given protein sequence.

        This method computes a set of features based on amino acid composition
        and sequence-order correlations using physicochemical properties, as
        described in the Pseudo Amino Acid Composition (PseAAC) model. The protein
        sequence should be of length greater than `self.lambda_val`.

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
            A 1D NumPy array of length (20 + `self.lambda_val) * number of normalized
            physiochemical (NP) property groups of amino acids (7).
            Each element consists of:
            - 20 normalized amino acid composition features
            - `self.lambda_val` normalized sequence-order correlation factors (theta
            values)

        Raises
        ------
        ValueError
            If the input sequence contains invalid amino acids or is shorter than
            `self.lambda_val`.
        """
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
