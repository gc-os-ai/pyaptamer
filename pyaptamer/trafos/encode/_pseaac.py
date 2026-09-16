"""Pseudo amino acid composition of protein sequences."""

__author__ = ["nennomp", "satvshr"]
__all__ = ["PSeAAC"]

from collections import Counter

import numpy as np
import pandas as pd

from pyaptamer.trafos.base import BaseTransform
from pyaptamer.trafos.encode._pseaac_props import aa_props
from pyaptamer.utils._pseaac_utils import AMINO_ACIDS, clean_protein_seq


class PSeAAC(BaseTransform):
    """Pseudo Amino Acid Composition (PseAAC) features of protein sequences.

    Encodes each protein sequence as a vector of amino acid composition and
    sequence-order correlation factors, following the PseAAC model of Chou.

    The 21 normalized physicochemical properties of the amino acids are loaded
    with ``aa_props``. They can be selected and grouped in three ways:

    - ``prop_indices``: 0-based indices of the properties to use. ``None``
      selects all 21.
    - ``group_props``: group the selected properties into consecutive chunks
      of this size. ``None`` means chunks of 3 (7 groups for 21 properties).
    - ``custom_groups``: explicit groups as lists of local column indices into
      the selected properties. Overrides ``group_props``.

    The 21 properties, in column order, are:

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

    For each property group the vector holds 20 normalized amino acid
    frequencies followed by ``lambda_val`` sequence-order correlation
    factors, so the output width is ``(20 + lambda_val) * n_groups``.

    Input is a one-column DataFrame or a ``MoleculeLoader`` of protein
    strings. Invalid residues are replaced with ``N`` with a warning; every
    sequence must be longer than ``lambda_val``.

    Parameters
    ----------
    lambda_val : int, default=30
        Number of sequence-order correlation factors per property group.
        Every input sequence must be longer than this.
    weight : float, default=0.05
        Weight of the sequence-order correlation factors.
    prop_indices : list of int or None, default=None
        0-based indices of the properties to use. ``None`` uses all 21.
    group_props : int or None, default=None
        Size of consecutive property groups. ``None`` means 3.
    custom_groups : list of list of int or None, default=None
        Explicit property groups. Overrides ``group_props``.

    References
    ----------
    Shen HB, Chou KC. PseAAC: a flexible web server for generating various
    kinds of protein pseudo amino acid composition. Anal Biochem.
    2008;373(2):386-8. doi: 10.1016/j.ab.2007.10.012.

    Examples
    --------
    >>> import pandas as pd
    >>> from pyaptamer.trafos.encode import PSeAAC
    >>> X = pd.DataFrame({"protein": ["ACDEFGHIKLMNPQRHIKLMNPQRSTVWHIKLMNPQRSTVWY"]})
    >>> PSeAAC().fit_transform(X).shape
    (1, 350)
    >>> PSeAAC(prop_indices=[0, 1, 2, 3, 4, 5], group_props=2).fit_transform(X).shape
    (1, 150)
    """

    _tags = {
        "authors": ["nennomp", "satvshr"],
        "maintainers": ["siddharth7113"],
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
        super().__init__()

    def _transform(self, X):
        """Encode every sequence in the single column of X.

        Parameters
        ----------
        X : pd.DataFrame
            One column of protein strings.

        Returns
        -------
        pd.DataFrame
            Shape ``(len(X), (20 + lambda_val) * n_groups)``, indexed like X.
        """
        np_matrix, prop_groups = self._resolve_groups()
        rows = [self._encode(seq, np_matrix, prop_groups) for seq in X.iloc[:, 0]]
        return pd.DataFrame(np.vstack(rows), index=X.index)

    def _resolve_groups(self):
        """Return the normalized property matrix and the property groups."""
        if self.group_props is not None and self.custom_groups is not None:
            raise ValueError(
                "Specify only one of `group_props` or `custom_groups`, not both."
            )

        np_matrix = aa_props(prop_indices=self.prop_indices).to_numpy()
        n_cols = np_matrix.shape[1]

        if self.custom_groups is not None:
            if len(self.custom_groups) == 0:
                raise ValueError("`custom_groups` must contain at least one group.")
            return np_matrix, self.custom_groups

        if self.group_props is None:
            if n_cols % 3 != 0:
                raise ValueError(
                    "Default grouping expects number of properties divisible by 3."
                )
            size = 3
        else:
            size = self.group_props
            if n_cols % size != 0:
                raise ValueError(
                    f"Number of properties ({n_cols}) must be divisible by "
                    f"group_props ({size})."
                )

        groups = [list(range(i, i + size)) for i in range(0, n_cols, size)]
        return np_matrix, groups

    def _encode(self, protein_sequence, np_matrix, prop_groups):
        """Return the PseAAC vector of one protein sequence."""
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
        for prop_group in prop_groups:
            props = np_matrix[:, prop_group]
            all_theta_val = np.array(
                [
                    self._avg_theta_val(seq_vec, seq_len, n, props)
                    for n in range(1, self.lambda_val + 1)
                ]
            )

            sum_all_theta_val = np.sum(all_theta_val)
            denominator_val = sum_all_aa_freq + (self.weight * sum_all_theta_val)

            all_pseaac.extend(np.round(aa_freq / denominator_val, 3))
            all_pseaac.extend(
                np.round((self.weight * all_theta_val) / denominator_val, 3)
            )

        return np.array(all_pseaac)

    def _normalized_aa(self, seq):
        """Return the frequency of each of the 20 amino acids in seq."""
        counts = Counter(seq)
        total = len(seq)
        return np.array([counts.get(aa, 0) / total for aa in AMINO_ACIDS])

    def _avg_theta_val(self, seq_vec, seq_len, n, props):
        """Return the mean squared property difference at sequence offset n."""
        ri = props[seq_vec[: seq_len - n]]
        rj = props[seq_vec[n:]]
        diffs = rj - ri
        return np.mean(diffs**2)

    @classmethod
    def get_test_params(cls):
        """Return parameter sets for the shared transformer tests."""
        return [{}, {"prop_indices": [0, 1, 2, 3, 4, 5], "group_props": 2}]
