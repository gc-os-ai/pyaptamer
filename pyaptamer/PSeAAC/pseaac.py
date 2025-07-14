import numpy as np

from pyaptamer.PSeAAC._props import aa_props
from pyaptamer.utils import AMINO_ACIDS, is_valid_aa


class PSeAAC:
    """
    Compute Pseudo Amino Acid Composition (PseAAC) features for a protein sequence.

    This class generates a feature vector for a given protein sequence using selected
    physicochemical properties and sequence-order information.

    Parameters
    ----------
    None (see `transform` method for usage)

    Example
    -------
    >>> pse = PSeAAC()
    >>> features = pse.transform("ACDEFGHIKLMNPQRSTVWY")
    >>> print(features[:10])
    [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]

    Methods
    -------
    transform(protein_sequence)
        Generate the PseAAC feature vector for the given protein sequence.
    """

    def __init__(self):
        """
        Initialize PSeAAC with a protein sequence.
        """

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

    def transform(self, protein_sequence):
        """
        Generate the PseAAC feature vector for the protein sequence.
        """
        if not is_valid_aa(protein_sequence):
            raise ValueError(
                "Invalid amino acid found in protein_sequence. Only "
                f"{''.join(AMINO_ACIDS)} are allowed."
            )

        lambda_val = 30
        weight = 0.15
        all_pseaac = []

        seq_len = len(protein_sequence)
        if seq_len <= lambda_val:
            raise ValueError(
                f"Sequence too short for Lambda={lambda_val}. Must be > {lambda_val}."
            )

        for prop_group in self.prop_groups:
            aa_freq = self._average_aa(protein_sequence)
            sum_all_aa_freq = sum(aa_freq.values())

            all_theta_val = np.array(
                [
                    self._sum_theta_val(
                        protein_sequence, seq_len, lambda_val, n, prop_group
                    )
                    for n in range(1, lambda_val + 1)
                ]
            )
            sum_all_theta_val = np.sum(all_theta_val)

            denominator_val = sum_all_aa_freq + (weight * sum_all_theta_val)

            # First 20 features: normalized amino acid composition
            aa_composition = np.array([aa_freq[aa] for aa in AMINO_ACIDS])
            all_pseaac.extend(np.round(aa_composition / denominator_val, 3))

            # Next 30 features: theta values
            all_pseaac.extend(np.round((weight * all_theta_val) / denominator_val, 3))

        return all_pseaac
