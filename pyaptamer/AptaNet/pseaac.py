import numpy as np
from pyaptamer.AptaNet.utils import is_valid_aa
from pyaptamer.AptaNet._props import (
    NP1, NP2, NP3, NP4, NP5, NP6, NP7, NP8, NP9,
    NP10, NP11, NP12, NP13, NP14, NP15, NP16, NP17, NP18, NP19, NP20, NP21
)

class PSeAAC:
    """
    Compute Pseudo Amino Acid Composition (PseAAC) features for a protein sequence.

    This class generates a feature vector for a given protein sequence using selected
    physicochemical properties and sequence-order information.

    Parameters
    ----------
    None (see `vectorize` method for usage)

    Example
    -------
    >>> pse = PSeAAC()
    >>> features = pse.vectorize("ACDEFGHIKLMNPQRSTVWY")
    >>> print(features[:10])
    [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]

    Methods
    -------
    vectorize(protein_sequence)
        Generate the PseAAC feature vector for the given protein sequence.
    """

    def __init__(self):
        """
        Initialize PSeAAC with a protein sequence.
        """
        self.amino_acid = set("ACDEFGHIKLMNPQRSTVWY")

        # Define 7 selected groups of 3 properties each
        self.prop_groups = [
            (NP1, NP2, NP3),
            (NP4, NP5, NP6),
            (NP7, NP8, NP9),
            (NP10, NP11, NP12),
            (NP13, NP14, NP15),
            (NP16, NP17, NP18),
            (NP19, NP20, NP21),
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
        total = len(self.amino_acid)
        return {aa: counts.get(aa, 0) / total if total > 0 else 0 for aa in self.amino_acid}

    def _theta_RiRj(self, Ri, Rj, prop_group):
        """
        Compute the theta value between two amino acids for a group of properties.

        Parameters
        ----------
        Ri : str
            First amino acid.
        Rj : str
            Second amino acid.
        prop_group : tuple of dict
            Tuple of property dictionaries.

        Returns
        -------
        float
            Theta value.
        """
        diffs = np.array([prop[Rj] - prop[Ri] for prop in prop_group], dtype=float)
        return np.mean(diffs ** 2)

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
        prop_group : tuple of dict
            Tuple of property dictionaries.

        Returns
        -------
        float
            Average theta value.
        """
        return sum(
            self._theta_RiRj(seq[i], seq[i + n], prop_group)
            for i in range(seq_len - lambda_val)
        ) / (seq_len - n)

    def vectorize(self, protein_sequence):
        """
        Generate the PseAAC feature vector for the protein sequence.
        """
        if not is_valid_aa(protein_sequence):
            raise ValueError(
                f"Invalid amino acid found in protein_sequence. Only {''.join(sorted(self.amino_acid))} are allowed."
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

            all_theta_val = np.array([
                self._sum_theta_val(protein_sequence, seq_len, lambda_val, n, prop_group)
                for n in range(1, lambda_val + 1)
            ])
            sum_all_theta_val = np.sum(all_theta_val)

            denominator_val = sum_all_aa_freq + (weight * sum_all_theta_val)
            print(f"Denominator value: {denominator_val}")

            # First 20 features: normalized amino acid composition
            aa_composition = np.array([aa_freq[aa] for aa in self.amino_acid])
            all_pseaac.extend(np.round(aa_composition / denominator_val, 3))

            # Next 30 features: theta values
            all_pseaac.extend(np.round((weight * all_theta_val) / denominator_val, 3))

        return all_pseaac