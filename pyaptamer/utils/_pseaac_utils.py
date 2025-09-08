__author__ = "satvshr"
__all__ = ["is_valid_aa"]

"""
Utility functions for amino acid sequence validation in PSeAAC.

This module provides helper functions for checking whether a string consists only of
valid amino acid single-letter codes (the 20 standard amino acids). The list AMINO_ACIDS
is used for efficient membership testing.

Functions
---------
is_valid_aa(seq)
    Returns True if all characters in the input string are valid amino acid
    codes, False otherwise.
"""

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def is_valid_aa(seq):
    """
    Check if the sequence contains only valid amino acids.

    Parameters
    ----------
    seq : str
        Protein sequence.

    Returns
    -------
    bool
        True if all characters are valid amino acids, False otherwise.
    """
    return all(aa in AMINO_ACIDS for aa in seq)
