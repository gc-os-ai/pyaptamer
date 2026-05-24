__author__ = "satvshr"
__all__ = ["clean_protein_seq"]

"""
Utility functions for amino acid sequence validation in PSeAAC.

This module provides helper functions for checking whether a string consists only of
valid amino acid single-letter codes (the 20 standard amino acids). The list AMINO_ACIDS
is used for efficient membership testing.

Functions
---------
clean_protein_seq(seq)
    Remove invalid amino acids from the sequence and warn the user.
"""

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def clean_protein_seq(seq):
    """Remove invalid amino acids from the sequence and warn the user.

    Invalid characters (i.e. characters that are not one of the 20 standard
    amino acid one-letter codes) are stripped from the sequence. A
    ``UserWarning`` is issued whenever at least one character is removed.

    Parameters
    ----------
    seq : str
        Protein sequence.

    Returns
    -------
    str
        Cleaned protein sequence with all invalid characters removed.
    """
    import warnings

    cleaned = []
    invalid_found = False

    for aa in seq:
        if aa in AMINO_ACIDS:
            cleaned.append(aa)
        else:
            invalid_found = True

    if invalid_found:
        warnings.warn(
            "Invalid amino acid(s) found in sequence. "
            "These characters have been removed.",
            UserWarning,
            stacklevel=2,
        )

    return "".join(cleaned)
