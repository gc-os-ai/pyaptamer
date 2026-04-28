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
    Replaces invalid amino acids with "X" and warn the user.
"""

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
UNKNOWN_AMINO_ACID = "X"


def clean_protein_seq(seq):
    """
    Replace invalid amino acids with "X" and warn the user.

    Parameters
    ----------
    seq : str
        Protein sequence.

    Returns
    -------
    str
        Cleaned protein sequence where all invalid characters have been replaced
        with "X".
    """
    import warnings

    cleaned = []
    invalid_found = False

    for aa in seq:
        if aa in AMINO_ACIDS or aa == UNKNOWN_AMINO_ACID:
            cleaned.append(aa)
        else:
            cleaned.append(UNKNOWN_AMINO_ACID)
            invalid_found = True

    if invalid_found:
        warnings.warn(
            "Invalid amino acid(s) found in sequence. Replaced with 'X'.",
            UserWarning,
            stacklevel=2,
        )

    return "".join(cleaned)
