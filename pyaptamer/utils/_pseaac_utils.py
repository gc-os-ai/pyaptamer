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
    Replaces invalid amino acids with "N" and warn the user.
"""

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def clean_protein_seq(seq):
    """
    Normalize valid amino acids to uppercase.

    Invalid amino acids are replaced with "N" and trigger a warning.

    Parameters
    ----------
    seq : str
        Protein sequence.

    Returns
    -------
    str
        Cleaned protein sequence where valid amino acids have been uppercased and
        invalid characters have been replaced with "N".
    """
    import warnings

    cleaned = []
    invalid_found = False

    for aa in seq.upper():
        if aa in AMINO_ACIDS:
            cleaned.append(aa)
        else:
            cleaned.append("N")
            invalid_found = True

    if invalid_found:
        warnings.warn(
            "Invalid amino acid(s) found in sequence. Replaced with 'N'.",
            UserWarning,
            stacklevel=2,
        )

    return "".join(cleaned)
