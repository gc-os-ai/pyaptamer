"""
Utility functions for amino acid sequence validation in PSeAAC.

This module provides helper functions for checking whether a string consists only of
valid amino acid single-letter codes (the 20 standard amino acids). The set AMINO_ACIDS
is used for efficient membership testing.
"""

import warnings

__author__ = "satvshr"
__all__ = ["clean_protein_seq"]

# Using a set for O(1) membership lookup
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def clean_protein_seq(seq: str) -> str:
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
        with "X" (standard IUPAC code for unknown amino acid).

    Examples
    --------
    >>> from pyaptamer.utils import clean_protein_seq
    >>> print(clean_protein_seq("ACDZE"))
    ACDXE
    """
    cleaned = []
    invalid_found = False

    for aa in seq:
        if aa in AMINO_ACIDS:
            cleaned.append(aa)
        else:
            cleaned.append("X")
            invalid_found = True

    if invalid_found:
        warnings.warn(
            "Invalid amino acid(s) found in sequence. Replaced with 'X'.",
            UserWarning,
            stacklevel=2,
        )

    return "".join(cleaned)