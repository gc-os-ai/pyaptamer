__author__ = "satvshr"
__all__ = ["is_valid_aa"]

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
