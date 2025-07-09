AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

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