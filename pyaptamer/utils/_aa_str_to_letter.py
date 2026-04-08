"""Utility to convert amino acid three-letter codes to one-letter codes."""

three_to_one = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
    "SEC": "U",  # Selenocysteine
    "PYL": "O",  # Pyrrolysine
    # Add more mappings if necessary
}

def aa_str_to_letter(aa_str: str) -> str:
    """Convert a single three-letter amino acid code to its one-letter code.

    Parameters
    ----------
    aa_str : str
        A single three-letter amino acid code (e.g., 'ALA').

    Returns
    -------
    str
        The corresponding one-letter amino acid code. Returns 'X' if the
        input code is not found in the mapping.

    Examples
    --------
    >>> from pyaptamer.utils import aa_str_to_letter
    >>> print(aa_str_to_letter("ALA"))
    A
    >>> print(aa_str_to_letter("XYZ"))
    X
    """
    return three_to_one.get(aa_str.upper(), "X")