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


def aa_str_to_letter(aa_str):
    """Convert a string of amino acid three-letter codes to one-letter codes.

    Parameters
    ----------
    aa_str : str
        A three-letter amino acid code string.

    Returns
    -------
    str
        The corresponding one-letter amino acid code.
    """
    return three_to_one.get(aa_str.upper(), "X")  # Return 'X' for unknown codes
