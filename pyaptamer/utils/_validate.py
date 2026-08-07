"""Sequence validation utilities for DNA, RNA, and protein sequences."""

__author__ = ["Vaishnav88sk"]
__all__ = ["validate_sequence", "is_valid_sequence"]

_VALID_CHARS = {
    "dna": set("ACGTacgt"),
    "rna": set("ACGUacgu"),
    "protein": set("ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy"),
}

_MOLECULE_TYPES = tuple(_VALID_CHARS.keys())


def validate_sequence(sequence, molecule_type="rna"):
    """Validate that a sequence contains only valid characters.

    Checks every character in ``sequence`` against the allowed alphabet
    for the given ``molecule_type`` and raises a clear error listing
    invalid characters and their positions.

    Parameters
    ----------
    sequence : str
        The nucleotide or amino-acid sequence to validate.
    molecule_type : str, optional, default="rna"
        Type of molecule. Must be one of ``"dna"``, ``"rna"``, or
        ``"protein"``.

    Returns
    -------
    str
        The original sequence (unchanged) if validation passes.

    Raises
    ------
    TypeError
        If ``sequence`` is not a string.
    ValueError
        If ``molecule_type`` is not one of the supported types.
    ValueError
        If ``sequence`` contains characters not in the allowed alphabet.

    Examples
    --------
    >>> from pyaptamer.utils._validate import validate_sequence
    >>> validate_sequence("AUGCUAGC", molecule_type="rna")
    'AUGCUAGC'
    >>> validate_sequence("ATGCTAGC", molecule_type="dna")
    'ATGCTAGC'
    """
    if not isinstance(sequence, str):
        raise TypeError(
            f"sequence must be a string, got {type(sequence).__name__}."
        )

    molecule_type = molecule_type.lower()
    if molecule_type not in _VALID_CHARS:
        raise ValueError(
            f"molecule_type must be one of {_MOLECULE_TYPES}, "
            f"got '{molecule_type}'."
        )

    valid = _VALID_CHARS[molecule_type]
    invalid_positions = []
    invalid_chars = []

    for i, char in enumerate(sequence):
        if char not in valid:
            invalid_positions.append(i)
            invalid_chars.append(char)

    if invalid_chars:
        unique_invalid = sorted(set(invalid_chars))
        raise ValueError(
            f"Invalid characters {unique_invalid} found at positions "
            f"{invalid_positions} in {molecule_type.upper()} sequence. "
            f"Allowed characters: {''.join(sorted(valid & set('ACDEFGHIKLMNPQRSTUVWY')))}"
        )

    return sequence


def is_valid_sequence(sequence, molecule_type="rna"):
    """Check whether a sequence contains only valid characters.

    This is a non-raising alternative to :func:`validate_sequence`.

    Parameters
    ----------
    sequence : str
        The nucleotide or amino-acid sequence to check.
    molecule_type : str, optional, default="rna"
        Type of molecule. Must be one of ``"dna"``, ``"rna"``, or
        ``"protein"``.

    Returns
    -------
    bool
        ``True`` if every character in ``sequence`` belongs to the
        allowed alphabet, ``False`` otherwise.

    Examples
    --------
    >>> from pyaptamer.utils._validate import is_valid_sequence
    >>> is_valid_sequence("AUGC", molecule_type="rna")
    True
    >>> is_valid_sequence("ATXG", molecule_type="dna")
    False
    """
    try:
        validate_sequence(sequence, molecule_type)
        return True
    except (TypeError, ValueError):
        return False
