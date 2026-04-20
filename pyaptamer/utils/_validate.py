__author__ = ["github.com/ritankarsaha"]
__all__ = ["validate_sequence"]

from typing import Literal

RNA_ALPHABET: frozenset[str] = frozenset("ACGU")
DNA_ALPHABET: frozenset[str] = frozenset("ACGT")
PROTEIN_ALPHABET: frozenset[str] = frozenset("ACDEFGHIKLMNPQRSTVWY")
SS_ALPHABET: frozenset[str] = frozenset("SHMBIXE")

_ALPHABETS: dict[str, frozenset[str]] = {
    "rna": RNA_ALPHABET,
    "dna": DNA_ALPHABET,
    "protein": PROTEIN_ALPHABET,
    "ss": SS_ALPHABET,
}

SequenceType = Literal["rna", "dna", "protein", "ss"]


def validate_sequence(sequence: str, sequence_type: SequenceType) -> None:
    """Validate a biological sequence against its allowed alphabet.

    Raises a :class:`ValueError` with a clear, actionable message when the
    sequence contains characters outside the expected alphabet, preventing
    silent token corruption that would otherwise propagate into model training.

    Parameters
    ----------
    sequence : str
        The sequence to validate. Characters are case-sensitive; lowercase
        letters are treated as invalid.
    sequence_type : {"rna", "dna", "protein", "ss"}
        The type of the sequence:

        * ``"rna"``     – RNA nucleotides: A, C, G, U
        * ``"dna"``     – DNA nucleotides: A, C, G, T
        * ``"protein"`` – 20 standard amino acids:
                          A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
        * ``"ss"``      – RNA secondary-structure symbols: B, E, H, I, M, S, X

    Raises
    ------
    TypeError
        If *sequence* is not a :class:`str`.
    ValueError
        If *sequence_type* is not one of the recognised types, or if *sequence*
        contains characters outside the allowed alphabet.

    Examples
    --------
    >>> from pyaptamer.utils import validate_sequence
    >>> validate_sequence("ACGU", "rna")  # valid – no exception raised
    >>> validate_sequence("ACGT", "dna")  # valid – no exception raised
    >>> validate_sequence("MKTLL", "protein")  # valid – no exception raised
    >>> validate_sequence("ACGX", "rna")
    Traceback (most recent call last):
        ...
    ValueError: Invalid character(s) 'X' found in RNA sequence.
    Allowed alphabet: A, C, G, U.
    """
    if not isinstance(sequence, str):
        raise TypeError(f"`sequence` must be a str, got {type(sequence).__name__!r}.")

    if sequence_type not in _ALPHABETS:
        valid_types = ", ".join(f"'{t}'" for t in sorted(_ALPHABETS))
        raise ValueError(
            f"`sequence_type` must be one of {valid_types}, got {sequence_type!r}."
        )

    allowed = _ALPHABETS[sequence_type]
    invalid = sorted({char for char in sequence if char not in allowed})

    if invalid:
        invalid_str = ", ".join(f"'{c}'" for c in invalid)
        allowed_str = ", ".join(sorted(allowed))
        raise ValueError(
            f"Invalid character(s) {invalid_str} found in "
            f"{sequence_type.upper()} sequence. "
            f"Allowed alphabet: {allowed_str}."
        )
