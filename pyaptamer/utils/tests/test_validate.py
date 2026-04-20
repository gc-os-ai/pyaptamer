"""Test suite for the validate_sequence utility."""

__author__ = ["github.com/ritankarsaha"]

import pytest

from pyaptamer.utils import validate_sequence
from pyaptamer.utils._validate import (
    DNA_ALPHABET,
    PROTEIN_ALPHABET,
    RNA_ALPHABET,
    SS_ALPHABET,
)


def test_rna_alphabet_contents():
    assert RNA_ALPHABET == frozenset("ACGU")


def test_dna_alphabet_contents():
    assert DNA_ALPHABET == frozenset("ACGT")


def test_protein_alphabet_contents():
    assert PROTEIN_ALPHABET == frozenset("ACDEFGHIKLMNPQRSTVWY")
    assert len(PROTEIN_ALPHABET) == 20


def test_ss_alphabet_contents():
    assert SS_ALPHABET == frozenset("SHMBIXE")


@pytest.mark.parametrize(
    "sequence",
    [
        "ACGU",
        "AAAA",
        "UUUU",
        "ACGACGU",
        "",
    ],
)
def test_valid_rna_sequences(sequence):
    validate_sequence(sequence, "rna")


@pytest.mark.parametrize(
    "sequence",
    [
        "ACGT",
        "AAAA",
        "TTTT",
        "ACGACGT",
        "",
    ],
)
def test_valid_dna_sequences(sequence):
    validate_sequence(sequence, "dna")


@pytest.mark.parametrize(
    "sequence",
    [
        "ACDEFGHIKLMNPQRSTVWY",
        "MKTLL",
        "ACGT",
        "AAAA",
        "",
    ],
)
def test_valid_protein_sequences(sequence):
    validate_sequence(sequence, "protein")


@pytest.mark.parametrize(
    "sequence",
    [
        "SHMBIXE",
        "SSSS",
        "HHHH",
        "SSHHM",
        "",
    ],
)
def test_valid_ss_sequences(sequence):
    validate_sequence(sequence, "ss")


@pytest.mark.parametrize(
    "sequence, sequence_type, invalid_chars",
    [
        ("ACGX", "rna", ["X"]),
        ("ACGT", "rna", ["T"]),  # T is DNA-only; not in RNA alphabet
        ("ACGU", "dna", ["U"]),  # U is RNA-only; not in DNA alphabet
        ("ACGX", "dna", ["X"]),
        ("MKTLLZ", "protein", ["Z"]),
        ("MKT1L", "protein", ["1"]),
        ("SSHHQ", "ss", ["Q"]),
        ("ACGXYZ", "rna", ["X", "Y", "Z"]),
    ],
)
def test_invalid_characters_raise_value_error(sequence, sequence_type, invalid_chars):
    with pytest.raises(ValueError) as exc_info:
        validate_sequence(sequence, sequence_type)
    error_msg = str(exc_info.value)
    for char in invalid_chars:
        assert f"'{char}'" in error_msg


def test_error_message_contains_allowed_alphabet():
    with pytest.raises(ValueError, match="Allowed alphabet"):
        validate_sequence("ACGX", "rna")


def test_error_message_contains_sequence_type():
    with pytest.raises(ValueError, match="RNA"):
        validate_sequence("ACGX", "rna")
    with pytest.raises(ValueError, match="DNA"):
        validate_sequence("ACGU", "dna")
    with pytest.raises(ValueError, match="PROTEIN"):
        validate_sequence("MKTLLZ", "protein")
    with pytest.raises(ValueError, match="SS"):
        validate_sequence("SSHHQ", "ss")


@pytest.mark.parametrize(
    "sequence, sequence_type",
    [
        ("acgu", "rna"),
        ("acgt", "dna"),
        ("mktll", "protein"),
        ("sshh", "ss"),
    ],
)
def test_lowercase_raises_value_error(sequence, sequence_type):
    with pytest.raises(ValueError):
        validate_sequence(sequence, sequence_type)


@pytest.mark.parametrize(
    "bad_input",
    [
        123,
        ["A", "C", "G", "U"],
        None,
        b"ACGU",
    ],
)
def test_non_string_sequence_raises_type_error(bad_input):
    with pytest.raises(TypeError, match="`sequence` must be a str"):
        validate_sequence(bad_input, "rna")


@pytest.mark.parametrize(
    "bad_type",
    ["nucleotide", "amino_acid", "RNA", "DNA", "", "mrna"],
)
def test_invalid_sequence_type_raises_value_error(bad_type):
    with pytest.raises(ValueError, match="`sequence_type` must be one of"):
        validate_sequence("ACGU", bad_type)


def test_rna2vec_raises_on_invalid_chars():
    from pyaptamer.utils import rna2vec

    with pytest.raises(ValueError, match="Invalid character"):
        rna2vec(["ACGX"], sequence_type="rna")


def test_rna2vec_accepts_dna_input():
    """rna2vec converts DNA to RNA internally; T is a valid input character."""
    from pyaptamer.utils import rna2vec

    result = rna2vec(["ACGT"], sequence_type="rna", max_sequence_length=4)
    assert result.shape[0] == 1


def test_rna2vec_accepts_unknown_nucleotide_n():
    """N is a standard unknown-nucleotide placeholder and must remain accepted."""
    from pyaptamer.utils import rna2vec

    result = rna2vec(["ACGN"], sequence_type="rna", max_sequence_length=4)
    assert result.shape[0] == 1


def test_rna2vec_raises_on_invalid_ss_chars():
    from pyaptamer.utils import rna2vec

    with pytest.raises(ValueError, match="Invalid character"):
        rna2vec(["SSHHZ"], sequence_type="ss")


@pytest.mark.parametrize(
    "sequence",
    ["ACGXYZ", "123456", "ACGU!", "ACG U"],
)
def test_rna2vec_raises_on_various_invalid_inputs(sequence):
    from pyaptamer.utils import rna2vec

    with pytest.raises(ValueError):
        rna2vec([sequence], sequence_type="rna")
