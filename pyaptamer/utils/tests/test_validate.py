"""Tests for sequence validation utilities."""

import pytest

from pyaptamer.utils._validate import is_valid_sequence, validate_sequence


class TestValidateSequence:
    """Tests for validate_sequence()."""

    def test_valid_rna(self):
        """Valid RNA sequence passes."""
        assert validate_sequence("AUGCUAGC", "rna") == "AUGCUAGC"

    def test_valid_dna(self):
        """Valid DNA sequence passes."""
        assert validate_sequence("ATGCTAGC", "dna") == "ATGCTAGC"

    def test_valid_protein(self):
        """Valid protein sequence passes."""
        assert validate_sequence("ACDEFGHIKLMNPQRSTVWY", "protein") == "ACDEFGHIKLMNPQRSTVWY"

    def test_case_insensitive(self):
        """Lowercase characters are accepted."""
        assert validate_sequence("augc", "rna") == "augc"

    def test_invalid_rna_with_t(self):
        """T is not valid in RNA."""
        with pytest.raises(ValueError, match="Invalid characters"):
            validate_sequence("AUTGC", "rna")

    def test_invalid_dna_with_u(self):
        """U is not valid in DNA."""
        with pytest.raises(ValueError, match="Invalid characters"):
            validate_sequence("AUGC", "dna")

    def test_invalid_chars_shows_positions(self):
        """Error message includes positions of invalid characters."""
        with pytest.raises(ValueError, match="positions"):
            validate_sequence("AX1C", "rna")

    def test_non_string_input(self):
        """Non-string input raises TypeError."""
        with pytest.raises(TypeError, match="must be a string"):
            validate_sequence(12345, "rna")

    def test_invalid_molecule_type(self):
        """Invalid molecule_type raises ValueError."""
        with pytest.raises(ValueError, match="molecule_type must be"):
            validate_sequence("AUGC", "lipid")

    def test_empty_sequence(self):
        """Empty string is valid (no invalid characters)."""
        assert validate_sequence("", "rna") == ""

    def test_single_valid_char(self):
        """Single valid character passes."""
        assert validate_sequence("A", "rna") == "A"

    def test_protein_invalid_chars(self):
        """Numbers are not valid in protein sequences."""
        with pytest.raises(ValueError, match="Invalid characters"):
            validate_sequence("ACD123", "protein")

    def test_molecule_type_case_insensitive(self):
        """molecule_type should be case-insensitive."""
        assert validate_sequence("AUGC", "RNA") == "AUGC"
        assert validate_sequence("ATGC", "DNA") == "ATGC"


class TestIsValidSequence:
    """Tests for is_valid_sequence()."""

    def test_valid_returns_true(self):
        assert is_valid_sequence("AUGC", "rna") is True

    def test_invalid_returns_false(self):
        assert is_valid_sequence("AXGC", "rna") is False

    def test_non_string_returns_false(self):
        assert is_valid_sequence(None, "rna") is False

    def test_invalid_molecule_type_returns_false(self):
        assert is_valid_sequence("AUGC", "lipid") is False

    def test_dna_valid(self):
        assert is_valid_sequence("ATGC", "dna") is True

    def test_protein_valid(self):
        assert is_valid_sequence("ACDEFGHIKLMNPQRSTVWY", "protein") is True
