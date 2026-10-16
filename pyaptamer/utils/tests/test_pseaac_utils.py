"""Tests for clean_protein_seq."""

import warnings

import pytest

from pyaptamer.utils._pseaac_utils import clean_protein_seq


def test_clean_protein_seq_valid_sequence_no_warning(recwarn):
    """Check a valid sequence is returned unchanged and raises no warning."""
    result = clean_protein_seq("ACDEFG")
    assert result == "ACDEFG"
    assert len(recwarn) == 0


def test_clean_protein_seq_invalid_chars_replaced_and_warns():
    """Check invalid characters are replaced with 'N' and a UserWarning is raised."""
    with pytest.warns(UserWarning, match="Invalid amino acid"):
        result = clean_protein_seq("ACXZ1")
    assert result == "ACNNN"


def test_clean_protein_seq_normalizes_lowercase_without_warning():
    """Check an all-lowercase valid sequence is uppercased without warning."""
    seq = "acdefghiklmnpqrstvwy"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cleaned = clean_protein_seq(seq)
    assert cleaned == seq.upper()
    assert caught == []


def test_clean_protein_seq_warns_only_for_truly_invalid_residues():
    """Check only truly invalid residues trigger a warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cleaned = clean_protein_seq("Acd?z")
    assert cleaned == "ACDNN"
    assert len(caught) == 1
    assert "Invalid amino acid(s) found in sequence" in str(caught[0].message)
