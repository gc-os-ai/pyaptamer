"""Tests for clean_protein_seq."""

import pytest

from pyaptamer.utils._pseaac_utils import clean_protein_seq


def test_clean_protein_seq_valid_sequence_no_warning(recwarn):
    result = clean_protein_seq("ACDEFG")
    assert result == "ACDEFG"
    assert len(recwarn) == 0


def test_clean_protein_seq_invalid_chars_replaced_and_warns():
    with pytest.warns(UserWarning, match="Invalid amino acid"):
        result = clean_protein_seq("ACXZ1")
    assert result == "ACNNN"
