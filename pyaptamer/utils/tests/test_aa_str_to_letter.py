"""Tests for aa_str_to_letter."""

import pytest

from pyaptamer.utils._aa_str_to_letter import aa_str_to_letter


@pytest.mark.parametrize(
    "aa_str, expected",
    [
        ("ALA", "A"),
        ("cys", "C"),  # lowercase input, exercises .upper()
        ("XYZ", "X"),  # unknown code falls back to "X"
    ],
)
def test_aa_str_to_letter(aa_str, expected):
    """Check known codes, lowercase input, and unknown-code fallback."""
    assert aa_str_to_letter(aa_str) == expected
