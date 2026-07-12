"""Tests for the amino-acid three-letter to one-letter conversion utility.

Targets the missing branch in ``pyaptamer/utils/_aa_str_to_letter.py``
(issue #725): the fallback that maps an unknown three-letter code to "X".
"""

__author__ = ["gandzekas"]

import pytest

from pyaptamer.utils._aa_str_to_letter import aa_str_to_letter


@pytest.mark.parametrize(
    "code, expected",
    [
        ("ALA", "A"),
        ("ala", "A"),  # lowercase is normalised
        ("Trp", "W"),
        ("TYR", "Y"),
        ("SEC", "U"),  # selenocysteine
        ("PYL", "O"),  # pyrrolysine
    ],
)
def test_known_codes(code, expected):
    """Known three-letter codes map to their one-letter equivalents."""
    assert aa_str_to_letter(code) == expected


def test_unknown_code_falls_back_to_x():
    """Unknown three-letter codes return "X" (previously uncovered line)."""
    assert aa_str_to_letter("ZZZ") == "X"
    assert aa_str_to_letter("???") == "X"
    # lowercase unknown is normalised before the lookup
    assert aa_str_to_letter("xyz") == "X"
