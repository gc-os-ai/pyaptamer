"""Tests for the PSeAAC protein-sequence cleaning utility.

Targets the missing branches in ``pyaptamer/utils/_pseaac_utils.py``
(issue #725): replacing an invalid amino acid with "N" (lines 44-45)
and emitting the UserWarning (line 48).
"""

__author__ = ["gandzekas"]

import warnings

import pytest

from pyaptamer.utils._pseaac_utils import clean_protein_seq


class _AssertNoWarning:
    """Context manager that fails if any warning is emitted."""

    def __enter__(self):
        self._catcher = warnings.catch_warnings(record=True)
        self._records = self._catcher.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(self, *exc):
        self._catcher.__exit__(*exc)
        assert not self._records, f"unexpected warning emitted: {self._records[0]}"


def test_clean_valid_sequence_passthrough():
    """A fully valid sequence is returned unchanged and warns nothing."""
    seq = "ACDEFGHIKLMNPQRSTVWY"
    with _AssertNoWarning():
        result = clean_protein_seq(seq)
    assert result == seq


@pytest.mark.parametrize(
    "seq, expected",
    [
        ("AXC", "ANC"),  # single invalid residue
        ("X", "N"),  # entirely invalid
        ("AC*DE", "ACNDE"),  # stray non-standard char
        ("BZJ", "NNN"),  # ambiguous codes are not among the 20 standard
    ],
)
def test_invalid_residues_replaced_with_n(seq, expected):
    """Invalid residues are replaced with "N" and a warning is raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = clean_protein_seq(seq)

    assert result == expected
    assert any(issubclass(w.category, UserWarning) for w in caught), (
        "expected a UserWarning for invalid residues"
    )
