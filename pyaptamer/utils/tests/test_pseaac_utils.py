import pytest
from pyaptamer.utils._pseaac_utils import clean_protein_seq


def test_clean_protein_seq_valid():
    seq = "ACDEFGHIKLMNPQRSTVWY"
    result = clean_protein_seq(seq)
    assert result == seq


def test_clean_protein_seq_invalid_replaced_with_X():
    seq = "ABZ*"
    result = clean_protein_seq(seq)
    assert result == "AXXX"


def test_clean_protein_seq_warns():
    seq = "ABZ*"
    with pytest.warns(UserWarning, match="Replaced with 'X'"):
        clean_protein_seq(seq)