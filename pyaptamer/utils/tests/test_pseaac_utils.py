import pytest
from pyaptamer.utils._pseaac_utils import clean_protein_seq


def test_clean_protein_seq_valid():
    assert clean_protein_seq("ACD") == "ACD"


def test_clean_protein_seq_invalid():
    with pytest.warns(UserWarning):
        assert clean_protein_seq("ACDZ") == "ACDX"