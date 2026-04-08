from pyaptamer.utils._aa_str_to_letter import aa_str_to_letter


def test_aa_str_to_letter_valid():
    assert aa_str_to_letter("ALA") == "A"
    assert aa_str_to_letter("ala") == "A"


def test_aa_str_to_letter_unknown():
    assert aa_str_to_letter("XYZ") == "X"
    assert aa_str_to_letter("") == "X"