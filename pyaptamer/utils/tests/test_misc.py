"""Test suite for the generic utiliesi."""

__author__ = ["nennomp"]

from itertools import product

from pyaptamer.utils._misc import generate_triplets


def test_generate_triplets():
    """Check generation of all possible triplets."""
    letters = ["A", "C", "G", "U", "N"]
    triplets = generate_triplets(letters=letters)
    expected_count = len(letters) ** 3  # 5^3 = 125 triplets

    assert isinstance(triplets, dict)
    assert len(triplets) == expected_count

    # check that all combinations are present
    for triplet in product(letters, repeat=3):
        triplet_str = "".join(triplet)
        assert triplet_str in triplets
        assert isinstance(triplets[triplet_str], int)
