"""Test suite for the generic utiliesi."""

__author__ = ["nennomp"]

from itertools import product

from pyaptamer.utils._base import filter_words, generate_triplets


def test_filter_words_basic_filtering():
    """Test filter_words with basic filtering logic."""
    # mean = (5 + 2 + 8 + 1) / 4 = 4.0
    # words above mean: apple (5.0), cherry (8.0)
    words = {"apple": 5.0, "banana": 2.0, "cherry": 8.0, "date": 1.0}

    result = filter_words(words)

    expected = {"apple": 1, "cherry": 2}
    assert result == expected


def test_filter_words_all_below_mean():
    """Test filter_words when all words are below the mean."""
    # mean = 1.0, no words above mean
    words = {"word1": 1.0, "word2": 1.0, "word3": 1.0}

    result = filter_words(words)

    assert result == {}


def test_filter_words_preserves_order():
    """Test filter_words preserves the order of words."""
    # mean = (10 + 8 + 6 + 2) / 4 = 6.5
    # words above mean: zebra (10.0), alpha (8.0)
    words = {"zebra": 10.0, "alpha": 8.0, "beta": 6.0, "gamma": 2.0}

    result = filter_words(words)

    # indices should reflect order of iteration over dict
    expected = {"zebra": 1, "alpha": 2}
    assert result == expected


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
