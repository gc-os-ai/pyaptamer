"""Test suite for aptatrans utilities."""

__author__ = ["nennomp"]

from itertools import product

import pytest
import torch

from pyaptamer.utils._aptatrans_utils import (
    encode_protein,
    filter_words,
    generate_triplets,
)


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


@pytest.mark.parametrize(
    "sequences, words, max_len, word_max_len, expected",
    [
        # single sequence with exact matches
        (
            "ACD",
            {"A": 1, "C": 2, "D": 3, "AC": 4},
            5,
            3,
            torch.tensor([[4, 3, 0, 0, 0]], dtype=torch.int64),
        ),
        # multiple sequences with padding
        (
            ["ACG", "UGC"],
            {"A": 1, "C": 2, "G": 3, "U": 4, "AC": 5, "UG": 6},
            6,
            3,
            torch.tensor([[5, 3, 0, 0, 0, 0], [6, 2, 0, 0, 0, 0]], dtype=torch.int64),
        ),
        # sequence with truncation
        (
            "ACGUACGU",
            {"A": 1, "C": 2, "G": 3, "U": 4},
            4,
            3,
            torch.tensor([[1, 2, 3, 4]], dtype=torch.int64),
        ),
        # sequence with unknown tokens
        (
            "ACXGU",
            {"A": 1, "C": 2, "G": 3, "U": 4},
            6,
            3,
            torch.tensor([[1, 2, 0, 3, 4, 0]], dtype=torch.int64),
        ),
        # greedy matching preference for longer patterns
        (
            "ACGU",
            {"A": 1, "C": 2, "G": 3, "U": 4, "AC": 5, "ACG": 6, "ACGU": 7},
            3,
            4,
            torch.tensor([[7, 0, 0]], dtype=torch.int64),
        ),
    ],
)
def test_encode_protein(sequences, words, max_len, word_max_len, expected):
    """Check correct encoding of RNA sequences."""
    encoded = encode_protein(sequences, words, max_len, word_max_len)

    # check output type and shape
    assert isinstance(encoded, torch.Tensor)
    assert encoded.dtype == torch.int64
    if isinstance(sequences, str):
        assert encoded.shape == (1, max_len)
    else:
        assert encoded.shape == (len(sequences), max_len)

    # check encoded values match expected
    assert torch.equal(encoded, expected)

    # verify all values are non-negative
    assert (encoded >= 0).all()
