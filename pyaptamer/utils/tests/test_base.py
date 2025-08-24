"""Test suite for the base generic utilities."""

__author__ = ["nennomp"]

from itertools import product

import numpy as np
import pytest

from pyaptamer.utils._aptatrans_utils import rna2vec
from pyaptamer.utils._base import dna2rna, filter_words, generate_triplets


@pytest.mark.parametrize(
    "dna, expected_rna",
    [
        ("AAA", "AAA"),
        ("ACG", "ACG"),
        ("AAT", "AAU"),
        ("TTT", "UUU"),
        ("AAX", "AAN"),
        ("XXX", "NNN"),
    ],
)
def test_dna2rna(dna, expected_rna):
    """Check conversion of DNA to RNA nucleotides."""
    assert dna2rna(dna) == expected_rna


def test_dna2rna_edge_cases():
    """Check edge cases of DNA to RNA conversion."""
    # empty sequence
    assert dna2rna("") == ""
    # mixed lowercase/uppercase
    assert dna2rna("aAtT") == "NANU"
    assert dna2rna("AcGt") == "ANGN"


def test_rna2vec():
    """Check conversion of RNA sequences."""
    # test sequences with known outcomes
    sequences = ["AAAA", "ACGT", "ACGU", "GGGN"]
    result = rna2vec(sequences, max_sequence_length=275)

    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(sequences)
    assert result.shape[1] == 275  # default `max_sequence_length`

    letters = ["A", "C", "G", "U", "N"]
    triplets = {}
    for idx, triplet in enumerate(product(letters, repeat=3)):
        triplets["".join(triplet)] = idx + 1

    # 'AAAA' -> triplets: 'AAA'
    expected_aaa = triplets["AAA"]
    assert result[0][0] == expected_aaa
    assert result[0][1] == expected_aaa
    assert np.all(result[0][2:] == 0)  # rest should be padding

    # 'ACGT' -> 'ACGU' -> triplets: 'ACG', 'CGU'
    expected_acg = triplets["ACG"]
    expected_cgu = triplets["CGU"]
    assert result[1][0] == expected_acg
    assert result[1][1] == expected_cgu
    assert np.all(result[1][2:] == 0)  # rest should be padding

    # 'ACGU' -> triplets: 'ACG', 'CGU'
    assert result[2][0] == expected_acg
    assert result[2][1] == expected_cgu
    assert np.all(result[2][2:] == 0)  # rest should be padding

    # 'GGGX' -> 'GGGN' -> triplets: 'GGG', 'GGN'
    expected_ggg_index = triplets["GGG"]
    expected_ggn_index = triplets["GGN"]
    assert result[3][0] == expected_ggg_index
    assert result[3][1] == expected_ggn_index
    assert np.all(result[3][2:] == 0)  # rest should be padding


def test_rna2vec_edge_cases():
    """Check edge cases for RNA to vector conversion."""
    # `max_sequence_length` is <= 0
    with pytest.raises(ValueError):
        rna2vec(["ACGU"], max_sequence_length=0)
    with pytest.raises(ValueError):
        rna2vec(["ACGU"], max_sequence_length=-1)

    # `max_sequence_length` is smallet than the number of triplets
    result = rna2vec(["AAACGU"], max_sequence_length=4)
    assert result.shape[1] == 4  # should truncate to 4 triplets

    # empty sequence
    result = rna2vec([""])
    assert len(result) == 0

    # single character sequence (can't form triplet)
    result = rna2vec(["A"])
    assert len(result) == 0

    # double character sequence (can't form triplet)
    result = rna2vec(["AA"])
    assert len(result) == 0


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
