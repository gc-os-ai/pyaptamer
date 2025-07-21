"""Test suite for the RNA to vector (rna2vec) conversion utilities."""

__author__ = ["nennomp"]

from itertools import product

import numpy as np

from pyaptamer.data.rna2vec_utils import dna2rna, rna2vec, word2idx


def test_word2idx_exists():
    """Check word to index when the word exists in the dictionary."""
    dummy_words = {"AAA": 1, "AAC": 2, "AAG": 3}
    assert word2idx("AAA", dummy_words) == 1
    assert word2idx("AAC", dummy_words) == 2
    assert word2idx("AAG", dummy_words) == 3


def test_word2idx_not_exists():
    """Check word to index when the word does not exist in the dictionary."""
    dummy_words = {"AAA": 1, "AAC": 2, "AAG": 3}

    # check that 0 is returned for missing words
    assert word2idx("BBB", dummy_words) == 0
    assert word2idx("CCC", dummy_words) == 0
    assert word2idx("GGG", dummy_words) == 0


def test_word2idx_empty_dictionary():
    """Check behavior with empty dictionary."""
    dummy_words = {}
    assert word2idx("AAA", dummy_words) == 0


def test_dna2rna():
    """Check conversion of DNA to RNA nucleotides."""
    # no 'T' to convert
    assert dna2rna("AAA") == "AAA"
    assert dna2rna("ACG") == "ACG"
    # conversion of 'T' to 'U'
    assert dna2rna("AAT") == "AAU"
    assert dna2rna("TTT") == "UUU"
    # conversion of unknown nucleotides
    assert dna2rna("AAX") == "AAN"
    assert dna2rna("XXX") == "NNN"


def test_rna2vec_complete_conversion():
    """Check complete conversion of RNA sequences"""
    # test sequences with known outcomes
    sequences = ["AAAA", "ACGT", "ACGU", "GGGN"]
    result = rna2vec(sequences)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int32
    assert result.shape[0] == len(sequences)
    assert result.shape[1] == 275  # default `max_sequence_length`

    nucleotides = ["A", "C", "G", "U", "N"]
    words = {
        "".join(triplet): i + 1
        for i, triplet in enumerate(product(nucleotides, repeat=3))
    }

    # 'AAAA' -> triplets: 'AAA'
    expected_aaa = words["AAA"]
    assert result[0][0] == expected_aaa
    assert result[0][1] == expected_aaa
    assert np.all(result[0][2:] == 0)  # rest should be padding

    # 'ACGT' -> 'ACGU' -> triplets: 'ACG', 'CGU'
    exoecred_acg = words["ACG"]
    exoecred_cgu = words["CGU"]
    assert result[1][0] == exoecred_acg
    assert result[1][1] == exoecred_cgu
    assert np.all(result[0][2:] == 0)  # rest should be padding

    # 'ACGU' -> triplets: 'ACG', 'CGU'
    assert result[2][0] == exoecred_acg
    assert result[2][1] == exoecred_cgu
    assert np.all(result[0][2:] == 0)  # rest should be padding

    # 'GGGX' -> 'GGGN' -> triplets: 'GGG', 'GGN'
    expected_ggg_index = words["GGG"]
    expected_ggn_index = words["GGN"]
    assert result[3][0] == expected_ggg_index
    assert result[3][1] == expected_ggn_index
    assert np.all(result[0][2:] == 0)  # rest should be padding
