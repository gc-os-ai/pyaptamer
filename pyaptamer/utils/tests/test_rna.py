"""Test suite for the RNA to vector utilities."""

__author__ = ["nennomp"]

from itertools import product

import numpy as np
import pytest
import torch

from pyaptamer.utils import dna2rna, rna2vec

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
def test_encode_rna(sequences, words, max_len, word_max_len, expected):
    """Check correct encoding of RNA sequences."""
    encoded = encode_rna(sequences, words, max_len, word_max_len)

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
