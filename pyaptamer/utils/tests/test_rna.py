"""Test suite for the RNA to vector utilities."""

__author__ = ["nennomp"]

from itertools import product

import numpy as np
import pytest
import torch

from pyaptamer.utils import dna2rna, encode_rna, generate_all_aptamer_triplets, rna2vec


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


def test_generate_all_aptamer_triplets():
    """Check generation of all possible 3-mer RNA subsequences (triplets)."""
    nucleotides = ["A", "C", "G", "U", "N"]
    words = generate_all_aptamer_triplets(letters=nucleotides)
    expected_count = len(nucleotides) ** 3  # 5^3 = 125 triplets

    assert isinstance(words, dict)
    assert len(words) == expected_count

    # check that all combinations are present
    for triplet in product(nucleotides, repeat=3):
        triplet_str = "".join(triplet)
        assert triplet_str in words
        assert isinstance(words[triplet_str], int)


def test_rna2vec_rna_sequences():
    """Check conversion of RNA sequences."""
    # test sequences with known outcomes
    sequences = ["AAAA", "ACGT", "ACGU", "GGGN"]
    result = rna2vec(sequences, sequence_type="rna", max_sequence_length=275)

    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(sequences)
    assert result.shape[1] == 275  # default `max_sequence_length`

    nucleotides = ["A", "C", "G", "U", "N"]
    words = {
        "".join(triplet): i + 1
        for i, triplet in enumerate(product(nucleotides, repeat=3))
    }

    # 'AAAA' -> triplets: 'AAA', 'AAA'
    expected_aaa = words["AAA"]
    assert result[0][0] == expected_aaa
    assert result[0][1] == expected_aaa
    assert np.all(result[0][2:] == 0)  # rest should be padding

    # 'ACGT' -> 'ACGU' -> triplets: 'ACG', 'CGU'
    expected_acg = words["ACG"]
    expected_cgu = words["CGU"]
    assert result[1][0] == expected_acg
    assert result[1][1] == expected_cgu
    assert np.all(result[1][2:] == 0)  # rest should be padding

    # 'ACGU' -> triplets: 'ACG', 'CGU'
    assert result[2][0] == expected_acg
    assert result[2][1] == expected_cgu
    assert np.all(result[2][2:] == 0)  # rest should be padding

    # 'GGGN' -> triplets: 'GGG', 'GGN'
    expected_ggg_index = words["GGG"]
    expected_ggn_index = words["GGN"]
    assert result[3][0] == expected_ggg_index
    assert result[3][1] == expected_ggn_index
    assert np.all(result[3][2:] == 0)  # rest should be padding


def test_rna2vec_secondary_structure():
    """Check conversion of secondary structure sequences."""
    # test secondary structure sequences
    sequences = ["SSSS", "SSHH", "SSHHM"]
    result = rna2vec(sequences, sequence_type="ss", max_sequence_length=10)

    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(sequences)
    assert result.shape[1] == 10

    ss_letters = ["S", "H", "M", "I", "B", "X", "E"]
    words = {
        "".join(triplet): i + 1
        for i, triplet in enumerate(product(ss_letters, repeat=3))
    }

    # 'SSSS' -> triplets: 'SSS', 'SSS'
    expected_sss = words["SSS"]
    assert result[0][0] == expected_sss
    assert result[0][1] == expected_sss
    assert np.all(result[0][2:] == 0)  # rest should be padding

    # 'SSHH' -> triplets: 'SSH', 'SHH'
    expected_ssh = words["SSH"]
    expected_shh = words["SHH"]
    assert result[1][0] == expected_ssh
    assert result[1][1] == expected_shh
    assert np.all(result[1][2:] == 0)  # rest should be padding

    # 'SSHHM' -> triplets: 'SSH', 'SHH', 'HHM'
    expected_hhm = words["HHM"]
    assert result[2][0] == expected_ssh
    assert result[2][1] == expected_shh
    assert result[2][2] == expected_hhm
    assert np.all(result[2][3:] == 0)  # rest should be padding


def test_rna2vec_edge_cases():
    """Check edge cases for RNA to vector conversion."""
    # `max_sequence_length` is <= 0
    with pytest.raises(
        ValueError, match="`max_sequence_length` must be greater than 0"
    ):
        rna2vec(["ACGU"], sequence_type="rna", max_sequence_length=0)
    with pytest.raises(
        ValueError, match="`max_sequence_length` must be greater than 0"
    ):
        rna2vec(["ACGU"], sequence_type="rna", max_sequence_length=-1)

    # invalid sequence_type
    with pytest.raises(
        ValueError, match="`sequence_type` must be either 'rna' or 'ss'"
    ):
        rna2vec(["ACGU"], sequence_type="invalid")

    # `max_sequence_length` is smaller than the number of triplets
    result = rna2vec(["AAACGU"], sequence_type="rna", max_sequence_length=4)
    assert result.shape[1] == 4  # should truncate to 4 triplets

    # empty sequence
    result = rna2vec([""], sequence_type="rna")
    assert len(result) == 0

    # single character sequence (can't form triplet)
    result = rna2vec(["A"], sequence_type="rna")
    assert len(result) == 0

    # double character sequence (can't form triplet)
    result = rna2vec(["AA"], sequence_type="rna")
    assert len(result) == 0

    # test with secondary structure sequences - edge cases
    result = rna2vec(["S"], sequence_type="ss")
    assert len(result) == 0

    result = rna2vec(["SS"], sequence_type="ss")
    assert len(result) == 0


def test_rna2vec_default_parameters():
    """Check that default parameters work correctly."""
    sequences = ["ACGU"]
    # test default sequence_type="rna"
    result = rna2vec(sequences)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 275)  # default max_sequence_length

    # test explicit defaults
    result_explicit = rna2vec(sequences, sequence_type="rna", max_sequence_length=275)
    assert np.array_equal(result, result_explicit)


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
