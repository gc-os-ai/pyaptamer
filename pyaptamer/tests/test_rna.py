"""Test suite for the RNA to vector (rna2vec) conversion utilities."""

__author__ = ["nennomp"]

from itertools import product

import numpy as np
import pytest

from pyaptamer.utils.rna import dna2rna, rna2vec


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
    expected_acg = words["ACG"]
    expected_cgu = words["CGU"]
    assert result[1][0] == expected_acg
    assert result[1][1] == expected_cgu
    assert np.all(result[1][2:] == 0)  # rest should be padding

    # 'ACGU' -> triplets: 'ACG', 'CGU'
    assert result[2][0] == expected_acg
    assert result[2][1] == expected_cgu
    assert np.all(result[2][2:] == 0)  # rest should be padding

    # 'GGGX' -> 'GGGN' -> triplets: 'GGG', 'GGN'
    expected_ggg_index = words["GGG"]
    expected_ggn_index = words["GGN"]
    assert result[3][0] == expected_ggg_index
    assert result[3][1] == expected_ggn_index
    assert np.all(result[3][2:] == 0)  # rest should be padding


def test_rna2vec_edge_cases():
    """Check edge cases for RNA to vector conversion."""
    # empty sequence
    result = rna2vec([""])
    assert len(result) == 0

    # single character sequence (can't form triplet)
    result = rna2vec(["A"])
    assert len(result) == 0

    # double character sequence (can't form triplet)
    result = rna2vec(["AA"])
    assert len(result) == 0
