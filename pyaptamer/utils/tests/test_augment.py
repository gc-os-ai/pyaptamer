"""Test suite for the data augmentation utilities."""

__author__ = ["nennomp"]

import numpy as np

from pyaptamer.utils._augment import augment_reverse


def test_augment_reverse_single_array():
    sequences = np.array(["AAC", "BBB", "ATCG"])
    result = augment_reverse(sequences)

    expected = (np.array(["AAC", "BBB", "ATCG", "CAA", "BBB", "GCTA"]),)
    assert len(result) == 1
    assert len(result[0]) == 6
    np.testing.assert_array_equal(result[0], expected[0])


def test_augment_reverse_multiple_arrays():
    seq1 = np.array(["ABC", "DEF"])
    seq2 = np.array(["XYZ"])
    seq3 = np.array(["123", "456", "789"])

    result = augment_reverse(seq1, seq2, seq3)

    expected = (
        np.array(["ABC", "DEF", "CBA", "FED"]),
        np.array(["XYZ", "ZYX"]),
        np.array(["123", "456", "789", "321", "654", "987"]),
    )
    assert len(result) == 3
    assert len(result[0]) == 4
    assert len(result[1]) == 2
    assert len(result[2]) == 6

    np.testing.assert_array_equal(result[0], expected[0])
    np.testing.assert_array_equal(result[1], expected[1])
    np.testing.assert_array_equal(result[2], expected[2])
