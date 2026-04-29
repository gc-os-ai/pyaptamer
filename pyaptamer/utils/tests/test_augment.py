"""Test suite for the data augmentation utilities."""

__author__ = ["nennomp"]

import numpy as np

from pyaptamer.utils._augment import augment_reverse


def test_augment_reverse_single_array():
    # "AUCG" -> reverse "GCUA" -> complement "CGAU"
    # "AAAA" -> reverse "AAAA" -> complement "UUUU"
    # "GCGC" -> reverse "CGCG" -> complement "GCGC" (palindrome)
    sequences = np.array(["AUCG", "AAAA", "GCGC"])
    result = augment_reverse(sequences)

    expected = (np.array(["AUCG", "AAAA", "GCGC", "CGAU", "UUUU", "GCGC"]),)
    assert len(result) == 1
    assert len(result[0]) == 6
    np.testing.assert_array_equal(result[0], expected[0])


def test_augment_reverse_multiple_arrays():
    # "AAUG" -> reverse "GUAA" -> complement "CAUU"
    # "CCGG" -> reverse "GGCC" -> complement "CCGG"
    seq1 = np.array(["AAUG", "CCGG"])
    # "UGCA" -> reverse "ACGU" -> complement "UGCA"
    seq2 = np.array(["UGCA"])

    result = augment_reverse(seq1, seq2)

    expected = (
        np.array(["AAUG", "CCGG", "CAUU", "CCGG"]),
        np.array(["UGCA", "UGCA"]),
    )
    assert len(result) == 2
    assert len(result[0]) == 4
    assert len(result[1]) == 2

    np.testing.assert_array_equal(result[0], expected[0])
    np.testing.assert_array_equal(result[1], expected[1])
