"""Test suite for the data augmentation utilities."""

__author__ = ["nennomp"]

import numpy as np
import pytest

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


def test_augment_reverse_validation():
    """Check input validation for augment_reverse."""
    # No arguments should raise ValueError
    with pytest.raises(
        ValueError, match="At least one sequence array must be provided"
    ):
        augment_reverse()

    # Empty array should raise ValueError
    empty_array = np.array([])
    with pytest.raises(ValueError, match="is empty, at least one sequence required"):
        augment_reverse(empty_array)

    # Non-numpy array input should raise TypeError
    with pytest.raises(TypeError, match="All arguments must be numpy arrays"):
        augment_reverse(["ABC", "DEF"])

    with pytest.raises(TypeError, match="All arguments must be numpy arrays"):
        augment_reverse("ABC")

    with pytest.raises(TypeError, match="All arguments must be numpy arrays"):
        augment_reverse(None)

    with pytest.raises(TypeError, match="All arguments must be numpy arrays"):
        augment_reverse(np.array(["ABC"]), ["XYZ"])


def test_augment_reverse_single_element():
    """Test with single element arrays."""
    result = augment_reverse(np.array(["A"]))
    assert len(result) == 1
    assert len(result[0]) == 2
    np.testing.assert_array_equal(result[0], np.array(["A", "A"]))


def test_augment_reverse_empty_strings():
    """Test with arrays containing empty strings."""
    # Empty strings are valid, should work
    result = augment_reverse(np.array(["", "ABC"]))
    assert len(result) == 1
    assert len(result[0]) == 4  # original 2 + reversed 2
