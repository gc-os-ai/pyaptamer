"""Tests for the KMerFrequencies transformer."""

__author__ = ["siddharth7113"]

import numpy as np
import pandas as pd
import pytest

from pyaptamer.trafos.encode import KMerFrequencies


def _frame(*seqs):
    return pd.DataFrame({"seq": list(seqs)})


@pytest.mark.parametrize("k,width", [(2, 20), (4, 340)])
def test_width(k, width):
    """Width is the number of k-mers of length 1 to k over a 4-letter alphabet."""
    Xt = KMerFrequencies(k=k).fit_transform(_frame("ACGTACGT"))
    assert Xt.shape == (1, width)


def test_frequencies_by_hand():
    """k=1 on ACGTT gives the letter frequencies in alphabet order."""
    Xt = KMerFrequencies(k=1).fit_transform(_frame("ACGTT"))
    np.testing.assert_allclose(Xt.to_numpy()[0], [0.2, 0.2, 0.2, 0.4])


def test_k2_by_hand():
    """k=2 on ACGT counts 4 monomers and 3 dimers once each; a tail slice is not a dimer."""  # noqa: E501
    Xt = KMerFrequencies(k=2).fit_transform(_frame("ACGT"))
    expected = np.zeros(20)
    expected[[0, 1, 2, 3, 5, 10, 15]] = 1 / 7
    np.testing.assert_allclose(Xt.to_numpy()[0], expected)


def test_letters_outside_alphabet_not_counted():
    """Substrings containing a letter outside the alphabet do not contribute."""
    Xt = KMerFrequencies(k=1).fit_transform(_frame("ACGU"))
    np.testing.assert_allclose(Xt.to_numpy()[0], [1 / 3, 1 / 3, 1 / 3, 0.0])


def test_rna_alphabet():
    """alphabet="ACGU" counts uracil."""
    Xt = KMerFrequencies(k=1, alphabet="ACGU").fit_transform(_frame("ACGU"))
    np.testing.assert_allclose(Xt.to_numpy()[0], [0.25, 0.25, 0.25, 0.25])


def test_empty_sequence_is_all_zero():
    """A sequence with no countable k-mers gives a zero vector, not NaN."""
    Xt = KMerFrequencies(k=2).fit_transform(_frame(""))
    assert not Xt.isna().any().any()
    assert Xt.to_numpy().sum() == 0


def test_normalize_by_length():
    """normalize="length" divides every count by the sequence length."""
    Xt = KMerFrequencies(k=2, normalize="length").fit_transform(_frame("ACGT"))
    expected = np.zeros(20)
    expected[[0, 1, 2, 3, 5, 10, 15]] = 1 / 4
    np.testing.assert_allclose(Xt.to_numpy()[0], expected)


def test_normalize_by_length_empty_sequence_is_all_zero():
    """An empty sequence gives a zero vector under normalize="length"."""
    Xt = KMerFrequencies(k=2, normalize="length").fit_transform(_frame(""))
    assert not Xt.isna().any().any()
    assert Xt.to_numpy().sum() == 0


def test_unknown_normalize_raises():
    """A normalize value other than "total" or "length" is an error."""
    with pytest.raises(ValueError, match="normalize"):
        KMerFrequencies(normalize="rows").fit_transform(_frame("ACGT"))
