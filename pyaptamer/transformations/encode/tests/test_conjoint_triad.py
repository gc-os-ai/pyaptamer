"""Tests for the ConjointTriad transformer."""

__author__ = ["siddharth7113"]

import numpy as np
import pandas as pd
import pytest

from pyaptamer.transformations.encode import ConjointTriad


def _frame(*seqs):
    return pd.DataFrame({"protein": list(seqs)})


@pytest.mark.parametrize("k,width", [(1, 7), (2, 56), (3, 399)])
def test_width(k, width):
    """Width is the number of k-mers of length 1 to k over the 7 groups."""
    Xt = ConjointTriad(k=k).fit_transform(_frame("ACDEFGHIKLMNPQRSTVWY"))
    assert Xt.shape == (1, width)


def test_one_residue_per_group():
    """One residue from each of the 7 groups gives 1/7 per group."""
    Xt = ConjointTriad(k=1).fit_transform(_frame("AIYHRDC"))
    np.testing.assert_allclose(Xt.to_numpy()[0], np.ones(7) / 7)


def test_same_group_residues_share_a_column():
    """A, G and V are one group, so AGV puts all weight on the first column."""
    Xt = ConjointTriad(k=1).fit_transform(_frame("AGV"))
    np.testing.assert_allclose(Xt.to_numpy()[0], [1, 0, 0, 0, 0, 0, 0])


def test_k2_normalized_by_length():
    """Counts are divided by the sequence length: AGV has 3 monomers and 2 dimers."""
    Xt = ConjointTriad(k=2).fit_transform(_frame("AGV"))
    expected = np.zeros(56)
    expected[0] = 3 / 3
    expected[7] = 2 / 3
    np.testing.assert_allclose(Xt.to_numpy()[0], expected)


def test_lowercase_is_uppercased():
    """Lower-case letters encode the same as upper-case."""
    lower = ConjointTriad().fit_transform(_frame("aiyhrdc"))
    upper = ConjointTriad().fit_transform(_frame("AIYHRDC"))
    pd.testing.assert_frame_equal(lower, upper)


@pytest.mark.parametrize("letter", ["B", "J", "O", "U", "X", "Z"])
def test_unknown_letter_counts_toward_length_only(letter):
    """A non amino acid letter is not counted but adds to the length."""
    Xt = ConjointTriad(k=1).fit_transform(_frame("A" + letter))
    np.testing.assert_allclose(Xt.to_numpy()[0], [0.5, 0, 0, 0, 0, 0, 0])


def test_empty_sequence_is_all_zero():
    """An empty sequence gives a zero vector, not NaN."""
    Xt = ConjointTriad(k=2).fit_transform(_frame(""))
    assert not Xt.isna().any().any()
    assert Xt.to_numpy().sum() == 0


def test_empty_batch_keeps_feature_width():
    """An empty frame retains the conjoint triad feature width."""
    X = pd.DataFrame({"protein": pd.Series([], dtype=object)})
    assert ConjointTriad(k=3).fit_transform(X).shape == (0, 399)
