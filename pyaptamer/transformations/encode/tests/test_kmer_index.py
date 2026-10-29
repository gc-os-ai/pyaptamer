"""Tests for the KMerIndexEncoder transform."""

__author__ = ["siddharth7113"]

import pandas as pd
import pytest

from pyaptamer.transformations.encode import KMerIndexEncoder


def test_ids_follow_alphabet_product_order():
    """With alphabet AC and k=2 the ids are AA=1, AC=2, CA=3, CC=4."""
    Xt = KMerIndexEncoder(k=2, alphabet="AC").fit_transform(
        pd.DataFrame({"seq": ["aacca"]})
    )
    assert Xt.iloc[0].tolist() == [1, 2, 4, 3]


def test_kmer_outside_alphabet_gets_id_zero():
    """A k-mer with a letter outside the alphabet is encoded as 0."""
    Xt = KMerIndexEncoder(k=2, alphabet="AC").fit_transform(
        pd.DataFrame({"seq": ["ACXA"]})
    )
    assert Xt.iloc[0].tolist() == [2, 0, 0]


def test_default_width_is_longest_and_short_rows_stay():
    """Rows pad to the longest sequence; one shorter than k is a zero row."""
    X = pd.DataFrame({"seq": ["AAAA", "A", "AAA"]}, index=[10, 11, 12])
    Xt = KMerIndexEncoder(k=2, alphabet="AC").fit_transform(X)
    assert Xt.index.tolist() == [10, 11, 12]
    assert Xt.to_numpy().tolist() == [[1, 1, 1], [0, 0, 0], [1, 1, 0]]


def test_max_len_truncates():
    """Only the first max_len ids are kept."""
    Xt = KMerIndexEncoder(k=2, alphabet="AC", max_len=2).fit_transform(
        pd.DataFrame({"seq": ["AACC"]})
    )
    assert Xt.iloc[0].tolist() == [1, 2]


def test_unknown_replaces_letters_outside_alphabet():
    """With unknown set, a foreign letter encodes like the unknown letter."""
    with_n = KMerIndexEncoder(k=3, alphabet="ACGTN", unknown="N").fit_transform(
        pd.DataFrame({"seq": ["AC-GT"]})
    )
    expected = KMerIndexEncoder(k=3, alphabet="ACGTN").fit_transform(
        pd.DataFrame({"seq": ["ACNGT"]})
    )
    pd.testing.assert_frame_equal(with_n, expected)


def test_unknown_outside_alphabet_raises():
    """unknown must be a letter of alphabet."""
    with pytest.raises(ValueError, match="unknown"):
        KMerIndexEncoder(alphabet="ACGT", unknown="N").fit_transform(
            pd.DataFrame({"seq": ["ACGT"]})
        )
