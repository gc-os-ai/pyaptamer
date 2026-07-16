"""Test suite for the GreedyEncoder."""

__author__ = ["Ishiezz"]

import pandas as pd
import pytest

from pyaptamer.trafos.encode._greedy import GreedyEncoder


def _make_df(sequences):
    """Helper to create input DataFrame."""
    return pd.DataFrame({"seq": [list(seq) for seq in sequences]})


def test_greedy_encoder_basic():
    """Check basic encoding with single-character words."""
    words = {"A": 1, "C": 2, "G": 3, "U": 4}
    enc = GreedyEncoder(words=words, max_len=4)
    X = _make_df(["ACGU"])
    result = enc.fit_transform(X)

    assert result.shape == (1, 4)
    assert list(result.iloc[0]) == [1, 2, 3, 4]


def test_greedy_encoder_longest_match():
    """Check greedy longest-match preference."""
    words = {"A": 1, "AC": 2, "G": 3}
    enc = GreedyEncoder(words=words, max_len=3)
    X = _make_df(["ACG"])
    result = enc.fit_transform(X)

    # "AC" should be matched over "A"
    assert list(result.iloc[0]) == [2, 3, 0]


def test_greedy_encoder_unknown_token():
    """Check that unknown characters produce token 0."""
    words = {"A": 1, "C": 2}
    enc = GreedyEncoder(words=words, max_len=4)
    X = _make_df(["AXCX"])
    result = enc.fit_transform(X)

    assert list(result.iloc[0]) == [1, 0, 2, 0]


def test_greedy_encoder_padding():
    """Check that sequences shorter than max_len are zero-padded."""
    words = {"A": 1, "C": 2}
    enc = GreedyEncoder(words=words, max_len=5)
    X = _make_df(["AC"])
    result = enc.fit_transform(X)

    assert result.shape == (1, 5)
    assert list(result.iloc[0]) == [1, 2, 0, 0, 0]


def test_greedy_encoder_truncation():
    """Check that sequences longer than max_len are truncated."""
    words = {"A": 1, "C": 2, "G": 3, "U": 4}
    enc = GreedyEncoder(words=words, max_len=2)
    X = _make_df(["ACGU"])
    result = enc.fit_transform(X)

    assert result.shape == (1, 2)
    assert list(result.iloc[0]) == [1, 2]


def test_greedy_encoder_multiple_sequences():
    """Check encoding of multiple sequences."""
    words = {"A": 1, "C": 2, "G": 3, "U": 4}
    enc = GreedyEncoder(words=words, max_len=3)
    X = _make_df(["ACG", "UAC"])
    result = enc.fit_transform(X)

    assert result.shape == (2, 3)
    assert list(result.iloc[0]) == [1, 2, 3]
    assert list(result.iloc[1]) == [4, 1, 2]


def test_greedy_encoder_empty_words():
    """Check that empty words dict raises ValueError at init."""
    with pytest.raises(ValueError, match="`words` must not be empty"):
        GreedyEncoder(words={})
