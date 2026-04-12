"""Tests for GreedyEncoder."""

import pandas as pd

from pyaptamer.trafos.encode import GreedyEncoder

WORDS = {"A": 1, "C": 2, "G": 3, "U": 4, "AC": 5, "GU": 6}


def test_get_test_params_instantiation():
    """Both param sets from get_test_params() must produce a valid instance.

    Regression test for issue #376: param0 was missing max_len,
    causing a TypeError on instantiation.
    """
    params = GreedyEncoder(words=WORDS).get_test_params()
    for i, param in enumerate(params):
        enc = GreedyEncoder(**param)
        assert isinstance(enc, GreedyEncoder), f"param{i} failed to instantiate"


def test_max_len_none_by_default():
    """max_len should default to None and pad to longest sequence length."""
    enc = GreedyEncoder(words=WORDS)
    assert enc.max_len is None

    # "ACGU" -> greedy encodes as ["AC", "GU"] = 2 tokens
    # "A"    -> 1 token, padded to 2
    X = pd.DataFrame([["ACGU"], ["A"]])
    result = enc.fit_transform(X)

    # output length = longest encoded sequence = 2
    assert result.shape[1] == 2
    # shorter sequence should be zero-padded
    assert result.iloc[1, 1] == 0


def test_max_len_truncates():
    """Sequences longer than max_len should be truncated."""
    enc = GreedyEncoder(words=WORDS, max_len=2)
    X = pd.DataFrame([["ACGU"]])
    result = enc.fit_transform(X)
    assert result.shape[1] == 2


def test_max_len_pads():
    """Sequences shorter than max_len should be zero-padded."""
    enc = GreedyEncoder(words=WORDS, max_len=6)
    X = pd.DataFrame([["AC"]])
    result = enc.fit_transform(X)
    assert result.shape[1] == 6
    assert result.iloc[0, 2:].eq(0).all()


def test_unknown_token_maps_to_zero():
    """Characters not in words dict should map to unknown token (0)."""
    enc = GreedyEncoder(words={"A": 1}, max_len=3)
    X = pd.DataFrame([["AXA"]])
    result = enc.fit_transform(X)
    assert result.iloc[0, 1] == 0  # X is unknown
