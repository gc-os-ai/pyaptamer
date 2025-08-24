"""Test suite for aptatrans utilities."""

__author__ = ["nennomp"]

import pytest
import torch

from pyaptamer.utils._aptatrans_utils import encode_protein


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
def test_encode_protein(sequences, words, max_len, word_max_len, expected):
    """Check correct encoding of RNA sequences."""
    encoded = encode_protein(sequences, words, max_len, word_max_len)

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
