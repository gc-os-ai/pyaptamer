"""Tests for AptaNet utility helpers."""

__author__ = ["nennomp"]

import numpy as np
import pandas as pd
import pytest

from pyaptamer.utils._aptanet_utils import generate_kmer_vecs, pairs_to_features


def test_pairs_to_features_happy_path():
    """Valid pairs should be converted to a float32 feature matrix."""
    pairs = [
        ("ATGC", "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQ"),
        ("AAAA", "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQ"),
    ]

    feats = pairs_to_features(pairs, k=2)

    assert isinstance(feats, np.ndarray)
    assert feats.dtype == np.float32
    assert feats.shape[0] == 2


def test_pairs_to_features_rejects_empty_input():
    """Empty input should raise a clear ValueError."""
    with pytest.raises(ValueError, match="requires at least one pair"):
        pairs_to_features([])


@pytest.mark.parametrize(
    "pairs",
    [
        [("ATGC",)],
        [("ATGC", "ACDE", "EXTRA")],
        [("ATGC", 123)],
        [("ATGC", None)],
        ["ATGC"],
    ],
)
def test_pairs_to_features_rejects_malformed_pairs(pairs):
    """Malformed input pairs should raise a clear ValueError."""
    with pytest.raises(ValueError, match="Each input pair must contain"):
        pairs_to_features(pairs)


def test_pairs_to_features_rejects_missing_dataframe_columns():
    """DataFrame input must provide both aptamer and protein columns."""
    df = pd.DataFrame({"aptamer": ["ATGC"]})

    with pytest.raises(ValueError, match="missing required column"):
        pairs_to_features(df)


def test_pairs_to_features_accepts_dataframe_input():
    """DataFrame input with required columns should still work."""
    df = pd.DataFrame(
        {
            "aptamer": ["ATGC", "AAAA"],
            "protein": [
                "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQ",
                "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQ",
            ],
        }
    )

    feats = pairs_to_features(df, k=2)

    assert feats.shape[0] == 2
    assert feats.dtype == np.float32


def test_generate_kmer_vecs_basic_smoke():
    """A small smoke test keeps the k-mer helper covered."""
    vec = generate_kmer_vecs("ATGC", k=2)
    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
