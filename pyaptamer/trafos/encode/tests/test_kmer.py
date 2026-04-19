"""Tests for KMerEncoder."""

__author__ = ["Jayant-kernel"]

import numpy as np
import pandas as pd
import pytest

from pyaptamer.trafos.encode import KMerEncoder

APTAMER_SEQ = "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"
X = pd.DataFrame({"sequence": [APTAMER_SEQ, "ACGTACGT"]})


def test_kmer_encoder_output_shape_k4():
    """Check KMerEncoder output shape for k=4."""
    enc = KMerEncoder(k=4)
    Xt = enc.fit_transform(X)
    # sum(4^i for i in 1..4) = 4+16+64+256 = 340
    assert isinstance(Xt, pd.DataFrame)
    assert Xt.shape == (2, 340)


def test_kmer_encoder_output_shape_k1():
    """Check KMerEncoder output shape for k=1."""
    enc = KMerEncoder(k=1)
    Xt = enc.fit_transform(X)
    assert Xt.shape == (2, 4)


def test_kmer_encoder_output_shape_k2():
    """Check KMerEncoder output shape for k=2."""
    enc = KMerEncoder(k=2)
    Xt = enc.fit_transform(X)
    # 4 + 16 = 20
    assert Xt.shape == (2, 20)


def test_kmer_encoder_frequencies_sum_to_one():
    """Check that k-mer frequency rows sum to 1."""
    enc = KMerEncoder(k=2)
    Xt = enc.fit_transform(X)
    sums = Xt.values.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-5)


def test_kmer_encoder_output_dtype_float():
    """Check KMerEncoder output values are float."""
    enc = KMerEncoder(k=2)
    Xt = enc.fit_transform(X)
    assert np.issubdtype(Xt.values.dtype, np.floating)


def test_kmer_encoder_fit_is_empty():
    """Check fit() returns self without error (fit_is_empty tag)."""
    enc = KMerEncoder(k=2)
    result = enc.fit(X)
    assert result is enc


def test_kmer_encoder_preserves_index():
    """Check KMerEncoder preserves DataFrame index."""
    X_idx = pd.DataFrame({"sequence": [APTAMER_SEQ]}, index=[42])
    enc = KMerEncoder(k=1)
    Xt = enc.fit_transform(X_idx)
    assert list(Xt.index) == [42]


@pytest.mark.parametrize("k", [1, 2, 3])
def test_kmer_encoder_different_k(k):
    """Check KMerEncoder works for k=1, 2, 3."""
    enc = KMerEncoder(k=k)
    Xt = enc.fit_transform(X)
    expected_cols = sum(4**i for i in range(1, k + 1))
    assert Xt.shape == (2, expected_cols)
