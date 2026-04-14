"""Tests for FCSWordTransformer."""

import numpy as np
import pandas as pd
import pytest
from pyaptamer.trafos.encode import FCSWordTransformer


def test_fcs_transformer_basic():
    """Test basic fit and transform with a simple dataset."""
    X = pd.DataFrame({"seq": ["ABC", "ABD", "ABC"]})
    transformer = FCSWordTransformer(k_max=2)
    transformer.fit(X)

    # Check that counts_ exists and contains expected k-mers
    assert "AB" in transformer.counts_
    assert "BC" in transformer.counts_
    assert "BD" in transformer.counts_
    assert "A" in transformer.counts_
    
    # Check words_ indices (1-indexed)
    assert len(transformer.words_) > 0
    for idx in transformer.words_.values():
        assert idx > 0

    # Transform
    Xt = transformer.transform(X)
    assert isinstance(Xt, pd.DataFrame)
    assert Xt.shape[0] == 3
    assert Xt.shape[1] >= 2  # Padded to max len


def test_fcs_transformer_filtering():
    """Test that below-average frequency words are filtered out."""
    # "A": 10, "B": 1 -> mean=5.5 -> B should be filtered out
    X = pd.DataFrame({"seq": ["A" * 10, "B"]})
    transformer = FCSWordTransformer(k_max=1)
    transformer.fit(X)
    
    # "A" should be in words_, "B" should not
    assert "A" in transformer.words_
    assert "B" not in transformer.words_


def test_fcs_transformer_greedy_match():
    """Test greedy longest-match tokenization."""
    # Setup words manually to ensure predictable greedy behavior
    X = pd.DataFrame({"seq": ["ABC"]})
    transformer = FCSWordTransformer(k_max=3)
    transformer.fit(X)
    
    # If "ABC" is in words, it should be matched as a single token
    if "ABC" in transformer.words_:
        Xt = transformer.transform(X)
        # Sequence length 1 after tokenization
        # (excluding padding zeros if any)
        row = Xt.iloc[0].values
        non_zero = row[row != 0]
        assert len(non_zero) == 1
        assert non_zero[0] == transformer.words_["ABC"]


def test_fcs_transformer_paddiing():
    """Test padding behavior."""
    X = pd.DataFrame({"seq": ["ABC", "A"]})
    transformer = FCSWordTransformer(k_max=1)
    transformer.fit(X)
    Xt = transformer.transform(X)
    
    assert Xt.shape == (2, 3)
    assert (Xt.iloc[1, 1:] == 0).all()


def test_fcs_transformer_max_len():
    """Test max_len truncation."""
    X = pd.DataFrame({"seq": ["ABCDE"]})
    transformer = FCSWordTransformer(k_max=1, max_len=3)
    transformer.fit(X)
    Xt = transformer.transform(X)
    
    assert Xt.shape == (1, 3)
