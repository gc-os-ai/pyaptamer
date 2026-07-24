"""Tests for the AptamerOneHotEncoder transform."""

__author__ = ["aditi-dsi"]

import pandas as pd
import pytest
import torch

from pyaptamer.aptadiff import AptamerOneHotEncoder
from pyaptamer.data import MoleculeLoader
from pyaptamer.datasets import load_sample_fastq

APTAMER = "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"


def test_aptamer_one_hot_encoder_from_moleculeloader():
    """
    A MoleculeLoader of aptamers is correctly encoded into a tensor
    of shape (batch_size, num_classes, seq_len).
    """
    X = MoleculeLoader(data={"aptamer": [APTAMER, APTAMER.lower()]})

    encoder = AptamerOneHotEncoder()
    Xt = encoder.fit_transform(X)

    assert isinstance(Xt, torch.Tensor)
    assert Xt.shape == (2, 4, len(APTAMER))
    assert Xt.dtype == torch.float32
    assert torch.equal(Xt[0], Xt[1])


def test_aptamer_one_hot_encoder_from_fastq():
    """A FASTQ file loaded via MoleculeLoader is correctly parsed and encoded."""
    loader = load_sample_fastq()
    original_df = loader.to_dataframe()
    num_seqs = len(original_df)
    seq_len = len(original_df["aptamer"].iloc[0])

    encoder = AptamerOneHotEncoder()
    Xt = encoder.fit_transform(loader)

    assert isinstance(Xt, torch.Tensor)
    assert Xt.shape == (num_seqs, 4, seq_len)
    assert Xt.dtype == torch.float32


def test_aptamer_one_hot_encoder_custom_columns():
    """Column names are configurable, not hardcoded to aptamer."""
    X = MoleculeLoader(data={"custom_seq": [APTAMER] * 2})
    encoder = AptamerOneHotEncoder(aptamer_col="custom_seq")
    Xt = encoder.fit_transform(X)

    assert len(Xt) == 2
    assert Xt.shape == (2, 4, len(APTAMER))


def test_aptamer_one_hot_encoder_inverse_transform():
    """A numeric tensor decodes back to string sequences"""
    encoder = AptamerOneHotEncoder()

    vocab = {"A": 0, "T": 1, "G": 2, "C": 3}
    real_indices = [vocab[char] for char in APTAMER]

    dummy_indices = torch.tensor([real_indices])
    decoded_df = encoder.inverse_transform(dummy_indices)

    assert isinstance(decoded_df, pd.DataFrame)
    assert decoded_df["aptamer"].iloc[0] == APTAMER


def test_aptamer_one_hot_encoder_inverse_transform_from_fastq():
    """
    A numeric tensor decodes back to string sequences matching the
    original FASTQ file.
    """
    loader = load_sample_fastq()
    encoder = AptamerOneHotEncoder()
    Xt = encoder.fit_transform(loader)

    decoded_df = encoder.inverse_transform(Xt)
    original_df = loader.to_dataframe()
    num_seqs = len(original_df)

    assert isinstance(decoded_df, pd.DataFrame)
    assert len(decoded_df) == num_seqs

    for idx in range(num_seqs):
        assert decoded_df["aptamer"].iloc[idx] == original_df["aptamer"].iloc[idx]


def test_aptamer_one_hot_encoder_inverse_transform_unknown_token():
    """An out-of-bounds index safely decodes to the fallback token 'X'."""
    encoder = AptamerOneHotEncoder()

    vocab = {"A": 0, "T": 1, "G": 2, "C": 3}
    real_indices = [vocab[char] for char in APTAMER]

    row_with_unknown = real_indices.copy()
    row_with_unknown[10] = 99

    dummy_indices = torch.tensor([row_with_unknown])
    decoded_df = encoder.inverse_transform(dummy_indices)

    expected_mutated = APTAMER[:10] + "X" + APTAMER[11:]
    assert decoded_df["aptamer"].iloc[0] == expected_mutated


def test_aptamer_one_hot_encoder_rejects_non_moleculeloader():
    """Only a MoleculeLoader is accepted; a plain DataFrame is rejected."""
    X = pd.DataFrame({"aptamer": [APTAMER]})
    with pytest.raises(TypeError, match="only a MoleculeLoader"):
        AptamerOneHotEncoder().fit_transform(X)
