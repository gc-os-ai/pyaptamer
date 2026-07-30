"""Tests for the AptamerOneHotEncoder transform."""

__author__ = ["aditi-dsi"]

from pathlib import Path

import pandas as pd
import pytest
import torch

from pyaptamer.aptadiff import AptamerOneHotEncoder
from pyaptamer.data import MoleculeLoader

APTAMER = "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"
FASTQ_PATH = Path(__file__).parents[2] / "datasets" / "data" / "sample.fastq"
PAD_TOKEN = "P"


@pytest.fixture
def load_fastq():
    """Fixture to provide a standard MoleculeLoader utilizing the FASTQ file."""
    assert FASTQ_PATH.exists(), f"FASTQ file not found at {FASTQ_PATH}"
    return MoleculeLoader(data={"aptamer": [str(FASTQ_PATH)]}, tiling="samples")


@pytest.mark.parametrize("target_length", [45, 40])
def test_aptamer_one_hot_encoder_from_moleculeloader(target_length):
    """A MoleculeLoader of aptamers is truncated, padded, and encoded."""
    X = MoleculeLoader(data={"aptamer": [APTAMER, APTAMER.lower()]})

    encoder = AptamerOneHotEncoder(target_length=target_length, pad_token=PAD_TOKEN)
    Xt = encoder.fit_transform(X)

    assert isinstance(Xt, torch.Tensor)
    assert Xt.shape == (2, target_length, 5)
    assert Xt.dtype == torch.float32

    if target_length == 45:
        expected_pads = torch.tensor([4, 4, 4])
        assert torch.equal(Xt[0, 42:].argmax(dim=-1), expected_pads)
        assert torch.equal(Xt[0], Xt[1])


@pytest.mark.parametrize("target_length", [50, 150])
def test_aptamer_one_hot_encoder_from_fastq(load_fastq, target_length):
    """A FASTQ file loaded via MoleculeLoader is correctly parsed and encoded."""
    original_df = load_fastq.to_dataframe()
    num_seqs = len(original_df)

    encoder = AptamerOneHotEncoder(target_length=target_length, pad_token=PAD_TOKEN)
    Xt = encoder.fit_transform(load_fastq)

    assert isinstance(Xt, torch.Tensor)
    assert Xt.shape == (num_seqs, target_length, 5)
    assert Xt.dtype == torch.float32


def test_aptamer_one_hot_encoder_custom_columns():
    """Column names are configurable, not hardcoded to aptamer."""
    X = MoleculeLoader(data={"custom_seq": [APTAMER] * 2})
    encoder = AptamerOneHotEncoder(target_length=40, aptamer_col="custom_seq")
    Xt = encoder.fit_transform(X)

    assert len(Xt) == 2
    assert Xt.shape == (2, 40, 5)


def test_aptamer_one_hot_encoder_inverse_transform():
    """A numeric tensor decodes back to string sequences, stripping padding."""
    encoder = AptamerOneHotEncoder(target_length=45, pad_token=PAD_TOKEN)

    vocab = {"A": 0, "T": 1, "G": 2, "C": 3}
    real_indices = [vocab[char] for char in APTAMER]

    row_0 = real_indices + [4, 4, 4]

    row_1 = row_0.copy()
    row_1[10] = 99

    dummy_indices = torch.tensor([row_0, row_1])
    decoded_df = encoder.inverse_transform(dummy_indices)

    assert isinstance(decoded_df, pd.DataFrame)
    assert decoded_df["aptamer"].iloc[0] == APTAMER

    expected_mutated = APTAMER[:10] + "X" + APTAMER[11:]
    assert decoded_df["aptamer"].iloc[1] == expected_mutated


def test_aptamer_one_hot_encoder_inverse_transform_from_fastq(load_fastq):
    """
    A numeric tensor decodes back to string sequences matching the
    original FASTQ file.
    """
    encoder = AptamerOneHotEncoder(target_length=120, pad_token=PAD_TOKEN)
    Xt = encoder.fit_transform(load_fastq)

    decoded_df = encoder.inverse_transform(Xt)
    original_df = load_fastq.to_dataframe()
    num_seqs = len(original_df)

    assert isinstance(decoded_df, pd.DataFrame)
    assert len(decoded_df) == num_seqs

    for idx in range(num_seqs):
        assert decoded_df["aptamer"].iloc[idx] == original_df["aptamer"].iloc[idx]


def test_aptamer_one_hot_encoder_rejects_non_moleculeloader():
    """Only a MoleculeLoader is accepted; a plain DataFrame is rejected."""
    X = pd.DataFrame({"aptamer": [APTAMER]})
    with pytest.raises(TypeError, match="only a MoleculeLoader"):
        AptamerOneHotEncoder().fit_transform(X)
