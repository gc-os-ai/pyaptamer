"""Tests for the AptamerOneHotEncoder transform."""

__author__ = ["aditi-dsi"]

import pandas as pd
import pytest
import torch

from pyaptamer.aptadiff import AptamerOneHotEncoder
from pyaptamer.data import MoleculeLoader

APTAMER = "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"


def test_aptamer_one_hot_encoder_from_moleculeloader():
    """A MoleculeLoader of aptamers is truncated, padded, and encoded."""
    X = MoleculeLoader(data={"aptamer": [APTAMER, APTAMER.lower()]})

    encoder = AptamerOneHotEncoder(target_length=45, pad_token="P")
    Xt = encoder.fit_transform(X)

    assert isinstance(Xt, torch.Tensor)
    assert Xt.shape == (2, 45, 5)
    assert Xt.dtype == torch.float32

    expected_pads = torch.tensor([4, 4, 4])
    assert torch.equal(Xt[0, 42:].argmax(dim=-1), expected_pads)

    assert torch.equal(Xt[0], Xt[1])

    encoder_trunc = AptamerOneHotEncoder(target_length=40)
    Xt_trunc = encoder_trunc.fit_transform(X)
    assert Xt_trunc.shape == (2, 40, 5)


def test_aptamer_one_hot_encoder_custom_columns():
    """Column names are configurable, not hardcoded to aptamer."""
    X = MoleculeLoader(data={"custom_seq": [APTAMER] * 2})
    encoder = AptamerOneHotEncoder(target_length=40, aptamer_col="custom_seq")
    Xt = encoder.fit_transform(X)

    assert len(Xt) == 2
    assert Xt.shape == (2, 40, 5)


def test_aptamer_one_hot_encoder_inverse_transform():
    """A numeric tensor decodes back to string sequences, stripping padding."""
    encoder = AptamerOneHotEncoder(target_length=45, pad_token="P")

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


def test_aptamer_one_hot_encoder_rejects_non_moleculeloader():
    """Only a MoleculeLoader is accepted; a plain DataFrame is rejected."""
    X = pd.DataFrame({"aptamer": [APTAMER]})
    with pytest.raises(TypeError, match="only a MoleculeLoader"):
        AptamerOneHotEncoder().fit_transform(X)
